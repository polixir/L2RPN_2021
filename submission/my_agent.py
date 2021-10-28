import copy
import os
import numpy as np
import random
import functools
import copy
from copy import deepcopy
import _pickle as cPickle
import datetime

from grid2op.Agent import BaseAgent

import torch
import torch.nn as nn
import torch.nn.functional as F

class unitary_action_network(nn.Module):
    def __init__(self, params_dict, scalar_dict):
        super(unitary_action_network, self).__init__()

        mean = scalar_dict["mean"]
        std = scalar_dict["std"]

        self.zero_std = np.where(std == 0.0)[0]
        self.mean = np.array([mean[i] for i in range(len(mean)) if i not in self.zero_std])
        self.std = np.array([std[i] for i in range(len(std)) if i not in self.zero_std])

        self.month_embedding = nn.Embedding(12, 64, _weight=torch.Tensor(params_dict['emb_month.w_0']))
        self.hour_embedding = nn.Embedding(24, 64, _weight=torch.Tensor(params_dict['emb_hour.w_0']))

        self.fc1 = nn.Linear(267, 512)
        self.fc1.weight.data = torch.Tensor(params_dict['fc1.w_0']).T
        self.fc1.bias.data = torch.Tensor(params_dict['fc1.b_0'])

        self.fc2 = nn.Linear(512, 512)
        self.fc2.weight.data = torch.Tensor(params_dict['fc2.w_0']).T
        self.fc2.bias.data = torch.Tensor(params_dict['fc2.b_0'])

        self.fc3 = nn.Linear(512, 512)
        self.fc3.weight.data = torch.Tensor(params_dict['fc3.w_0']).T
        self.fc3.bias.data = torch.Tensor(params_dict['fc3.b_0'])

        self.fc4 = nn.Linear(512, 512)
        self.fc4.weight.data = torch.Tensor(params_dict['fc4.w_0']).T
        self.fc4.bias.data = torch.Tensor(params_dict['fc4.b_0'])

        self.fc5 = nn.Linear(512, 512)
        self.fc5.weight.data = torch.Tensor(params_dict['fc5.w_0']).T
        self.fc5.bias.data = torch.Tensor(params_dict['fc5.b_0'])

        self.fc6 = nn.Linear(512, 500)
        self.fc6.weight.data = torch.Tensor(params_dict['fc6.w_0']).T
        self.fc6.bias.data = torch.Tensor(params_dict['fc6.b_0'])

    def feature_process(self, raw_obs):
        obs = raw_obs.to_dict()

        loads = []
        for key in ['q', 'v']:
            loads.append(obs['loads'][key])
        loads = np.concatenate(loads)

        prods = []
        for key in ['q', 'v']:
            prods.append(obs['prods'][key])
        prods = np.concatenate(prods)

        features = np.concatenate([loads, prods])
        features = np.array([features[i] for i in range(len(features)) if i not in self.zero_std])
        norm_features = (features - self.mean) / self.std

        rho = obs['rho']

        time_info = np.array([raw_obs.month - 1, raw_obs.hour_of_day])

        return np.concatenate([norm_features, rho, time_info]).tolist()

    def forward(self, raw_obs):

        obs_vec = np.array(self.feature_process(raw_obs)).reshape([1, -1])
        dense_input = torch.tensor(obs_vec[:, :-2], dtype=torch.float32)
        month = torch.tensor(obs_vec[:, -2], dtype=torch.int32)
        hour = torch.tensor(obs_vec[:, -1], dtype=torch.int32)

        month_emb = self.month_embedding(month)
        hour_emb = self.hour_embedding(hour)
        obs_emb = torch.cat([dense_input, month_emb, hour_emb], axis=1)

        x = F.relu(self.fc1(obs_emb))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        predicts = self.fc6(x)

        return predicts.detach().cpu().numpy()

class Track1PowerNetAgent(BaseAgent):
    def __init__(self, env, submission_path):
        BaseAgent.__init__(self, action_space=env.action_space)
        # global action
        self.simulate_times = 0
        self.alive_step = 0
        self.action_space = env.action_space
        self.last_step = datetime.datetime.now()

        # construct action_to_sub_topo
        offset = 59
        self.action_to_sub_topo = {}
        for sub_id, sub_elem_num in enumerate(self.action_space.sub_info):
            self.action_to_sub_topo[sub_id] = (offset, offset + sub_elem_num)
            offset += sub_elem_num

        # ============================= load baidu unitary action 500 =======================
        unitary_actions_vec = np.load(os.path.join(submission_path, "unitary_action.npz"), 'r')["actions"]
        self.unitary_actions = [self.action_space.from_vect(action) for action in list(unitary_actions_vec)]

        params_dict = np.load(os.path.join(submission_path, 'unitary_action_network.npz'))
        scalar_dict = np.load(os.path.join(submission_path, 'unitary_action_scalar.npz'))
        self.unitary_es_agent = unitary_action_network(params_dict, scalar_dict)

        # construct action_to_sub_id
        self.action_to_sub_id = {}
        for act_idx, action in enumerate(self.unitary_actions):
            act_dict = action.impact_on_objects()
            if act_dict["redispatch"]['changed']:
                self.action_to_sub_id[act_idx] = "redispatch"
            elif act_dict["topology"]['changed']:
                self.action_to_sub_id[act_idx] = act_dict["topology"]["assigned_bus"][0]['substation']

        # ============================= load alarm action =======================
        self.alarms_lines_area = env.alarms_lines_area
        self.alarms_area_names = env.alarms_area_names
        self.alarm_overflow_flag = False
        self.alarm_cool_down = True
        self.alarm_cool_time = 3  # needed to be finetuned
        self.alarm_count = 0

        # ============================= load redispatch action 80 =======================
        self.redispatch_cnt = 0
        self.not_redispatch_month = [5, 6, 7, 8]
        self.max_redispatch_cnt = 3  # needed to be finetuned

        act_matrix = cPickle.load(open(os.path.join(submission_path, 'redispatch_action.pkl'), 'rb'))['act_matrix']
        self.redispatch_actions = [self.action_space.from_vect(act_matrix[i, :]) for i in range(len(act_matrix))]

        # ============================= define combo indexes ============================
        self.combo_actions = []
        self.allow_combo = True
        with open(os.path.join(submission_path, 'combos.txt'), 'r') as f:
            self.combo_indexes = [tuple(map(int, line.strip().split())) for line in f.readlines()]


    def act_without_alarm(self, observation):
        if len(self.combo_actions) > 0:
            combo_results = [(i, self.simulate_combo(observation, combo_indexes)) for i, combo_indexes in enumerate(self.combo_actions)]
            combo_results = sorted(combo_results, key=lambda x: x[1])
            if combo_results[0][1] < 2.0: # can live for extra steps
                best_combo = self.combo_actions[combo_results[0][0]]
                if len(best_combo) == 1:
                    self.combo_actions = []
                else:
                    self.combo_actions = [indexes[1:] for indexes in self.combo_actions if indexes[0] == best_combo[0]]
                action = self.get_action_from_index(observation, best_combo[0])
                if action is None:
                    self.combo_actions = []
                else:
                    return action
            else:
                self.combo_actions = []

        # if there is a disconnected line, then try to reconnect it
        action = self.reconnect_action(observation)
        if action is not None:
            return action

        # if there is no disconnected line, try to reset topology and redispatch
        if np.all(observation.topo_vect != -1):
            self.redispatch_cnt = 0

            self.sub_topo_dict = self.calc_sub_topo_dict(observation)
            # try to reset the topology to the orginal state
            action = self.reset_topology(observation)
            if action is not None:
                return action

            # # try to reset the redispatch to the orginal state
            # action = self.reset_redispatch(observation)
            # if action is not None:
            #     return action

        # if there is a overflow line, then try to fix it
        if np.any(observation.rho > 1.0):
            if (observation.line_status[45] == False or observation.line_status[56] == False):
                action, least_overflow_1, need_combo = self.unitary_actions_simulate_seq(observation)
                if need_combo and self.allow_combo:
                    self.allow_combo = False
                    # try combo action
                    combo_results = [(i, self.simulate_combo(observation, combo_indexes)) for i, combo_indexes in enumerate(self.combo_indexes)]
                    combo_results = sorted(combo_results, key=lambda x: x[1])
                    if combo_results[0][1] < 2.0: # can live for extra steps
                        best_combo = self.combo_indexes[combo_results[0][0]]
                        self.combo_actions = [indexes[1:] for indexes in self.combo_indexes if indexes[0] == best_combo[0]]
                        action = self.get_action_from_index(observation, best_combo[0])
                        if action is None: action = self.action_space({})
            else:
                action, least_overflow_1 = self.unitary_actions_simulate(observation)

            # if the action has not solved the overflow
            if least_overflow_1 > 1.0:
                self.alarm_overflow_flag = True
        else:
            action = self.action_space({})

        if (observation.line_status[45] == False or observation.line_status[56] == False):

            if observation.month not in self.not_redispatch_month:
                if action != self.action_space({}) \
                    and self.redispatch_cnt < self.max_redispatch_cnt \
                    and action.impact_on_objects()['topology']['changed']:

                    action, least_overflow_2 = self.combined_redispatch_simulate(observation, action)

            if observation.attention_budget[0] >= 2:
                self.alarm_overflow_flag = True

        return action

    def act(self, observation, reward, done):
        tnow = observation.get_time_stamp()
        if self.last_step + datetime.timedelta(minutes=5) != tnow:
            self.redispatch_cnt = 0
            self.alarm_cool_down = True
            self.alarm_count = 0
            self.allow_combo = True

        self.last_step = tnow

        self.alive_step += 1
        self.alarm_overflow_flag = False

        if not self.alarm_cool_down:
            self.alarm_count += 1
            if self.alarm_count >= self.alarm_cool_time:
                self.alarm_cool_down = True
                self.alarm_count = 0

        action = self.act_without_alarm(observation)

        if np.any(observation.rho > 1.0) or np.any(observation.line_status==False):
            if observation.attention_budget[0] >= 2:
                self.alarm_overflow_flag = True

        if self.alarm_overflow_flag and not observation.is_alarm_illegal and self.alarm_cool_down:
            zones_alert = self.get_region_alert(observation)
            action.raise_alarm = zones_alert
            self.alarm_cool_down = False
            self.alarm_count = 0

        # if this action will cause game over, the try disconnected action
        sim_obs, sim_reward, sim_done, sim_inf = observation.simulate(action)
        observation._obs_env._reset_to_orig_state()

        if sim_done:
            for dis_line in range(59): # from the left to the right
                try:
                    dis_action = self.action_space.disconnect_powerline(dis_line)
                    dis_obs, dis_reward, dis_done, dis_inf = observation.simulate(dis_action)
                    observation._obs_env._reset_to_orig_state()

                    if not dis_done:

                        # use redispatch action to decrease the overflow
                        dis_action, least_overflow_3 = self.combined_redispatch_simulate(observation, dis_action)

                        if self.alarm_overflow_flag and not observation.is_alarm_illegal and self.alarm_cool_down:
                            zones_alert = self.get_region_alert(observation)
                            dis_action.raise_alarm = zones_alert
                            self.alarm_cool_down = False
                            self.alarm_count = 0

                        return dis_action

                except BaseException:
                    print('disconnect_action error')
                    continue

        return action

    def calc_sub_topo_dict(self, observation):
        offset = 0
        sub_topo_dict = {}

        for sub_id, sub_elem_num in enumerate(observation.sub_info):
            sub_topo = observation.topo_vect[offset:offset + sub_elem_num]
            offset += sub_elem_num
            sub_topo_dict[sub_id] = sub_topo

        return sub_topo_dict

    def reconnect_action(self, observation):
        # reconnect the line if the line will not cause overflow
        # return a reconnect action or None
        disconnected_lines = np.where(observation.line_status == False)[0].tolist()

        for line_id in disconnected_lines:  # from the left to the right
            if observation.time_before_cooldown_line[line_id] == 0:
                action = self.action_space({"set_line_status": [(line_id, +1)]})

                try:
                    obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(action)
                    observation._obs_env._reset_to_orig_state()
                    if np.max(observation.rho) < 1.0 and np.max(obs_simulate.rho) >= 1.0:
                        continue

                except BaseException:
                    print('reconnect_action error')
                    continue

                finally:
                    self.simulate_times += 1

                return action

        return None

    def reset_topology(self, observation):
        # if there is no overflow then try to reset the topology
        if np.max(observation.rho) < 0.95:

            for sub_id, sub_elem_num in enumerate(observation.sub_info):
                sub_topo = self.sub_topo_dict[sub_id]

                if sub_id == 28:
                    sub_28_topo = np.array([2.0, 1.0, 2.0, 1.0, 1.0]).astype(np.int32)

                    if not np.all(sub_topo.astype(np.int32) == sub_28_topo.astype(np.int32)) \
                            and observation.time_before_cooldown_sub[sub_id] == 0:

                        act = self.action_space({
                            "set_bus": {
                                "substations_id": [(sub_id, sub_28_topo.astype(np.int32).tolist())]
                            }
                        })

                        try:
                            obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(act)
                            observation._obs_env._reset_to_orig_state()
                            if info_simulate['is_illegal'] or info_simulate['is_ambiguous']:
                                return None
                            if not done_simulate and obs_simulate is not None and not any(np.isnan(obs_simulate.rho)):
                                if np.max(obs_simulate.rho) < 0.95:
                                    return act

                        except BaseException:
                            print('reset_topology error')
                            continue

                        finally:
                            self.simulate_times += 1

                    continue

                if np.any(sub_topo == 2) and observation.time_before_cooldown_sub[sub_id] == 0:
                    sub_topo = np.where(sub_topo == 2, 1, sub_topo)  # bus 2 to bus 1
                    sub_topo = np.where(sub_topo == -1, 0, sub_topo)  # don't do action in bus=-1
                    reconfig_sub = self.action_space({
                        "set_bus": {
                            "substations_id": [(sub_id, sub_topo.astype(np.int32).tolist())]
                        }
                    })

                    try:
                        obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(reconfig_sub)
                        observation._obs_env._reset_to_orig_state()
                        if info_simulate['is_illegal'] or info_simulate['is_ambiguous']:
                            return None
                        if not done_simulate:
                            if not np.any(obs_simulate.topo_vect != observation.topo_vect):  # have some impact
                                return None
                        if not done_simulate and obs_simulate is not None and not any(np.isnan(obs_simulate.rho)):
                            if np.max(obs_simulate.rho) < 0.95:
                                return reconfig_sub

                    except BaseException:
                        print('reset_topology error')
                        return None

                    finally:
                        self.simulate_times += 1

        elif np.max(observation.rho) >= 1.0:
            sub_id = 28
            sub_topo = self.sub_topo_dict[sub_id]
            if np.any(sub_topo == 2) and observation.time_before_cooldown_sub[sub_id] == 0:
                sub_28_topo = np.array([1.0, 1.0, 1.0, 1.0, 1.0]).astype(int)
                act = self.action_space({
                    "set_bus": {
                        "substations_id": [(sub_id, sub_28_topo.astype(int))]
                    }
                })

                try:
                    obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(act)
                    observation._obs_env._reset_to_orig_state()
                    if info_simulate['is_illegal'] or info_simulate['is_ambiguous']:
                        return None
                    if not done_simulate and obs_simulate is not None and not any(np.isnan(obs_simulate.rho)):
                        if np.max(obs_simulate.rho) < 0.99:
                            return act

                except BaseException:
                    print('reset_topology error')
                    return None

        return None

    def reset_redispatch(self, observation):
        # if there is no overflow then try to reset the redispatch
        if np.max(observation.rho) < 0.95:
            # reset redispatch
            if not np.all(observation.target_dispatch == 0.0):
                gen_ids = np.where(observation.gen_redispatchable)[0]
                gen_ramp = observation.gen_max_ramp_up[gen_ids]
                changed_idx = np.where(observation.target_dispatch[gen_ids] != 0.0)[0]
                redispatchs = []
                for idx in changed_idx:
                    target_value = observation.target_dispatch[gen_ids][idx]

                    if abs(target_value) < 0.1 * gen_ramp[idx]:
                        value = -1 * target_value / abs(target_value) * abs(target_value)
                    else:
                        value = -1 * target_value / abs(target_value) * 0.1 * gen_ramp[idx]

                    # value = min(abs(target_value), gen_ramp[idx])
                    # value = -1 * target_value / abs(target_value) * value

                    redispatchs.append((gen_ids[idx], value))
                action = self.action_space({"redispatch": redispatchs})

                try:
                    obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(action)
                    observation._obs_env._reset_to_orig_state()
                    if info_simulate['is_illegal'] or info_simulate['is_ambiguous']:
                        return None
                    if not done_simulate and obs_simulate is not None and not any(np.isnan(obs_simulate.rho)):
                        if np.max(obs_simulate.rho) < 0.95:
                            return action

                except BaseException:
                    print('reset_redispatch error')
                    return None

                finally:
                    self.simulate_times += 1

    def get_action_from_index(self, observation, idx):
        # this function is used to get a legal action from the action space
        action = self.unitary_actions[idx]
        sub_id = self.action_to_sub_id[idx]
        if sub_id != "redispatch":  # topology change action
            if observation.time_before_cooldown_sub[int(sub_id)] != 0:
                return None

            legal_action_vec = np.array(action.to_vect()).copy()
            sub_topo = self.sub_topo_dict[int(sub_id)]

            if np.any(sub_topo == -1):  # line disconnected
                start, end = self.action_to_sub_topo[int(sub_id)]
                action_topo = legal_action_vec[start:end].astype(np.int32)  # reference
                action_topo[np.where(sub_topo == -1)[0]] = 0  # done't change bus=-1
                legal_action_vec[start:end] = action_topo
                legal_action = self.action_space.from_vect(legal_action_vec)

            else:
                legal_action = action

        else:
            legal_action = action

        return legal_action

    def make_simulatable_obs(self, new_obs, old_obs):
        # make the simulated obs simulatable
        new_obs.action_helper = old_obs.action_helper
        new_obs._obs_env = old_obs._obs_env

        # direct copy
        new_obs._forecasted_inj = old_obs._forecasted_inj

        # if len(new_obs._forecasted_inj) > 2:
        #     new_obs._forecasted_inj = new_obs._forecasted_inj[1:]

        # linear extrapolation of the injection
        # new_injection = {k : 2 * old_obs._forecasted_inj[1][1]['injection'][k] - old_obs._forecasted_inj[0][1]['injection'][k] for k in old_obs._forecasted_inj[1][1]['injection']}
        # new_datetime = old_obs._forecasted_inj[1][0] + datetime.timedelta(minutes=5)
        # new_obs._forecasted_inj = [old_obs._forecasted_inj[1], (new_datetime, {'injection' : new_injection})]

        return new_obs

    def simulate_combo(self, obs, combo_indexes):
        sim_obs = obs
        for index in combo_indexes:
            action = self.get_action_from_index(obs, index)
            if action is None: return 2.0
            sim_obs, _, done, _ = sim_obs.simulate(action)
            if done: return 2.0
            sim_obs = self.make_simulatable_obs(sim_obs, obs)
        max_rho = sim_obs.rho.max()
        return 2.0 if max_rho == 0 else max_rho

    def unitary_actions_simulate_seq(self, observation):
        # get a baseline by do nothing
        try:
            obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(self.action_space({}))
            observation._obs_env._reset_to_orig_state()

            least_overflow = 2.0
            if obs_simulate is not None and not any(np.isnan(obs_simulate.rho)):
                least_overflow = float(np.max(obs_simulate.rho))

        except BaseException:
            least_overflow = 2.0

        finally:
            self.simulate_times += 1

        if least_overflow < 1.0:
            return self.action_space({}), least_overflow, False

        # candidate in format (max_rho, action_sequence, simulatable_obs, expanded_flag)
        line_overflow = observation.rho >= 1.0
        candidates = [(observation.rho.max(), [], observation, False)]

        def compare(a, b):
            penalty_factor = 0.05

            a_score = a[0] + penalty_factor * len(a[1])
            b_score = b[0] + penalty_factor * len(b[1])

            a_solve = a[0] < 1.0
            b_solve = b[0] < 1.0

            if not a_solve == b_solve:
                return -1 if a_solve else 1

            a_fix_line = np.all(np.logical_or(a[2].rho >= 1.0, line_overflow))
            b_fix_line = np.all(np.logical_or(b[2].rho >= 1.0, line_overflow))

            if a_fix_line == b_fix_line:
                if a_score == b_score:
                    return 0
                else:
                    return -1 if a_score < b_score else 1
            else:
                return -1 if a_fix_line else 1

        planning_horizon = 4 - observation.timestep_overflow.max()
        for depth in range(planning_horizon):
            new_candidates = []
            expand_num = 150 if depth == 0 else 50
            select_num = 50 if depth == 0 else 50

            for candidate in candidates:
                current_max_rho, current_actions, simulatable_obs, expanded_flag = candidate

                if current_max_rho < 1.0:  # already in control, do not simulate further
                    new_candidates.append(candidate)
                    continue

                if expanded_flag == True:  # the candidate is already expanded
                    new_candidates.append(candidate)
                    continue
                else:
                    new_candidates.append((current_max_rho, current_actions, simulatable_obs, True))  # mark as expanded

                predicted_rho = self.unitary_es_agent(simulatable_obs)[0, :]  # 500
                sorted_idx = np.argsort(predicted_rho).tolist()
                top_idx = sorted_idx[:expand_num]
                top_idx.sort()

                for idx in top_idx:
                    legal_action = self.get_action_from_index(simulatable_obs, idx)
                    if legal_action is None: continue
                    if legal_action in current_actions: continue

                    obs_simulate, reward_simulate, done_simulate, info_simulate = simulatable_obs.simulate(legal_action)
                    observation._obs_env._reset_to_orig_state()

                    if info_simulate['is_illegal'] or info_simulate['is_ambiguous']:
                        continue

                    if obs_simulate is not None and not any(np.isnan(obs_simulate.rho)) and not done_simulate:
                        overflow_value = float(np.max(obs_simulate.rho))
                        action_sequence = deepcopy(current_actions)
                        action_sequence.append(legal_action)
                        new_candidates.append((overflow_value, action_sequence,
                                               self.make_simulatable_obs(obs_simulate, simulatable_obs), False))

                    self.simulate_times += 1

            # new_candidates = sorted(new_candidates, key=lambda x: x[0])
            new_candidates = sorted(new_candidates, key=functools.cmp_to_key(compare))
            candidates = new_candidates[:select_num]  # only consider the top 30

            if candidates[0][0] < 1.0: break


        if len(candidates) > 0:
            best_candidate = candidates[0]

            if len(best_candidate[1]) == 0:
                return self.action_space({}), least_overflow, False

            if best_candidate[0] >= least_overflow:
                return self.action_space({}), least_overflow, False

            best_action = best_candidate[1][0]
            simulated_obs, _, _, _ = observation.simulate(best_action)
            rho = simulated_obs.rho.max()

            return best_action, rho, best_candidate[0] >= 1.15 and planning_horizon == 3
        else:
            return self.action_space({}), least_overflow, False

    def unitary_actions_simulate(self, observation):

        self.simulate_times += 1

        try:
            obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(self.action_space({}))
            observation._obs_env._reset_to_orig_state()

            least_overflow = 2.0
            best_action = self.action_space({})
            if obs_simulate is not None and not any(np.isnan(obs_simulate.rho)):
                least_overflow = float(np.max(obs_simulate.rho))

        except BaseException:
            best_action = self.action_space({})
            least_overflow = 2.0

        finally:
            self.simulate_times += 1

        if least_overflow < 1.0:
            return best_action, least_overflow

        predicted_rho = self.unitary_es_agent(observation)[0, :] # 500
        sorted_idx = np.argsort(predicted_rho).tolist()
        top_idx = sorted_idx[:150]  # needed to be finetuned
        top_idx.sort()

        for idx in top_idx:
            action = self.unitary_actions[idx]
            sub_id = self.action_to_sub_id[idx]
            if sub_id != "redispatch":  # topology change action
                if observation.time_before_cooldown_sub[int(sub_id)] != 0:
                    continue

                legal_action_vec = np.array(action.to_vect()).copy()
                sub_topo = self.sub_topo_dict[int(sub_id)]

                if np.any(sub_topo == -1):  # line disconnected
                    start, end = self.action_to_sub_topo[int(sub_id)]
                    action_topo = legal_action_vec[start:end].astype(np.int32)  # reference
                    action_topo[np.where(sub_topo == -1)[0]] = 0  # done't change bus=-1
                    legal_action_vec[start:end] = action_topo
                    legal_action = self.action_space.from_vect(legal_action_vec)

                else:
                    legal_action = action

            else:
                legal_action = action

            try:
                obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(legal_action)
                observation._obs_env._reset_to_orig_state()
                if info_simulate['is_illegal'] or info_simulate['is_ambiguous']:
                    continue
                if obs_simulate is not None and not any(np.isnan(obs_simulate.rho)):
                    if not done_simulate:
                        overflow_value = float(np.max(obs_simulate.rho))
                        if overflow_value < least_overflow:
                            least_overflow = overflow_value
                            best_action = legal_action

            except BaseException:
                print('unitary_actions_simulate error')
                continue

            finally:
                self.simulate_times += 1

        return best_action, least_overflow

    def combined_redispatch_simulate(self, observation, action):

        obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(action)
        observation._obs_env._reset_to_orig_state()

        origin_rho = 10.0
        if not done_simulate:
            origin_rho = obs_simulate.rho.max()

        least_rho = origin_rho
        best_action = None
        for redispatch_action in self.redispatch_actions:
            try:
                redispatch_action = copy.deepcopy(redispatch_action)
                redispatch_action._curtail = np.zeros(22)
                action = copy.deepcopy(action)
                action._redispatch = np.zeros(22)

                combine_action = self.action_space.from_vect(action.to_vect() + redispatch_action.to_vect())

                obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(combine_action)
                observation._obs_env._reset_to_orig_state()

            except BaseException:
                print('redispatch_action error')
                continue

            max_rho = 10.0
            if not done_simulate:
                max_rho = obs_simulate.rho.max()
            if max_rho < least_rho:
                least_rho = max_rho
                best_action = combine_action

        if least_rho < origin_rho:
            action = best_action
            self.redispatch_cnt += 1

        return action, least_rho

    def get_region_alert(self, observation):
        # extract the zones they belong too
        zones_these_lines = set()
        zone_for_each_lines = self.alarms_lines_area

        lines_overloaded = np.where(observation.rho >= 1)[0].tolist()  # obs.rho>0.6
        # print(lines_overloaded)
        for line_id in lines_overloaded:
            line_name = observation.name_line[line_id]
            for zone_name in zone_for_each_lines[line_name]:
                zones_these_lines.add(zone_name)

        zones_these_lines = list(zones_these_lines)
        zones_ids_these_lines = [self.alarms_area_names.index(zone) for zone in zones_these_lines]
        return zones_ids_these_lines

def make_agent(env, submission_path):
    # my_agent = MyAgent(env.action_space,  env.alarms_lines_area, env.alarms_area_names, submission_path)
    my_agent = Track1PowerNetAgent(env, submission_path)
    return my_agent
