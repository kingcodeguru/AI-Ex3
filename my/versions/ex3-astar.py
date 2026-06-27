# astar version with ex2.py - gone terrably wrong
"""AI assistance disclosure: drafted with Gemini.

Blazingly Fast Dynamic A* / Expectimax Controller.
Uses O(N) tuple state transitions and lightweight internal action 
representations to maximize nodes-per-second while updating empirical 
probabilities at every step.
"""

import math
import heapq
import itertools
import time

import ext_elev

INF = float("inf")

id = ["123456789"]

class Controller:
    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self._max_steps = int(game.get_max_steps())
        self._initial_state = game.get_initial_state()
        self._goal_reward = float(game.get_goal_reward())

        self._capacities = game.get_capacities()
        self._reachable = {
            eid: tuple(sorted(floors))
            for eid, floors in game.get_reachable().items()
        }
        self._reachable_sets = {
            eid: frozenset(floors)
            for eid, floors in self._reachable.items()
        }

        _, persons_t, _ = self._initial_state
        self._person_ids = tuple(sorted(pid for pid, _ in persons_t))
        self._elevator_ids = tuple(sorted(self._capacities))

        self._person_weight = {
            pid: game.get_person_weight(pid) for pid in self._person_ids
        }
        self._person_goal = {
            pid: game.get_person_goal(pid) for pid in self._person_ids
        }
        self._full_targets = frozenset(self._person_ids)
        self._many_elevators = len(self._elevator_ids) >= 3

        # ---- Dynamic Probability Tracking ----
        self.att_move = {e: 0 for e in self._elevator_ids}
        self.ok_move = {e: 0 for e in self._elevator_ids}
        
        self.att_pax = {p: 0 for p in self._person_ids}
        self.ok_pax = {p: 0 for p in self._person_ids}
        
        self.rew_sum = {p: 0.0 for p in self._person_ids}
        self.rew_cnt = {p: 0 for p in self._person_ids}
        self.max_rew = 40.0
        
        self.t = 0
        self.last_state = None
        self.last_action_tuple = None

        # Caches
        self._state_info_cache = {}
        self._legal_action_cache = {}
        self._transition_cache = {}
        self._value_cache = {}
        self._policy_cache = {}
        self._heuristic_cache = {}
        self._plan_cache = {}

        self._shared_floors = set()
        _reaches = [self._reachable_sets[e] for e in self._elevator_ids]
        for _i in range(len(_reaches)):
            for _j in range(_i + 1, len(_reaches)):
                self._shared_floors |= (_reaches[_i] & _reaches[_j])

        self._plan_node_cap = 60000

    # ------------------------------------------------------------------
    # Dynamic Probability Estimators
    # ------------------------------------------------------------------
    def ep(self, eid):
        return (self.ok_move[eid] + 1.0) / (self.att_move[eid] + 2.0)

    def pp(self, pid):
        return (self.ok_pax[pid] + 1.0) / (self.att_pax[pid] + 2.0)

    def pr(self, pid):
        n = self.rew_cnt[pid]
        if n == 0:
            return self.max_rew
        mean = self.rew_sum[pid] / n
        bonus = math.sqrt(2.0 * math.log(max(2, self.t)) / n)
        return mean + (self.max_rew * 0.5 * bonus)

    # ------------------------------------------------------------------
    # Feedback Engine
    # ------------------------------------------------------------------
    def _update_stats(self, state):
        if not self.last_action_tuple or self.last_action_tuple[0] == "RESET": return
        
        kind = self.last_action_tuple[0]
        p1, p2 = self.last_action_tuple[1], self.last_action_tuple[2]
        
        last_e, last_p, last_rem = self.last_state
        curr_e, curr_p, curr_rem = state

        if kind == "MOVE":
            self.att_move[p1] += 1
            curr_f = next((f for e, f, w in curr_e if e == p1), None)
            if curr_f == p2:
                self.ok_move[p1] += 1
                
        elif kind == "ENTER":
            self.att_pax[p1] += 1
            curr_loc = next((loc for p, loc in curr_p if p == p1), None)
            if curr_loc == ("in", p2):
                self.ok_pax[p1] += 1
                
        elif kind == "EXIT":
            self.att_pax[p1] += 1
            curr_loc = next((loc for p, loc in curr_p if p == p1), None)
            delivered = (curr_loc is None) or (state == self._initial_state and last_rem == 1)
            
            if delivered or (curr_loc and curr_loc[0] == "floor"):
                self.ok_pax[p1] += 1
                
            if delivered:
                rew = float(self.game.get_last_gained_reward())
                if state == self._initial_state and last_rem == 1:
                    rew -= float(self.game.get_goal_reward())
                self.rew_sum[p1] += rew
                self.rew_cnt[p1] += 1
                self.max_rew = max(self.max_rew, rew)

    # ------------------------------------------------------------------
    # Main Decision Loop
    # ------------------------------------------------------------------
    def choose_next_action(self, state):
        steps_left = self._max_steps - self.game.get_current_steps()
        if steps_left <= 0: return "RESET"

        self._update_stats(state)
        self.t += 1
        self.last_state = state

        self._state_info_cache.clear()
        self._legal_action_cache.clear()
        self._transition_cache.clear()
        self._value_cache.clear()
        self._policy_cache.clear()
        self._heuristic_cache.clear()
        self._plan_cache.clear()

        self._precompute_plan_costs()
        self._precompute_move_costs()

        _min_w = min(self._person_weight.values()) if self._person_weight else 1
        _single = all(self._capacities[e] < 2 * _min_w for e in self._elevator_ids)
        _perfect = all(self.ep(e) >= 0.95 for e in self._elevator_ids)
        
        self._collapse_eligible = _single and _perfect
        mode = self._select_mode()
        self._collapse_moves = self._collapse_eligible and mode == "planner"

        if mode == "expectimax":
            action_tuple = self._expectimax_action(state, steps_left)
        else:
            ps = (state[0], state[1])
            plan = self._astar_plan(ps, self._full_targets)
            if plan and len(plan) <= steps_left:
                action_tuple = plan[0]
            else:
                action_tuple = self._expectimax_action(state, steps_left)

        self.last_action_tuple = action_tuple
        
        if action_tuple[0] == "RESET":
            return "RESET"
        return f"{action_tuple[0]}{{{action_tuple[1]},{action_tuple[2]}}}"

    def _select_mode(self):
        not_all_reliable = any(self.ep(e) < 0.95 for e in self._elevator_ids)
        if len(self._person_ids) > 3 or not not_all_reliable:
            return "planner"
        return "expectimax"

    # ------------------------------------------------------------------
    # Expectimax Core (ex2.py)
    # ------------------------------------------------------------------
    def _expectimax_action(self, state, steps_left):
        depth = self._choose_depth(state, steps_left)
        self._best_value(state, steps_left, depth)
        return self._policy_cache.get((state, steps_left, depth), ("RESET",))

    def _choose_depth(self, state, steps_left):
        remaining = state[2]
        if steps_left <= 2: return steps_left
        if remaining <= 1: return min(steps_left, 8)
        if remaining <= 2: return min(steps_left, 7)
        if remaining <= 3: return min(steps_left, 7)
        return min(steps_left, 5)

    def _best_value(self, state, steps_left, depth_left):
        key = (state, steps_left, depth_left)
        if key in self._value_cache: return self._value_cache[key]

        if steps_left <= 0:
            self._value_cache[key] = 0.0
            self._policy_cache[key] = ("RESET",)
            return 0.0

        if depth_left <= 0:
            val = self._heuristic(state, steps_left)
            self._value_cache[key] = val
            self._policy_cache[key] = ("RESET",)
            return val

        best_value = -INF
        best_action = ("RESET",)

        for action in self._legal_actions(state):
            q_value = self._action_value(state, action, steps_left, depth_left)
            if q_value > best_value:
                best_value = q_value
                best_action = action

        self._value_cache[key] = best_value
        self._policy_cache[key] = best_action
        return best_value

    def _action_value(self, state, action, steps_left, depth_left):
        outcomes = self._transition_outcomes(state, action)
        next_steps = steps_left - 1
        next_depth = depth_left - 1
        total = 0.0

        if self._collapse_moves and action[0] == "MOVE":
            for idx, (prob, next_state, imm_reward) in enumerate(outcomes):
                if prob <= 0.0: continue
                if next_steps <= 0: cont = 0.0
                elif idx == 0: cont = self._best_value(next_state, next_steps, next_depth)
                else: cont = self._heuristic(next_state, next_steps)
                total += prob * (imm_reward + cont)
            return total

        for prob, next_state, imm_reward in outcomes:
            if prob <= 0.0: continue
            cont = self._best_value(next_state, next_steps, next_depth) if next_steps > 0 else 0.0
            total += prob * (imm_reward + cont)

        if action[0] == "RESET":
            total -= self._reset_penalty(state)

        return total

    # ------------------------------------------------------------------
    # Legal Actions & Fast Tuple Transitions
    # ------------------------------------------------------------------
    def _decode_state(self, state):
        if state in self._state_info_cache: return self._state_info_cache[state]

        elevators_t, persons_t, remaining = state
        elevator_floor, elevator_load = {}, {}
        for eid, floor, load in elevators_t:
            elevator_floor[eid] = floor
            elevator_load[eid] = load

        person_loc, people_on_floor, people_in_elevator = {}, {}, {}
        for pid, loc in persons_t:
            person_loc[pid] = loc
            if loc[0] == "floor":
                people_on_floor.setdefault(loc[1], []).append(pid)
            else:
                people_in_elevator.setdefault(loc[1], []).append(pid)

        info = (elevator_floor, elevator_load, person_loc, people_on_floor, people_in_elevator, remaining)
        self._state_info_cache[state] = info
        return info

    def _legal_actions(self, state):
        if state in self._legal_action_cache: return self._legal_action_cache[state]

        elevator_floor, elevator_load, _, people_on_floor, people_in_elevator, _ = self._decode_state(state)
        actions = []

        for eid in self._elevator_ids:
            floor = elevator_floor[eid]
            inside = people_in_elevator.get(eid, [])
            goal_first, other = [], []
            for pid in inside:
                if floor == self._person_goal[pid]: goal_first.append(pid)
                else: other.append(pid)
            goal_first.sort(key=lambda pid: (-self.pr(pid), pid))
            other.sort()
            for pid in goal_first + other:
                actions.append(("EXIT", pid, eid))

        for eid in self._elevator_ids:
            floor = elevator_floor[eid]
            load = elevator_load[eid]
            candidates = [pid for pid in people_on_floor.get(floor, []) if load + self._person_weight[pid] <= self._capacities[eid]]
            candidates.sort(key=lambda pid: (-self.pr(pid), pid))
            for pid in candidates:
                actions.append(("ENTER", pid, eid))

        floor_scores = self._floor_scores(state)
        for eid in self._elevator_ids:
            floor = elevator_floor[eid]
            targets = []
            for target in self._reachable[eid]:
                score = floor_scores.get(target, 0.0) + (0.1 if target == floor else 0.0)
                targets.append((score, target))
            targets.sort(key=lambda item: (-item[0], item[1]))
            for _, target in targets:
                actions.append(("MOVE", eid, target))

        actions.append(("RESET",))
        self._legal_action_cache[state] = tuple(actions)
        return self._legal_action_cache[state]

    def _floor_scores(self, state):
        elevators_t, persons_t, _ = state
        score = {}
        total_reward = sum(self.pr(pid) for pid, _ in persons_t)

        for eid, _, _ in elevators_t:
            for floor in self._reachable[eid]:
                score[floor] = score.get(floor, 0.0) + 0.15 * total_reward

        for pid, loc in persons_t:
            r = self.pr(pid)
            goal = self._person_goal[pid]
            score[goal] = score.get(goal, 0.0) + 1.50 * r
            if loc[0] == "floor":
                score[loc[1]] = score.get(loc[1], 0.0) + 2.50 * r
            else:
                f = self._decode_state(state)[0][loc[1]]
                score[f] = score.get(f, 0.0) + 1.20 * r

        for floor in score:
            reach_count = sum(1 for eid in self._elevator_ids if floor in self._reachable_sets[eid])
            score[floor] += 0.40 * reach_count

        return score

    def _transition_outcomes(self, state, action):
        key = (state, action)
        if key in self._transition_cache: return self._transition_cache[key]

        if action[0] == "RESET":
            outcomes = ((1.0, self._initial_state, 0.0),)
        elif action[0] == "MOVE":
            outcomes = self._move_outcomes(state, action[1], action[2])
        elif action[0] == "ENTER":
            outcomes = self._enter_outcomes(state, action[1], action[2])
        elif action[0] == "EXIT":
            outcomes = self._exit_outcomes(state, action[1], action[2])
        else:
            outcomes = ((1.0, state, 0.0),)

        self._transition_cache[key] = outcomes
        return outcomes

    def _move_outcomes(self, state, eid, target_floor):
        success_prob = self.ep(eid)
        current_floor = next(f for e, f, _ in state[0] if e == eid)

        success_state = self._replace_elevator(state, eid, target_floor)
        if success_prob >= 1.0: return ((1.0, success_state, 0.0),)

        failure_floors = tuple(sorted((self._reachable_sets[eid] - {target_floor}) | {current_floor}))
        if not failure_floors: return ((1.0, success_state, 0.0),)

        fail_prob = (1.0 - success_prob) / len(failure_floors)
        outcomes = [(success_prob, success_state, 0.0)]
        for floor in failure_floors:
            outcomes.append((fail_prob, self._replace_elevator(state, eid, floor), 0.0))
        return tuple(outcomes)

    def _enter_outcomes(self, state, pid, eid):
        success_prob = self.pp(pid)
        success_state = self._enter_success_state(state, pid, eid)
        if success_prob >= 1.0: return ((1.0, success_state, 0.0),)
        return ((success_prob, success_state, 0.0), (1.0 - success_prob, state, 0.0))

    def _exit_outcomes(self, state, pid, eid):
        success_prob = self.pp(pid)
        success_state, reward = self._exit_success_state(state, pid, eid)
        if success_prob >= 1.0: return ((1.0, success_state, reward),)
        return ((success_prob, success_state, reward), (1.0 - success_prob, state, 0.0))

    def _replace_elevator(self, state, eid, new_floor):
        elevs = tuple((e, new_floor, w) if e == eid else (e, f, w) for e, f, w in state[0])
        return (elevs, state[1], state[2])

    def _enter_success_state(self, state, pid, eid):
        w = self._person_weight[pid]
        elevs = tuple((e, f, l + w) if e == eid else (e, f, l) for e, f, l in state[0])
        pers = tuple((p, ("in", eid)) if p == pid else (p, loc) for p, loc in state[1])
        return (elevs, pers, state[2])

    def _exit_success_state(self, state, pid, eid):
        w = self._person_weight[pid]
        current_floor = next(f for e, f, _ in state[0] if e == eid)
        elevs = tuple((e, f, l - w) if e == eid else (e, f, l) for e, f, l in state[0])

        if current_floor == self._person_goal[pid]:
            next_remaining = state[2] - 1
            if next_remaining == 0:
                return self._initial_state, self._goal_reward + self.pr(pid)
            next_pers = tuple(p for p in state[1] if p[0] != pid)
            return (elevs, next_pers, next_remaining), self.pr(pid)

        pers = tuple((p, ("floor", current_floor)) if p == pid else (p, loc) for p, loc in state[1])
        return (elevs, pers, state[2]), 0.0

    # ------------------------------------------------------------------
    # Heuristic
    # ------------------------------------------------------------------
    def _heuristic(self, state, steps_left):
        key = (state, steps_left)
        if key in self._heuristic_cache: return self._heuristic_cache[key]

        _, persons_t, _ = state
        if not persons_t: return self._goal_reward

        elevator_floor, elevator_load, _, _, _, _ = self._decode_state(state)
        total, total_cost = 0.0, 0.0

        for pid, loc in persons_t:
            cost = self._estimate_person_cost(pid, loc, elevator_floor, elevator_load)
            if math.isinf(cost): return -1e9

            total += self.pr(pid) * (0.95 ** min(cost, steps_left))
            total_cost += cost

        total += self._goal_reward * (0.95 ** min(total_cost, steps_left))
        self._heuristic_cache[key] = total
        return total

    def _reset_penalty(self, state):
        return 3.0 + 1.5 * len(state[1])

    def _estimate_person_cost(self, pid, loc, elevator_floor, elevator_load):
        goal, weight, p_prob = self._person_goal[pid], self._person_weight[pid], self.pp(pid)
        
        if loc[0] == "floor":
            start_floor = loc[1]
            best = INF
            for eid in self._elevator_ids:
                if elevator_load[eid] + weight > self._capacities[eid] or start_floor not in self._reachable_sets[eid]:
                    continue
                move_to_pickup = 0.0 if elevator_floor[eid] == start_floor else (1.0 / self.ep(eid))
                enter_cost = 1.0 / p_prob

                if goal in self._reachable_sets[eid]:
                    move_to_goal = 0.0 if start_floor == goal else (1.0 / self.ep(eid))
                    best = min(best, move_to_pickup + enter_cost + move_to_goal + enter_cost)

                for eid2 in self._elevator_ids:
                    if eid2 == eid or goal not in self._reachable_sets[eid2] or elevator_load[eid2] + weight > self._capacities[eid2]:
                        continue
                    for meet in self._reachable_sets[eid] & self._reachable_sets[eid2]:
                        move_to_meet = 0.0 if meet == start_floor else (1.0 / self.ep(eid))
                        move_e2_to_meet = 0.0 if elevator_floor[eid2] == meet else (1.0 / self.ep(eid2))
                        move_e2_to_goal = 0.0 if meet == goal else (1.0 / self.ep(eid2))
                        best = min(best, move_to_pickup + enter_cost + move_to_meet + enter_cost + move_e2_to_meet + enter_cost + move_e2_to_goal + enter_cost)
            return best

        eid = loc[1]
        current_floor = elevator_floor[eid]
        best = INF
        if goal in self._reachable_sets[eid]:
            best = min(best, (0.0 if current_floor == goal else (1.0 / self.ep(eid))) + (1.0 / p_prob))

        for eid2 in self._elevator_ids:
            if eid2 == eid or goal not in self._reachable_sets[eid2] or elevator_load[eid2] + weight > self._capacities[eid2]:
                continue
            for meet in self._reachable_sets[eid] & self._reachable_sets[eid2]:
                move_to_meet = 0.0 if current_floor == meet else (1.0 / self.ep(eid))
                move_e2_to_meet = 0.0 if elevator_floor[eid2] == meet else (1.0 / self.ep(eid2))
                move_e2_to_goal = 0.0 if meet == goal else (1.0 / self.ep(eid2))
                best = min(best, move_to_meet + (1.0 / p_prob) + move_e2_to_meet + (1.0 / p_prob) + move_e2_to_goal + (1.0 / p_prob))

        return best

    # ------------------------------------------------------------------
    # Deterministic A* planner (Uses Dynamic Probabilities)
    # ------------------------------------------------------------------
    def _precompute_plan_costs(self):
        self._pcost = {}
        for pid in self._person_ids:
            goal, w, pprob = self._person_goal[pid], self._person_weight[pid], self.pp(pid)
            pc = (1.0 / pprob) if pprob > 0 else INF
            dist, pq = {}, []
            
            for eid in self._elevator_ids:
                if goal in self._reachable_sets[eid] and w <= self._capacities[eid]:
                    node = ("in", eid, goal)
                    if pc < dist.get(node, INF):
                        dist[node] = pc; heapq.heappush(pq, (pc, node))
                        
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, INF): continue
                if u[0] == "in":
                    eid, f = u[1], u[2]
                    ep = self.ep(eid)
                    mc = (1.0 / (ep * ep)) if ep > 0 else INF
                    for f2 in self._reachable_sets[eid]:
                        if f2 != f:
                            v = ("in", eid, f2); nd = d + mc
                            if nd < dist.get(v, INF):
                                dist[v] = nd; heapq.heappush(pq, (nd, v))
                    v = ("floor", f); nd = d + pc
                    if nd < dist.get(v, INF):
                        dist[v] = nd; heapq.heappush(pq, (nd, v))
                else:
                    f = u[1]
                    for eid in self._elevator_ids:
                        if f in self._reachable_sets[eid] and w <= self._capacities[eid]:
                            v = ("in", eid, f); nd = d + pc
                            if nd < dist.get(v, INF):
                                dist[v] = nd; heapq.heappush(pq, (nd, v))
            self._pcost[pid] = dist

    def _precompute_move_costs(self):
        self._mcost = {}
        for pid in self._person_ids:
            goal, w = self._person_goal[pid], self._person_weight[pid]
            dist, pq = {}, []
            for eid in self._elevator_ids:
                if goal in self._reachable_sets[eid] and w <= self._capacities[eid]:
                    node = ("in", eid, goal)
                    if 0.0 < dist.get(node, INF):
                        dist[node] = 0.0; heapq.heappush(pq, (0.0, node))
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, INF): continue
                if u[0] == "in":
                    eid, f = u[1], u[2]
                    ep = self.ep(eid)
                    mc = (1.0 / (ep * ep)) if ep > 0 else INF
                    for f2 in self._reachable_sets[eid]:
                        if f2 != f:
                            v = ("in", eid, f2); nd = d + mc
                            if nd < dist.get(v, INF):
                                dist[v] = nd; heapq.heappush(pq, (nd, v))
                    v = ("floor", f)
                    if d < dist.get(v, INF):
                        dist[v] = d; heapq.heappush(pq, (d, v))
                else:
                    f = u[1]
                    for eid in self._elevator_ids:
                        if f in self._reachable_sets[eid] and w <= self._capacities[eid]:
                            v = ("in", eid, f)
                            if d < dist.get(v, INF):
                                dist[v] = d; heapq.heappush(pq, (d, v))
            self._mcost[pid] = dist

    def _planning_heuristic(self, ps, targets):
        elevs, persons = ps
        ef = {e[0]: e[1] for e in elevs}
        if self._many_elevators:
            total = 0.0
            for pid, loc in persons:
                if pid not in targets: continue
                node = ("floor", loc[1]) if loc[0] == "floor" else ("in", loc[1], ef[loc[1]])
                c = self._pcost[pid].get(node, INF)
                if c == INF: return INF
                total += c
            return total
            
        ee_total, move_max = 0.0, 0.0
        for pid, loc in persons:
            if pid not in targets: continue
            pc = 1.0 / self.pp(pid)
            if loc[0] == "floor": node = ("floor", loc[1]); ee_total += 2.0 * pc
            else: node = ("in", loc[1], ef[loc[1]]); ee_total += pc
            mc = self._mcost[pid].get(node, INF)
            if mc == INF: return INF
            if mc > move_max: move_max = mc
        return ee_total + move_max

    def _plan_successors(self, ps, targets):
        elevs, persons = ps
        e_dict = {e[0]: (e[1], e[2]) for e in elevs}
        mandatory = []
        
        for pid, loc in persons:
            if pid not in targets or loc[0] != "in": continue
            eid = loc[1]
            if e_dict[eid][0] == self._person_goal[pid]:
                cost = 1.0 / self.pp(pid)
                new_elevs = tuple((e[0], e[1], e[2] - self._person_weight[pid]) if e[0] == eid else e for e in elevs)
                new_persons = tuple(p for p in persons if p[0] != pid)
                mandatory.append((("EXIT", pid, eid), (new_elevs, new_persons), cost))
        if mandatory: return mandatory

        succ = []
        for pid, loc in persons:
            if pid not in targets or loc[0] != "floor": continue
            f = loc[1]
            for eid, (ef, ew) in e_dict.items():
                if ef == f and ew + self._person_weight[pid] <= self._capacities[eid]:
                    cost = 1.0 / self.pp(pid)
                    new_elevs = tuple((e[0], e[1], e[2] + self._person_weight[pid]) if e[0] == eid else e for e in elevs)
                    new_persons = tuple((p[0], ("in", eid)) if p[0] == pid else p for p in persons)
                    succ.append((("ENTER", pid, eid), (new_elevs, new_persons), cost))

        for pid, loc in persons:
            if pid not in targets or loc[0] != "in": continue
            eid = loc[1]
            ef = e_dict[eid][0]
            if ef != self._person_goal[pid] and ef in self._shared_floors:
                cost = 1.0 / self.pp(pid)
                new_elevs = tuple((e[0], e[1], e[2] - self._person_weight[pid]) if e[0] == eid else e for e in elevs)
                new_persons = tuple((p[0], ("floor", ef)) if p[0] == pid else p for p in persons)
                succ.append((("EXIT", pid, eid), (new_elevs, new_persons), cost))

        interesting = set(self._shared_floors)
        for pid, loc in persons:
            if pid not in targets: continue
            if loc[0] == "floor": interesting.add(loc[1])
            else: interesting.add(self._person_goal[pid])
            
        for eid, (ef, _) in e_dict.items():
            ep = self.ep(eid)
            cost = (1.0 / (ep * ep)) if ep > 0 else INF
            for tf in self._reachable_sets[eid]:
                if tf != ef and tf in interesting:
                    new_elevs = tuple((e[0], tf, e[2]) if e[0] == eid else e for e in elevs)
                    succ.append((("MOVE", eid, tf), (new_elevs, persons), cost))
        return succ

    def _astar_plan(self, ps, targets):
        h0 = self._planning_heuristic(ps, targets)
        if h0 == INF: return None
        counter = itertools.count()
        pq = [(h0, next(counter), 0.0, ps, ())]
        visited = set()
        expansions = 0
        
        while pq:
            f, _, g, st, path = heapq.heappop(pq)
            if not any(pid in targets for pid, _ in st[1]): return list(path)
            if st in visited: continue
            visited.add(st)
            expansions += 1
            if expansions > self._plan_node_cap: return None
            
            for act, nxt, cost in self._plan_successors(st, targets):
                if nxt in visited: continue
                ng = g + cost
                hn = self._planning_heuristic(nxt, targets)
                if hn == INF: continue
                heapq.heappush(pq, (ng + hn, next(counter), ng, nxt, path + (act,)))
        return None