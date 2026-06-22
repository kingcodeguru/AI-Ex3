# gemini version + supposably fast + amit and yoav ideas
"""AI assistance disclosure: drafted with Gemini.

Blazingly Fast Hybrid RL controller with Advanced Heuristics.
Features: Decaying UCB, Laplace Smoothing, Subset RESET-Looping (Farming),
Synergy Positioning, Broken Elevator Avoidance, and Strict Priorities.
"""

import math
import heapq
import itertools

INF = float("inf")

# Replace with your actual student ID
id = ["123456789"]

class Controller:
    def __init__(self, game):
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

        self._person_weight = {pid: game.get_person_weight(pid) for pid in self._person_ids}
        self._person_goal = {pid: game.get_person_goal(pid) for pid in self._person_ids}
        
        # ---- RL Exploration Setup ----
        self._alpha = 0.20  # 20% exploration is enough with UCB
        self._explore_steps = int(self._max_steps * self._alpha)
        
        # Trackers
        self._stats_move_attempts = {eid: 0 for eid in self._elevator_ids}
        self._stats_move_successes = {eid: 0 for eid in self._elevator_ids}
        self._stats_person_attempts = {pid: 0 for pid in self._person_ids}
        self._stats_person_successes = {pid: 0 for pid in self._person_ids}
        self._stats_person_rewards = {pid: [] for pid in self._person_ids}
        
        # Current estimates
        self._elevator_prob = {eid: 1.0 for eid in self._elevator_ids}
        self._person_prob = {pid: 1.0 for pid in self._person_ids}
        self._person_mean_reward = {pid: 40.0 for pid in self._person_ids}
        
        self._last_action = None
        self._last_state = None

        # Subset Farming Lock
        self._farming_targets = None
        self._full_targets = frozenset(self._person_ids)

        # Caches
        self._state_info_cache = {}
        self._legal_action_cache = {}
        self._transition_cache = {}
        self._value_cache = {}
        self._policy_cache = {}
        self._heuristic_cache = {}
        self._plan_cache = {}
        
        self._mode = "expectimax"
        self._collapse_moves = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def choose_next_action(self, state):
        steps_left = self._max_steps - self.game.get_current_steps()
        if steps_left <= 0:
            return "RESET"
            
        self._update_stats(state)
        current_step = self.game.get_current_steps()
        
        # STRICT PRIORITY 0: Instant Exit if at goal (Short-circuit search)
        instant_action = self._check_instant_priority(state)
        if instant_action:
            action_tuple = instant_action
        else:
            if current_step < self._explore_steps:
                # PHASE 1: EXPLORATION (Decaying UCB & Laplace)
                self._update_ucb_estimates(current_step)
                self._clear_caches()
                action_tuple = self._expectimax_action(state, steps_left, self._full_targets, forced_depth=2)
                
            elif current_step == self._explore_steps:
                # TRANSITION: Evaluate Subset Farming
                self._freeze_empirical_estimates()
                self._evaluate_subset_farming()
                self._setup_exploitation_planner()
                action_tuple = self._choose(state, steps_left, self._mode)
                
            else:
                # PHASE 2: EXPLOITATION
                # Check if farming condition met (all targets delivered)
                if self._farming_targets and self._farming_complete(state):
                    action_tuple = ("RESET",)
                else:
                    action_tuple = self._choose(state, steps_left, self._mode)

        self._last_state = state
        self._last_action = action_tuple
        
        if action_tuple[0] == "RESET": return "RESET"
        return f"{action_tuple[0]}{{{action_tuple[1]},{action_tuple[2]}}}"

    def _check_instant_priority(self, state):
        """Priority 0: If someone is in an elevator at their goal, EXIT immediately."""
        elevators_t, persons_t, _ = state
        elev_floors = {e[0]: e[1] for e in elevators_t}
        for pid, loc in persons_t:
            if loc[0] == "in" and elev_floors.get(loc[1]) == self._person_goal[pid]:
                return ("EXIT", pid, loc[1])
        return None

    def _farming_complete(self, state):
        """Returns True if all target persons in the farming subset are delivered."""
        _, persons_t, _ = state
        remaining_pids = {p[0] for p in persons_t}
        return not remaining_pids.intersection(self._farming_targets)

    # ------------------------------------------------------------------
    # RL Learning & Subset Farming
    # ------------------------------------------------------------------
    def _update_stats(self, state):
        if self._last_action is None or self._last_action[0] == "RESET": return
            
        is_reset = (state[2] == len(self._person_ids) and self._last_state[2] == 1)
        kind = self._last_action[0]
        if kind == "RESET": return
        p1, p2 = self._last_action[1], self._last_action[2]
        
        if kind == "MOVE":
            eid, target_f = p1, p2
            if not is_reset:
                curr_f = next(e[1] for e in state[0] if e[0] == eid)
                self._stats_move_attempts[eid] += 1
                if curr_f == target_f: self._stats_move_successes[eid] += 1
                    
        elif kind == "ENTER":
            pid, eid = p1, p2
            if not is_reset:
                p_loc = next((p[1] for p in state[1] if p[0] == pid), None)
                self._stats_person_attempts[pid] += 1
                if p_loc == ("in", eid): self._stats_person_successes[pid] += 1
                    
        elif kind == "EXIT":
            pid, eid = p1, p2
            self._stats_person_attempts[pid] += 1
            if is_reset:
                self._stats_person_successes[pid] += 1
                self._stats_person_rewards[pid].append(self.game.get_last_gained_reward() - self._goal_reward)
            else:
                p_loc = next((p[1] for p in state[1] if p[0] == pid), None)
                if p_loc is None:
                    self._stats_person_successes[pid] += 1
                    self._stats_person_rewards[pid].append(self.game.get_last_gained_reward())
                elif p_loc[0] == "floor":
                    self._stats_person_successes[pid] += 1

    def _update_ucb_estimates(self, current_step):
        """Weapon 1: Decaying UCB + Laplace Smoothing"""
        decay = max(0.0, 1.0 - current_step / self._max_steps)
        
        for eid in self._elevator_ids:
            att = self._stats_move_attempts[eid]
            suc = self._stats_move_successes[eid]
            self._elevator_prob[eid] = (suc + 1.0) / (att + 2.0)
            
        for pid in self._person_ids:
            att = self._stats_person_attempts[pid]
            suc = self._stats_person_successes[pid]
            self._person_prob[pid] = (suc + 1.0) / (att + 2.0)
            
            rews = self._stats_person_rewards[pid]
            if not rews:
                self._person_mean_reward[pid] = 40.0
            else:
                mean_r = sum(rews) / len(rews)
                bonus = decay * (20.0 / math.sqrt(len(rews)))
                self._person_mean_reward[pid] = mean_r + bonus

    def _freeze_empirical_estimates(self):
        """Lock pure empirical averages using Laplace."""
        for eid in self._elevator_ids:
            self._elevator_prob[eid] = (self._stats_move_successes[eid] + 1.0) / (self._stats_move_attempts[eid] + 2.0)
        for pid in self._person_ids:
            self._person_prob[pid] = (self._stats_person_successes[pid] + 1.0) / (self._stats_person_attempts[pid] + 2.0)
            rews = self._stats_person_rewards[pid]
            self._person_mean_reward[pid] = sum(rews) / len(rews) if rews else 5.0

    def _evaluate_subset_farming(self):
        """Weapon 2: Subset RESET-Looping (Combinatorial Pricing)"""
        total_reward = sum(self._person_mean_reward.values()) + self._goal_reward
        total_cost = sum(self._estimate_ideal_cost(pid) for pid in self._person_ids) + self._reset_penalty(self._initial_state)
        baseline_roi = total_reward / total_cost if total_cost > 0 else 0

        best_roi = baseline_roi
        best_subset = None
        
        max_subset_size = min(4, len(self._person_ids) - 1)
        for r in range(1, max_subset_size + 1):
            for subset in itertools.combinations(self._person_ids, r):
                sub_reward = sum(self._person_mean_reward[pid] for pid in subset)
                sub_cost = sum(self._estimate_ideal_cost(pid) for pid in subset) + self._reset_penalty(self._initial_state)
                
                roi = sub_reward / sub_cost if sub_cost > 0 else 0
                if roi > best_roi:
                    best_roi = roi
                    best_subset = frozenset(subset)
        
        if best_subset and best_roi >= 1.15 * baseline_roi:
            self._farming_targets = best_subset
        else:
            self._farming_targets = None

    def _estimate_ideal_cost(self, pid):
        """Rough estimation of cost to deliver a person for subset pricing."""
        goal, w, start = self._person_goal[pid], self._person_weight[pid], None
        for p_id, loc in self._initial_state[1]:
            if p_id == pid: start = loc[1]
            
        best = INF
        for eid in self._elevator_ids:
            if start in self._reachable_sets[eid] and goal in self._reachable_sets[eid] and w <= self._capacities[eid]:
                p_m, p_p = self._elevator_prob[eid], self._person_prob[pid]
                cost = (1.0/p_m) + (1.0/p_p) + (1.0/p_m) + (1.0/p_p)
                best = min(best, cost)
        return best if best != INF else 50.0

    # ------------------------------------------------------------------
    # Engine Setup & Actions
    # ------------------------------------------------------------------
    def _setup_exploitation_planner(self):
        self._clear_caches()
        self._shared_floors = set()
        _reaches = [self._reachable_sets[e] for e in self._elevator_ids]
        for i in range(len(_reaches)):
            for j in range(i + 1, len(_reaches)):
                self._shared_floors |= (_reaches[i] & _reaches[j])
                
        self._plan_node_cap = 60000
        self._precompute_plan_costs()
        self._precompute_move_costs()
        self._many_elevators = len(self._elevator_ids) >= 3
        self._mode = "planner" if len(self._person_ids) >= 4 else "expectimax"
        self._collapse_moves = False

    def _clear_caches(self):
        self._state_info_cache.clear()
        self._legal_action_cache.clear()
        self._transition_cache.clear()
        self._value_cache.clear()
        self._policy_cache.clear()
        self._heuristic_cache.clear()
        self._plan_cache.clear()

    def _choose(self, state, steps_left, mode):
        targets = self._farming_targets if self._farming_targets else self._full_targets
        
        if mode == "expectimax":
            return self._expectimax_action(state, steps_left, targets)
        
        plan = self._get_plan((state[0], state[1]), targets)
        if plan and len(plan) <= steps_left:
            return plan[0]
        return self._expectimax_action(state, steps_left, targets)

    def _expectimax_action(self, state, steps_left, targets, forced_depth=None):
        depth = forced_depth if forced_depth is not None else 5
        self._best_value(state, steps_left, depth, targets)
        return self._policy_cache.get((state, steps_left, depth), ("RESET",))

    def _best_value(self, state, steps_left, depth_left, targets):
        key = (state, steps_left, depth_left)
        if key in self._value_cache: return self._value_cache[key]

        if steps_left <= 0:
            self._value_cache[key], self._policy_cache[key] = 0.0, ("RESET",)
            return 0.0

        if depth_left <= 0:
            val = self._heuristic(state, steps_left, targets)
            self._value_cache[key], self._policy_cache[key] = val, ("RESET",)
            return val

        best_value, best_action = -INF, ("RESET",)
        for action in self._legal_actions(state):
            q_value = self._action_value(state, action, steps_left, depth_left, targets)
            if q_value > best_value:
                best_value, best_action = q_value, action

        self._value_cache[key], self._policy_cache[key] = best_value, best_action
        return best_value

    def _action_value(self, state, action, steps_left, depth_left, targets):
        outcomes = self._transition_outcomes(state, action)
        total = 0.0
        for prob, next_state, immediate_reward in outcomes:
            if prob <= 0.0: continue
            continuation = self._best_value(next_state, steps_left - 1, depth_left - 1, targets) if steps_left > 1 else 0.0
            total += prob * (immediate_reward + continuation)
        
        if action[0] == "RESET": total -= self._reset_penalty(state)
        return total

    # ------------------------------------------------------------------
    # Optimized Legal Actions & Synergy
    # ------------------------------------------------------------------
    def _legal_actions(self, state):
        cached = self._legal_action_cache.get(state)
        if cached is not None: return cached

        elevators_t, persons_t, _ = state
        elevator_floor = {e[0]: e[1] for e in elevators_t}
        elevator_load = {e[0]: e[2] for e in elevators_t}
        
        people_on_floor, people_in_elevator = {}, {}
        for pid, loc in persons_t:
            if loc[0] == "floor": people_on_floor.setdefault(loc[1], []).append(pid)
            else: people_in_elevator.setdefault(loc[1], []).append(pid)

        actions = []
        
        # Weapon 4: Broken Elevator Avoidance
        best_pm = max(self._elevator_prob.values()) if self._elevator_prob else 1.0

        for eid in self._elevator_ids:
            floor = elevator_floor[eid]
            for pid in people_in_elevator.get(eid, []):
                actions.append(("EXIT", pid, eid))

            load = elevator_load[eid]
            for pid in people_on_floor.get(floor, []):
                if load + self._person_weight[pid] <= self._capacities[eid]:
                    actions.append(("ENTER", pid, eid))

            if self._elevator_prob[eid] < best_pm - 0.2 and not people_in_elevator.get(eid):
                continue
                
            targets = []
            for target in self._reachable[eid]:
                if target != floor: targets.append(target)
            for target in targets[:3]:
                actions.append(("MOVE", eid, target))

        actions.append(("RESET",))
        self._legal_action_cache[state] = tuple(actions)
        return self._legal_action_cache[state]

    def _heuristic(self, state, steps_left, targets):
        """Weapon 3: Synergy / Positioning Nudge in Heuristic"""
        key = (state, steps_left)
        if key in self._heuristic_cache: return self._heuristic_cache[key]

        _, persons_t, _ = state
        if not persons_t: return self._goal_reward

        elevator_floor = {e[0]: e[1] for e in state[0]}
        elevator_load = {e[0]: e[2] for e in state[0]}
        
        waiting_counts = {}
        for p, loc in persons_t:
            if loc[0] == "floor": waiting_counts[loc[1]] = waiting_counts.get(loc[1], 0) + 1

        total, total_cost = 0.0, 0.0
        for pid, loc in persons_t:
            if pid not in targets: continue
            
            cost = self._estimate_person_cost(state, pid, loc, elevator_floor, elevator_load)
            if math.isinf(cost): return -1e9

            synergy = 1.0 + 0.30 * waiting_counts.get(self._person_goal[pid], 0)
            reward = self._person_mean_reward[pid] * synergy
            
            # Weapon 5: Discount Factor (0.90) favors quick operations
            total += reward * (0.90 ** min(cost, steps_left))
            total_cost += cost

        if not self._farming_targets:
            total += self._goal_reward * (0.90 ** min(total_cost, steps_left))
            
        self._heuristic_cache[key] = total
        return total

    def _reset_penalty(self, state): return 3.0 + 1.5 * len(state[1])

    # ------------------------------------------------------------------
    # Transition Model
    # ------------------------------------------------------------------
    def _transition_outcomes(self, state, action):
        key = (state, action)
        cached = self._transition_cache.get(key)
        if cached is not None: return cached

        if action[0] == "RESET":
            outcomes = ((1.0, self._initial_state, 0.0),)
        elif action[0] == "MOVE": outcomes = self._move_outcomes(state, action[1], action[2])
        elif action[0] == "ENTER": outcomes = self._enter_outcomes(state, action[1], action[2])
        elif action[0] == "EXIT": outcomes = self._exit_outcomes(state, action[1], action[2])
        else: outcomes = ((1.0, state, 0.0),)

        self._transition_cache[key] = outcomes
        return outcomes

    def _move_outcomes(self, state, eid, target_floor):
        success_prob = self._elevator_prob[eid]
        current_floor = next(e[1] for e in state[0] if e[0] == eid)

        success_state = self._replace_elevator(state, eid, target_floor)
        if success_prob >= 1.0:
            return ((1.0, success_state, 0.0),)

        failure_floors = tuple(sorted((self._reachable_sets[eid] - {target_floor}) | {current_floor}))
        if not failure_floors:
            return ((1.0, success_state, 0.0),)

        fail_prob = (1.0 - success_prob) / len(failure_floors)
        outcomes = [(success_prob, success_state, 0.0)]
        for floor in failure_floors:
            outcomes.append((fail_prob, self._replace_elevator(state, eid, floor), 0.0))
        return tuple(outcomes)

    def _enter_outcomes(self, state, pid, eid):
        success_prob = self._person_prob[pid]
        success_state = self._enter_success_state(state, pid, eid)
        if success_prob >= 1.0: return ((1.0, success_state, 0.0),)
        return ((success_prob, success_state, 0.0), (1.0 - success_prob, state, 0.0))

    def _exit_outcomes(self, state, pid, eid):
        success_prob = self._person_prob[pid]
        success_state, reward = self._exit_success_state(state, pid, eid)
        if success_prob >= 1.0: return ((1.0, success_state, reward),)
        return ((success_prob, success_state, reward), (1.0 - success_prob, state, 0.0))

    def _replace_elevator(self, state, eid, new_floor):
        elevators_t, persons_t, remaining = state
        new_elevs = tuple((e[0], new_floor, e[2]) if e[0] == eid else e for e in elevators_t)
        return (new_elevs, persons_t, remaining)

    def _enter_success_state(self, state, pid, eid):
        elevators_t, persons_t, remaining = state
        weight = self._person_weight[pid]
        new_elevs = tuple((e[0], e[1], e[2] + weight) if e[0] == eid else e for e in elevators_t)
        new_persons = tuple((p[0], ("in", eid)) if p[0] == pid else p for p in persons_t)
        return (new_elevs, new_persons, remaining)

    def _exit_success_state(self, state, pid, eid):
        elevators_t, persons_t, remaining = state
        weight = self._person_weight[pid]
        current_floor = next(e[1] for e in elevators_t if e[0] == eid)
        new_elevs = tuple((e[0], e[1], e[2] - weight) if e[0] == eid else e for e in elevators_t)

        if current_floor == self._person_goal[pid]:
            next_remaining = remaining - 1
            if next_remaining == 0:
                return self._initial_state, self._goal_reward + self._person_mean_reward[pid]
            new_persons = tuple(p for p in persons_t if p[0] != pid)
            return (new_elevs, new_persons, next_remaining), self._person_mean_reward[pid]

        new_persons = tuple((p[0], ("floor", current_floor)) if p[0] == pid else p for p in persons_t)
        return (new_elevs, new_persons, remaining), 0.0

    def _estimate_person_cost(self, state, pid, loc, elevator_floor, elevator_load):
        goal, weight, p_prob = self._person_goal[pid], self._person_weight[pid], self._person_prob[pid]
        
        if loc[0] == "floor":
            start_floor = loc[1]
            best = INF
            for eid in self._elevator_ids:
                if elevator_load[eid] + weight > self._capacities[eid] or start_floor not in self._reachable_sets[eid]:
                    continue
                move_to_pickup = 0.0 if elevator_floor[eid] == start_floor else (1.0 / self._elevator_prob[eid])
                enter_cost = 1.0 / p_prob

                if goal in self._reachable_sets[eid]:
                    move_to_goal = 0.0 if start_floor == goal else (1.0 / self._elevator_prob[eid])
                    best = min(best, move_to_pickup + enter_cost + move_to_goal + enter_cost)

                for eid2 in self._elevator_ids:
                    if eid2 == eid or goal not in self._reachable_sets[eid2] or elevator_load[eid2] + weight > self._capacities[eid2]:
                        continue
                    for meet in self._reachable_sets[eid] & self._reachable_sets[eid2]:
                        move_to_meet = 0.0 if meet == start_floor else (1.0 / self._elevator_prob[eid])
                        move_e2_to_meet = 0.0 if elevator_floor[eid2] == meet else (1.0 / self._elevator_prob[eid2])
                        move_e2_to_goal = 0.0 if meet == goal else (1.0 / self._elevator_prob[eid2])
                        best = min(best, move_to_pickup + enter_cost + move_to_meet + enter_cost + move_e2_to_meet + enter_cost + move_e2_to_goal + enter_cost)
            return best

        eid = loc[1]
        current_floor = elevator_floor[eid]
        best = INF
        if goal in self._reachable_sets[eid]:
            best = min(best, (0.0 if current_floor == goal else (1.0 / self._elevator_prob[eid])) + (1.0 / p_prob))

        for eid2 in self._elevator_ids:
            if eid2 == eid or goal not in self._reachable_sets[eid2] or elevator_load[eid2] + weight > self._capacities[eid2]:
                continue
            for meet in self._reachable_sets[eid] & self._reachable_sets[eid2]:
                move_to_meet = 0.0 if current_floor == meet else (1.0 / self._elevator_prob[eid])
                move_e2_to_meet = 0.0 if elevator_floor[eid2] == meet else (1.0 / self._elevator_prob[eid2])
                move_e2_to_goal = 0.0 if meet == goal else (1.0 / self._elevator_prob[eid2])
                best = min(best, move_to_meet + (1.0 / p_prob) + move_e2_to_meet + (1.0 / p_prob) + move_e2_to_goal + (1.0 / p_prob))

        return best

    # ------------------------------------------------------------------
    # Deterministic A* planner (Exploitation Phase Only)
    # ------------------------------------------------------------------
    def _precompute_plan_costs(self):
        self._pcost = {}
        for pid in self._person_ids:
            goal, w, pprob = self._person_goal[pid], self._person_weight[pid], self._person_prob[pid]
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
                    ep = self._elevator_prob[eid]
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
                    ep = self._elevator_prob[eid]
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
            pc = 1.0 / self._person_prob[pid]
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
                cost = 1.0 / self._person_prob[pid]
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
                    cost = 1.0 / self._person_prob[pid]
                    new_elevs = tuple((e[0], e[1], e[2] + self._person_weight[pid]) if e[0] == eid else e for e in elevs)
                    new_persons = tuple((p[0], ("in", eid)) if p[0] == pid else p for p in persons)
                    succ.append((("ENTER", pid, eid), (new_elevs, new_persons), cost))

        for pid, loc in persons:
            if pid not in targets or loc[0] != "in": continue
            eid = loc[1]
            ef = e_dict[eid][0]
            if ef != self._person_goal[pid] and ef in self._shared_floors:
                cost = 1.0 / self._person_prob[pid]
                new_elevs = tuple((e[0], e[1], e[2] - self._person_weight[pid]) if e[0] == eid else e for e in elevs)
                new_persons = tuple((p[0], ("floor", ef)) if p[0] == pid else p for p in persons)
                succ.append((("EXIT", pid, eid), (new_elevs, new_persons), cost))

        interesting = set(self._shared_floors)
        for pid, loc in persons:
            if pid not in targets: continue
            if loc[0] == "floor": interesting.add(loc[1])
            else: interesting.add(self._person_goal[pid])
            
        for eid, (ef, _) in e_dict.items():
            ep = self._elevator_prob[eid]
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

    def _get_plan(self, ps, targets):
        cached = self._plan_cache.get(ps)
        if cached is not None: return cached
        plan = self._astar_plan(ps, targets)
        if plan is None: plan = []
        self._plan_cache[ps] = plan
        return plan