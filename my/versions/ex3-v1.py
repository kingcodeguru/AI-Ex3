# gemini version
"""AI assistance disclosure: drafted with Gemini.

Hybrid RL controller for the stochastic multi-elevator MDP.
Features an initial UCB-driven Exploration Phase to learn hidden parameters
(transition probabilities and reward distributions), followed by an Exact/A*
Exploitation Phase based on the learned empirical models.
"""

import math
import heapq
import itertools

INF = float("inf")

# Replace with your actual student ID
id = ["123456789"]

class Controller:
    """Stochastic multi-elevator RL controller.
    
    Phases:
      1. Explore (Step 0 to alpha * max_steps): Uses shallow Expectimax with
         Upper Confidence Bound (UCB) optimistic estimates to actively seek out
         unknown probabilities and rewards.
      2. Exploit (Step > alpha * max_steps): Freezes empirical models, runs heavy
         precomputations, and exploits using the exact search / deterministic planner.
    """

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

        self._person_weight = {
            pid: game.get_person_weight(pid) for pid in self._person_ids
        }
        self._person_goal = {
            pid: game.get_person_goal(pid) for pid in self._person_ids
        }
        
        # ---- RL Exploration Setup ----
        self._alpha = 0.25  # 25% of the steps dedicated to active exploration
        self._explore_steps = int(self._max_steps * self._alpha)
        
        # Trackers for hidden model learning
        self._stats_move_attempts = {eid: 0 for eid in self._elevator_ids}
        self._stats_move_successes = {eid: 0 for eid in self._elevator_ids}
        self._stats_person_attempts = {pid: 0 for pid in self._person_ids}
        self._stats_person_successes = {pid: 0 for pid in self._person_ids}
        self._stats_person_rewards = {pid: [] for pid in self._person_ids}
        
        # Current working estimates (start optimistic)
        self._elevator_prob = {eid: 1.0 for eid in self._elevator_ids}
        self._person_prob = {pid: 1.0 for pid in self._person_ids}
        self._person_mean_reward = {pid: 40.0 for pid in self._person_ids}
        
        self._last_action = None
        self._last_state = None

        # Caches
        self._state_info_cache = {}
        self._legal_action_cache = {}
        self._transition_cache = {}
        self._value_cache = {}
        self._policy_cache = {}
        self._heuristic_cache = {}
        self._plan_cache = {}
        
        # Default mode during exploration
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
        
        if current_step < self._explore_steps:
            # PHASE 1: EXPLORATION
            self._update_ucb_estimates()
            self._clear_caches()
            action = self._expectimax_action(state, steps_left, forced_depth=2)
            
        elif current_step == self._explore_steps:
            # TRANSITION TO EXPLOITATION
            self._freeze_empirical_estimates()
            self._setup_exploitation_planner()
            action = self._choose(state, steps_left, self._mode)
            
        else:
            # PHASE 2: EXPLOITATION
            action = self._choose(state, steps_left, self._mode)

        self._last_state = state
        self._last_action = action
        return action

    # ------------------------------------------------------------------
    # RL Learning & Phase Management
    # ------------------------------------------------------------------
    def _update_stats(self, state):
        """Monitors state transitions to track successes, failures, and rewards."""
        if self._last_action is None or self._last_action == "RESET":
            return
            
        # Detect if episode auto-reset because the goal was completed
        is_reset = (state[2] == len(self._person_ids) and self._last_state[2] == 1)
        
        kind, p1, p2 = self._parse_action(self._last_action)
        
        if kind == "MOVE":
            eid, target_f = p1, p2
            if not is_reset:
                curr_f = next(e[1] for e in state[0] if e[0] == eid)
                success = (curr_f == target_f)
                self._stats_move_attempts[eid] += 1
                if success:
                    self._stats_move_successes[eid] += 1
                    
        elif kind == "ENTER":
            pid, eid = p1, p2
            if not is_reset:
                p_loc = next((p[1] for p in state[1] if p[0] == pid), None)
                success = (p_loc == ("in", eid))
                self._stats_person_attempts[pid] += 1
                if success:
                    self._stats_person_successes[pid] += 1
                    
        elif kind == "EXIT":
            pid, eid = p1, p2
            self._stats_person_attempts[pid] += 1
            
            if is_reset:
                # Successfully delivered the last person
                self._stats_person_successes[pid] += 1
                reward = self.game.get_last_gained_reward() - self._goal_reward
                self._stats_person_rewards[pid].append(reward)
            else:
                p_loc = next((p[1] for p in state[1] if p[0] == pid), None)
                if p_loc is None:
                    # Delivered normally (not last)
                    self._stats_person_successes[pid] += 1
                    reward = self.game.get_last_gained_reward()
                    self._stats_person_rewards[pid].append(reward)
                else:
                    # Not delivered
                    if p_loc[0] == "floor":
                        self._stats_person_successes[pid] += 1  # Successfully exited to floor

    def _update_ucb_estimates(self):
        """Uses Upper Confidence Bound to encourage exploration of unknowns."""
        for eid in self._elevator_ids:
            att = self._stats_move_attempts[eid]
            suc = self._stats_move_successes[eid]
            self._elevator_prob[eid] = (suc + 1.0) / (att + 1.0)  # Laplace smoothing
            
        for pid in self._person_ids:
            att = self._stats_person_attempts[pid]
            suc = self._stats_person_successes[pid]
            self._person_prob[pid] = (suc + 1.0) / (att + 1.0)
            
            rews = self._stats_person_rewards[pid]
            if not rews:
                self._person_mean_reward[pid] = 40.0  # High optimistic start
            else:
                mean_r = sum(rews) / len(rews)
                bonus = 20.0 / math.sqrt(len(rews))
                self._person_mean_reward[pid] = mean_r + bonus

    def _freeze_empirical_estimates(self):
        """Locks in the pure empirical averages for the exploitation phase."""
        for eid in self._elevator_ids:
            att = self._stats_move_attempts[eid]
            suc = self._stats_move_successes[eid]
            self._elevator_prob[eid] = suc / att if att > 0 else 0.5
            
        for pid in self._person_ids:
            att = self._stats_person_attempts[pid]
            suc = self._stats_person_successes[pid]
            self._person_prob[pid] = suc / att if att > 0 else 0.5
            
            rews = self._stats_person_rewards[pid]
            if not rews:
                self._person_mean_reward[pid] = 5.0  # Fallback
            else:
                self._person_mean_reward[pid] = sum(rews) / len(rews)

    def _setup_exploitation_planner(self):
        """Runs the heavy initialization from ex2 now that models are known."""
        self._clear_caches()
        
        _RELIABLE = 0.80
        _min_w = min(self._person_weight.values()) if self._person_weight else 1
        _single_passenger = all(self._capacities[eid] < 2 * _min_w for eid in self._elevator_ids)
        _perfectly_reliable = all(self._elevator_prob[eid] >= 0.95 for eid in self._elevator_ids)
        self._collapse_eligible = _single_passenger and _perfectly_reliable
        
        # Shared floors setup
        self._shared_floors = set()
        _reaches = [self._reachable_sets[e] for e in self._elevator_ids]
        for _i in range(len(_reaches)):
            for _j in range(_i + 1, len(_reaches)):
                self._shared_floors |= (_reaches[_i] & _reaches[_j])
                
        self._full_targets = frozenset(self._person_ids)
        self._plan_node_cap = 60000
        
        self._precompute_plan_costs()
        self._precompute_move_costs()
        self._many_elevators = len(self._elevator_ids) >= 3
        
        _vals = list(self._person_mean_reward.values())
        _mean = (sum(_vals) / len(_vals)) if _vals else 0.0
        _mx = max(_vals) if _vals else 0.0
        _rl_like = (_mx > 1.5 * _mean)
        
        _floors_all = set()
        for _e in self._elevator_ids:
            _floors_all |= self._reachable_sets[_e]
        _maxf = max(_floors_all) if _floors_all else 0
        _full = frozenset(range(_maxf + 1))
        _full_range = all(self._reachable_sets[_e] == _full for _e in self._elevator_ids)
        _multi = any(self._capacities[_e] >= 2 * _min_w for _e in self._elevator_ids)
        
        self._force_expectimax = _rl_like or (_full_range and _multi)
        
        if self._force_expectimax:
            self._mode = "expectimax"
        else:
            self._mode = self._select_mode()
            
        self._collapse_moves = self._collapse_eligible and self._mode == "planner"

    def _clear_caches(self):
        self._state_info_cache.clear()
        self._legal_action_cache.clear()
        self._transition_cache.clear()
        self._value_cache.clear()
        self._policy_cache.clear()
        self._heuristic_cache.clear()
        self._plan_cache.clear()

    # ------------------------------------------------------------------
    # Search Engine
    # ------------------------------------------------------------------
    def _choose(self, state, steps_left, mode):
        if mode == "expectimax":
            return self._expectimax_action(state, steps_left)
        
        ps = (state[0], state[1])
        plan = self._get_plan(ps)
        if plan and len(plan) <= steps_left:
            return plan[0]
        return self._expectimax_action(state, steps_left)

    def _expectimax_action(self, state, steps_left, forced_depth=None):
        depth = forced_depth if forced_depth is not None else self._choose_depth(state, steps_left)
        self._best_value(state, steps_left, depth)
        return self._policy_cache.get((state, steps_left, depth), "RESET")

    def _choose_depth(self, state, steps_left):
        _, persons_t, _ = state
        remaining = len(persons_t)
        if steps_left <= 2: return steps_left
        if remaining <= 1: return min(steps_left, 8)
        if remaining <= 2: return min(steps_left, 7)
        if remaining <= 3: return min(steps_left, 7)
        return min(steps_left, 5)

    def _best_value(self, state, steps_left, depth_left):
        key = (state, steps_left, depth_left)
        cached = self._value_cache.get(key)
        if cached is not None:
            return cached

        if steps_left <= 0:
            self._value_cache[key] = 0.0
            self._policy_cache[key] = "RESET"
            return 0.0

        if depth_left <= 0:
            value = self._heuristic(state, steps_left)
            self._value_cache[key] = value
            self._policy_cache[key] = "RESET"
            return value

        best_value = -math.inf
        best_action = "RESET"

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

        if self._collapse_moves and action[0] == "M":
            for idx, (prob, next_state, immediate_reward) in enumerate(outcomes):
                if prob <= 0.0: continue
                if next_steps <= 0: continuation = 0.0
                elif idx == 0: continuation = self._best_value(next_state, next_steps, next_depth)
                else: continuation = self._heuristic(next_state, next_steps)
                total += prob * (immediate_reward + continuation)
            return total

        for prob, next_state, immediate_reward in outcomes:
            if prob <= 0.0: continue
            continuation = self._best_value(next_state, next_steps, next_depth) if next_steps > 0 else 0.0
            total += prob * (immediate_reward + continuation)

        if action == "RESET":
            total -= self._reset_penalty(state)

        return total

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------
    def _decode_state(self, state):
        cached = self._state_info_cache.get(state)
        if cached is not None: return cached

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
        cached = self._legal_action_cache.get(state)
        if cached is not None: return cached

        elevator_floor, elevator_load, _, people_on_floor, people_in_elevator, _ = self._decode_state(state)
        actions = []

        for eid in self._elevator_ids:
            floor = elevator_floor[eid]
            inside = people_in_elevator.get(eid, [])
            goal_first, other = [], []
            for pid in inside:
                if floor == self._person_goal[pid]: goal_first.append(pid)
                else: other.append(pid)
            goal_first.sort(key=lambda pid: (-self._person_mean_reward[pid], pid))
            other.sort()
            for pid in goal_first + other:
                actions.append(f"EXIT{{{pid},{eid}}}")

        for eid in self._elevator_ids:
            floor = elevator_floor[eid]
            load = elevator_load[eid]
            candidates = []
            for pid in people_on_floor.get(floor, []):
                if load + self._person_weight[pid] <= self._capacities[eid]:
                    candidates.append(pid)
            candidates.sort(key=lambda pid: (-self._person_mean_reward[pid], pid))
            for pid in candidates:
                actions.append(f"ENTER{{{pid},{eid}}}")

        floor_scores = self._floor_scores(state)
        for eid in self._elevator_ids:
            floor = elevator_floor[eid]
            targets = []
            for target in self._reachable[eid]:
                score = floor_scores.get(target, 0.0)
                if target == floor: score += 0.1
                targets.append((score, target))
            targets.sort(key=lambda item: (-item[0], item[1]))
            keep_count = len(targets) if len(targets) <= 4 else 3
            for _, target in targets[:keep_count]:
                actions.append(f"MOVE{{{eid},{target}}}")

        actions.append("RESET")
        actions = tuple(actions)
        self._legal_action_cache[state] = actions
        return actions

    def _floor_scores(self, state):
        elevators_t, persons_t, _ = state
        score = {}
        total_reward = sum(self._person_mean_reward[pid] for pid, _ in persons_t)

        for eid, _, _ in elevators_t:
            for floor in self._reachable[eid]:
                score[floor] = score.get(floor, 0.0) + 0.15 * total_reward

        for pid, loc in persons_t:
            reward = self._person_mean_reward[pid]
            goal = self._person_goal[pid]
            score[goal] = score.get(goal, 0.0) + 1.50 * reward
            if loc[0] == "floor":
                score[loc[1]] = score.get(loc[1], 0.0) + 2.50 * reward
            else:
                floor = self._decode_state(state)[0][loc[1]]
                score[floor] = score.get(floor, 0.0) + 1.20 * reward

        for floor in score:
            reach_count = sum(1 for eid in self._elevator_ids if floor in self._reachable_sets[eid])
            score[floor] += 0.40 * reach_count

        return score

    # ------------------------------------------------------------------
    # Transition model
    # ------------------------------------------------------------------
    def _transition_outcomes(self, state, action):
        key = (state, action)
        cached = self._transition_cache.get(key)
        if cached is not None: return cached

        if action == "RESET":
            outcomes = ((1.0, self._initial_state, 0.0),)
        else:
            kind, first, second = self._parse_action(action)
            if kind == "MOVE": outcomes = self._move_outcomes(state, first, second)
            elif kind == "ENTER": outcomes = self._enter_outcomes(state, first, second)
            elif kind == "EXIT": outcomes = self._exit_outcomes(state, first, second)
            else: outcomes = ((1.0, state, 0.0),)

        self._transition_cache[key] = outcomes
        return outcomes

    def _move_outcomes(self, state, eid, target_floor):
        success_prob = self._elevator_prob[eid]
        elevator_floor, _, _, _, _, _ = self._decode_state(state)
        current_floor = elevator_floor[eid]

        success_state = self._replace_elevator(state, eid, target_floor)
        if success_prob >= 1.0:
            return ((1.0, success_state, 0.0),)

        failure_floors = tuple(sorted((self._reachable_sets[eid] - {target_floor}) | {current_floor}))
        if not failure_floors:
            return ((1.0, success_state, 0.0),)

        fail_prob = (1.0 - success_prob) / len(failure_floors)
        outcomes = [(success_prob, success_state, 0.0)]
        for floor in failure_floors:
            fail_state = self._replace_elevator(state, eid, floor)
            outcomes.append((fail_prob, fail_state, 0.0))
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
        elevators = list(elevators_t)
        for idx, (cur_eid, floor, load) in enumerate(elevators):
            if cur_eid == eid:
                elevators[idx] = (cur_eid, new_floor, load)
                break
        return (tuple(sorted(elevators)), persons_t, remaining)

    def _enter_success_state(self, state, pid, eid):
        elevators_t, persons_t, remaining = state
        elevators, persons = list(elevators_t), list(persons_t)
        weight = self._person_weight[pid]
        
        for idx, (cur_eid, floor, load) in enumerate(elevators):
            if cur_eid == eid:
                elevators[idx] = (cur_eid, floor, load + weight)
                break
        for idx, (cur_pid, loc) in enumerate(persons):
            if cur_pid == pid:
                persons[idx] = (cur_pid, ("in", eid))
                break
                
        return (tuple(sorted(elevators)), tuple(sorted(persons)), remaining)

    def _exit_success_state(self, state, pid, eid):
        elevators_t, persons_t, remaining = state
        elevators, persons = list(elevators_t), list(persons_t)
        current_floor = None
        
        for idx, (cur_eid, floor, load) in enumerate(elevators):
            if cur_eid == eid:
                current_floor = floor
                elevators[idx] = (cur_eid, floor, load - self._person_weight[pid])
                break

        if current_floor == self._person_goal[pid]:
            next_remaining = remaining - 1
            if next_remaining == 0:
                return self._initial_state, self._goal_reward + self._person_mean_reward[pid]
            next_persons = tuple(sorted((cur_pid, loc) for cur_pid, loc in persons if cur_pid != pid))
            return (tuple(sorted(elevators)), next_persons, next_remaining), self._person_mean_reward[pid]

        for idx, (cur_pid, loc) in enumerate(persons):
            if cur_pid == pid:
                persons[idx] = (cur_pid, ("floor", current_floor))
                break

        return (tuple(sorted(elevators)), tuple(sorted(persons)), remaining), 0.0

    # ------------------------------------------------------------------
    # Heuristic
    # ------------------------------------------------------------------
    def _heuristic(self, state, steps_left):
        key = (state, steps_left)
        cached = self._heuristic_cache.get(key)
        if cached is not None: return cached

        _, persons_t, _ = state
        if not persons_t:
            self._heuristic_cache[key] = self._goal_reward
            return self._goal_reward

        elevator_floor, elevator_load, _, _, _, _ = self._decode_state(state)
        total, total_cost = 0.0, 0.0

        for pid, loc in persons_t:
            cost = self._estimate_person_cost(state, pid, loc, elevator_floor, elevator_load)
            if math.isinf(cost):
                self._heuristic_cache[key] = -1e9
                return -1e9

            reward = self._person_mean_reward[pid]
            total += reward * (0.95 ** min(cost, steps_left))
            total_cost += cost

        total += self._goal_reward * (0.95 ** min(total_cost, steps_left))
        self._heuristic_cache[key] = total
        return total

    def _reset_penalty(self, state):
        _, persons_t, _ = state
        return 3.0 + 1.5 * len(persons_t)

    def _estimate_person_cost(self, state, pid, loc, elevator_floor, elevator_load):
        goal = self._person_goal[pid]
        weight = self._person_weight[pid]
        p_prob = self._person_prob[pid]

        if loc[0] == "floor":
            start_floor = loc[1]
            best = math.inf
            for eid in self._elevator_ids:
                if elevator_load[eid] + weight > self._capacities[eid] or start_floor not in self._reachable_sets[eid]:
                    continue
                move_to_pickup = 0.0 if elevator_floor[eid] == start_floor else (1.0 / self._elevator_prob[eid])
                enter_cost = 1.0 / p_prob

                if goal in self._reachable_sets[eid]:
                    move_to_goal = 0.0 if start_floor == goal else (1.0 / self._elevator_prob[eid])
                    cost = move_to_pickup + enter_cost + move_to_goal + enter_cost
                    best = min(best, cost)

                for eid2 in self._elevator_ids:
                    if eid2 == eid or goal not in self._reachable_sets[eid2] or elevator_load[eid2] + weight > self._capacities[eid2]:
                        continue
                    for meet in self._reachable_sets[eid] & self._reachable_sets[eid2]:
                        move_to_meet = 0.0 if meet == start_floor else (1.0 / self._elevator_prob[eid])
                        move_e2_to_meet = 0.0 if elevator_floor[eid2] == meet else (1.0 / self._elevator_prob[eid2])
                        move_e2_to_goal = 0.0 if meet == goal else (1.0 / self._elevator_prob[eid2])
                        cost = (move_to_pickup + enter_cost + move_to_meet + enter_cost
                                + move_e2_to_meet + enter_cost + move_e2_to_goal + enter_cost)
                        best = min(best, cost)
            return best

        eid = loc[1]
        current_floor = elevator_floor[eid]
        best = math.inf
        if goal in self._reachable_sets[eid]:
            move_to_goal = 0.0 if current_floor == goal else (1.0 / self._elevator_prob[eid])
            best = min(best, move_to_goal + (1.0 / p_prob))

        for eid2 in self._elevator_ids:
            if eid2 == eid or goal not in self._reachable_sets[eid2] or elevator_load[eid2] + weight > self._capacities[eid2]:
                continue
            for meet in self._reachable_sets[eid] & self._reachable_sets[eid2]:
                move_to_meet = 0.0 if current_floor == meet else (1.0 / self._elevator_prob[eid])
                move_e2_to_meet = 0.0 if elevator_floor[eid2] == meet else (1.0 / self._elevator_prob[eid2])
                move_e2_to_goal = 0.0 if meet == goal else (1.0 / self._elevator_prob[eid2])
                cost = move_to_meet + (1.0 / p_prob) + move_e2_to_meet + (1.0 / p_prob) + move_e2_to_goal + (1.0 / p_prob)
                best = min(best, cost)

        return best

    # ------------------------------------------------------------------
    # Deterministic A* planner (Exploitation Phase Only)
    # ------------------------------------------------------------------
    def _precompute_plan_costs(self):
        self._pcost = {}
        for pid in self._person_ids:
            goal = self._person_goal[pid]
            w = self._person_weight[pid]
            pprob = self._person_prob[pid]
            pc = (1.0 / pprob) if pprob > 0 else INF
            dist = {}
            pq = []
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
            goal = self._person_goal[pid]
            w = self._person_weight[pid]
            dist = {}
            pq = []
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
        ee_total = 0.0; move_max = 0.0
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
            ef = e_dict[eid][0]
            if ef == self._person_goal[pid]:
                cost = 1.0 / self._person_prob[pid]
                new_elevs = tuple((e[0], e[1], e[2] - self._person_weight[pid]) if e[0] == eid else e for e in elevs)
                new_persons = tuple(p for p in persons if p[0] != pid)
                mandatory.append((f"EXIT{{{pid},{eid}}}", (new_elevs, new_persons), cost))
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
                    succ.append((f"ENTER{{{pid},{eid}}}", (new_elevs, new_persons), cost))

        for pid, loc in persons:
            if pid not in targets or loc[0] != "in": continue
            eid = loc[1]
            ef = e_dict[eid][0]
            if ef != self._person_goal[pid] and ef in self._shared_floors:
                cost = 1.0 / self._person_prob[pid]
                new_elevs = tuple((e[0], e[1], e[2] - self._person_weight[pid]) if e[0] == eid else e for e in elevs)
                new_persons = tuple((p[0], ("floor", ef)) if p[0] == pid else p for p in persons)
                succ.append((f"EXIT{{{pid},{eid}}}", (new_elevs, new_persons), cost))

        interesting = set(self._shared_floors)
        for pid, loc in persons:
            if pid not in targets: continue
            if loc[0] == "floor": interesting.add(loc[1])
            else: interesting.add(self._person_goal[pid])
            
        for eid, (ef, ew) in e_dict.items():
            ep = self._elevator_prob[eid]
            cost = (1.0 / (ep * ep)) if ep > 0 else INF
            for tf in self._reachable_sets[eid]:
                if tf != ef and tf in interesting:
                    new_elevs = tuple((e[0], tf, e[2]) if e[0] == eid else e for e in elevs)
                    succ.append((f"MOVE{{{eid},{tf}}}", (new_elevs, persons), cost))
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

    def _get_plan(self, ps):
        cached = self._plan_cache.get(ps)
        if cached is not None: return cached
        plan = self._astar_plan(ps, self._full_targets)
        if plan is None: plan = []
        self._plan_cache[ps] = plan
        return plan

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def _select_mode(self):
        not_all_reliable = any(self._elevator_prob[e] < 0.95 for e in self._elevator_ids)
        if len(self._person_ids) > 3 or not not_all_reliable:
            return "planner"
        return "expectimax"

    @staticmethod
    def _parse_action(action):
        if action == "RESET": return ("RESET", None, None)
        kind, rest = action.split("{", 1)
        rest = rest[:-1]
        left, right = rest.split(",")
        return kind, int(left), int(right)