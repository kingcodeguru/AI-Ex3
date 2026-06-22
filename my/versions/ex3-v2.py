# claude version
"""
ex3-2.py  –  Reinforcement-Learning controller for Assignment 3.

AI assistance disclosure: drafted with Claude (Anthropic).

Design overview
---------------
The horizon is split into two consecutive phases:

  EXPLORATION  (first alpha * horizon steps)
    The agent uses the budget to learn the three hidden quantities that matter
    most for planning:
      (a) each elevator's move success probability
      (b) each person's enter/exit success probability
      (c) the mean delivery reward for each person

    The exploration policy is *not* random – it is goal-directed:
      • It always delivers people whenever it can (exit on destination, enter
        when co-located) so it never wastes a step that gives free information.
      • It selects elevator-move targets and person-enter attempts that are
        still uncertain (fewest observations), prioritising the most
        "decision-relevant" entities (see _exploration_priority).
      • When enough observations have been gathered for a given entity
        (>= CONFIDENT_OBS) the agent stops deliberately probing it.

  EXPLOITATION  (remaining steps)
    Once the exploration budget is exhausted the controller switches to the
    hybrid expectimax + A* planner from Assignment 2, now parameterised with
    the *estimated* probabilities and reward means learned during exploration.
    The planner is essentially ex2.py transplanted here with the hidden
    probabilities replaced by our posterior estimates.

Parameters you can tune
-----------------------
ALPHA           fraction of horizon dedicated to exploration (default 0.25)
CONFIDENT_OBS   how many observations per entity before we stop targeting it
                deliberately (default 8)
UCB_C           exploration bonus coefficient (UCB1-style, used to rank which
                elevator/person to probe next)
"""

import math
import heapq
import itertools
import time

import ext_elev

# ---------------------------------------------------------------------------
# Tuneable hyper-parameters
# ---------------------------------------------------------------------------
ALPHA         = 0.25   # fraction of horizon used for exploration
CONFIDENT_OBS = 8      # observations needed before an entity is "known enough"
UCB_C         = 1.5    # UCB bonus weight for targeting uncertain entities
INF           = float("inf")

id = ["000000000"]


# ===========================================================================
# Statistics tracker
# ===========================================================================
class Stats:
    """Maintains running estimates for all hidden quantities."""

    def __init__(self, elevator_ids, person_ids):
        # Move success: successes / attempts per elevator
        self.move_success  = {e: 0 for e in elevator_ids}
        self.move_attempts = {e: 0 for e in elevator_ids}

        # Person action success: successes / attempts per person
        self.person_success  = {p: 0 for p in person_ids}
        self.person_attempts = {p: 0 for p in person_ids}

        # Delivery reward: list of observed samples per person
        self.reward_samples = {p: [] for p in person_ids}

        self.elevator_ids = list(elevator_ids)
        self.person_ids   = list(person_ids)

    # ---- update methods ---------------------------------------------------

    def record_move(self, eid, intended_floor, actual_floor):
        self.move_attempts[eid] += 1
        if actual_floor == intended_floor:
            self.move_success[eid] += 1

    def record_person_action(self, pid, succeeded):
        self.person_attempts[pid] += 1
        if succeeded:
            self.person_success[pid] += 1

    def record_delivery_reward(self, pid, reward):
        self.reward_samples[pid].append(reward)

    # ---- query methods ----------------------------------------------------

    def elevator_prob(self, eid):
        """Posterior estimate of elevator move success prob (Laplace smoothing)."""
        n = self.move_attempts[eid]
        s = self.move_success[eid]
        # Start optimistic (1.0) so unexplored elevators are tried first;
        # once we have observations, Laplace smoothing with alpha=1 prior.
        if n == 0:
            return 0.85  # optimistic prior
        return (s + 1) / (n + 2)

    def person_prob(self, pid):
        """Posterior estimate of person action success prob."""
        n = self.person_attempts[pid]
        s = self.person_success[pid]
        if n == 0:
            return 0.85  # optimistic prior
        return (s + 1) / (n + 2)

    def person_mean_reward(self, pid):
        """Estimated mean delivery reward."""
        samples = self.reward_samples[pid]
        if not samples:
            return 5.0  # optimistic prior; keeps unexplored people attractive
        return sum(samples) / len(samples)

    def elevator_confident(self, eid):
        return self.move_attempts[eid] >= CONFIDENT_OBS

    def person_confident(self, pid):
        return self.person_attempts[pid] >= CONFIDENT_OBS

    def all_confident(self):
        return (all(self.elevator_confident(e) for e in self.elevator_ids) and
                all(self.person_confident(p) for p in self.person_ids))

    # ---- UCB priority (higher = more worth exploring) --------------------

    def elevator_ucb(self, eid, total_moves):
        n = max(self.move_attempts[eid], 1)
        return UCB_C * math.sqrt(math.log(max(total_moves, 1)) / n)

    def person_ucb(self, pid, total_actions):
        n = max(self.person_attempts[pid], 1)
        return UCB_C * math.sqrt(math.log(max(total_actions, 1)) / n)


# ===========================================================================
# Controller
# ===========================================================================
class Controller:
    """
    Two-phase RL controller.

    Phase 1 (exploration): smart probing of all unknown quantities.
    Phase 2 (exploitation): hybrid expectimax / A* planner with learned model.
    """

    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self._max_steps    = int(game.get_max_steps())
        self._initial_state = game.get_initial_state()
        self._goal_reward  = float(game.get_goal_reward())

        self._capacities      = game.get_capacities()
        self._reachable       = {
            eid: tuple(sorted(floors))
            for eid, floors in game.get_reachable().items()
        }
        self._reachable_sets  = {
            eid: frozenset(floors)
            for eid, floors in self._reachable.items()
        }

        _, persons_t, _ = self._initial_state
        self._person_ids   = tuple(sorted(pid for pid, _ in persons_t))
        self._elevator_ids = tuple(sorted(self._capacities))

        self._person_weight = {
            pid: game.get_person_weight(pid) for pid in self._person_ids
        }
        self._person_goal = {
            pid: game.get_person_goal(pid) for pid in self._person_ids
        }

        # Statistics tracker (the heart of exploration)
        self.stats = Stats(self._elevator_ids, self._person_ids)

        # Exploration budget
        self._explore_budget = int(ALPHA * self._max_steps)
        self._exploration_done = False

        # Pending move: we need to remember what elevator we told to move
        # and to which floor, so we can record success/failure in next call.
        self._pending_move = None   # (eid, intended_floor)
        self._pending_person = None # (pid, action_type)
        self._prev_state    = None

        # Planner caches (populated lazily when exploitation starts)
        self._planner_ready = False
        self._state_info_cache  = {}
        self._legal_action_cache = {}
        self._transition_cache  = {}
        self._value_cache       = {}
        self._policy_cache      = {}
        self._heuristic_cache   = {}
        self._plan_cache        = {}
        self._plan_node_cap     = 60000

        # Shared floors (for transfer logic)
        self._shared_floors = set()
        _reaches = [self._reachable_sets[e] for e in self._elevator_ids]
        for _i in range(len(_reaches)):
            for _j in range(_i + 1, len(_reaches)):
                self._shared_floors |= (_reaches[_i] & _reaches[_j])

        self._full_targets = frozenset(self._person_ids)
        self._many_elevators = len(self._elevator_ids) >= 3

        # Counters for UCB
        self._total_moves   = 0
        self._total_actions = 0

    # =======================================================================
    # Public entry point
    # =======================================================================
    def choose_next_action(self, state):
        steps_done = self.game.get_current_steps()
        steps_left = self._max_steps - steps_done

        if steps_left <= 0:
            return "RESET"

        # ---- Update statistics from last action's outcome ---------------
        self._update_stats(state)

        # ---- Decide phase -----------------------------------------------
        in_exploration = (steps_done < self._explore_budget
                          and not self.stats.all_confident())

        if in_exploration:
            action = self._exploration_action(state)
        else:
            if not self._planner_ready:
                self._init_planner()
            action = self._exploitation_action(state, steps_left)

        # ---- Record what we're about to do so we can observe outcome -----
        self._prev_state = state
        self._pending_move   = None
        self._pending_person = None

        if action.startswith("MOVE"):
            _, eid, tgt = _parse_action(action)
            self._pending_move = (eid, tgt)
            self._total_moves += 1
        elif action.startswith("ENTER") or action.startswith("EXIT"):
            kind, pid, eid = _parse_action(action)
            self._pending_person = (pid, kind)
            self._total_actions += 1

        return action

    # =======================================================================
    # Statistics update
    # =======================================================================
    def _update_stats(self, new_state):
        """Observe outcome of the last action and update estimates."""
        if self._prev_state is None:
            return

        # -- Elevator move outcome -----------------------------------------
        if self._pending_move is not None:
            eid, intended_floor = self._pending_move
            # Find actual floor of elevator in new_state
            actual_floor = self._elevator_floor(new_state, eid)
            self.stats.record_move(eid, intended_floor, actual_floor)
            # Invalidate planner caches since probabilities changed
            self._planner_ready = False

        # -- Person action outcome -----------------------------------------
        if self._pending_person is not None:
            pid, action_kind = self._pending_person
            prev_loc = self._person_loc(self._prev_state, pid)
            new_loc  = self._person_loc(new_state, pid)

            if new_loc is None:
                # Person was delivered (disappeared from state)
                # Determine success (they exited onto goal floor)
                succeeded = True
            elif action_kind == "ENTER":
                succeeded = (prev_loc[0] == "floor" and new_loc[0] == "in")
            else:  # EXIT
                succeeded = (prev_loc[0] == "in" and new_loc[0] == "floor")

            self.stats.record_person_action(pid, succeeded)

            # Reward signal: only meaningful on EXIT at goal
            if new_loc is None:
                raw = self.game.get_last_gained_reward()
                # last_gained_reward may include goal_reward if this delivery
                # completed all persons; subtract it to isolate delivery reward
                delivery_reward = raw - (
                    self._goal_reward
                    if self._prev_state[2] == 1  # was last person
                    else 0.0
                )
                if delivery_reward > 0:
                    self.stats.record_delivery_reward(pid, delivery_reward)

            self._planner_ready = False

    # =======================================================================
    # Exploration phase
    # =======================================================================
    def _exploration_action(self, state):
        """
        Smart exploration: always grab free wins (deliver/board when co-located)
        but otherwise choose actions that maximise information about the
        least-known, most decision-relevant entities.
        """
        elevators_t, persons_t, total_remaining = state
        elevator_floor, elevator_load, person_loc, people_on_floor, \
            people_in_elevator, _ = self._decode_state(state)

        # 1. Opportunistic delivery: if someone is in an elevator on their goal
        #    floor, exit them. This is always the right thing to do AND teaches
        #    us their person_prob and reward at zero extra cost.
        for eid in self._elevator_ids:
            floor = elevator_floor[eid]
            for pid in people_in_elevator.get(eid, []):
                if floor == self._person_goal[pid]:
                    return f"EXIT{{{pid},{eid}}}"

        # 2. Opportunistic boarding: if someone is on the same floor as an
        #    elevator and not yet known (few observations), board them.
        best_enter = self._best_enter_action(
            elevator_floor, elevator_load, people_on_floor)
        if best_enter:
            return best_enter

        # 3. Opportunistic exit: move someone closer to goal or to transfer
        #    floor even if not yet at goal (teaches person_prob).
        non_goal_exit = self._best_nongf_exit(
            elevator_floor, people_in_elevator)
        if non_goal_exit:
            return non_goal_exit

        # 4. Move an elevator we haven't characterised yet, or that is
        #    most uncertain, to a useful floor (pickup or goal).
        return self._best_exploration_move(
            state, elevator_floor, elevator_load, people_on_floor,
            person_loc, people_in_elevator)

    def _best_enter_action(self, elevator_floor, elevator_load,
                           people_on_floor):
        """Return the ENTER with highest exploration value."""
        best_score = -INF
        best_action = None
        total = max(self._total_actions, 1)
        for eid in self._elevator_ids:
            floor = elevator_floor[eid]
            for pid in people_on_floor.get(floor, []):
                if elevator_load[eid] + self._person_weight[pid] > self._capacities[eid]:
                    continue
                # Score: UCB uncertainty + estimated mean reward
                ucb  = self.stats.person_ucb(pid, total)
                base = self.stats.person_mean_reward(pid)
                score = base + ucb
                if score > best_score:
                    best_score = score
                    best_action = f"ENTER{{{pid},{eid}}}"
        return best_action

    def _best_nongf_exit(self, elevator_floor, people_in_elevator):
        """
        Exit a person not at goal onto the floor they're on, prioritising
        persons we haven't observed enough, only if the floor is a useful
        stopping point (goal of person or a transfer floor) – otherwise
        skip to avoid wasted repositioning.
        """
        total = max(self._total_actions, 1)
        best_score = -INF
        best_action = None
        for eid in self._elevator_ids:
            floor = elevator_floor[eid]
            inside = people_in_elevator.get(eid, [])
            if not inside:
                continue
            for pid in inside:
                if floor == self._person_goal[pid]:
                    continue  # handled above
                # Only exit onto a useful floor
                if (floor not in self._shared_floors and
                        floor != self._person_goal[pid]):
                    continue
                if self.stats.person_confident(pid):
                    continue
                ucb   = self.stats.person_ucb(pid, total)
                score = ucb
                if score > best_score:
                    best_score = score
                    best_action = f"EXIT{{{pid},{eid}}}"
        return best_action

    def _best_exploration_move(self, state, elevator_floor, elevator_load,
                               people_on_floor, person_loc,
                               people_in_elevator):
        """
        Choose a MOVE that targets:
          • the most uncertain elevator, OR
          • the floor most likely to let us observe a person action next step.

        We rank each (elevator, floor) pair by a composite score:
            = elevator_ucb  +  floor_utility
        where floor_utility counts how many uncertain persons are on that floor
        or have it as a goal (we'd learn something useful next step).
        """
        total_moves   = max(self._total_moves, 1)
        total_actions = max(self._total_actions, 1)

        # Build floor utility map: floors that give future learning value
        floor_utility = {}
        for pid in self._person_ids:
            loc = person_loc.get(pid)
            if loc is None:
                continue
            if not self.stats.person_confident(pid):
                p_ucb = self.stats.person_ucb(pid, total_actions)
                r     = self.stats.person_mean_reward(pid)
                val   = p_ucb + r * 0.1
                # Floor where person stands → ENTER next step
                if loc[0] == "floor":
                    floor_utility[loc[1]] = floor_utility.get(loc[1], 0.0) + val
                # Person's goal → EXIT at goal next step
                g = self._person_goal[pid]
                floor_utility[g] = floor_utility.get(g, 0.0) + val

        best_score  = -INF
        best_action = "RESET"

        for eid in self._elevator_ids:
            cur_floor = elevator_floor[eid]
            e_ucb     = self.stats.elevator_ucb(eid, total_moves)
            e_conf    = self.stats.elevator_confident(eid)

            for target in self._reachable[eid]:
                if target == cur_floor:
                    continue  # moving to current floor is a no-op attempt

                # If elevator is confident, only move it to a useful floor
                if e_conf and floor_utility.get(target, 0.0) == 0.0:
                    continue

                fut = floor_utility.get(target, 0.0)
                score = e_ucb + fut

                if score > best_score:
                    best_score  = score
                    best_action = f"MOVE{{{eid},{target}}}"

        # Fallback: if no move found (everyone is confident or no useful floor)
        # just pick the first legal move.
        if best_action == "RESET":
            for eid in self._elevator_ids:
                for target in self._reachable[eid]:
                    if target != elevator_floor[eid]:
                        return f"MOVE{{{eid},{target}}}"

        return best_action

    # =======================================================================
    # Exploitation phase  (ex2-style planner with learned parameters)
    # =======================================================================
    def _init_planner(self):
        """Build planner structures from learned statistics."""
        self._state_info_cache   = {}
        self._legal_action_cache = {}
        self._transition_cache   = {}
        self._value_cache        = {}
        self._policy_cache       = {}
        self._heuristic_cache    = {}
        self._plan_cache         = {}

        # Snapshot of current estimates
        self._ep = {e: self.stats.elevator_prob(e)     for e in self._elevator_ids}
        self._pp = {p: self.stats.person_prob(p)       for p in self._person_ids}
        self._pr = {p: self.stats.person_mean_reward(p) for p in self._person_ids}

        self._precompute_plan_costs()
        self._precompute_move_costs()

        # Decide mode (same heuristic as ex2)
        _rl_like = self._is_reward_skewed()
        _floors_all = set()
        for e in self._elevator_ids:
            _floors_all |= self._reachable_sets[e]
        _maxf  = max(_floors_all) if _floors_all else 0
        _full  = frozenset(range(_maxf + 1))
        _full_range = all(self._reachable_sets[e] == _full for e in self._elevator_ids)
        _minw  = min(self._person_weight.values()) if self._person_weight else 1
        _multi = any(self._capacities[e] >= 2 * _minw for e in self._elevator_ids)
        self._force_expectimax = _rl_like or (_full_range and _multi)

        _min_w   = min(self._person_weight.values()) if self._person_weight else 1
        _single  = all(self._capacities[e] < 2 * _min_w for e in self._elevator_ids)
        _perfect = all(self._ep[e] >= 0.95 for e in self._elevator_ids)
        self._collapse_eligible = _single and _perfect
        self._collapse_moves    = self._collapse_eligible and not self._force_expectimax

        if self._force_expectimax:
            self._mode = "expectimax"
        else:
            self._mode = self._select_mode()
            self._collapse_moves = self._collapse_eligible and self._mode == "planner"

        self._planner_ready = True

    def _is_reward_skewed(self):
        vals = [self._pr[p] for p in self._person_ids]
        if not vals:
            return False
        mean = sum(vals) / len(vals)
        mx   = max(vals)
        return mx > 1.5 * mean

    def _exploitation_action(self, state, steps_left):
        return self._choose(state, steps_left, self._mode)

    def _choose(self, state, steps_left, mode):
        if mode == "expectimax":
            return self._expectimax_action(state, steps_left)
        ps   = (state[0], state[1])
        plan = self._get_plan(ps)
        if plan and len(plan) <= steps_left:
            return plan[0]
        return self._expectimax_action(state, steps_left)

    def _expectimax_action(self, state, steps_left):
        depth = self._choose_depth(state, steps_left)
        self._best_value(state, steps_left, depth)
        return self._policy_cache.get((state, steps_left, depth), "RESET")

    # ------------------------------------------------------------------
    # Search (identical structure to ex2)
    # ------------------------------------------------------------------
    def _choose_depth(self, state, steps_left):
        _, persons_t, _ = state
        remaining = len(persons_t)
        if steps_left <= 2:  return steps_left
        if remaining <= 1:   return min(steps_left, 8)
        if remaining <= 2:   return min(steps_left, 7)
        if remaining <= 3:   return min(steps_left, 7)
        return min(steps_left, 5)

    def _best_value(self, state, steps_left, depth_left):
        key = (state, steps_left, depth_left)
        cached = self._value_cache.get(key)
        if cached is not None:
            return cached

        if steps_left <= 0:
            self._value_cache[key]  = 0.0
            self._policy_cache[key] = "RESET"
            return 0.0

        if depth_left <= 0:
            v = self._heuristic(state, steps_left)
            self._value_cache[key]  = v
            self._policy_cache[key] = "RESET"
            return v

        best_value  = -math.inf
        best_action = "RESET"

        for action in self._legal_actions(state):
            q = self._action_value(state, action, steps_left, depth_left)
            if q > best_value:
                best_value  = q
                best_action = action

        self._value_cache[key]  = best_value
        self._policy_cache[key] = best_action
        return best_value

    def _action_value(self, state, action, steps_left, depth_left):
        outcomes    = self._transition_outcomes(state, action)
        next_steps  = steps_left - 1
        next_depth  = depth_left - 1
        total       = 0.0

        if self._collapse_moves and action[0] == "M":
            for idx, (prob, next_state, imm) in enumerate(outcomes):
                if prob <= 0.0:
                    continue
                if next_steps <= 0:
                    cont = 0.0
                elif idx == 0:
                    cont = self._best_value(next_state, next_steps, next_depth)
                else:
                    cont = self._heuristic(next_state, next_steps)
                total += prob * (imm + cont)
            return total

        for prob, next_state, imm in outcomes:
            if prob <= 0.0:
                continue
            cont = (self._best_value(next_state, next_steps, next_depth)
                    if next_steps > 0 else 0.0)
            total += prob * (imm + cont)

        if action == "RESET":
            total -= self._reset_penalty(state)
        return total

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------
    def _decode_state(self, state):
        cached = self._state_info_cache.get(state)
        if cached is not None:
            return cached

        elevators_t, persons_t, remaining = state
        elevator_floor = {}
        elevator_load  = {}
        for eid, floor, load in elevators_t:
            elevator_floor[eid] = floor
            elevator_load[eid]  = load

        person_loc        = {}
        people_on_floor   = {}
        people_in_elevator = {}
        for pid, loc in persons_t:
            person_loc[pid] = loc
            if loc[0] == "floor":
                people_on_floor.setdefault(loc[1], []).append(pid)
            else:
                people_in_elevator.setdefault(loc[1], []).append(pid)

        info = (elevator_floor, elevator_load, person_loc,
                people_on_floor, people_in_elevator, remaining)
        self._state_info_cache[state] = info
        return info

    def _legal_actions(self, state):
        cached = self._legal_action_cache.get(state)
        if cached is not None:
            return cached

        elevator_floor, elevator_load, _, people_on_floor, \
            people_in_elevator, _ = self._decode_state(state)

        actions = []

        # Exit: goal-floor exits first
        for eid in self._elevator_ids:
            floor  = elevator_floor[eid]
            inside = people_in_elevator.get(eid, [])
            goal_first, other = [], []
            for pid in inside:
                (goal_first if floor == self._person_goal[pid] else other).append(pid)
            goal_first.sort(key=lambda p: (-self._pr[p], p))
            other.sort()
            for pid in goal_first + other:
                actions.append(f"EXIT{{{pid},{eid}}}")

        # Enter
        for eid in self._elevator_ids:
            floor  = elevator_floor[eid]
            load   = elevator_load[eid]
            cands  = [pid for pid in people_on_floor.get(floor, [])
                      if load + self._person_weight[pid] <= self._capacities[eid]]
            cands.sort(key=lambda p: (-self._pr[p], p))
            for pid in cands:
                actions.append(f"ENTER{{{pid},{eid}}}")

        # Move (pruned)
        floor_scores = self._floor_scores(state)
        for eid in self._elevator_ids:
            floor   = elevator_floor[eid]
            targets = [(floor_scores.get(t, 0.0) + (0.1 if t == floor else 0.0), t)
                       for t in self._reachable[eid]]
            targets.sort(key=lambda x: (-x[0], x[1]))
            keep = len(targets) if len(targets) <= 4 else 3
            for _, t in targets[:keep]:
                actions.append(f"MOVE{{{eid},{t}}}")

        actions.append("RESET")
        actions = tuple(actions)
        self._legal_action_cache[state] = actions
        return actions

    def _floor_scores(self, state):
        elevators_t, persons_t, _ = state
        score = {}
        total_reward = sum(self._pr[pid] for pid, _ in persons_t)

        for eid, _, _ in elevators_t:
            for floor in self._reachable[eid]:
                score[floor] = score.get(floor, 0.0) + 0.15 * total_reward

        for pid, loc in persons_t:
            r    = self._pr[pid]
            goal = self._person_goal[pid]
            score[goal] = score.get(goal, 0.0) + 1.50 * r
            if loc[0] == "floor":
                score[loc[1]] = score.get(loc[1], 0.0) + 2.50 * r
            else:
                f = self._decode_state(state)[0][loc[1]]
                score[f] = score.get(f, 0.0) + 1.20 * r

        for floor in score:
            rc = sum(1 for e in self._elevator_ids if floor in self._reachable_sets[e])
            score[floor] += 0.40 * rc

        return score

    # ------------------------------------------------------------------
    # Transition model
    # ------------------------------------------------------------------
    def _transition_outcomes(self, state, action):
        key = (state, action)
        cached = self._transition_cache.get(key)
        if cached is not None:
            return cached

        if action == "RESET":
            outcomes = ((1.0, self._initial_state, 0.0),)
            self._transition_cache[key] = outcomes
            return outcomes

        kind, first, second = _parse_action(action)
        if kind == "MOVE":
            outcomes = self._move_outcomes(state, first, second)
        elif kind == "ENTER":
            outcomes = self._enter_outcomes(state, first, second)
        elif kind == "EXIT":
            outcomes = self._exit_outcomes(state, first, second)
        else:
            outcomes = ((1.0, state, 0.0),)

        self._transition_cache[key] = outcomes
        return outcomes

    def _move_outcomes(self, state, eid, target_floor):
        success_prob  = self._ep[eid]
        elevator_floor, _, _, _, _, _ = self._decode_state(state)
        current_floor = elevator_floor[eid]
        success_state = self._replace_elevator(state, eid, target_floor)

        if success_prob >= 1.0:
            return ((1.0, success_state, 0.0),)

        failure_floors = tuple(sorted(
            (self._reachable_sets[eid] - {target_floor}) | {current_floor}))
        if not failure_floors:
            return ((1.0, success_state, 0.0),)

        fail_prob = (1.0 - success_prob) / len(failure_floors)
        outcomes  = [(success_prob, success_state, 0.0)]
        for floor in failure_floors:
            outcomes.append((fail_prob, self._replace_elevator(state, eid, floor), 0.0))
        return tuple(outcomes)

    def _enter_outcomes(self, state, pid, eid):
        sp = self._pp[pid]
        ss = self._enter_success_state(state, pid, eid)
        if sp >= 1.0:
            return ((1.0, ss, 0.0),)
        return ((sp, ss, 0.0), (1.0 - sp, state, 0.0))

    def _exit_outcomes(self, state, pid, eid):
        sp = self._pp[pid]
        ss, reward = self._exit_success_state(state, pid, eid)
        if sp >= 1.0:
            return ((1.0, ss, reward),)
        return ((sp, ss, reward), (1.0 - sp, state, 0.0))

    def _replace_elevator(self, state, eid, new_floor):
        elevators_t, persons_t, remaining = state
        elevs = [(eid_, new_floor, l) if eid_ == eid else (eid_, f, l)
                 for eid_, f, l in elevators_t]
        return (tuple(sorted(elevs)), persons_t, remaining)

    def _enter_success_state(self, state, pid, eid):
        elevators_t, persons_t, remaining = state
        w    = self._person_weight[pid]
        elevs = [(e, f, l + w) if e == eid else (e, f, l) for e, f, l in elevators_t]
        pers  = [(p, ("in", eid)) if p == pid else (p, loc) for p, loc in persons_t]
        return (tuple(sorted(elevs)), tuple(sorted(pers)), remaining)

    def _exit_success_state(self, state, pid, eid):
        elevators_t, persons_t, remaining = state
        w = self._person_weight[pid]
        current_floor = None
        for e, f, l in elevators_t:
            if e == eid:
                current_floor = f
                break
        if current_floor is None:
            raise ValueError(f"Unknown elevator {eid}")

        elevs = [(e, f, l - w) if e == eid else (e, f, l) for e, f, l in elevators_t]

        if current_floor == self._person_goal[pid]:
            next_remaining = remaining - 1
            if next_remaining == 0:
                return self._initial_state, self._goal_reward + self._pr[pid]
            next_pers = tuple(sorted((p, loc) for p, loc in persons_t if p != pid))
            return (tuple(sorted(elevs)), next_pers, next_remaining), self._pr[pid]

        pers = [(p, ("floor", current_floor)) if p == pid else (p, loc)
                for p, loc in persons_t]
        return (tuple(sorted(elevs)), tuple(sorted(pers)), remaining), 0.0

    # ------------------------------------------------------------------
    # Heuristic
    # ------------------------------------------------------------------
    def _heuristic(self, state, steps_left):
        key = (state, steps_left)
        cached = self._heuristic_cache.get(key)
        if cached is not None:
            return cached

        _, persons_t, _ = state
        if not persons_t:
            self._heuristic_cache[key] = self._goal_reward
            return self._goal_reward

        elevator_floor, elevator_load, _, _, _, _ = self._decode_state(state)
        total = 0.0
        total_cost = 0.0

        for pid, loc in persons_t:
            cost = self._estimate_person_cost(pid, loc, elevator_floor, elevator_load)
            if math.isinf(cost):
                self._heuristic_cache[key] = -1e9
                return -1e9
            r = self._pr[pid]
            total      += r * (0.95 ** min(cost, steps_left))
            total_cost += cost

        total += self._goal_reward * (0.95 ** min(total_cost, steps_left))
        self._heuristic_cache[key] = total
        return total

    def _reset_penalty(self, state):
        _, persons_t, _ = state
        return 3.0 + 1.5 * len(persons_t)

    def _estimate_person_cost(self, pid, loc, elevator_floor, elevator_load):
        goal   = self._person_goal[pid]
        weight = self._person_weight[pid]
        p_prob = self._pp[pid]

        if loc[0] == "floor":
            start_floor = loc[1]
            best = INF
            for eid in self._elevator_ids:
                if elevator_load[eid] + weight > self._capacities[eid]:
                    continue
                if start_floor not in self._reachable_sets[eid]:
                    continue
                ep = self._ep[eid]
                m2p = 0.0 if elevator_floor[eid] == start_floor else 1.0 / ep
                ec  = 1.0 / p_prob
                if goal in self._reachable_sets[eid]:
                    m2g = 0.0 if start_floor == goal else 1.0 / ep
                    best = min(best, m2p + ec + m2g + ec)
                for eid2 in self._elevator_ids:
                    if eid2 == eid:
                        continue
                    shared = self._reachable_sets[eid] & self._reachable_sets[eid2]
                    if goal not in self._reachable_sets[eid2]:
                        continue
                    if elevator_load[eid2] + weight > self._capacities[eid2]:
                        continue
                    ep2 = self._ep[eid2]
                    for meet in shared:
                        m2m  = 0.0 if meet == start_floor else 1.0 / ep
                        em2m = 0.0 if elevator_floor[eid2] == meet else 1.0 / ep2
                        em2g = 0.0 if meet == goal else 1.0 / ep2
                        cost = m2p + ec + m2m + ec + em2m + ec + em2g + ec
                        best = min(best, cost)
            return best

        eid = loc[1]
        cur = elevator_floor[eid]
        ep  = self._ep[eid]
        best = INF
        if goal in self._reachable_sets[eid]:
            best = min(best, (0.0 if cur == goal else 1.0 / ep) + 1.0 / p_prob)
        for eid2 in self._elevator_ids:
            if eid2 == eid or goal not in self._reachable_sets[eid2]:
                continue
            shared = self._reachable_sets[eid] & self._reachable_sets[eid2]
            if elevator_load[eid2] + weight > self._capacities[eid2]:
                continue
            ep2 = self._ep[eid2]
            for meet in shared:
                m2m  = 0.0 if cur  == meet else 1.0 / ep
                em2m = 0.0 if elevator_floor[eid2] == meet else 1.0 / ep2
                em2g = 0.0 if meet == goal else 1.0 / ep2
                cost = m2m + 1.0 / p_prob + em2m + 1.0 / p_prob + em2g + 1.0 / p_prob
                best = min(best, cost)
        return best

    # ------------------------------------------------------------------
    # A* planner
    # ------------------------------------------------------------------
    def _precompute_plan_costs(self):
        self._pcost = {}
        for pid in self._person_ids:
            goal  = self._person_goal[pid]
            w     = self._person_weight[pid]
            pprob = self._pp[pid]
            pc    = (1.0 / pprob) if pprob > 0 else INF
            dist  = {}
            pq    = []
            for eid in self._elevator_ids:
                if goal in self._reachable_sets[eid] and w <= self._capacities[eid]:
                    node = ("in", eid, goal)
                    if pc < dist.get(node, INF):
                        dist[node] = pc
                        heapq.heappush(pq, (pc, node))
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, INF):
                    continue
                if u[0] == "in":
                    eid, f = u[1], u[2]
                    ep = self._ep[eid]
                    mc = (1.0 / (ep * ep)) if ep > 0 else INF
                    for f2 in self._reachable_sets[eid]:
                        if f2 != f:
                            v  = ("in", eid, f2)
                            nd = d + mc
                            if nd < dist.get(v, INF):
                                dist[v] = nd
                                heapq.heappush(pq, (nd, v))
                    v  = ("floor", f)
                    nd = d + pc
                    if nd < dist.get(v, INF):
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
                else:
                    f = u[1]
                    for eid in self._elevator_ids:
                        if f in self._reachable_sets[eid] and w <= self._capacities[eid]:
                            v  = ("in", eid, f)
                            nd = d + pc
                            if nd < dist.get(v, INF):
                                dist[v] = nd
                                heapq.heappush(pq, (nd, v))
            self._pcost[pid] = dist

    def _precompute_move_costs(self):
        self._mcost = {}
        for pid in self._person_ids:
            goal = self._person_goal[pid]
            w    = self._person_weight[pid]
            dist = {}
            pq   = []
            for eid in self._elevator_ids:
                if goal in self._reachable_sets[eid] and w <= self._capacities[eid]:
                    node = ("in", eid, goal)
                    if 0.0 < dist.get(node, INF):
                        dist[node] = 0.0
                        heapq.heappush(pq, (0.0, node))
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, INF):
                    continue
                if u[0] == "in":
                    eid, f = u[1], u[2]
                    ep = self._ep[eid]
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
                if pid not in targets:
                    continue
                node = (("floor", loc[1]) if loc[0] == "floor"
                        else ("in", loc[1], ef[loc[1]]))
                c = self._pcost[pid].get(node, INF)
                if c == INF:
                    return INF
                total += c
            return total
        ee_total = 0.0
        move_max = 0.0
        for pid, loc in persons:
            if pid not in targets:
                continue
            pc = 1.0 / self._pp[pid]
            if loc[0] == "floor":
                node = ("floor", loc[1])
                ee_total += 2.0 * pc
            else:
                node = ("in", loc[1], ef[loc[1]])
                ee_total += pc
            mc = self._mcost[pid].get(node, INF)
            if mc == INF:
                return INF
            if mc > move_max:
                move_max = mc
        return ee_total + move_max

    def _plan_successors(self, ps, targets):
        elevs, persons = ps
        e_dict = {e[0]: (e[1], e[2]) for e in elevs}

        mandatory = []
        for pid, loc in persons:
            if pid not in targets or loc[0] != "in":
                continue
            eid = loc[1]
            ef  = e_dict[eid][0]
            if ef == self._person_goal[pid]:
                cost     = 1.0 / self._pp[pid]
                new_elevs = tuple(
                    (e[0], e[1], e[2] - self._person_weight[pid]) if e[0] == eid else e
                    for e in elevs)
                new_persons = tuple(p for p in persons if p[0] != pid)
                mandatory.append((f"EXIT{{{pid},{eid}}}", (new_elevs, new_persons), cost))
        if mandatory:
            return mandatory

        succ = []
        for pid, loc in persons:
            if pid not in targets or loc[0] != "floor":
                continue
            f = loc[1]
            for eid, (ef, ew) in e_dict.items():
                if ef == f and ew + self._person_weight[pid] <= self._capacities[eid]:
                    cost = 1.0 / self._pp[pid]
                    new_elevs = tuple(
                        (e[0], e[1], e[2] + self._person_weight[pid]) if e[0] == eid else e
                        for e in elevs)
                    new_persons = tuple(
                        (p[0], ("in", eid)) if p[0] == pid else p for p in persons)
                    succ.append((f"ENTER{{{pid},{eid}}}", (new_elevs, new_persons), cost))

        for pid, loc in persons:
            if pid not in targets or loc[0] != "in":
                continue
            eid = loc[1]
            ef  = e_dict[eid][0]
            if ef != self._person_goal[pid] and ef in self._shared_floors:
                cost = 1.0 / self._pp[pid]
                new_elevs = tuple(
                    (e[0], e[1], e[2] - self._person_weight[pid]) if e[0] == eid else e
                    for e in elevs)
                new_persons = tuple(
                    (p[0], ("floor", ef)) if p[0] == pid else p for p in persons)
                succ.append((f"EXIT{{{pid},{eid}}}", (new_elevs, new_persons), cost))

        interesting = set(self._shared_floors)
        for pid, loc in persons:
            if pid not in targets:
                continue
            if loc[0] == "floor":
                interesting.add(loc[1])
            else:
                interesting.add(self._person_goal[pid])

        for eid, (ef, ew) in e_dict.items():
            ep   = self._ep[eid]
            cost = (1.0 / (ep * ep)) if ep > 0 else INF
            for tf in self._reachable_sets[eid]:
                if tf != ef and tf in interesting:
                    new_elevs = tuple(
                        (e[0], tf, e[2]) if e[0] == eid else e for e in elevs)
                    succ.append((f"MOVE{{{eid},{tf}}}", (new_elevs, persons), cost))

        return succ

    def _astar_plan(self, ps, targets):
        h0 = self._planning_heuristic(ps, targets)
        if h0 == INF:
            return None
        counter = itertools.count()
        pq      = [(h0, next(counter), 0.0, ps, ())]
        visited = set()
        expansions = 0
        while pq:
            f, _, g, st, path = heapq.heappop(pq)
            if not any(pid in targets for pid, _ in st[1]):
                return list(path)
            if st in visited:
                continue
            visited.add(st)
            expansions += 1
            if expansions > self._plan_node_cap:
                return None
            for act, nxt, cost in self._plan_successors(st, targets):
                if nxt in visited:
                    continue
                ng = g + cost
                hn = self._planning_heuristic(nxt, targets)
                if hn == INF:
                    continue
                heapq.heappush(pq, (ng + hn, next(counter), ng, nxt, path + (act,)))
        return None

    def _get_plan(self, ps):
        cached = self._plan_cache.get(ps)
        if cached is not None:
            return cached
        plan = self._astar_plan(ps, self._full_targets)
        if plan is None:
            plan = []
        self._plan_cache[ps] = plan
        return plan

    # ------------------------------------------------------------------
    # Mode selection
    # ------------------------------------------------------------------
    def _select_mode(self):
        not_all_reliable = any(self._ep[e] < 0.95 for e in self._elevator_ids)
        if len(self._person_ids) > 3 or not not_all_reliable:
            return "planner"
        t0    = time.time()
        self._expectimax_action(self._initial_state, self._max_steps)
        t_dec = time.time() - t0
        limit = 20.0 + 0.5 * self._max_steps
        affordable = (t_dec < 0.8) and (t_dec * self._max_steps < 0.5 * limit)
        return "expectimax" if affordable else "planner"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _elevator_floor(self, state, eid):
        for e, f, _ in state[0]:
            if e == eid:
                return f
        return None

    def _person_loc(self, state, pid):
        """Return loc tuple for pid in state, or None if delivered."""
        for p, loc in state[1]:
            if p == pid:
                return loc
        return None


# ===========================================================================
# Module-level helpers
# ===========================================================================
def _parse_action(action):
    """Parse 'KIND{a,b}' → (kind_str, int_a, int_b)."""
    if action == "RESET":
        return ("RESET", None, None)
    kind, rest = action.split("{", 1)
    rest       = rest[:-1]
    left, right = rest.split(",")
    return kind, int(left), int(right)