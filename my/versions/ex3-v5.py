# version 2 + yoav's ideas to make it faster.
"""AI assistance disclosure: drafted with Gemini.

Blazingly Fast Reactive RL Controller for the stochastic multi-elevator MDP.
Combines the speed of Greedy Action Selection (Static Priorities) with the 
accuracy of Probability-Weighted Dijkstra Maps (Expected Cost) for dynamic routing.
"""

import math
import heapq

import ext_elev

# Replace with your actual student ID
id = ["123456789"]

class Controller:
    """A blazingly fast greedy RL controller using Dynamic Expected Cost maps."""

    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self.reachable = self.game.get_reachable()
        self.capacities = self.game.get_capacities()
        
        initial_elevators, initial_persons, _ = self.game.get_initial_state()
        
        self.elevator_ids = tuple(sorted(self.capacities.keys()))
        self.person_ids = tuple(sorted(pid for pid, _ in initial_persons))

        self.all_floors = set()
        for floors in self.reachable.values():
            self.all_floors.update(floors)
        for _, floor, _ in initial_elevators:
            self.all_floors.add(floor)
        for pid, loc in initial_persons:
            if isinstance(loc, tuple) and loc[0] == "floor":
                self.all_floors.add(loc[1])
            self.all_floors.add(self.game.get_person_goal(pid))

        # ---- Statistics Tracking ----
        self.move_tries = {eid: 0 for eid in self.elevator_ids}
        self.move_succ = {eid: 0 for eid in self.elevator_ids}
        
        self.person_tries = {pid: 0 for pid in self.person_ids}
        self.person_succ = {pid: 0 for pid in self.person_ids}
        
        self.reward_sum = {pid: 0.0 for pid in self.person_ids}
        self.reward_count = {pid: 0 for pid in self.person_ids}

        self.max_obs_reward = 50.0
        self.t = 0
        
        self.last_state = None
        self.last_action_tuple = None

        # Distance map: p_cost[pid][node] = expected cost to goal
        self.p_cost = {}
        self._dirty_dist = True

    # ------------------------------------------------------------------
    # Probability & UCB Estimators
    # ------------------------------------------------------------------
    def _get_ep(self, eid):
        """Elevator Move Success Probability (Laplace Smoothed)"""
        return (self.move_succ[eid] + 1.0) / (self.move_tries[eid] + 2.0)

    def _get_pp(self, pid):
        """Person Action Success Probability (Laplace Smoothed)"""
        return (self.person_succ[pid] + 1.0) / (self.person_tries[pid] + 2.0)

    def _get_reward(self, pid):
        """Optimistic Expected Reward (UCB1)"""
        cnt = self.reward_count[pid]
        if cnt == 0:
            return self.max_obs_reward
        mean = self.reward_sum[pid] / cnt
        bonus = math.sqrt(2.0 * math.log(max(2, self.t)) / cnt)
        return mean + self.max_obs_reward * bonus

    # ------------------------------------------------------------------
    # Fast Stats Update
    # ------------------------------------------------------------------
    def _update_stats(self, curr_state):
        if not self.last_action_tuple or self.last_action_tuple[0] == "RESET":
            return
            
        kind, p1, p2 = self.last_action_tuple
        last_e, last_p, last_rem = self.last_state
        curr_e, curr_p, curr_rem = curr_state

        if kind == "MOVE":
            eid, tgt = p1, p2
            self.move_tries[eid] += 1
            curr_floor = next(f for e, f, w in curr_e if e == eid)
            if curr_floor == tgt:
                self.move_succ[eid] += 1
                self._dirty_dist = True
                
        elif kind == "ENTER":
            pid, eid = p1, p2
            self.person_tries[pid] += 1
            curr_loc = next((loc for p, loc in curr_p if p == pid), None)
            if curr_loc == ("in", eid):
                self.person_succ[pid] += 1
                self._dirty_dist = True
                
        elif kind == "EXIT":
            pid, eid = p1, p2
            self.person_tries[pid] += 1
            curr_loc = next((loc for p, loc in curr_p if p == pid), None)
            
            if curr_loc is None:
                # Successfully Delivered
                self.person_succ[pid] += 1
                self._dirty_dist = True
                reward = float(self.game.get_last_gained_reward())
                
                # Discount Goal Reward if it was the last person
                if curr_rem == 0 and last_rem == 1:
                    reward -= float(self.game.get_goal_reward())
                    
                self.reward_sum[pid] += reward
                self.reward_count[pid] += 1
                self.max_obs_reward = max(self.max_obs_reward, reward)
                
            elif curr_loc[0] == "floor":
                # Exited successfully to a transfer floor
                self.person_succ[pid] += 1
                self._dirty_dist = True

    # ------------------------------------------------------------------
    # Probability-Weighted Dijkstra (The Upgrade over yoav.py)
    # ------------------------------------------------------------------
    def _recompute_distances(self):
        """Builds a map of expected costs from any location to the goal for each person."""
        self.p_cost = {}
        for pid in self.person_ids:
            goal = self.game.get_person_goal(pid)
            weight = self.game.get_person_weight(pid)
            pc = 1.0 / self._get_pp(pid)  # Cost to Enter/Exit

            dist = {}
            pq = []

            # Initialize goals (exiting from any feasible elevator at the goal floor)
            for eid in self.elevator_ids:
                if goal in self.reachable[eid] and weight <= self.capacities[eid]:
                    node = ("in", eid, goal)
                    dist[node] = pc
                    heapq.heappush(pq, (pc, node))

            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, float('inf')):
                    continue
                
                if u[0] == "in":
                    eid, f = u[1], u[2]
                    ep = self._get_ep(eid)
                    mc = 1.0 / (ep * ep)  # Heavily penalize broken elevators
                    
                    # 1. Reverse MOVE: could have come from another reachable floor
                    for f2 in self.reachable[eid]:
                        if f2 != f:
                            v = ("in", eid, f2)
                            nd = d + mc
                            if nd < dist.get(v, float('inf')):
                                dist[v] = nd
                                heapq.heappush(pq, (nd, v))
                                
                    # 2. Reverse ENTER: could have boarded from this floor
                    v = ("floor", f)
                    nd = d + pc
                    if nd < dist.get(v, float('inf')):
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
                        
                else:
                    f = u[1]
                    # 3. Reverse EXIT: could have alighted from another elevator here
                    for eid in self.elevator_ids:
                        if f in self.reachable[eid] and weight <= self.capacities[eid]:
                            v = ("in", eid, f)
                            nd = d + pc
                            if nd < dist.get(v, float('inf')):
                                dist[v] = nd
                                heapq.heappush(pq, (nd, v))
                                
            self.p_cost[pid] = dist
            
        self._dirty_dist = False

    # ------------------------------------------------------------------
    # Engine Setup & Actions
    # ------------------------------------------------------------------
    def choose_next_action(self, state):
        self._update_stats(state)
        self.t += 1
        
        # Recompute dynamically only when probabilities change, or periodically to ensure convergence
        if self._dirty_dist or self.t % 25 == 0:
            self._recompute_distances()

        elevators_t, persons_t, _ = state
        elev_by_id = {e: {"floor": f, "weight": w} for e, f, w in elevators_t}

        passengers = {e: [] for e in self.elevator_ids}
        waiting = []
        for pid, loc in persons_t:
            if loc[0] == "in":
                passengers[loc[1]].append(pid)
            elif loc[0] == "floor":
                waiting.append((pid, loc[1]))

        candidates = []

        def get_cost(pid, node):
            return self.p_cost[pid].get(node, float('inf'))

        def score(pid, cost):
            if math.isinf(cost): return -float('inf')
            val = self._get_reward(pid)
            # Discount by Expected Cost (not flat hop count)
            return val * (0.95 ** cost)

        # --------------------------------------------------------------
        # Priority 3: Finish passengers at their goal
        # --------------------------------------------------------------
        for eid, pids in passengers.items():
            floor = elev_by_id[eid]["floor"]
            for pid in pids:
                if floor == self.game.get_person_goal(pid):
                    candidates.append((score(pid, 0), 3, ("EXIT", pid, eid)))

        # --------------------------------------------------------------
        # Priority 2: Board valuable waiting people
        # --------------------------------------------------------------
        for pid, floor in waiting:
            weight = self.game.get_person_weight(pid)
            cost_from_floor = get_cost(pid, ("floor", floor))
            for eid, info in elev_by_id.items():
                if info["floor"] != floor: continue
                if info["weight"] + weight > self.capacities[eid]: continue

                cost_from_in = get_cost(pid, ("in", eid, floor))
                # Only board if it actually makes routing progress!
                if cost_from_in < cost_from_floor or floor == self.game.get_person_goal(pid):
                    candidates.append((score(pid, cost_from_in), 2, ("ENTER", pid, eid)))

        # --------------------------------------------------------------
        # Priority 1: Move occupied elevators towards goal
        # --------------------------------------------------------------
        for eid, pids in passengers.items():
            if not pids: continue
            current_floor = elev_by_id[eid]["floor"]

            # Find best passenger inside
            best_pid = max(pids, key=lambda p: score(p, get_cost(p, ("in", eid, current_floor))))
            current_cost = get_cost(best_pid, ("in", eid, current_floor))

            improving_targets = []
            for target in self.reachable[eid]:
                if target == current_floor: continue
                tgt_cost = get_cost(best_pid, ("in", eid, target))
                if tgt_cost < current_cost:
                    improving_targets.append(target)

            for target in improving_targets:
                tgt_cost = get_cost(best_pid, ("in", eid, target))
                candidates.append((score(best_pid, tgt_cost), 1, ("MOVE", eid, target)))

            # Priority 0: If elevator is useless for passenger, exit to transfer
            if not improving_targets and current_floor != self.game.get_person_goal(best_pid):
                candidates.append((score(best_pid, current_cost), 0, ("EXIT", best_pid, eid)))

        # --------------------------------------------------------------
        # Priority 1: Empty elevators move towards waiting people
        # --------------------------------------------------------------
        for eid, info in elev_by_id.items():
            if passengers[eid]: continue
            current_floor = info["floor"]
            for pid, floor in waiting:
                if floor == current_floor: continue
                if floor not in self.reachable[eid]: continue
                if self.game.get_person_weight(pid) > self.capacities[eid]: continue

                ep = self._get_ep(eid)
                # Cost is distance from pickup + move cost
                dist_cost = get_cost(pid, ("floor", floor)) + (1.0 / (ep * ep))
                candidates.append((score(pid, dist_cost), 1, ("MOVE", eid, floor)))

        # --------------------------------------------------------------
        # Selection
        # --------------------------------------------------------------
        if candidates:
            # Sort by Priority (x[1]) first, then Score (x[0]), tiebreaker PID/EID (x[2][1])
            best_cand = max(candidates, key=lambda x: (x[1], x[0], x[2][1]))
            action_tuple = best_cand[2]
        else:
            action_tuple = ("RESET",)

        self.last_state = state
        self.last_action_tuple = action_tuple

        if action_tuple[0] == "RESET":
            return "RESET"
        return f"{action_tuple[0]}{{{action_tuple[1]},{action_tuple[2]}}}"