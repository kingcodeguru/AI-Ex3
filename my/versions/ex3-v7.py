# gemini + yoav + tirgulim
"""AI assistance disclosure: drafted with Gemini.

Model-Based RL Controller using UCB Exploration and Expected-Cost Planning.
"""

import math
import re
import heapq

import ext_elev

id = ["123456789"]  # Insert your ID here

class Controller:
    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self.reachable = self.game.get_reachable()
        self.capacities = self.game.get_capacities()
        
        init_elevators, init_persons, _ = self.game.get_initial_state()
        self.elev_ids = tuple(sorted(self.capacities.keys()))
        self.person_ids = tuple(sorted(pid for pid, _ in init_persons))

        self.all_floors = set()
        for floors in self.reachable.values():
            self.all_floors.update(floors)
        for _, floor, _ in init_elevators:
            self.all_floors.add(floor)
        for pid, loc in init_persons:
            if isinstance(loc, tuple) and loc[0] == "floor":
                self.all_floors.add(loc[1])
            self.all_floors.add(self.game.get_person_goal(pid))

        # Model Estimation (Empirical Tracking)
        self.stat_move_att = {e: 0 for e in self.elev_ids}
        self.stat_move_ok = {e: 0 for e in self.elev_ids}
        
        self.stat_pax_att = {p: 0 for p in self.person_ids}
        self.stat_pax_ok = {p: 0 for p in self.person_ids}
        
        self.reward_sum = {p: 0.0 for p in self.person_ids}
        self.reward_cnt = {p: 0 for p in self.person_ids}

        self.highest_seen_reward = 40.0
        self.time_step = 0
        
        self.last_s = None
        self.last_a = None

        self.expected_cost_map = {}
        self.gamma = 0.90  # MDP Discount Factor

    # ------------------------------------------------------------------
    # RL Parameter Estimation (Slide 6 - Multi-Armed Bandits)
    # ------------------------------------------------------------------
    def _prob_elev(self, eid):
        """Laplace Smoothed Transition Probability for Elevators."""
        return (self.stat_move_ok[eid] + 1.0) / (self.stat_move_att[eid] + 2.0)

    def _prob_pax(self, pid):
        """Laplace Smoothed Transition Probability for Persons."""
        return (self.stat_pax_ok[pid] + 1.0) / (self.stat_pax_att[pid] + 2.0)

    def _ucb_value(self, pid):
        """UCB1 Exploration for Delivery Rewards."""
        n = self.reward_cnt[pid]
        if n == 0:
            return self.highest_seen_reward
        empirical_mean = self.reward_sum[pid] / n
        exploration_bonus = math.sqrt(2.0 * math.log(max(2, self.time_step)) / n)
        return empirical_mean + (self.highest_seen_reward * 0.5 * exploration_bonus)

    # ------------------------------------------------------------------
    # Environment Tracking (Model Building)
    # ------------------------------------------------------------------
    def _parse_action_str(self, action_str):
        action_str = str(action_str).strip()
        if action_str == "RESET": return "RESET", None, None
        match = re.fullmatch(r"(MOVE|ENTER|EXIT)\s*\{\s*(-?\d+)\s*,\s*(-?\d+)\s*\}", action_str)
        if match:
            return match.group(1), int(match.group(2)), int(match.group(3))
        return None, None, None

    def _update_model(self, curr_state):
        if self.last_s is None or self.last_a is None or self.last_a == "RESET":
            return

        a_type, param1, param2 = self._parse_action_str(self.last_a)
        if not a_type: return

        last_elevs, last_persons, last_rem = self.last_s
        curr_elevs, curr_persons, curr_rem = curr_state

        curr_e_dict = {e: f for e, f, w in curr_elevs}
        curr_p_dict = dict(curr_persons)
        last_p_dict = dict(last_persons)

        if a_type == "MOVE":
            eid, tgt = param1, param2
            self.stat_move_att[eid] += 1
            if curr_e_dict.get(eid) == tgt:
                self.stat_move_ok[eid] += 1
                
        elif a_type == "ENTER":
            pid, eid = param1, param2
            self.stat_pax_att[pid] += 1
            if curr_p_dict.get(pid) == ("in", eid):
                self.stat_pax_ok[pid] += 1
                
        elif a_type == "EXIT":
            pid, eid = param1, param2
            self.stat_pax_att[pid] += 1
            
            is_reset = (curr_state == self.game.get_initial_state())
            was_last = (last_rem == 1)
            
            delivered = (pid not in curr_p_dict) or (is_reset and was_last)
            if delivered or curr_p_dict.get(pid, (None, None))[0] == "floor":
                self.stat_pax_ok[pid] += 1
            
            if delivered:
                gained = float(self.game.get_last_gained_reward())
                if is_reset and was_last:
                    gained -= float(self.game.get_goal_reward())
                
                self.reward_sum[pid] += gained
                self.reward_cnt[pid] += 1
                self.highest_seen_reward = max(self.highest_seen_reward, gained)

    # ------------------------------------------------------------------
    # Model-Based Planning: Expected Cost via Dijkstra (Slide 5 - MDP)
    # ------------------------------------------------------------------
    def _compute_mdp_costs(self):
        """Builds a backward map of the *expected number of steps* to goal."""
        self.expected_cost_map.clear()
        
        for pid in self.person_ids:
            goal = self.game.get_person_goal(pid)
            weight = self.game.get_person_weight(pid)
            cost_pax_action = 1.0 / self._prob_pax(pid)

            dist = {}
            pq = []

            for eid in self.elev_ids:
                if goal in self.reachable[eid] and weight <= self.capacities[eid]:
                    node = ("in", eid, goal)
                    dist[node] = cost_pax_action
                    heapq.heappush(pq, (cost_pax_action, node))

            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, float('inf')): continue
                
                if u[0] == "in":
                    eid, f = u[1], u[2]
                    ep = self._prob_elev(eid)
                    # Square the penalty to aggressively avoid broken elevators
                    cost_move = 1.0 / (ep * ep) 
                    
                    for f2 in self.reachable[eid]:
                        if f2 != f:
                            v = ("in", eid, f2)
                            new_d = d + cost_move
                            if new_d < dist.get(v, float('inf')):
                                dist[v] = new_d
                                heapq.heappush(pq, (new_d, v))
                                
                    v = ("floor", f)
                    new_d = d + cost_pax_action
                    if new_d < dist.get(v, float('inf')):
                        dist[v] = new_d
                        heapq.heappush(pq, (new_d, v))
                        
                else:
                    f = u[1]
                    for eid in self.elev_ids:
                        if f in self.reachable[eid] and weight <= self.capacities[eid]:
                            v = ("in", eid, f)
                            new_d = d + cost_pax_action
                            if new_d < dist.get(v, float('inf')):
                                dist[v] = new_d
                                heapq.heappush(pq, (new_d, v))
                                
            self.expected_cost_map[pid] = dist

    # ------------------------------------------------------------------
    # Reactive Action Engine
    # ------------------------------------------------------------------
    def _calc_score(self, pid, expected_cost):
        if math.isinf(expected_cost): return -float('inf')
        # Utility = UCB_Reward * (Discount_Factor ^ Expected_Steps)
        return self._ucb_value(pid) * (self.gamma ** expected_cost)

    def choose_next_action(self, state):
        self._update_model(state)
        self.time_step += 1
        
        # Periodically re-evaluate the MDP as probabilities converge
        if self.time_step % 10 == 1 or not self.expected_cost_map:
            self._compute_mdp_costs()

        elevs, persons, _ = state
        elev_state = {e: {"f": f, "w": w} for e, f, w in elevs}
        
        passengers = {e: [] for e in self.elev_ids}
        waiting = []
        for p, loc in persons:
            if loc[0] == "in":
                passengers[loc[1]].append(p)
            elif loc[0] == "floor":
                waiting.append((p, loc[1]))

        candidates = []

        # Tier 3: Exit at exact goal
        for eid, pids in passengers.items():
            f = elev_state[eid]["f"]
            for pid in pids:
                if f == self.game.get_person_goal(pid):
                    candidates.append((self._calc_score(pid, 0), 3, f"EXIT{{{pid},{eid}}}"))

        # Tier 2: Board waiting passengers if it progresses them
        for pid, p_floor in waiting:
            weight = self.game.get_person_weight(pid)
            cost_from_hall = self.expected_cost_map[pid].get(("floor", p_floor), float('inf'))
            
            for eid, info in elev_state.items():
                if info["f"] != p_floor or info["w"] + weight > self.capacities[eid]:
                    continue
                    
                cost_from_in = self.expected_cost_map[pid].get(("in", eid, p_floor), float('inf'))
                
                # Only board if it mathematically shortens the expected path to the goal
                if cost_from_in < cost_from_hall or p_floor == self.game.get_person_goal(pid):
                    candidates.append((self._calc_score(pid, cost_from_in), 2, f"ENTER{{{pid},{eid}}}"))

        # Tier 1: Move occupied elevators to minimize expected cost
        for eid, pids in passengers.items():
            if not pids: continue
            curr_f = elev_state[eid]["f"]
            
            # Focus the elevator on the passenger with the highest immediate utility
            focus_pid = max(pids, key=lambda p: self._calc_score(p, self.expected_cost_map[p].get(("in", eid, curr_f), float('inf'))))
            curr_cost = self.expected_cost_map[focus_pid].get(("in", eid, curr_f), float('inf'))
            
            moved = False
            for target in self.reachable[eid]:
                if target == curr_f: continue
                tgt_cost = self.expected_cost_map[focus_pid].get(("in", eid, target), float('inf'))
                if tgt_cost < curr_cost:
                    moved = True
                    candidates.append((self._calc_score(focus_pid, tgt_cost), 1, f"MOVE{{{eid},{target}}}"))
                    
            # Tier 0: If the elevator can't progress this passenger, drop them off (Transfer)
            if not moved and curr_f != self.game.get_person_goal(focus_pid):
                candidates.append((self._calc_score(focus_pid, curr_cost), 0, f"EXIT{{{focus_pid},{eid}}}"))

        # Tier 1: Move empty elevators towards highest-utility waiting passengers
        for eid, info in elev_state.items():
            if passengers[eid]: continue
            curr_f = info["f"]
            
            for pid, p_floor in waiting:
                if p_floor == curr_f or p_floor not in self.reachable[eid]:
                    continue
                if self.game.get_person_weight(pid) > self.capacities[eid]:
                    continue
                    
                ep = self._prob_elev(eid)
                cost_of_travel = 1.0 / (ep * ep)
                cost_from_pickup = self.expected_cost_map[pid].get(("floor", p_floor), float('inf'))
                
                total_cost = cost_from_pickup + cost_of_travel
                candidates.append((self._calc_score(pid, total_cost), 1, f"MOVE{{{eid},{p_floor}}}"))

        if candidates:
            best_action = max(candidates, key=lambda item: (item[1], item[0]))[2]
        else:
            best_action = "RESET"

        self.last_s = state
        self.last_a = best_action
        return best_action