"""AI assistance disclosure: drafted with Gemini.

Algorithm: Bayesian Thompson Sampling + Expected Cost A* Routing.
Maintains Beta distributions for all hidden probabilities. At each step, 
it draws a sample from the posterior distributions and runs a greedy A* expected-cost planner over the sampled MDP.
"""

import math
import re
import heapq
import random

import ext_elev

id = ["123456789"]  # Insert your ID here

class Controller:
    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self.reachable = self.game.get_reachable()
        self.capacities = self.game.get_capacities()
        
        init_elevs, init_persons, _ = self.game.get_initial_state()
        self.elev_ids = tuple(sorted(self.capacities.keys()))
        self.person_ids = tuple(sorted(pid for pid, _ in init_persons))

        # ---- Bayesian Tracking (Alpha = Successes, Beta = Failures) ----
        # We use Beta(1.0, 0.1) as a highly optimistic prior to encourage exploration
        self.e_alpha = {e: 1.0 for e in self.elev_ids}
        self.e_beta = {e: 0.1 for e in self.elev_ids}
        
        self.p_alpha = {p: 1.0 for p in self.person_ids}
        self.p_beta = {p: 0.1 for p in self.person_ids}
        
        self.rewards = {p: [] for p in self.person_ids}
        self.max_reward = 40.0
        
        self.last_state = None
        self.last_action = None

        self.expected_cost_map = {}
        self.gamma = 0.90  # Discount factor for routing

    # ------------------------------------------------------------------
    # Thompson Sampling: Draw a hallucinated model from the Posteriors
    # ------------------------------------------------------------------
    def _sample_model(self):
        """Samples probabilities from the Beta distributions."""
        sampled_e_prob = {}
        for e in self.elev_ids:
            # random.betavariate samples a value between 0.0 and 1.0
            p = random.betavariate(self.e_alpha[e], self.e_beta[e])
            # Clamp minimally to avoid division by zero in cost calculations
            sampled_e_prob[e] = max(p, 0.01)
            
        sampled_p_prob = {}
        for p in self.person_ids:
            p_prob = random.betavariate(self.p_alpha[p], self.p_beta[p])
            sampled_p_prob[p] = max(p_prob, 0.01)
            
        sampled_rewards = {}
        for p in self.person_ids:
            if not self.rewards[p]:
                sampled_rewards[p] = self.max_reward
            else:
                # Thompson sampling for rewards: randomly pick an observed sample
                sampled_rewards[p] = random.choice(self.rewards[p])
                
        return sampled_e_prob, sampled_p_prob, sampled_rewards

    # ------------------------------------------------------------------
    # A* Expected Cost Planner (Calculated on the Sampled Model)
    # ------------------------------------------------------------------
    def _build_dijkstra_astar(self, sampled_e_prob, sampled_p_prob):
        """Builds expected step costs to the goal based on the sampled model."""
        self.expected_cost_map.clear()
        
        for pid in self.person_ids:
            goal = self.game.get_person_goal(pid)
            weight = self.game.get_person_weight(pid)
            
            # Expected cost to board/alight based on sampled probability
            cost_pax = 1.0 / sampled_p_prob[pid]

            dist = {}
            pq = []

            for eid in self.elev_ids:
                if goal in self.reachable[eid] and weight <= self.capacities[eid]:
                    node = ("in", eid, goal)
                    dist[node] = cost_pax
                    heapq.heappush(pq, (cost_pax, node))

            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, float('inf')): continue
                
                if u[0] == "in":
                    eid, f = u[1], u[2]
                    ep = sampled_e_prob[eid]
                    
                    # Expected cost to move (penalized heavily by failure chance)
                    cost_move = 1.0 / (ep * ep) 
                    
                    for f2 in self.reachable[eid]:
                        if f2 != f:
                            v = ("in", eid, f2)
                            if d + cost_move < dist.get(v, float('inf')):
                                dist[v] = d + cost_move
                                heapq.heappush(pq, (d + cost_move, v))
                                
                    v = ("floor", f)
                    if d + cost_pax < dist.get(v, float('inf')):
                        dist[v] = d + cost_pax
                        heapq.heappush(pq, (d + cost_pax, v))
                        
                else:
                    f = u[1]
                    for eid in self.elev_ids:
                        if f in self.reachable[eid] and weight <= self.capacities[eid]:
                            v = ("in", eid, f)
                            if d + cost_pax < dist.get(v, float('inf')):
                                dist[v] = d + cost_pax
                                heapq.heappush(pq, (d + cost_pax, v))
                                
            self.expected_cost_map[pid] = dist

    # ------------------------------------------------------------------
    # Feedback Engine: Update Bayesian Priors
    # ------------------------------------------------------------------
    def _parse_action_str(self, action_str):
        if action_str == "RESET": return "RESET", None, None
        match = re.fullmatch(r"(MOVE|ENTER|EXIT)\s*\{\s*(-?\d+)\s*,\s*(-?\d+)\s*\}", action_str)
        if match:
            return match.group(1), int(match.group(2)), int(match.group(3))
        return None, None, None

    def _update_distributions(self, curr_state):
        if not self.last_action or self.last_action == "RESET": return

        a_type, p1, p2 = self._parse_action_str(self.last_action)
        if not a_type: return

        last_elevs, last_persons, last_rem = self.last_state
        curr_elevs, curr_persons, curr_rem = curr_state

        curr_e_dict = {e: f for e, f, w in curr_elevs}
        curr_p_dict = dict(curr_persons)
        
        is_reset = (curr_state == self.game.get_initial_state())

        if a_type == "MOVE":
            eid, target = p1, p2
            if curr_e_dict.get(eid) == target:
                self.e_alpha[eid] += 1.0  # Success
            else:
                self.e_beta[eid] += 1.0   # Failure
                
        elif a_type == "ENTER":
            pid, eid = p1, p2
            if curr_p_dict.get(pid) == ("in", eid):
                self.p_alpha[pid] += 1.0  # Success
            else:
                self.p_beta[pid] += 1.0   # Failure
                
        elif a_type == "EXIT":
            pid, eid = p1, p2
            was_last = (last_rem == 1)
            delivered = (pid not in curr_p_dict) or (is_reset and was_last)
            
            if delivered or curr_p_dict.get(pid, (None, None))[0] == "floor":
                self.p_alpha[pid] += 1.0  # Success
            else:
                self.p_beta[pid] += 1.0   # Failure
            
            if delivered:
                gained = float(self.game.get_last_gained_reward())
                if is_reset and was_last:
                    gained -= float(self.game.get_goal_reward())
                if gained > 0:
                    self.rewards[pid].append(gained)
                    self.max_reward = max(self.max_reward, gained)

    # ------------------------------------------------------------------
    # Master Decision Function
    # ------------------------------------------------------------------
    def choose_next_action(self, state):
        # 1. Update the Beta distributions based on the previous outcome
        self._update_distributions(state)
        
        # 2. Thompson Sampling: Draw a hallucinated reality
        s_ep, s_pp, s_rew = self._sample_model()
        
        # 3. Build the optimal A* paths for this hallucinated reality
        self._build_dijkstra_astar(s_ep, s_pp)

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

        # Scorer evaluates action utility based on Thompson sampled rewards and Expected Cost
        def score(pid, expected_cost):
            if math.isinf(expected_cost): return -float('inf')
            return s_rew[pid] * (self.gamma ** expected_cost)

        # Evaluate EXIT actions (Highest Priority)
        for eid, pids in passengers.items():
            f = elev_state[eid]["f"]
            for pid in pids:
                if f == self.game.get_person_goal(pid):
                    candidates.append((score(pid, 0), 3, f"EXIT{{{pid},{eid}}}"))

        # Evaluate ENTER actions
        for pid, p_floor in waiting:
            weight = self.game.get_person_weight(pid)
            cost_from_hall = self.expected_cost_map[pid].get(("floor", p_floor), float('inf'))
            for eid, info in elev_state.items():
                if info["f"] == p_floor and info["w"] + weight <= self.capacities[eid]:
                    cost_from_in = self.expected_cost_map[pid].get(("in", eid, p_floor), float('inf'))
                    if cost_from_in < cost_from_hall or p_floor == self.game.get_person_goal(pid):
                        candidates.append((score(pid, cost_from_in), 2, f"ENTER{{{pid},{eid}}}"))

        # Evaluate MOVE actions for occupied elevators
        for eid, pids in passengers.items():
            if not pids: continue
            curr_f = elev_state[eid]["f"]
            
            focus_pid = max(pids, key=lambda p: score(p, self.expected_cost_map[p].get(("in", eid, curr_f), float('inf'))))
            curr_cost = self.expected_cost_map[focus_pid].get(("in", eid, curr_f), float('inf'))
            
            moved = False
            for target in self.reachable[eid]:
                if target == curr_f: continue
                tgt_cost = self.expected_cost_map[focus_pid].get(("in", eid, target), float('inf'))
                if tgt_cost < curr_cost:
                    moved = True
                    candidates.append((score(focus_pid, tgt_cost), 1, f"MOVE{{{eid},{target}}}"))
                    
            if not moved and curr_f != self.game.get_person_goal(focus_pid):
                candidates.append((score(focus_pid, curr_cost), 0, f"EXIT{{{focus_pid},{eid}}}"))

        # Evaluate MOVE actions for empty elevators
        for eid, info in elev_state.items():
            if passengers[eid]: continue
            for pid, p_floor in waiting:
                if p_floor != info["f"] and p_floor in self.reachable[eid]:
                    if self.game.get_person_weight(pid) <= self.capacities[eid]:
                        cost_travel = 1.0 / (s_ep[eid] * s_ep[eid])
                        cost_pickup = self.expected_cost_map[pid].get(("floor", p_floor), float('inf'))
                        candidates.append((score(pid, cost_pickup + cost_travel), 1, f"MOVE{{{eid},{p_floor}}}"))

        if candidates:
            # Sort by Priority Tier first, then Score
            best_action = max(candidates, key=lambda item: (item[1], item[0]))[2]
        else:
            best_action = "RESET"

        self.last_state = state
        self.last_action = best_action
        return best_action