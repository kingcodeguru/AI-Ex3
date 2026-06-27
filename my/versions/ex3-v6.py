"""AI assistance disclosure: drafted with Gemini.

High-Performance Reactive Controller.
Implements dynamic cost tensors based on empirical Laplace smoothing
and a unified single-scalar utility function for action selection.
"""

import math
import heapq

import ext_elev

id = ["123456789"]

class Controller:
    def __init__(self, game: ext_elev.GameAPI):
        self.api = game
        self._cfg_reach = self.api.get_reachable()
        self._cfg_cap = self.api.get_capacities()
        
        start_lifts, start_pax, _ = self.api.get_initial_state()
        
        self._l_keys = tuple(sorted(self._cfg_cap.keys()))
        self._p_keys = tuple(sorted(p for p, _ in start_pax))

        self._domain_nodes = set()
        for r_set in self._cfg_reach.values():
            self._domain_nodes.update(r_set)
        for _, f, _ in start_lifts:
            self._domain_nodes.add(f)
        for p, loc in start_pax:
            if isinstance(loc, tuple) and loc[0] == "floor":
                self._domain_nodes.add(loc[1])
            self._domain_nodes.add(self.api.get_person_goal(p))

        self._l_att = {k: 0 for k in self._l_keys}
        self._l_ok = {k: 0 for k in self._l_keys}
        
        self._p_att = {k: 0 for k in self._p_keys}
        self._p_ok = {k: 0 for k in self._p_keys}
        
        self._yield_acc = {k: 0.0 for k in self._p_keys}
        self._yield_cnt = {k: 0 for k in self._p_keys}

        self._peak_bounty = 50.0
        self._tick = 0
        
        self._prev_s = None
        self._prev_a = None

        self._exp_cost = {}
        self._matrix_stale = True

    def _eval_lift_rel(self, l_id):
        return (self._l_ok[l_id] + 1.0) / (self._l_att[l_id] + 2.0)

    def _eval_pax_rel(self, p_id):
        return (self._p_ok[p_id] + 1.0) / (self._p_att[p_id] + 2.0)

    def _eval_bounty(self, p_id):
        n = self._yield_cnt[p_id]
        if n == 0:
            return self._peak_bounty
        avg = self._yield_acc[p_id] / n
        ucb_shift = math.sqrt(2.0 * math.log(max(2, self._tick)) / n)
        return avg + (self._peak_bounty * ucb_shift)

    def _incorporate_feedback(self, current_s):
        if not self._prev_a or self._prev_a == "RESET":
            return
            
        a_type, param_a, param_b = self._parse_instruction(self._prev_a)
        _, p_curr, rem_curr = current_s
        _, p_last, rem_last = self._prev_s

        if a_type == "MOVE":
            self._l_att[param_a] += 1
            f_now = next(f for e, f, w in current_s[0] if e == param_a)
            if f_now == param_b:
                self._l_ok[param_a] += 1
                self._matrix_stale = True
                
        elif a_type == "ENTER":
            self._p_att[param_a] += 1
            loc_now = next((loc for p, loc in p_curr if p == param_a), None)
            if loc_now == ("in", param_b):
                self._p_ok[param_a] += 1
                self._matrix_stale = True
                
        elif a_type == "EXIT":
            self._p_att[param_a] += 1
            loc_now = next((loc for p, loc in p_curr if p == param_a), None)
            
            if loc_now is None:
                self._p_ok[param_a] += 1
                self._matrix_stale = True
                raw_val = float(self.api.get_last_gained_reward())
                
                if rem_curr == 0 and rem_last == 1:
                    raw_val -= float(self.api.get_goal_reward())
                    
                self._yield_acc[param_a] += raw_val
                self._yield_cnt[param_a] += 1
                if raw_val > self._peak_bounty:
                    self._peak_bounty = raw_val
                
            elif loc_now[0] == "floor":
                self._p_ok[param_a] += 1
                self._matrix_stale = True

    def _solve_mdp_costs(self):
        self._exp_cost.clear()
        
        for p_id in self._p_keys:
            dest = self.api.get_person_goal(p_id)
            mass = self.api.get_person_weight(p_id)
            transit_penalty = 1.0 / self._eval_pax_rel(p_id)

            local_dist = {}
            fringe = []

            for l_id in self._l_keys:
                if dest in self._cfg_reach[l_id] and mass <= self._cfg_cap[l_id]:
                    tag = f"I_{l_id}_{dest}"
                    local_dist[tag] = transit_penalty
                    heapq.heappush(fringe, (transit_penalty, tag))

            while fringe:
                acc_d, vertex = heapq.heappop(fringe)
                if acc_d > local_dist.get(vertex, float('inf')):
                    continue
                
                parts = vertex.split("_")
                if parts[0] == "I":
                    curr_l = int(parts[1])
                    curr_f = int(parts[2])
                    
                    l_rel = self._eval_lift_rel(curr_l)
                    nav_penalty = 1.0 / (l_rel * l_rel)
                    
                    for adj_f in self._cfg_reach[curr_l]:
                        if adj_f != curr_f:
                            nbr = f"I_{curr_l}_{adj_f}"
                            alt_d = acc_d + nav_penalty
                            if alt_d < local_dist.get(nbr, float('inf')):
                                local_dist[nbr] = alt_d
                                heapq.heappush(fringe, (alt_d, nbr))
                                
                    nbr_floor = f"F_{curr_f}"
                    alt_d = acc_d + transit_penalty
                    if alt_d < local_dist.get(nbr_floor, float('inf')):
                        local_dist[nbr_floor] = alt_d
                        heapq.heappush(fringe, (alt_d, nbr_floor))
                        
                else:
                    curr_f = int(parts[1])
                    for curr_l in self._l_keys:
                        if curr_f in self._cfg_reach[curr_l] and mass <= self._cfg_cap[curr_l]:
                            nbr = f"I_{curr_l}_{curr_f}"
                            alt_d = acc_d + transit_penalty
                            if alt_d < local_dist.get(nbr, float('inf')):
                                local_dist[nbr] = alt_d
                                heapq.heappush(fringe, (alt_d, nbr))
                                
            self._exp_cost[p_id] = local_dist
            
        self._matrix_stale = False

    def choose_next_action(self, state):
        self._incorporate_feedback(state)
        self._tick += 1
        
        if self._matrix_stale or self._tick % 25 == 0:
            self._solve_mdp_costs()

        s_lifts, s_pax, _ = state
        
        lift_data = {}
        for l, f, w in s_lifts:
            lift_data[l] = {"f": f, "w": w, "manifest": []}
            
        hallway_queue = []
        for p, loc in s_pax:
            if loc[0] == "in":
                lift_data[loc[1]]["manifest"].append(p)
            elif loc[0] == "floor":
                hallway_queue.append((p, loc[1]))

        legal_moves = []

        def _calc_utility(target_p, v_tag):
            req_steps = self._exp_cost[target_p].get(v_tag, float('inf'))
            if math.isinf(req_steps): 
                return -1e9
            base_val = self._eval_bounty(target_p)
            return base_val * (0.95 ** req_steps)

        for l_id, data in lift_data.items():
            l_floor = data["f"]
            
            for p_id in data["manifest"]:
                if l_floor == self.api.get_person_goal(p_id):
                    utility = _calc_utility(p_id, f"I_{l_id}_{l_floor}")
                    legal_moves.append((utility + 1000000.0, f"EXIT{{{p_id},{l_id}}}"))
                    
            for p_id, p_floor in hallway_queue:
                if p_floor == l_floor and data["w"] + self.api.get_person_weight(p_id) <= self._cfg_cap[l_id]:
                    cost_boarded = self._exp_cost[p_id].get(f"I_{l_id}_{p_floor}", float('inf'))
                    cost_waiting = self._exp_cost[p_id].get(f"F_{p_floor}", float('inf'))
                    if cost_boarded < cost_waiting or p_floor == self.api.get_person_goal(p_id):
                        utility = _calc_utility(p_id, f"I_{l_id}_{p_floor}")
                        legal_moves.append((utility + 10000.0, f"ENTER{{{p_id},{l_id}}}"))

            if data["manifest"]:
                focal_p = max(data["manifest"], key=lambda p: _calc_utility(p, f"I_{l_id}_{l_floor}"))
                focal_u = _calc_utility(focal_p, f"I_{l_id}_{l_floor}")
                
                has_route = False
                for adj in self._cfg_reach[l_id]:
                    if adj != l_floor:
                        alt_u = _calc_utility(focal_p, f"I_{l_id}_{adj}")
                        if alt_u > focal_u:
                            has_route = True
                            legal_moves.append((alt_u, f"MOVE{{{l_id},{adj}}}"))
                            
                if not has_route and l_floor != self.api.get_person_goal(focal_p):
                    legal_moves.append((focal_u - 10000.0, f"EXIT{{{focal_p},{l_id}}}"))
                    
            else:
                for p_id, p_floor in hallway_queue:
                    if p_floor != l_floor and p_floor in self._cfg_reach[l_id] and self.api.get_person_weight(p_id) <= self._cfg_cap[l_id]:
                        ep_factor = 1.0 / (self._eval_lift_rel(l_id) ** 2)
                        raw_cost = self._exp_cost[p_id].get(f"F_{p_floor}", float('inf')) + ep_factor
                        base_val = self._eval_bounty(p_id)
                        proxy_u = base_val * (0.95 ** raw_cost)
                        legal_moves.append((proxy_u, f"MOVE{{{l_id},{p_floor}}}"))

        if legal_moves:
            best_directive = max(legal_moves, key=lambda item: item[0])[1]
        else:
            best_directive = "RESET"

        self._prev_s = state
        self._prev_a = best_directive
        return best_directive

    def _parse_instruction(self, inst):
        if inst == "RESET": return "RESET", None, None
        head, tail = inst.split("{", 1)
        left, right = tail[:-1].split(",")
        return head, int(left), int(right)