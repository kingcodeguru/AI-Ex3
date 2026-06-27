# thompson version with 5 more optimization ideas
"""AI assistance disclosure: drafted with Gemini.

The Ultimate Bayesian RL Controller.
Integrates Ensemble Thompson Sampling, Deep Lookahead A* Planning,
Gaussian Reward Posteriors, Horizon-Aware Exploitation, and Subset Farming.
"""

import math
import heapq
import itertools
import random
from collections import Counter
import re

import ext_elev

INF = float("inf")

# Replace with your actual student ID
id = ["123456789"]

class Controller:
    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self.reachable = self.game.get_reachable()
        self.capacities = self.game.get_capacities()
        self._max_steps = int(game.get_max_steps())
        self._goal_reward = float(game.get_goal_reward())
        
        init_elevs, init_persons, _ = self.game.get_initial_state()
        self.elev_ids = tuple(sorted(self.capacities.keys()))
        self.person_ids = tuple(sorted(pid for pid, _ in init_persons))

        self.reachable_sets = {e: frozenset(f) for e, f in self.reachable.items()}
        self.all_targets = frozenset(self.person_ids)
        self.weights = {p: game.get_person_weight(p) for p in self.person_ids}
        self.goals = {p: game.get_person_goal(p) for p in self.person_ids}

        # 1. Beta Distributions (Transition Probabilities)
        # Prior Beta(1.0, 0.1) for optimistic exploration
        self.e_alpha = {e: 1.0 for e in self.elev_ids}
        self.e_beta = {e: 0.1 for e in self.elev_ids}
        
        self.p_alpha = {p: 1.0 for p in self.person_ids}
        self.p_beta = {p: 0.1 for p in self.person_ids}
        
        # 2. Normal-Gaussian Tracking (Rewards)
        self.r_n = {p: 0 for p in self.person_ids}
        self.r_sum = {p: 0.0 for p in self.person_ids}
        self.r_sum_sq = {p: 0.0 for p in self.person_ids}
        self.max_obs_reward = 40.0
        
        self.last_state = None
        self.last_action = None

        self.pcost = {}
        self.mcost = {}
        
        self.ensemble_size = 7
        self.astar_node_limit = 2000  # Bound to prevent TLE during ensemble

    # ------------------------------------------------------------------
    # Idea 4: True Bayesian Sampling (Beta & Normal-Gaussian)
    # ------------------------------------------------------------------
    def _sample_model(self, use_mean=False):
        """Draws a hallucinated MDP from the Bayesian posteriors."""
        s_ep, s_pp, s_rew = {}, {}, {}
        
        for e in self.elev_ids:
            if use_mean:
                s_ep[e] = self.e_alpha[e] / (self.e_alpha[e] + self.e_beta[e])
            else:
                s_ep[e] = random.betavariate(self.e_alpha[e], self.e_beta[e])
            s_ep[e] = max(s_ep[e], 0.01)
            
        for p in self.person_ids:
            if use_mean:
                s_pp[p] = self.p_alpha[p] / (self.p_alpha[p] + self.p_beta[p])
            else:
                s_pp[p] = random.betavariate(self.p_alpha[p], self.p_beta[p])
            s_pp[p] = max(s_pp[p], 0.01)
            
            # Normal-Gaussian Reward Sampling
            n = self.r_n[p]
            if n == 0:
                s_rew[p] = self.max_obs_reward
            else:
                mu = self.r_sum[p] / n
                if use_mean or n < 2:
                    s_rew[p] = mu
                else:
                    # Sample variance: (E[X^2] - E[X]^2) / (n-1)
                    var = (self.r_sum_sq[p] - (self.r_sum[p]**2)/n) / (n - 1)
                    var = max(var, 0.1)  # floor variance
                    s_rew[p] = random.gauss(mu, math.sqrt(var / n))
                    
        return s_ep, s_pp, s_rew

    # ------------------------------------------------------------------
    # Feedback Update (Posteriors)
    # ------------------------------------------------------------------
    def _parse_action_str(self, action_str):
        if action_str == "RESET": return "RESET", None, None
        match = re.fullmatch(r"(MOVE|ENTER|EXIT)\s*\{\s*(-?\d+)\s*,\s*(-?\d+)\s*\}", action_str)
        if match: return match.group(1), int(match.group(2)), int(match.group(3))
        return None, None, None

    def _update_posteriors(self, curr_state):
        if not self.last_action or self.last_action == "RESET": return

        a_type, p1, p2 = self._parse_action_str(self.last_action)
        if not a_type: return

        last_elevs, last_persons, last_rem = self.last_state
        curr_elevs, curr_persons, curr_rem = curr_state

        curr_e_dict = {e: f for e, f, w in curr_elevs}
        curr_p_dict = dict(curr_persons)
        is_reset = (curr_state == self.game.get_initial_state())

        if a_type == "MOVE":
            if curr_e_dict.get(p1) == p2:
                self.e_alpha[p1] += 1.0
            else:
                self.e_beta[p1] += 1.0
                
        elif a_type == "ENTER":
            if curr_p_dict.get(p1) == ("in", p2):
                self.p_alpha[p1] += 1.0
            else:
                self.p_beta[p1] += 1.0
                
        elif a_type == "EXIT":
            was_last = (last_rem == 1)
            delivered = (p1 not in curr_p_dict) or (is_reset and was_last)
            
            if delivered or curr_p_dict.get(p1, (None, None))[0] == "floor":
                self.p_alpha[p1] += 1.0
            else:
                self.p_beta[p1] += 1.0
            
            if delivered:
                gained = float(self.game.get_last_gained_reward())
                if is_reset and was_last:
                    gained -= float(self.game.get_goal_reward())
                if gained > 0:
                    self.r_n[p1] += 1
                    self.r_sum[p1] += gained
                    self.r_sum_sq[p1] += gained * gained
                    self.max_obs_reward = max(self.max_obs_reward, gained)

    # ------------------------------------------------------------------
    # Master Decision Loop
    # ------------------------------------------------------------------
    def choose_next_action(self, state):
        steps_left = self._max_steps - self.game.get_current_steps()
        if steps_left <= 0: return "RESET"

        self._update_posteriors(state)
        
        # Idea 3: Endgame Exploitation (Crush variance to 0)
        endgame = (steps_left < 45)

        # Idea 5: Subset Farming Evaluation
        # Draw the TRUE MEAN expected model to evaluate subsets
        m_ep, m_pp, m_rew = self._sample_model(use_mean=True)
        self._precompute_dijkstra(m_ep, m_pp)
        farming_targets = self._evaluate_subsets(state, m_rew)

        # Idea 1: Ensemble Voting Loop
        votes = []
        for _ in range(self.ensemble_size):
            s_ep, s_pp, s_rew = self._sample_model(use_mean=endgame)
            self._precompute_dijkstra(s_ep, s_pp)
            
            ps = (state[0], state[1])
            # Idea 2: Deep Lookahead A* Planning on the hallucinated MDP
            plan = self._astar_plan(ps, farming_targets, s_ep, s_pp, s_rew)
            
            if plan:
                votes.append(plan[0])
            else:
                # Fallback to lightning-fast Reactive Routing if A* hits node limit
                votes.append(self._reactive_greedy(state, farming_targets, s_ep, s_pp, s_rew))

        # Majority Vote
        best_action = Counter(votes).most_common(1)[0][0]

        self.last_state = state
        self.last_action = best_action
        return best_action

    # ------------------------------------------------------------------
    # Idea 5: Combinatorial Subset Farming
    # ------------------------------------------------------------------
    def _evaluate_subsets(self, state, m_rew):
        ps = (state[0], state[1])
        base_h = self._heuristic(ps, self.all_targets, m_rew)
        if base_h <= 0 or base_h == INF: return self.all_targets
        
        base_roi = sum(m_rew[p] for p in self.all_targets) / base_h
        best_roi = base_roi
        best_subset = self.all_targets
        
        max_sub = min(3, len(self.person_ids) - 1)
        for r in range(1, max_sub + 1):
            for subset in itertools.combinations(self.person_ids, r):
                subset = frozenset(subset)
                h = self._heuristic(ps, subset, m_rew)
                if h > 0 and h != INF:
                    roi = sum(m_rew[p] for p in subset) / h
                    # Require a 20% ROI improvement to abandon the rest of the map
                    if roi > best_roi * 1.20:
                        best_roi = roi
                        best_subset = subset
                        
        return best_subset

    # ------------------------------------------------------------------
    # Deterministic MDP Builders (Dijkstra Expected Cost)
    # ------------------------------------------------------------------
    def _precompute_dijkstra(self, e_prob, p_prob):
        self.pcost = {}
        self.mcost = {}
        
        for pid in self.person_ids:
            goal, w, p_pr = self.goals[pid], self.weights[pid], p_prob[pid]
            pc = (1.0 / p_pr) if p_pr > 0 else INF
            
            dist_p, dist_m = {}, {}
            pq_p, pq_m = [], []
            
            for eid in self.elev_ids:
                if goal in self.reachable_sets[eid] and w <= self.capacities[eid]:
                    node = ("in", eid, goal)
                    dist_p[node] = pc; heapq.heappush(pq_p, (pc, node))
                    dist_m[node] = 0.0; heapq.heappush(pq_m, (0.0, node))
                    
            self.pcost[pid] = self._run_dijkstra(dist_p, pq_p, w, pc, e_prob)
            self.mcost[pid] = self._run_dijkstra(dist_m, pq_m, w, 0.0, e_prob)

    def _run_dijkstra(self, dist, pq, w, base_cost, e_prob):
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, INF): continue
            
            if u[0] == "in":
                eid, f = u[1], u[2]
                ep = e_prob[eid]
                mc = (1.0 / (ep * ep)) if ep > 0 else INF
                
                for f2 in self.reachable_sets[eid]:
                    if f2 != f:
                        v = ("in", eid, f2); nd = d + mc
                        if nd < dist.get(v, INF):
                            dist[v] = nd; heapq.heappush(pq, (nd, v))
                            
                v = ("floor", f); nd = d + base_cost
                if nd < dist.get(v, INF):
                    dist[v] = nd; heapq.heappush(pq, (nd, v))
            else:
                f = u[1]
                for eid in self.elev_ids:
                    if f in self.reachable_sets[eid] and w <= self.capacities[eid]:
                        v = ("in", eid, f); nd = d + base_cost
                        if nd < dist.get(v, INF):
                            dist[v] = nd; heapq.heappush(pq, (nd, v))
        return dist

    def _heuristic(self, ps, targets, rew):
        elevs, persons = ps
        ef = {e[0]: e[1] for e in elevs}
        
        total_cost, max_move = 0.0, 0.0
        for pid, loc in persons:
            if pid not in targets: continue
            
            if loc[0] == "floor": 
                node = ("floor", loc[1])
                total_cost += 2.0 * self.pcost[pid].get(node, INF)
            else: 
                node = ("in", loc[1], ef[loc[1]])
                total_cost += self.pcost[pid].get(node, INF)
                
            mc = self.mcost[pid].get(node, INF)
            if mc == INF: return INF
            if mc > max_move: max_move = mc
            
        return total_cost + max_move

    # ------------------------------------------------------------------
    # Idea 2: Deep Lookahead (Bounded A* Planner)
    # ------------------------------------------------------------------
    def _plan_successors(self, ps, targets, p_prob, e_prob):
        elevs, persons = ps
        e_dict = {e[0]: (e[1], e[2]) for e in elevs}
        
        mandatory = []
        for pid, loc in persons:
            if pid not in targets or loc[0] != "in": continue
            eid = loc[1]
            if e_dict[eid][0] == self.goals[pid]:
                cost = 1.0 / p_prob[pid]
                n_elevs = tuple((e[0], e[1], e[2] - self.weights[pid]) if e[0] == eid else e for e in elevs)
                n_pers = tuple(p for p in persons if p[0] != pid)
                mandatory.append((f"EXIT{{{pid},{eid}}}", (n_elevs, n_pers), cost))
        if mandatory: return mandatory

        succ = []
        for pid, loc in persons:
            if pid not in targets or loc[0] != "floor": continue
            f = loc[1]
            for eid, (ef, ew) in e_dict.items():
                if ef == f and ew + self.weights[pid] <= self.capacities[eid]:
                    cost = 1.0 / p_prob[pid]
                    n_elevs = tuple((e[0], e[1], e[2] + self.weights[pid]) if e[0] == eid else e for e in elevs)
                    n_pers = tuple((p[0], ("in", eid)) if p[0] == pid else p for p in persons)
                    succ.append((f"ENTER{{{pid},{eid}}}", (n_elevs, n_pers), cost))

        interesting = set()
        for pid, loc in persons:
            if pid not in targets: continue
            if loc[0] == "floor": interesting.add(loc[1])
            else: interesting.add(self.goals[pid])
            
        for eid, (ef, _) in e_dict.items():
            ep = e_prob[eid]
            cost = 1.0 / (ep * ep)
            for tf in self.reachable_sets[eid]:
                if tf != ef and tf in interesting:
                    n_elevs = tuple((e[0], tf, e[2]) if e[0] == eid else e for e in elevs)
                    succ.append((f"MOVE{{{eid},{tf}}}", (n_elevs, persons), cost))
        return succ

    def _astar_plan(self, ps, targets, e_prob, p_prob, rew):
        h0 = self._heuristic(ps, targets, rew)
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
            if expansions > self.astar_node_limit: return None
            
            for act, nxt, cost in self._plan_successors(st, targets, p_prob, e_prob):
                if nxt in visited: continue
                ng = g + cost
                hn = self._heuristic(nxt, targets, rew)
                if hn == INF: continue
                heapq.heappush(pq, (ng + hn, next(counter), ng, nxt, path + (act,)))
        return None

    # ------------------------------------------------------------------
    # Fallback Reactive Greedy Router (O(N) backup when A* is too deep)
    # ------------------------------------------------------------------
    def _reactive_greedy(self, state, targets, e_prob, p_prob, rew):
        elevs, persons, _ = state
        elev_st = {e: {"f": f, "w": w} for e, f, w in elevs}
        
        pax = {e: [] for e in self.elev_ids}
        waiting = []
        for p, loc in persons:
            if loc[0] == "in": pax[loc[1]].append(p)
            elif loc[0] == "floor": waiting.append((p, loc[1]))

        candidates = []
        gamma = 0.90

        def score(pid, expected_cost):
            if math.isinf(expected_cost): return -INF
            return rew[pid] * (gamma ** expected_cost)

        for eid, pids in pax.items():
            f = elev_st[eid]["f"]
            for pid in pids:
                if f == self.goals[pid]:
                    candidates.append((score(pid, 0), 3, f"EXIT{{{pid},{eid}}}"))

        for pid, p_floor in waiting:
            if pid not in targets: continue
            c_hall = self.pcost[pid].get(("floor", p_floor), INF)
            for eid, info in elev_st.items():
                if info["f"] == p_floor and info["w"] + self.weights[pid] <= self.capacities[eid]:
                    c_in = self.pcost[pid].get(("in", eid, p_floor), INF)
                    if c_in < c_hall or p_floor == self.goals[pid]:
                        candidates.append((score(pid, c_in), 2, f"ENTER{{{pid},{eid}}}"))

        for eid, pids in pax.items():
            if not pids: continue
            curr_f = elev_st[eid]["f"]
            f_pid = max(pids, key=lambda p: score(p, self.pcost[p].get(("in", eid, curr_f), INF)))
            c_curr = self.pcost[f_pid].get(("in", eid, curr_f), INF)
            
            moved = False
            for tgt in self.reachable_sets[eid]:
                if tgt == curr_f: continue
                c_tgt = self.pcost[f_pid].get(("in", eid, tgt), INF)
                if c_tgt < c_curr:
                    moved = True
                    candidates.append((score(f_pid, c_tgt), 1, f"MOVE{{{eid},{tgt}}}"))
                    
            if not moved and curr_f != self.goals[f_pid]:
                candidates.append((score(f_pid, c_curr), 0, f"EXIT{{{f_pid},{eid}}}"))

        for eid, info in elev_st.items():
            if pax[eid]: continue
            for pid, p_floor in waiting:
                if pid not in targets: continue
                if p_floor != info["f"] and p_floor in self.reachable_sets[eid]:
                    if self.weights[pid] <= self.capacities[eid]:
                        c_travel = 1.0 / (e_prob[eid] * e_prob[eid])
                        c_pickup = self.pcost[pid].get(("floor", p_floor), INF)
                        candidates.append((score(pid, c_pickup + c_travel), 1, f"MOVE{{{eid},{p_floor}}}"))

        if candidates:
            return max(candidates, key=lambda i: (i[1], i[0]))[2]
        return "RESET"