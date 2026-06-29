# thompson version
# 5 more optimization ideas
# better hyperparameters
# enable exit not to goal floor
# fix bug on ex4_easy

"""AI assistance disclosure: drafted with Gemini.

The Ultimate Bayesian RL Controller (Max-Reward Edition).
Replaced standard A* with a Discounted Max-Reward Best-First Search.
Perfectly coordinates multi-elevator transfers and optimal delivery schedules.
"""

import math
import heapq
import itertools
import random
from collections import Counter
import re

import ext_elev

INF = float("inf")
id = ["123456789"]

# ==============================================================================
# ⚙️ HYPERPARAMETERS & CONFIGURATION 
# ==============================================================================
ENSEMBLE_SIZE = 9               # Voted universes. 9 is the sweet spot for stability/speed.
ASTAR_NODE_LIMIT = 4000         # Max nodes expanded per universe before greedy fallback.
GAMMA = 0.95                    # Patience factor. 0.95 balances quick wins and long routes.
PRIOR_BETA_FAILURES = 0.1       # Extreme initial optimism for exploration.
INITIAL_ASSUMED_REWARD = 40.0   # Baseline reward assumption.
ENDGAME_STEPS_THRESHOLD = 50    # Lock in exact means (0 variance) for the final 50 steps.
SUBSET_FARMING_ROI_BOOST = 1.05 # Aggressive farming: loop if a subset is just 5% better.
MIN_PROBABILITY_CLAMP = 0.005   # Prevents expected cost explosions.
MIN_REWARD_VARIANCE = 0.05      # Minimal variance for Gaussian sampling.
# ==============================================================================

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

        # Shared floors for complex e3 transfers
        self._shared_floors = set()
        _reaches = list(self.reachable_sets.values())
        for i in range(len(_reaches)):
            for j in range(i + 1, len(_reaches)):
                self._shared_floors |= (_reaches[i] & _reaches[j])

        # Bayesian Posteriors
        self.e_alpha = {e: 1.0 for e in self.elev_ids}
        self.e_beta = {e: PRIOR_BETA_FAILURES for e in self.elev_ids}
        
        self.p_alpha = {p: 1.0 for p in self.person_ids}
        self.p_beta = {p: PRIOR_BETA_FAILURES for p in self.person_ids}
        
        self.r_n = {p: 0 for p in self.person_ids}
        self.r_sum = {p: 0.0 for p in self.person_ids}
        self.r_sum_sq = {p: 0.0 for p in self.person_ids}
        self.max_obs_reward = INITIAL_ASSUMED_REWARD
        
        self.last_state = None
        self.last_action = None

        self.pcost = {}
        self.mcost = {}

    # ------------------------------------------------------------------
    # Bayesian Sampling
    # ------------------------------------------------------------------
    def _sample_model(self, use_mean=False):
        s_ep, s_pp, s_rew = {}, {}, {}
        
        for e in self.elev_ids:
            if use_mean:
                s_ep[e] = self.e_alpha[e] / (self.e_alpha[e] + self.e_beta[e])
            else:
                s_ep[e] = random.betavariate(self.e_alpha[e], self.e_beta[e])
            s_ep[e] = max(s_ep[e], MIN_PROBABILITY_CLAMP)
            
        for p in self.person_ids:
            if use_mean:
                s_pp[p] = self.p_alpha[p] / (self.p_alpha[p] + self.p_beta[p])
            else:
                s_pp[p] = random.betavariate(self.p_alpha[p], self.p_beta[p])
            s_pp[p] = max(s_pp[p], MIN_PROBABILITY_CLAMP)
            
            n = self.r_n[p]
            if n == 0:
                s_rew[p] = self.max_obs_reward
            else:
                mu = self.r_sum[p] / n
                if use_mean or n < 2:
                    s_rew[p] = mu
                else:
                    var = (self.r_sum_sq[p] - (self.r_sum[p]**2)/n) / (n - 1)
                    var = max(var, MIN_REWARD_VARIANCE)  
                    s_rew[p] = random.gauss(mu, math.sqrt(var / n))
                    
        return s_ep, s_pp, s_rew

    # ------------------------------------------------------------------
    # Feedback Update
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
        endgame = (steps_left < ENDGAME_STEPS_THRESHOLD)

        m_ep, m_pp, m_rew = self._sample_model(use_mean=True)
        self._precompute_dijkstra(m_ep, m_pp)
        farming_targets = self._evaluate_subsets(m_rew)

        votes = []
        for _ in range(ENSEMBLE_SIZE):
            s_ep, s_pp, s_rew = self._sample_model(use_mean=endgame)
            self._precompute_dijkstra(s_ep, s_pp)
            
            ps = (state[0], state[1])
            plan = self._max_reward_plan(ps, farming_targets, s_ep, s_pp, s_rew)
            
            if plan:
                votes.append(plan[0])
            else:
                votes.append(self._reactive_greedy(state, farming_targets, s_ep, s_pp, s_rew))

        best_action = Counter(votes).most_common(1)[0][0]
        self.last_state = state
        self.last_action = best_action
        return best_action

    # ------------------------------------------------------------------
    # Static Combinatorial Subset Farming
    # ------------------------------------------------------------------
    def _evaluate_subsets(self, m_rew):
        init_s = self.game.get_initial_state()
        cost_cache = {}
        for pid, loc in init_s[1]:
            cost_cache[pid] = self.pcost[pid].get(("floor", loc[1]), INF)
            
        best_roi = -1.0
        best_subset = self.all_targets
        
        base_cost = sum(cost_cache[p] for p in self.all_targets)
        if base_cost < INF:
            base_reward = sum(m_rew[p] for p in self.all_targets) + self._goal_reward
            best_roi = base_reward / base_cost
            
        max_sub = min(5, len(self.person_ids) - 1)
        if max_sub < 1: return self.all_targets
        
        for r in range(1, max_sub + 1):
            for subset in itertools.combinations(self.person_ids, r):
                subset_cost = sum(cost_cache[p] for p in subset)
                if subset_cost < INF:
                    subset_cost += 1.0  
                    subset_reward = sum(m_rew[p] for p in subset)
                    roi = subset_reward / subset_cost
                    
                    if roi > best_roi * SUBSET_FARMING_ROI_BOOST:
                        best_roi = roi
                        best_subset = frozenset(subset)
                        
        return best_subset

    # ------------------------------------------------------------------
    # Deterministic MDP Builders (Dijkstra Expected Cost)
    # ------------------------------------------------------------------
    def _precompute_dijkstra(self, e_prob, p_prob):
        self.pcost = {}
        for pid in self.person_ids:
            goal, w, p_pr = self.goals[pid], self.weights[pid], p_prob[pid]
            pc = (1.0 / p_pr) if p_pr > 0 else INF
            
            dist, pq = {}, []
            
            for eid in self.elev_ids:
                if goal in self.reachable_sets[eid] and w <= self.capacities[eid]:
                    node = ("in", eid, goal)
                    dist[node] = pc; heapq.heappush(pq, (pc, node))
                    
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
                                
                    v = ("floor", f); nd = d + pc
                    if nd < dist.get(v, INF):
                        dist[v] = nd; heapq.heappush(pq, (nd, v))
                else:
                    f = u[1]
                    for eid in self.elev_ids:
                        if f in self.reachable_sets[eid] and w <= self.capacities[eid]:
                            v = ("in", eid, f); nd = d + pc
                            if nd < dist.get(v, INF):
                                dist[v] = nd; heapq.heappush(pq, (nd, v))
                                
            self.pcost[pid] = dist

    # ------------------------------------------------------------------
    # Max-Reward Deep Lookahead (Replaces Shortest-Path A*)
    # ------------------------------------------------------------------
    def _reward_heuristic(self, ps, targets, t, rew):
        """Calculates Maximum Possible Future Discounted Reward (Admissible)."""
        elevs, persons = ps
        ef = {e[0]: e[1] for e in elevs}
        
        h_reward = 0.0
        for pid, loc in persons:
            if pid not in targets: continue
            
            if loc[0] == "floor": 
                c = self.pcost[pid].get(("floor", loc[1]), INF)
            else: 
                c = self.pcost[pid].get(("in", loc[1], ef[loc[1]]), INF)
                
            if c == INF: return -INF  # Impossible to deliver someone
            
            # Future Reward = Base Reward * Discount^(Current_Time + Expected_Steps)
            h_reward += rew[pid] * (GAMMA ** (t + c))
            
        return h_reward

    def _plan_successors(self, ps, targets, p_prob, e_prob, rew):
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
                # Returns: Action, NextState, ExpectedTimeCost, EarnedReward
                mandatory.append((f"EXIT{{{pid},{eid}}}", (n_elevs, n_pers), cost, rew[pid]))
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
                    succ.append((f"ENTER{{{pid},{eid}}}", (n_elevs, n_pers), cost, 0.0))

        for pid, loc in persons:
            if pid not in targets or loc[0] != "in": continue
            eid = loc[1]
            ef = e_dict[eid][0]
            if ef != self.goals[pid] and ef in self._shared_floors:
                cost = 1.0 / p_prob[pid]
                n_elevs = tuple((e[0], e[1], e[2] - self.weights[pid]) if e[0] == eid else e for e in elevs)
                n_pers = tuple((p[0], ("floor", ef)) if p[0] == pid else p for p in persons)
                succ.append((f"EXIT{{{pid},{eid}}}", (n_elevs, n_pers), cost, 0.0))

        interesting = set(self._shared_floors)
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
                    succ.append((f"MOVE{{{eid},{tf}}}", (n_elevs, persons), cost, 0.0))
        return succ

    def _max_reward_plan(self, ps, targets, e_prob, p_prob, rew):
        """Best-First Search prioritizing absolute highest discounted reward."""
        t0 = 0.0
        g0 = 0.0
        h0 = self._reward_heuristic(ps, targets, t0, rew)
        if h0 == -INF: return None
        
        counter = itertools.count()
        # PQ format: (-f, counter, g, t, state, path) -> Pops maximum f (g+h)
        pq = [(-(g0 + h0), next(counter), g0, t0, ps, ())]
        
        visited = {} # maps state -> minimum expected time (t) reached
        expansions = 0
        
        best_plan = None
        best_g = -1.0
        
        while pq:
            neg_f, _, g, t, st, path = heapq.heappop(pq)
            f = -neg_f
            
            # Pruning: Optimistic maximum is worse than an established plan
            if best_plan and f <= best_g:
                break
                
            if not any(pid in targets for pid, _ in st[1]):
                if g > best_g:
                    best_g = g
                    best_plan = list(path)
                continue
                
            if st in visited and visited[st] <= t: continue
            visited[st] = t
            
            expansions += 1
            if expansions > ASTAR_NODE_LIMIT: break
            
            for act, nxt, cost, r_earned in self._plan_successors(st, targets, p_prob, e_prob, rew):
                t_nxt = t + cost
                # Apply the discount factor strictly based on expected timeline
                g_nxt = g + r_earned * (GAMMA ** t_nxt)
                
                h_nxt = self._reward_heuristic(nxt, targets, t_nxt, rew)
                if h_nxt == -INF: continue
                
                f_nxt = g_nxt + h_nxt
                heapq.heappush(pq, (-f_nxt, next(counter), g_nxt, t_nxt, nxt, path + (act,)))
                
        return best_plan if best_plan else None

    # ------------------------------------------------------------------
    # Fallback Reactive Greedy Router (Math-Aligned with A*)
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

        def score(pid, expected_cost):
            if expected_cost == INF: return -INF
            return rew[pid] * (GAMMA ** expected_cost)

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