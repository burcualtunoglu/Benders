"""lbbd_heuristics.py — incumbent generators (HEURISTIC ONLY: only ever produce a valid LB).

These NEVER produce cuts and NEVER touch the UB. Each returned decision is validated by the
solver via forward_pass + subproblem before its value is admitted to the LB.

With no pre-suppression, a burning cell forces all its neighbours to burn ((40)+(41)+(43) with
u^pre=0), so the only consistent "no pre-control" structure is the full ignition cascade from
the roots. That cascade (with an optional post-control set) is always master-feasible, giving a
guaranteed valid LB (u ≡ 0 ⇒ LB = Σ_{y_i=0} π_i).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lbbd_v2.benders_forward import forward_pass, FWD_OK
from lbbd_v2.lbbd_subproblem_resource import solve_resource_subproblem, SPStatus


@dataclass
class Incumbent:
    y: Dict[int, float]
    z: Dict[Tuple[int, int], float]
    q: Dict[Tuple[int, int], float]
    u_pre: Dict[int, float]
    u_post: Dict[int, float]
    C: List[int] = field(default_factory=list)
    regime: Dict[int, str] = field(default_factory=dict)
    value: Optional[float] = None
    # Optional full-integer decision for a monolithic MIPStart (var-name -> {key: value}).
    # Populated only by build_greedy_incumbent_fast; other builders leave it empty.
    start_vars: Dict[str, Dict] = field(default_factory=dict)


def _full_cascade(inst) -> Tuple[Dict, Dict, Dict, List[int]]:
    """Shortest-ignition-time (Dijkstra) forest from the roots; every reachable cell burns.

    The q-forest MUST be the shortest-ignition-time tree, NOT an arbitrary BFS tree. With
    u^pre=0 every burning cell keeps all its z-arcs ((40)), and forward-pass Control B (46)
    requires t^s_j ≤ t^m_i = t^s_i + α/λ_i on every z-arc. Taking q as the Dijkstra tree makes
    t^s_j = τ^lb_j, and Dijkstra optimality gives τ^lb_j ≤ τ^lb_i + α/λ_i for EVERY neighbour
    edge (i,j) — so (46) holds on all z-arcs and the cascade is forward-CONSISTENT. A BFS tree
    can make a tree path longer than the shortest path, breaking (46) on cross edges (the reason
    the old BFS cascade was FWD_TIME_INCONSISTENT on every 2-D grid → no usable incumbent).
    """
    INF = float("inf")
    dist: Dict[int, float] = {i: INF for i in inst.Nf}
    parent: Dict[int, int] = {}
    pq: List[Tuple[float, int]] = []
    for r in inst.Na:
        dist[r] = 0.0
        heapq.heappush(pq, (0.0, r))
    reached = set()
    while pq:
        du, u = heapq.heappop(pq)
        if u in reached:
            continue
        reached.add(u)
        w = inst.alpha / inst.lam[u]
        for j in inst.neighbors[u]:
            nd = du + w
            if nd < dist[j] - 1e-12:
                dist[j] = nd
                parent[j] = u
                heapq.heappush(pq, (nd, j))
    y = {i: (1.0 if i in reached else 0.0) for i in inst.Nf}
    q = {a: 0.0 for a in inst.arcs}
    for j, i in parent.items():
        q[(i, j)] = 1.0                      # ignition arc i -> j (shortest-time predecessor)
    # z: with u^pre=0, every burning cell spreads to all burning neighbours (40)+(41)
    z = {a: 0.0 for a in inst.arcs}
    for (i, j) in inst.arcs:
        if y[i] > 0.5 and y[j] > 0.5:
            z[(i, j)] = 1.0
    burning = sorted(i for i in inst.Nf if y[i] > 0.5)
    return y, z, q, burning


def build_fallback_incumbent(inst, pre, cfg) -> Incumbent:
    y, z, q, _ = _full_cascade(inst)
    u_pre = {i: 0.0 for i in inst.Nf}
    u_post = {i: 0.0 for i in inst.Nf}
    return Incumbent(y=y, z=z, q=q, u_pre=u_pre, u_post=u_post, C=[], regime={})


def build_greedy_incumbent(inst, pre, cfg, budget: Optional[float] = None,
                           sp_solve=None) -> Incumbent:
    """Full cascade + greedily post-control burning cells (high π first), validated by SP."""
    sp_solve = sp_solve or solve_resource_subproblem
    y, z, q, burning = _full_cascade(inst)
    u_pre = {i: 0.0 for i in inst.Nf}
    u_post = {i: 0.0 for i in inst.Nf}
    inc = Incumbent(y=y, z=z, q=q, u_pre=dict(u_pre), u_post=dict(u_post))

    fwd = forward_pass(inc, inst)
    if fwd.status != FWD_OK:
        return inc

    C: List[int] = []
    best_val: Optional[float] = None
    for i in sorted(burning, key=lambda c: -inst.pi[c]):
        if not pre.controllable[i]:
            continue
        trial = C + [i]
        regime = {c: "post" for c in trial}
        sp = sp_solve(fwd, trial, regime, inst, pre, cfg,
                      time_budget=(budget or cfg.heur_budget))
        if sp.status in (SPStatus.OPTIMAL, SPStatus.FEASIBLE_NOT_PROVEN_OPTIMAL,
                         SPStatus.TIME_LIMIT_WITH_INCUMBENT) and sp.obj is not None:
            val = sum(inst.pi[c] * (1 - y[c]) for c in inst.Nf) + sp.obj
            if best_val is None or val > best_val - 1e-9:
                C = trial
                best_val = val
    inc.u_post = {i: (1.0 if i in C else 0.0) for i in inst.Nf}
    inc.C = sorted(C)
    inc.regime = {i: "post" for i in inc.C}
    inc.value = best_val
    return inc


def _assign_cell(c, lo_win, hi_win, ts_c, tsmin_c, pool, inst, pre, midpoint,
                 max_veh: int = 16):
    """Closed-form vehicle assignment for controlling cell c in a given window [lo_win, hi_win]
    (pre: [t^s, t^m]; post: [t^m, t^e]). Returns a feasible record or None.

    Adds nearest-arriving free vehicles until the water balance (18)/(19) is met with a control
    time t^c in the window that also keeps reward p_c ≥ 0 (i.e. t^c − t^s ≤ π_c/β_c). Feasibility
    is by construction; the reward is the closed form π_c − β_c(t^c − t^s)."""
    a, mu, pi, beta, d, dbuf, dwat = (inst.a, inst.mu, inst.pi, inst.beta, inst.d,
                                      inst.delta_buf, inst.delta_wat)
    EPS = 1e-9

    def vmin(k):
        return max(tsmin_c + dbuf + d[(c, k)], ts_c)

    avail = sorted(pool, key=vmin)
    S: List[int] = []
    for k in avail[:max_veh]:
        S.append(k)
        vmins = {j: vmin(j) for j in S}
        lo = max(lo_win, max(vmins.values()))
        if lo > hi_win + 1e-6:
            return None                                  # cannot arrive within the window
        Dvals = {j: a[c] * (vmins[j] - ts_c) for j in S}
        Dmax, Dmin = max(Dvals.values()), min(Dvals.values())
        omega = (Dmax + Dmin) / 2.0 + dwat if midpoint else Dmax + dwat
        if omega > pre.M_s[c] + 1e-6:
            return None
        wmax = sum(mu[j] * (hi_win - vmins[j]) for j in S)
        if wmax + EPS < omega:
            continue                                     # need more water
        wlo = sum(mu[j] * (lo - vmins[j]) for j in S)
        if wlo + EPS >= omega:
            tc = lo
        else:
            sum_mu = sum(mu[j] for j in S)
            sum_muv = sum(mu[j] * vmins[j] for j in S)
            tc = min(max((omega + sum_muv) / sum_mu, lo), hi_win)
        # reward feasibility: p_c ≥ 0 requires t^c − t^s ≤ π_c/β_c
        if beta[c] > 1e-12 and (tc - ts_c) > pi[c] / beta[c] + 1e-9:
            continue                                     # more vehicles lower t^c toward `lo`
        p = max(0.0, pi[c] - beta[c] * (tc - ts_c))
        s_alloc = {k: max(0.0, tc - vmins[k]) for k in S}
        return {"S": list(S), "tc": tc, "p": p, "D": Dvals, "vmin": vmins,
                "omega": omega, "omega_max": Dmax, "omega_min": Dmin, "s": s_alloc,
                "hmax": max(S, key=lambda kk: Dvals[kk]),
                "hmin": min(S, key=lambda kk: Dvals[kk])}
    return None


def _build_full_start(inst, pre, cfg, y, z, q, u_pre, u_post, fwd, cell_sol, p_of):
    """Assemble a COMPLETE, feasible monolithic MIPStart (all vars) from a containment/greedy
    decision + its per-cell control records. Mirrors the (1)-(50) variable structure so Gurobi
    accepts it by a feasibility check alone."""
    midpoint = (cfg.omega_mode == "midpoint")
    d, pi = inst.d, inst.pi

    def ts_val(j):
        return fwd.ts.get(j, 0.0)

    bmin = {}
    ts_min_var = {}
    for i in inst.Nf:
        burning_nb = [j for j in inst.Nplus[i] if y[j] > 0.5]
        j_star = min(burning_nb, key=lambda j: (ts_val(j), j)) if burning_nb else i
        ts_min_var[i] = ts_val(j_star)
        for j in inst.Nplus[i]:
            bmin[(i, j)] = 1.0 if j == j_star else 0.0

    x = {(i, k): 0.0 for i in inst.Nf for k in inst.K}
    for c, rec in cell_sol.items():
        for k in rec["S"]:
            x[(c, k)] = 1.0

    start = {"y": dict(y), "z": dict(z), "q": dict(q),
             "u_pre": dict(u_pre), "u_post": dict(u_post),
             "x": x, "bmin": bmin, "ts_min": ts_min_var}
    if midpoint:
        hmax = {(i, k): 0.0 for i in inst.Nf for k in inst.K}
        hmin = {(i, k): 0.0 for i in inst.Nf for k in inst.K}
        for c, rec in cell_sol.items():
            hmax[(c, rec["hmax"])] = 1.0
            hmin[(c, rec["hmin"])] = 1.0
        start["hmax"] = hmax
        start["hmin"] = hmin

    ts_v, tm_v, te_v = {}, {}, {}
    for i in inst.Nf:
        if y[i] > 0.5:
            ts_v[i], tm_v[i], te_v[i] = fwd.ts[i], fwd.tm[i], fwd.te[i]
        else:
            ts_v[i] = 0.0
            tm_v[i] = inst.alpha / inst.lam[i]
            te_v[i] = tm_v[i] + inst.alpha / inst.sig[i]
    tc_v, p_v, omega_v, omax_v, omin_v = {}, {}, {}, {}, {}
    t_v = {(i, k): 0.0 for i in inst.Nf for k in inst.K}
    s_v = {(i, k): 0.0 for i in inst.Nf for k in inst.K}
    v_v = {(i, k): 0.0 for i in inst.Nf for k in inst.K}
    delta_v = {(i, k): 0.0 for i in inst.Nf for k in inst.K}
    for i in inst.Nf:
        if i in cell_sol:
            rec = cell_sol[i]
            tc_v[i] = rec["tc"]; p_v[i] = p_of.get(i, rec["p"])
            omega_v[i] = rec["omega"]; omax_v[i] = rec["omega_max"]; omin_v[i] = rec["omega_min"]
            for k in rec["S"]:
                vmk = rec["vmin"][k]
                v_v[(i, k)] = vmk
                t_v[(i, k)] = max(0.0, vmk - d[(i, k)])
                delta_v[(i, k)] = rec["D"][k]
                s_v[(i, k)] = rec["s"][k]
        else:
            tc_v[i] = 0.0
            omega_v[i] = omax_v[i] = omin_v[i] = 0.0
            p_v[i] = pi[i] if y[i] < 0.5 else 0.0
    start.update({"ts": ts_v, "tm": tm_v, "te": te_v, "tc": tc_v, "p": p_v,
                  "omega": omega_v, "omega_max": omax_v,
                  "t": t_v, "s": s_v, "v": v_v, "delta": delta_v})
    if midpoint:
        start["omega_min"] = omin_v
    return start


def build_containment_incumbent(inst, pre, cfg, max_vehicles_per_cell: int = 64,
                                max_expand: int = 8, verbose: int = 0) -> Incumbent:
    """Mono-FEASIBLE containment primal (HEURISTIC ONLY -> valid LB + accepted MIPStart).

    Structure (matches the exact optimum): roots ignite and spread one step; the boundary of the
    burning region is sealed by PRE-controlled firebreak cells ((40): u^pre_i=1 ⇒ no spread), so
    everything beyond stays UNBURNT and collects full π_i. Interior/root burning cells may be
    post-controlled for bonus reward. Only pre-controllable, water/timing-feasible cells can be
    firebreaks; a firebreak that cannot actually be pre-controlled is expanded (its neighbours
    burn and the next ring seals) up to `max_expand` rounds. Every produced point is monolithic-
    feasible by construction (small burning region ⇒ t^s ≤ burn·margin), so its objective is a
    valid LB and it seeds the monolithic MIPStart."""
    midpoint = (cfg.omega_mode == "midpoint")
    Na, Nf, nb = set(inst.Na), inst.Nf, inst.neighbors
    alpha, lam, a = inst.alpha, inst.lam, inst.a
    tau = pre.tau_lb                                    # earliest ignition (Dijkstra, full graph)
    INF = float("inf")
    cap = {i: (pre.M_i[i] / a[i] if a[i] > 1e-12 else INF) for i in Nf}

    def burnable(j):
        return math.isfinite(tau.get(j, INF)) and tau[j] <= cap[j] + 1e-6

    blocked: set = set()          # cells proven NOT firebreak-feasible -> must burn+spread
    inc = None

    for _ in range(max_expand):
        # ---- Phase 1: grow a BALL B in τ_lb order; seal boundary with firebreaks ----
        # Processing in τ order guarantees an interior cell's still-unburnt neighbours only ever
        # have LARGER τ (truly outside), so a cell is sealed only when it is genuinely on the
        # boundary. A ball keeps t^s = τ_lb (firebreaks never distort interior shortest paths).
        B = set(Na)
        firebreak: set = set()
        pq = [(tau.get(r, 0.0), r) for r in Na]
        heapq.heapify(pq)
        seen = set()
        while pq:
            _, c = heapq.heappop(pq)
            if c in seen:
                continue
            seen.add(c)
            # a cell (INCLUDING a root — see 6×6 optimum, root 4 pre-controlled at t^s=0) can seal
            # iff pre-control is possible for it and it has not been proven unassignable (blocked).
            seals = (c not in blocked) and pre.pre_possible[c]
            if seals:
                firebreak.add(c)
                continue                              # firebreak: do not spread
            for j in nb[c]:                           # spread to burnable neighbours
                if j in B:
                    continue
                if not burnable(j):
                    firebreak.add(c)                  # forced to seal (neighbour can't burn)
                    continue
                B.add(j)
                heapq.heappush(pq, (tau[j], j))

        # ---- Phase 2: q-forest (Dijkstra through NON-firebreak cells), z, forward pass ----
        INF = float("inf")
        dist = {r: 0.0 for r in Na}
        parent: Dict[int, int] = {}
        pq = [(0.0, r) for r in Na]
        heapq.heapify(pq)
        expanded = set()
        while pq:
            du, u = heapq.heappop(pq)
            if u in expanded:
                continue
            expanded.add(u)
            if u in firebreak:
                continue                              # firebreak does not propagate (z=0)
            w = alpha / lam[u]
            for j in nb[u]:
                if j not in B:
                    continue
                nd = du + w
                if nd < dist.get(j, INF) - 1e-12:
                    dist[j] = nd
                    parent[j] = u
                    heapq.heappush(pq, (nd, j))

        y = {i: (1.0 if i in B else 0.0) for i in Nf}
        q = {arc: 0.0 for arc in inst.arcs}
        for j, i in parent.items():
            q[(i, j)] = 1.0
        z = {arc: 0.0 for arc in inst.arcs}
        for (i, j) in inst.arcs:
            if i in B and j in B and i not in firebreak:
                z[(i, j)] = 1.0
        u_pre = {i: (1.0 if i in firebreak else 0.0) for i in Nf}
        u_post = {i: 0.0 for i in Nf}
        inc = Incumbent(y=y, z=z, q=q, u_pre=dict(u_pre), u_post=dict(u_post))
        fwd = forward_pass(inc, inst)
        if fwd.status != FWD_OK:
            # structural issue (e.g. an unreachable B cell) -> give up on containment this round
            inc.value = sum(inst.pi[i] * (1 - y[i]) for i in Nf)
            return inc

        # ---- Phase 3: assign vehicles — firebreaks (pre, mandatory) then post bonus ----
        pool = set(inst.K)
        cell_sol: Dict[int, dict] = {}
        p_of: Dict[int, float] = {}
        newly_blocked = set()
        # firebreaks first (high priority; their pre-control is what enables the unburnt π)
        for c in sorted(firebreak, key=lambda i: -inst.pi[i]):
            rec = _assign_cell(c, fwd.ts[c], fwd.tm[c], fwd.ts[c], fwd.ts_min.get(c, fwd.ts[c]),
                               pool, inst, pre, midpoint, max_vehicles_per_cell)
            if rec is None:
                newly_blocked.add(c)                  # cannot pre-control -> must expand
                if verbose > 1:
                    ts_c, tm_c = fwd.ts[c], fwd.tm[c]
                    tsmin = fwd.ts_min.get(c, ts_c)
                    ks = sorted(pool, key=lambda k: max(tsmin + inst.delta_buf + inst.d[(c, k)], ts_c))[:3]
                    vm = [round(max(tsmin + inst.delta_buf + inst.d[(c, k)], ts_c), 2) for k in ks]
                    win = round(tm_c - ts_c, 2)
                    print(f"    FAIL firebreak {c}: ts={ts_c:.2f} tm={tm_c:.2f} win={win} "
                          f"tsmin={tsmin:.2f} nearest_vmin={vm} pre_possible={pre.pre_possible[c]} "
                          f"cap={cap[c]:.1f} nbrs={sorted(nb[c])} "
                          f"nbr_burnable={[burnable(j) for j in sorted(nb[c])]}")
                continue
            cell_sol[c] = rec
            p_of[c] = rec["p"]
            pool.difference_update(rec["S"])
        if verbose:
            print(f"  round: |B|={len(B)} firebreak={len(firebreak)} "
                  f"assigned={len(cell_sol)} failed={len(newly_blocked)} "
                  f"blocked_total={len(blocked | newly_blocked)}")
        if newly_blocked:
            blocked |= newly_blocked
            continue                                  # rebuild with these cells forced to burn

        # post-control bonus on interior/root burning cells (optional, high π first)
        for c in sorted((i for i in B if i not in firebreak), key=lambda i: -inst.pi[i]):
            if not pre.controllable[c]:
                continue
            rec = _assign_cell(c, fwd.tm[c], fwd.te[c], fwd.ts[c], fwd.ts_min.get(c, fwd.ts[c]),
                               pool, inst, pre, midpoint, max_vehicles_per_cell)
            if rec is None:
                continue
            cell_sol[c] = rec
            p_of[c] = rec["p"]
            u_post[c] = 1.0
            pool.difference_update(rec["S"])

        # ---- mono-feasibility safety net: every burning cell must have t^s ≤ cap ----
        late = [i for i in B if i not in Na and a[i] > 1e-12 and fwd.ts[i] > cap[i] + 1e-6]
        if late:
            # A burning cell ignites past its c5 cap ⇒ monolithic-infeasible structure. This should
            # not happen for a τ-ordered ball; if it does, do NOT emit a seed (harmless, gated out).
            deg = Incumbent(y=y, z=z, q=q, u_pre=dict(u_pre), u_post=dict(u_post))
            deg.value = sum(inst.pi[i] * (1 - y[i]) for i in Nf)
            return deg

        # ---- finalize ----
        inc.u_post = u_post
        inc.C = sorted(cell_sol)
        inc.regime = {c: ("pre" if u_pre[c] > 0.5 else "post") for c in inc.C}
        inc.value = sum(inst.pi[i] * (1 - y[i]) for i in Nf) + sum(p_of.values())
        inc.start_vars = _build_full_start(inst, pre, cfg, y, z, q, u_pre, inc.u_post,
                                            fwd, cell_sol, p_of)
        return inc

    # Genişleme bütçesi tükendi ve GEÇERLİ bir atama tamamlanamadı. Bu noktadaki `inc`, DOĞRULANMAMIŞ
    # firebreak (u_pre=1) içeren yarı-kurulmuş bir yapı olabilir ve pi_free(y) değeri GEÇERSİZ bir LB
    # verir. Bu yüzden onu DÖNDÜRMEYİZ; geçerli fallback'e (tam kaskad, u≡0, recourse=0) döneriz.
    fb = build_fallback_incumbent(inst, pre, cfg)
    fb.value = sum(inst.pi[i] * (1 - fb.y[i]) for i in inst.Nf)
    fb.start_vars = {}
    return fb


def build_greedy_incumbent_fast(inst, pre, cfg, max_vehicles_per_cell: int = 16,
                                verbose: int = 0) -> Incumbent:
    """SP-FREE greedy primal (HEURISTIC ONLY -> valid LB, never a bound/cut).

    Full ignition cascade (u^pre=0) + closed-form greedy post-control. With the forward-pass
    times fixed, Lemma A makes v^min_ik and D_ik CONSTANTS, so a cell's reward, ω and water
    balance are all closed-form — no Gurobi solve. For each controllable burning cell (high π
    first) we assign nearest-arriving free vehicles until the water balance (18) is met within
    the post window [t^m_i, t^e_i], choosing the earliest feasible control time t^c_i (max
    reward). Every produced point is a genuine feasible solution of the exact model, so its
    objective is a valid lower bound; it is ALSO emitted as a full integer assignment for a
    monolithic MIPStart (``start_vars``). Exactness is untouched (seed only).
    """
    y, z, q, burning = _full_cascade(inst)
    u_pre = {i: 0.0 for i in inst.Nf}
    u_post = {i: 0.0 for i in inst.Nf}
    inc = Incumbent(y=y, z=z, q=q, u_pre=dict(u_pre), u_post=dict(u_post))

    fwd = forward_pass(inc, inst)
    if fwd.status != FWD_OK:
        inc.value = sum(inst.pi[i] * (1 - y[i]) for i in inst.Nf)
        return inc

    midpoint = (cfg.omega_mode == "midpoint")
    dbuf, dwat = inst.delta_buf, inst.delta_wat
    a, mu, pi, beta, d = inst.a, inst.mu, inst.pi, inst.beta, inst.d
    EPS = 1e-9

    pool = set(inst.K)
    assign: Dict[int, List[int]] = {}          # cell -> assigned vehicle ids
    tc_of: Dict[int, float] = {}
    p_of: Dict[int, float] = {}
    hmax_k: Dict[int, int] = {}
    hmin_k: Dict[int, int] = {}
    cell_sol: Dict[int, dict] = {}
    reasons: Dict[str, int] = {}
    dbg = 0

    def _r(tag):
        reasons[tag] = reasons.get(tag, 0) + 1

    for c in sorted(burning, key=lambda i: -pi[i]):
        if not pre.controllable[c]:
            _r("uncontrollable")
            continue
        ts_c, tm_c, te_c = fwd.ts[c], fwd.tm[c], fwd.te[c]
        avail = sorted(pool, key=lambda k: max(fwd.ts_min.get(c, 0.0) + dbuf + d[(c, k)], ts_c))

        def vmin(k):
            return max(fwd.ts_min.get(c, 0.0) + dbuf + d[(c, k)], ts_c)

        S: List[int] = []
        chosen = None
        fail = "water_or_reward"
        for k in avail[:max_vehicles_per_cell]:
            S.append(k)
            vmins = {j: vmin(j) for j in S}
            lo = max(tm_c, max(vmins.values()))
            if lo > te_c + 1e-6:
                fail = "window(lo>te)"
                break                                   # cannot arrive within the post window
            Dvals = {j: a[c] * (vmins[j] - ts_c) for j in S}
            Dmax, Dmin = max(Dvals.values()), min(Dvals.values())
            omega = (Dmax + Dmin) / 2.0 + dwat if midpoint else Dmax + dwat
            if omega > pre.M_s[c] + 1e-6:
                fail = "omega>M_s"
                break                                   # ω cannot be met; adding vehicles only grows it
            wmax = sum(mu[j] * (te_c - vmins[j]) for j in S)
            if wmax + EPS < omega:
                fail = "water_short"
                continue                                # need more water -> add another vehicle
            # earliest feasible control time in [lo, te_c] meeting the water balance
            wlo = sum(mu[j] * (lo - vmins[j]) for j in S)
            if wlo + EPS >= omega:
                tc = lo
            else:
                sum_mu = sum(mu[j] for j in S)
                sum_muv = sum(mu[j] * vmins[j] for j in S)
                tc = min(max((omega + sum_muv) / sum_mu, lo), te_c)
            p = pi[c] - beta[c] * (tc - ts_c)
            if p <= EPS:
                # tc forced too late for positive reward; MORE parallel vehicles lower tc
                # toward `lo`, so keep adding rather than abandoning the cell.
                fail = "reward<=0"
                continue
            chosen = (list(S), tc, p, dict(Dvals), dict(vmins), omega, Dmax, Dmin)
            break
        if verbose and dbg < verbose:
            dbg += 1
            k0 = avail[0]
            print(f"  cell {c}: pi={pi[c]:.1f} beta={beta[c]:.3f} a={a[c]:.3f} "
                  f"ts={ts_c:.2f} tm={tm_c:.2f} te={te_c:.2f} tsmin={fwd.ts_min.get(c,0.0):.2f} "
                  f"| k0={k0} d={d[(c,k0)]:.2f} mu={mu[k0]:.2f} vmin0={vmin(k0):.2f} "
                  f"| M_s={pre.M_s[c]:.1f} ctrl={pre.controllable[c]} "
                  f"-> {'ACCEPT' if chosen else 'REJECT:'+fail}")
        if chosen is None:
            _r(fail)
            continue
        _r("accepted")
        Sc, tc, p, Dvals, vmins_c, omega_c, Dmax_c, Dmin_c = chosen
        assign[c] = Sc
        tc_of[c] = tc
        p_of[c] = p
        hmax_k[c] = max(Sc, key=lambda k: Dvals[k])
        hmin_k[c] = min(Sc, key=lambda k: Dvals[k])
        # water allocation: fill each assigned vehicle to its full service window s_ik = t^c-v^min.
        # Then Σμ_k s_ik = water_avail(t^c) ≥ ω (t^c was chosen to meet the balance), satisfying
        # (18)/(19) with margin and (24) with equality — no floating-point under-fill risk. Water
        # constraints are lower bounds, so supplying the maximum is always feasible.
        s_alloc = {k: max(0.0, tc - vmins_c[k]) for k in Sc}
        cell_sol[c] = {"S": Sc, "tc": tc, "vmin": vmins_c, "D": Dvals,
                       "omega": omega_c, "omega_max": Dmax_c, "omega_min": Dmin_c,
                       "s": s_alloc}
        pool.difference_update(Sc)

    C = sorted(assign)
    for c in C:
        inc.u_post[c] = 1.0
    inc.C = C
    inc.regime = {c: "post" for c in C}
    inc.value = sum(pi[i] * (1 - y[i]) for i in inst.Nf) + sum(p_of[c] for c in C)
    if verbose:
        print(f"  reasons={reasons}  burning={len(burning)} controllable="
              f"{sum(1 for i in burning if pre.controllable[i])}")

    # ---- full integer decision for a monolithic MIPStart ----------------------
    def ts_val(j):
        return fwd.ts.get(j, 0.0)

    bmin: Dict[Tuple[int, int], float] = {}
    for i in inst.Nf:
        burning_nb = [j for j in inst.Nplus[i] if y[j] > 0.5]
        j_star = min(burning_nb, key=lambda j: (ts_val(j), j)) if burning_nb else i
        for j in inst.Nplus[i]:
            bmin[(i, j)] = 1.0 if j == j_star else 0.0

    x = {(i, k): 0.0 for i in inst.Nf for k in inst.K}
    for c, Sc in assign.items():
        for k in Sc:
            x[(c, k)] = 1.0

    # ts_min from the bmin selector (= t^s of the chosen j*, which is the min burning t^s)
    ts_min_var: Dict[int, float] = {}
    for i in inst.Nf:
        burning_nb = [j for j in inst.Nplus[i] if y[j] > 0.5]
        ts_min_var[i] = min((ts_val(j) for j in burning_nb), default=0.0)

    start = {"y": dict(y), "z": dict(z), "q": dict(q),
             "u_pre": dict(inc.u_pre), "u_post": dict(inc.u_post),
             "x": x, "bmin": bmin, "ts_min": ts_min_var}
    if midpoint:
        hmax = {(i, k): 0.0 for i in inst.Nf for k in inst.K}
        hmin = {(i, k): 0.0 for i in inst.Nf for k in inst.K}
        for c in C:
            hmax[(c, hmax_k[c])] = 1.0
            hmin[(c, hmin_k[c])] = 1.0
        start["hmax"] = hmax
        start["hmin"] = hmin

    # ---- full CONTINUOUS solution (a complete, feasible MIPStart -> instant Gurobi accept) ----
    ts_v, tm_v, te_v = {}, {}, {}
    for i in inst.Nf:
        if y[i] > 0.5:
            ts_v[i], tm_v[i], te_v[i] = fwd.ts[i], fwd.tm[i], fwd.te[i]
        else:                                            # unburnt: (23t) t^s=0, (49),(50)
            ts_v[i] = 0.0
            tm_v[i] = inst.alpha / inst.lam[i]
            te_v[i] = tm_v[i] + inst.alpha / inst.sig[i]
    tc_v, p_v, omega_v, omax_v, omin_v = {}, {}, {}, {}, {}
    t_v = {(i, k): 0.0 for i in inst.Nf for k in inst.K}
    s_v = {(i, k): 0.0 for i in inst.Nf for k in inst.K}
    v_v = {(i, k): 0.0 for i in inst.Nf for k in inst.K}
    delta_v = {(i, k): 0.0 for i in inst.Nf for k in inst.K}
    for i in inst.Nf:
        if i in cell_sol:
            rec = cell_sol[i]
            tc_v[i] = rec["tc"]
            p_v[i] = p_of[i]
            omega_v[i] = rec["omega"]
            omax_v[i] = rec["omega_max"]
            omin_v[i] = rec["omega_min"]
            for k in rec["S"]:
                vmk = rec["vmin"][k]
                v_v[(i, k)] = vmk
                t_v[(i, k)] = max(0.0, vmk - d[(i, k)])
                delta_v[(i, k)] = rec["D"][k]
                s_v[(i, k)] = rec["s"][k]
        else:
            tc_v[i] = 0.0
            omega_v[i] = omax_v[i] = omin_v[i] = 0.0
            p_v[i] = pi[i] if y[i] < 0.5 else 0.0        # unburnt keeps π; burnt-uncontrolled 0
    start.update({"ts": ts_v, "tm": tm_v, "te": te_v, "tc": tc_v, "p": p_v,
                  "omega": omega_v, "omega_max": omax_v,
                  "t": t_v, "s": s_v, "v": v_v, "delta": delta_v})
    if midpoint:
        start["omega_min"] = omin_v
    inc.start_vars = start
    return inc
