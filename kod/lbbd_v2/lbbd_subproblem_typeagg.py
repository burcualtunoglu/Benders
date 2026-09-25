"""lbbd_subproblem_typeagg.py — type-aggregated subproblem (04 §7, 05 §6). (Başlıktaki "PROVEN EXACT" iddiası
2026-09-24 incelemesinde DOĞRULANMADI: (4),(5) deaktivasyon sabiti hatası bulundu ve düzeltildi; bkz. docs/groupagg_sp.md.
Üretimde kullanılmaz; Geliştirme 1 için ayrı güçlü grup-toplulaştırılmış SP yazıldı: baseline/subproblem_groupagg.py.)

Vehicles in an interchangeability group g (identical d_·k and µ_k) are aggregated: n_ig counts
how many group-g vehicles serve cell i. The assignment layer is built to be BIT-FOR-BIT identical
to the certified aggregated monolithic (model_monolithic_typeagg) — same delta big-M form and
same ω^max/ω^min selectors (hmax/hmin) — only the structure/time variables are fixed constants
from the forward pass. This guarantees the subproblem is the exact restriction of that model, so
it reproduces the per-resource recourse exactly (Lemma 1, both ω-modes) while scaling to |K|=150.

(An earlier ω-selector shortcut — ω^max ≥ delay·used without the hmax upper selector, and a convex
ω^min — undercomputed the recourse on tightly-windowed pre-suppression configs, which produced an
invalid optimality cut. The mirror-the-monolithic construction below removes that gap.)
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from lbbd_v2.config import Config
from lbbd_v2.preprocessing import Preprocessed
from lbbd_v2.benders_forward import ForwardResult
from lbbd_v2.lbbd_subproblem_resource import SPResult, SPStatus, _tc_window


def _build(cells: List[int], regime: Dict[int, str], fwd: ForwardResult,
           inst, pre: Preprocessed, cfg: Config, integer: bool = True) -> Tuple[gp.Model, dict]:
    m = gp.Model("sp_typeagg")
    m.setParam("OutputFlag", 0)
    m.setParam("Seed", cfg.gurobi_seed)
    m.setParam("Threads", cfg.gurobi_threads)
    midpoint = (cfg.omega_mode == "midpoint")
    a, pi, beta = inst.a, inst.pi, inst.beta
    Md, Mi, Ms = pre.M_d, pre.M_i, pre.M_s
    Dwat, Dbuf = inst.delta_wat, inst.delta_buf
    Wcap = pre.te_ub                          # valid upper bound on win (win ≤ tc ≤ te ≤ te_ub)
    groups = pre.groups
    gmap = {g.gid: g for g in groups}
    vt = GRB.INTEGER if integer else GRB.CONTINUOUS
    bvt = GRB.BINARY if integer else GRB.CONTINUOUS

    # ALL (cell,group) pairs are built (no pruning) — infeasible ones settle to used=0,
    # exactly as in the certified aggregated monolithic.
    pairs = [(i, g.gid) for i in cells for g in groups]

    n = m.addVars(pairs, vtype=vt, lb=0.0, name="n")
    used = m.addVars(pairs, vtype=bvt, lb=0.0, ub=1.0, name="used")
    v = m.addVars(pairs, lb=0.0, name="v")
    win = m.addVars(pairs, lb=0.0, name="win")
    WS = m.addVars(pairs, lb=0.0, name="WS")
    delta = m.addVars(pairs, lb=0.0, name="delta")
    tc = m.addVars(cells, lb=0.0, name="tc")
    p = m.addVars(cells, lb=0.0, name="p")
    omega = m.addVars(cells, lb=0.0, name="omega")
    omega_max = m.addVars(cells, lb=0.0, name="omega_max")
    if midpoint:
        omega_min = m.addVars(cells, lb=0.0, name="omega_min")
        hmax = m.addVars(pairs, vtype=bvt, lb=0.0, ub=1.0, name="hmax")
        hmin = m.addVars(pairs, vtype=bvt, lb=0.0, ub=1.0, name="hmin")
    else:
        omega_min = hmax = hmin = None

    m.setObjective(gp.quicksum(p[i] for i in cells), GRB.MAXIMIZE)

    # ---- count linking + bit-expansion / McCormick WS = n·win ----
    for (i, gid) in pairs:
        Ng = gmap[gid].size
        n[i, gid].ub = Ng
        Bg = max(1, math.ceil(math.log2(Ng + 1)))
        bits, gmc = [], []
        for b in range(Bg):
            bb = m.addVar(vtype=bvt, lb=0.0, ub=1.0, name=f"beta[{i},{gid},{b}]")
            gg = m.addVar(lb=0.0, name=f"gmc[{i},{gid},{b}]")
            bits.append(bb); gmc.append(gg)
            m.addConstr(gg <= win[i, gid])
            m.addConstr(gg <= Wcap * bb)
            m.addConstr(gg >= win[i, gid] - Wcap * (1 - bb))
        m.addConstr(n[i, gid] == gp.quicksum((2 ** b) * bits[b] for b in range(Bg)))
        m.addConstr(WS[i, gid] == gp.quicksum((2 ** b) * gmc[b] for b in range(Bg)))
        m.addConstr(used[i, gid] <= gp.quicksum(bits))
        for b in range(Bg):
            m.addConstr(used[i, gid] >= bits[b])

    # ---- per (cell,group): arrival, service window, delay water, omega bounds ----
    for (i, gid) in pairs:
        dgi = gmap[gid].d[i]
        tsi = fwd.ts[i]
        tsmin = fwd.ts_min.get(i, 0.0)
        # arrival v ≥ max(ts_min+Δbuf+d, ts)   ((25),(29)+(21)); v=0 when unused
        m.addConstr(v[i, gid] >= tsmin + Dbuf + dgi - Md * (1 - used[i, gid]))
        m.addConstr(v[i, gid] >= tsi - Md * (1 - used[i, gid]))
        m.addConstr(v[i, gid] <= Md * used[i, gid])
        m.addConstr(win[i, gid] <= Md * used[i, gid])
        m.addConstr(v[i, gid] + win[i, gid] <= tc[i] + Md * (1 - used[i, gid]))   # (24)
        # delay water δ (4),(5),(6)
        # 2026-09-24 DÜZELTME (Geliştirme 1 incelemesi): (4),(5) deaktivasyon sabiti a_i·M_d olmalı
        # (subproblem_weak ile aynı). Eski M_i, kullanılmayan çiftte v=0 iken delta ≤ −a·ts + M_i < 0'a
        # düşüyor ve kullanılmayan (i,g) çiftleri modeli fizibilsiz kılıyordu (IIS: 4x4 yapı sd=3, hücre 11,
        # a=1925, ts=21,45 ⇒ a·ts=41 293 > M_i=22 053). Sonuç: Φ eksik hesaplanıyordu (124,98 vs 151,23).
        m.addConstr(delta[i, gid] >= a[i] * (v[i, gid] - tsi) - a[i] * Md * (1 - used[i, gid]))
        m.addConstr(delta[i, gid] <= a[i] * (v[i, gid] - tsi) + a[i] * Md * (1 - used[i, gid]))
        m.addConstr(delta[i, gid] <= Mi[i] * used[i, gid])
        # ω^max (7)
        m.addConstr(omega_max[i] >= delta[i, gid] - Ms[i] * (1 - used[i, gid]))
        if midpoint:
            m.addConstr(omega_max[i] <= delta[i, gid] + Ms[i] * (1 - hmax[i, gid]))   # (8)
            m.addConstr(hmax[i, gid] <= used[i, gid])                                 # (9)
            m.addConstr(omega_min[i] <= delta[i, gid] + Ms[i] * (1 - used[i, gid]))   # (10)
            m.addConstr(omega_min[i] >= delta[i, gid] - Ms[i] * (1 - hmin[i, gid]))   # (11)
            m.addConstr(hmin[i, gid] <= used[i, gid])                                 # (12)

    # ---- per cell: reward, omega definition, water, must-serve, selectors ----
    for i in cells:
        lo, hi = _tc_window(fwd, i, regime[i])
        gids = [g.gid for g in groups]
        m.addConstr(tc[i] >= lo)
        m.addConstr(tc[i] <= hi)
        m.addConstr(p[i] <= pi[i] - beta[i] * (tc[i] - fwd.ts[i]))
        m.addConstr(p[i] <= pi[i])
        m.addConstr(gp.quicksum(n[i, gid] for gid in gids) >= 1)          # act_i = 1, must serve
        if midpoint:
            m.addConstr(gp.quicksum(hmax[i, gid] for gid in gids) == 1)   # (14)
            m.addConstr(gp.quicksum(hmin[i, gid] for gid in gids) == 1)   # (15)
            m.addConstr(2 * omega[i] == omega_max[i] + omega_min[i] + 2 * Dwat)   # (16)
        else:
            m.addConstr(omega[i] == omega_max[i] + Dwat)                  # (16w)
        m.addConstr(omega[i] <= Ms[i])                                    # (17)
        m.addConstr(gp.quicksum(gmap[gid].mu * WS[i, gid] for gid in gids) >= omega[i])  # (18)

    # ---- (33) resource competition per group: Σ_i n_ig ≤ N_g ----
    for g in groups:
        m.addConstr(gp.quicksum(n[i, g.gid] for i in cells) <= g.size, f"cap[{g.gid}]")

    handles = dict(n=n, win=win, WS=WS, tc=tc, p=p, omega=omega, delta=delta,
                   pairs=pairs, gmap=gmap)
    return m, handles


def solve_typeagg_subproblem(fwd: ForwardResult, C: Iterable[int], regime: Dict[int, str],
                             inst, pre: Preprocessed, cfg: Config,
                             time_budget: Optional[float] = None) -> SPResult:
    cells = sorted(C)
    if not cells:
        return SPResult(SPStatus.OPTIMAL, 0.0, 0.0, 0.0, 0.0, {})
    m, h = _build(cells, regime, fwd, inst, pre, cfg, integer=True)
    if time_budget is not None:
        m.setParam("TimeLimit", time_budget)
    t0 = time.time()
    m.optimize()
    if m.Status == GRB.INF_OR_UNBD:
        m.setParam("DualReductions", 0)
        m.optimize()
    runtime = time.time() - t0

    from lbbd_v2.lbbd_subproblem_resource import _map_status
    status = _map_status(m)
    obj = m.ObjVal if m.SolCount > 0 else None
    obj_bound = m.ObjBound if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL) else None
    try:
        gap = m.MIPGap if m.SolCount > 0 else None
    except Exception:
        gap = None

    solution: Dict[str, object] = {}
    if m.SolCount > 0:
        solution["p"] = {i: h["p"][i].X for i in cells}
        solution["tc"] = {i: h["tc"][i].X for i in cells}
        solution["omega"] = {i: h["omega"][i].X for i in cells}
        solution["n"] = {(i, gid): int(round(h["n"][i, gid].X)) for (i, gid) in h["pairs"]
                         if h["n"][i, gid].X > 0.5}
    return SPResult(status, obj, obj_bound, gap, runtime, solution)
