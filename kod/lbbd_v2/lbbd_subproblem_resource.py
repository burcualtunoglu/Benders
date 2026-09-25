"""lbbd_subproblem_resource.py — strong (Big-M-free) per-resource subproblem (04 §7.2, cert V2).

With forward-pass times fixed, Lemma A (earliest-arrival dominance) lets every assigned
vehicle take v_ik = v_ik^min = max(t^{s,min}_i + Δ_buf + d_ik, t^s_i), making
D_ik := a_i(v_ik^min − t^s_i) a CONSTANT. Then δ_ik = D_ik x_ik and the only remaining Big-M
is the small time-scale one on the service link. Exact in both ω-modes (Lemma A, 02 §1-2).

The subproblem is the original model's exact restriction with the master decision fixed:
maximise Σ_{i∈C} p_i subject to (2),(4)-(38),(33) with C = {i : ū_i = 1} and regimes fixed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from lbbd_v2.config import Config
from lbbd_v2.data_loader import Instance
from lbbd_v2.preprocessing import Preprocessed
from lbbd_v2.benders_forward import ForwardResult


class SPStatus(Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE_NOT_PROVEN_OPTIMAL = "FEASIBLE_NOT_PROVEN_OPTIMAL"
    INFEASIBLE_PROVEN = "INFEASIBLE_PROVEN"
    TIME_LIMIT_WITH_INCUMBENT = "TIME_LIMIT_WITH_INCUMBENT"
    TIME_LIMIT_NO_INCUMBENT = "TIME_LIMIT_NO_INCUMBENT"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


@dataclass
class SPResult:
    status: SPStatus
    obj: Optional[float]                 # Φ = Σ_{i∈C} p_i
    obj_bound: Optional[float]           # valid UPPER bound on Φ (dual)
    mip_gap: Optional[float]
    runtime: float
    solution: Dict[str, Any] = field(default_factory=dict)


def _v_min(fwd: ForwardResult, inst: Instance, i: int, k: int) -> float:
    return max(fwd.ts_min.get(i, 0.0) + inst.delta_buf + inst.d[(i, k)], fwd.ts[i])


def _tc_window(fwd: ForwardResult, i: int, regime: str) -> Tuple[float, float]:
    if regime == "pre":
        return fwd.ts[i], fwd.tm[i]          # (35),(37)
    return fwd.tm[i], fwd.te[i]              # (38),(34)


def _build(cells: List[int], regime: Dict[int, str], fwd: ForwardResult,
           inst: Instance, pre: Preprocessed, cfg: Config,
           K: Iterable[int], integer: bool,
           warm_x: Optional[Dict[Tuple[int, int], float]] = None,
           symmetry_break: Optional[bool] = None,
           mip_gap: Optional[float] = None) -> Tuple[gp.Model, dict]:
    m = gp.Model("sp_strong")
    m.setParam("OutputFlag", 0)
    m.setParam("Seed", cfg.gurobi_seed)
    m.setParam("Threads", cfg.gurobi_threads)
    eff_gap = getattr(cfg, "sp_mip_gap", 0.0) if mip_gap is None else mip_gap
    m.setParam("MIPGap", eff_gap)   # 0.0=SP optimumu kanıtlanır; pozitif=erken dur; per-çağrı override
    # (uyarlamalı sıkılaştırma). KESİNLİK korunur: LB=ObjVal fizibil, kesme=ObjBound≥Φ; dış opt=LB-UB.
    K = list(K)
    midpoint = (cfg.omega_mode == "midpoint")
    Md = pre.M_d
    a, mu, pi, beta = inst.a, inst.mu, inst.pi, inst.beta
    vt = GRB.BINARY if integer else GRB.CONTINUOUS

    D = {(i, k): a[i] * (_v_min(fwd, inst, i, k) - fwd.ts[i]) for i in cells for k in K}
    vmin = {(i, k): _v_min(fwd, inst, i, k) for i in cells for k in K}

    x = m.addVars(cells, K, vtype=vt, lb=0.0, ub=1.0, name="x")
    s = m.addVars(cells, K, lb=0.0, name="s")
    tc = m.addVars(cells, lb=0.0, name="tc")
    p = m.addVars(cells, lb=0.0, name="p")
    omega = m.addVars(cells, lb=0.0, name="omega")
    omega_max = m.addVars(cells, lb=0.0, name="omega_max")
    if midpoint:
        omega_min = m.addVars(cells, lb=0.0, name="omega_min")
        w = m.addVars(cells, K, lb=0.0, ub=1.0, name="w")   # min-selector (continuous ok)
    else:
        omega_min = w = None

    m.setObjective(gp.quicksum(p[i] for i in cells), GRB.MAXIMIZE)

    for i in cells:
        lo, hi = _tc_window(fwd, i, regime[i])
        m.addConstr(tc[i] >= lo, f"tc_lo[{i}]")
        m.addConstr(tc[i] <= hi, f"tc_hi[{i}]")
        m.addConstr(p[i] <= pi[i] - beta[i] * (tc[i] - fwd.ts[i]), f"p_reward[{i}]")
        m.addConstr(p[i] <= pi[i], f"p_cap[{i}]")
        m.addConstr(gp.quicksum(x[i, k] for k in K) >= 1, f"must_serve[{i}]")   # (32), act=1

        for k in K:
            m.addConstr(omega_max[i] >= D[i, k] * x[i, k], f"omax[{i},{k}]")     # (7) strong
            m.addConstr(s[i, k] <= Md * x[i, k], f"s_off[{i},{k}]")              # time-scale M
            # (24): v_min + s ≤ tc when assigned; s→0 when not
            m.addConstr(s[i, k] <= tc[i] - vmin[i, k] + Md * (1 - x[i, k]), f"s_win[{i},{k}]")

        if midpoint:
            m.addConstr(omega_min[i] == gp.quicksum(D[i, k] * w[i, k] for k in K), f"omin[{i}]")
            m.addConstr(gp.quicksum(w[i, k] for k in K) == 1, f"omin_sel[{i}]")
            for k in K:
                m.addConstr(w[i, k] <= x[i, k], f"omin_le_x[{i},{k}]")
            m.addConstr(2 * omega[i] == omega_max[i] + omega_min[i] + 2 * inst.delta_wat,
                        f"omega_mid[{i}]")
        else:
            m.addConstr(omega[i] == omega_max[i] + inst.delta_wat, f"omega_wc[{i}]")

        m.addConstr(omega[i] <= pre.M_s[i], f"omega_cap[{i}]")                   # (17), act=1
        m.addConstr(gp.quicksum(mu[k] * s[i, k] for k in K) >= omega[i], f"water[{i}]")  # (18)

    # (33) global resource competition
    for k in K:
        m.addConstr(gp.quicksum(x[i, k] for i in cells) <= 1, f"cap[{k}]")

    # symmetry-breaking within interchangeability groups (identical d_ik, µ_k).
    # Canonical assignment: for consecutive group members k_a < k_b, pos(k_a) ≤ pos(k_b), where
    # pos(k) = Σ_i rank_i·x[i,k] + BIG·(1 − Σ_i x[i,k]); rank_i is the 1-based index of cell i in
    # `cells` and BIG = |cells|+1 (unused ⇒ pos = BIG). This forces used vehicles to be the
    # lowest-indexed members, ordered by non-decreasing served-cell rank — a full canonical form
    # of the vehicle-permutation symmetry. VALID: same-group vehicles are fully interchangeable
    # (d_ik, µ_k identical), so any solution relabels to this form with identical objective.
    sym = getattr(cfg, "sp_symmetry_break", True) if symmetry_break is None else symmetry_break
    if integer and sym:
        rank = {i: r for r, i in enumerate(cells, start=1)}
        BIG = len(cells) + 1
        Kset = set(K)
        for g in pre.groups:
            members = [k for k in g.members if k in Kset]
            for ka, kb in zip(members, members[1:]):
                used_a = gp.quicksum(x[i, ka] for i in cells)
                used_b = gp.quicksum(x[i, kb] for i in cells)
                pos_a = gp.quicksum(rank[i] * x[i, ka] for i in cells) + BIG * (1 - used_a)
                pos_b = gp.quicksum(rank[i] * x[i, kb] for i in cells) + BIG * (1 - used_b)
                m.addConstr(pos_a <= pos_b, f"symbreak[{ka},{kb}]")

    # optional MIPStart from a closed-form feasible assignment (seed only; a rejected start is
    # harmless). Only x is seeded — Gurobi completes the continuous vars by LP. Requires
    # symmetry_break=False, else a non-canonical assignment violates the symbreak constraints.
    if warm_x is not None:
        for i in cells:
            for k in K:
                x[i, k].Start = 1.0 if warm_x.get((i, k), 0.0) > 0.5 else 0.0
        m.update()

    handles = dict(x=x, s=s, tc=tc, p=p, omega=omega, omega_max=omega_max,
                   omega_min=omega_min, w=w, D=D, vmin=vmin)
    return m, handles


def _map_status(m: gp.Model) -> SPStatus:
    st, n = m.Status, m.SolCount
    if st == GRB.OPTIMAL and n >= 1:
        return SPStatus.OPTIMAL
    # INF_OR_UNBD -> INFEASIBLE_PROVEN GEÇERLİDİR çünkü bu SP SINIRLIDIR (amaç Σp_i; her p_i ≤ π_i
    # (c3/p_cap), ω ≤ M_s, s ≤ M_d·x) -> SINIRSIZ olamaz, dolayısıyla INF_OR_UNBD = INFEASIBLE.
    # Ayrıca solve_resource_subproblem, INF_OR_UNBD'de DualReductions=0 ile YENİDEN çözer; buraya
    # gelen durum o yeniden-çözümün sonucudur.
    if st in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return SPStatus.INFEASIBLE_PROVEN
    if st in (GRB.TIME_LIMIT, GRB.INTERRUPTED):   # INTERRUPTED = hard wall-clock terminate (see _optimize_capped)
        return SPStatus.TIME_LIMIT_WITH_INCUMBENT if n >= 1 else SPStatus.TIME_LIMIT_NO_INCUMBENT
    if st == GRB.SUBOPTIMAL and n >= 1:
        return SPStatus.FEASIBLE_NOT_PROVEN_OPTIMAL
    return SPStatus.NUMERICAL_FAILURE


def _optimize_capped(m: gp.Model, budget: Optional[float]) -> None:
    """Run ``m.optimize()`` honoring a HARD wall-clock ``budget`` (seconds).

    Gurobi's ``TimeLimit`` is a *soft* limit --- it is checked only at operation
    boundaries, so a single long root-LP / cut / presolve phase can overrun it
    (observed ~2x on hard subproblems). We set ``TimeLimit`` as the primary guard
    AND install a terminate callback as a hard backstop. A terminated solve reports
    ``INTERRUPTED`` (mapped to ``TIME_LIMIT_*`` by ``_map_status``); its ``ObjBound``
    is still a valid dual bound, so the dual-bound optimality cut stays available and
    exactness is unaffected.
    """
    if budget is None:
        m.optimize()
        return
    budget = max(1.0, budget)
    m.setParam("TimeLimit", budget)
    start = time.time()

    def _terminate_cb(model, where):
        if time.time() - start > budget:
            model.terminate()

    m.optimize(_terminate_cb)


def solve_resource_subproblem(fwd: ForwardResult, C: Iterable[int], regime: Dict[int, str],
                              inst: Instance, pre: Preprocessed, cfg: Config,
                              time_budget: Optional[float] = None,
                              warm_x: Optional[Dict[Tuple[int, int], float]] = None,
                              symmetry_break: Optional[bool] = None,
                              mip_gap: Optional[float] = None) -> SPResult:
    cells = sorted(C)
    if not cells:
        return SPResult(SPStatus.OPTIMAL, 0.0, 0.0, 0.0, 0.0, {})
    m, h = _build(cells, regime, fwd, inst, pre, cfg, inst.K, integer=True,
                  warm_x=warm_x, symmetry_break=symmetry_break, mip_gap=mip_gap)
    t0 = time.time()
    _optimize_capped(m, time_budget)
    if m.Status == GRB.INF_OR_UNBD:          # disambiguate (05 §10, note ¹)
        m.setParam("DualReductions", 0)
        rem = None if time_budget is None else max(1.0, time_budget - (time.time() - t0))
        _optimize_capped(m, rem)             # remaining budget only (was: full budget again -> ~2x overrun)
    runtime = time.time() - t0

    status = _map_status(m)
    obj = m.ObjVal if m.SolCount > 0 else None
    obj_bound = m.ObjBound if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL) else None
    try:
        gap = m.MIPGap if m.SolCount > 0 else None
    except Exception:
        gap = None

    solution: Dict[str, Any] = {}
    if m.SolCount > 0:
        solution["p"] = {i: h["p"][i].X for i in cells}
        solution["tc"] = {i: h["tc"][i].X for i in cells}
        solution["omega"] = {i: h["omega"][i].X for i in cells}
        solution["x"] = {(i, k): h["x"][i, k].X for i in cells for k in inst.K
                         if h["x"][i, k].X > 0.5}
        # 2026-09-25 (çözüm tanığı): atanan araçların servis süreleri de saklanır; amaç değeri ile
        # atama/zaman/servis AYNI Gurobi çözümünden okunur (bağımsız doğrulama için; davranış değişmez).
        solution["s"] = {(i, k): h["s"][i, k].X for (i, k) in solution["x"]}
    return SPResult(status, obj, obj_bound, gap, runtime, solution)


def solve_single_cell(i: int, regime_i: str, fwd: ForwardResult, inst: Instance,
                      pre: Preprocessed, cfg: Config,
                      time_budget: Optional[float] = None, return_status: bool = False,
                      timing: Optional[dict] = None):
    """Competition-free single-cell solve (all fleet, no rivals). Returns (feasible, p_solo)
    veya return_status=True ise (feasible, p_solo, kind).

    kind ∈ {"FEASIBLE_VERIFIED","INFEASIBLE_PROVEN","UNKNOWN"} — reviewer (a): Boolean 'feasible'
    tek başına yeterli değil (UNKNOWN'da da True,π_i döner). Önbellek bu üç durumu AYRI saklar.
    p_solo, tam problemde p_i üzerinde GEÇERLİ ÜST SINIR (04 §8.5): hücreye tüm filo verilir.
    timing (verilirse): {"build_s":..,"solve_s":..} — model kurma vs çözme süreleri AYRI (reviewer (b))."""
    _tb = time.time()
    m, h = _build([i], {i: regime_i}, fwd, inst, pre, cfg, inst.K, integer=True)
    build_s = time.time() - _tb
    t0 = time.time()
    _optimize_capped(m, time_budget)
    if m.Status == GRB.INF_OR_UNBD:
        m.setParam("DualReductions", 0)
        rem = None if time_budget is None else max(1.0, time_budget - (time.time() - t0))
        _optimize_capped(m, rem)
    if timing is not None:
        timing["build_s"] = build_s
        timing["solve_s"] = time.time() - t0
    if m.Status == GRB.OPTIMAL and m.SolCount > 0:
        # #3: p_solo, K5 kesmesinde p_i üzerinde GEÇERLİ ÜST SINIR olmalı. Gurobi varsayılan
        # MIPGap (1e-4) altında ObjVal (p.X) gerçek tek-hücre optimumunun ALTINDA olabilir ve
        # p_i(tam) < ObjVal durumunda K5 kesmesini geçersizleştirirdi. ObjBound ≥ optimum ≥ p_i(tam)
        # her toleransta geçerli üst sınırdır; p_i ≤ π_i (c3) ile min alınarak sıkılaştırılır.
        res = (True, min(m.ObjBound, inst.pi[i]), "FEASIBLE_VERIFIED")
    elif m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        res = (False, 0.0, "INFEASIBLE_PROVEN")
    else:
        # timeout/numeric: treat as "no information" -> feasible=True (ELEME), p_solo=π_i (safe UB)
        res = (True, inst.pi[i], "UNKNOWN")
    return res if return_status else res[:2]


def cell_feasible(i: int, regime_i: str, fwd: ForwardResult, inst: Instance,
                  pre: Preprocessed, cfg: Config) -> bool:
    feasible, _ = solve_single_cell(i, regime_i, fwd, inst, pre, cfg,
                                    time_budget=cfg.filter_budget)
    return feasible
