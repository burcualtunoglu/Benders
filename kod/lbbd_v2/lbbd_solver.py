"""lbbd_solver.py — the exact LBBD main loop (05 §13, 04 §9-§10).

Invariants (never violated):
  * UB comes ONLY from the master dual bound (ms.obj_bound).
  * LB comes ONLY from validated feasible incumbents (SP OPTIMAL / FEASIBLE / TL_WITH_INCUMBENT).
  * Cuts are emitted ONLY from proven SP states (INFEASIBLE_PROVEN → feasibility cut;
    OPTIMAL → optimality cut). Timeout/numeric never yield a cut and never mean INFEASIBLE.
  * LB ≤ OPT ≤ UB every iteration; OPTIMAL is declared only when UB − LB ≤ ε.

Baked-in exact accelerators (cert V2): strong SP, always-on per-cell optimality cut,
dual-bound optimality cut on timeouts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lbbd_v2.config import Config
from lbbd_v2.preprocessing import preprocess_instance
from lbbd_v2.lbbd_master import build_lbbd_master
from lbbd_v2.benders_forward import (
    forward_pass, FWD_OK, FWD_UNREACHABLE, FWD_TIME_INCONSISTENT)
from lbbd_v2.lbbd_subproblem_resource import (
    solve_resource_subproblem, solve_single_cell, SPStatus)
from lbbd_v2.lbbd_subproblem_typeagg import solve_typeagg_subproblem
from lbbd_v2.lbbd_conflicts import extract_conflict
from lbbd_v2.lbbd_heuristics import (
    build_fallback_incumbent, build_greedy_incumbent, build_greedy_incumbent_fast,
    build_containment_incumbent, _assign_cell)


def _closed_form_warm_x(C, regime, fwd, inst, pre, cfg):
    """Closed-form feasible x-assignment for the master's control set C (SEED ONLY), used to
    warm-start the multi-cell SP so it returns an incumbent fast even with 855 vehicles.
    Greedy over cells (high π first), sharing the global vehicle pool. Returns {(i,k): 1.0}."""
    midpoint = (cfg.omega_mode == "midpoint")
    pool = set(inst.K)
    wx = {}
    for c in sorted(C, key=lambda i: -inst.pi[i]):
        ts_c, tsmin_c = fwd.ts[c], fwd.ts_min.get(c, fwd.ts[c])
        lo, hi = (ts_c, fwd.tm[c]) if regime.get(c) == "pre" else (fwd.tm[c], fwd.te[c])
        rec = _assign_cell(c, lo, hi, ts_c, tsmin_c, pool, inst, pre, midpoint, 64)
        if rec is None:
            continue
        for k in rec["S"]:
            wx[(c, k)] = 1.0
        pool.difference_update(rec["S"])
    return wx or None
from lbbd_v2.lbbd_cuts import (
    cut_connectivity, cut_propagation, cut_singleton_infeasible,
    cut_conflict_infeasible, cut_cell_optimality, cut_aggregate_optimality,
    cut_dual_bound_optimality, cut_ignition_time)


def _late_ignition_cells(fwd, inst, pre, tol: float = 1e-6) -> List[int]:
    """Burning cells whose forward ignition time t^s_i exceeds the monolithic-feasibility cap
    M^i_i/a_i = burn_i·margin (from constraint (5) with the forced v_ik=0 at x_ik=0). Such a
    (y,z,q) is monolithic-INFEASIBLE, so its objective is not a valid incumbent (see
    cut_ignition_time). Cells with a_i≈0 are exempt (δ_ik≡0 ⇒ no cap)."""
    out: List[int] = []
    for i in fwd.reached:
        ai = inst.a[i]
        if ai > 1e-12 and fwd.ts.get(i, 0.0) > pre.M_i[i] / ai + tol:
            out.append(i)
    return out


@dataclass
class LBBDResult:
    status: str
    LB: float
    UB: float
    gap: float
    incumbent: Optional[dict]
    iterations: int
    sp_calls: Dict[str, int]
    cut_counts: Dict[str, int]
    runtime: float
    iteration_log: List[dict] = field(default_factory=list)
    cut_records: List[dict] = field(default_factory=list)
    max_LB_over_opt: Optional[float] = None
    max_opt_over_UB: Optional[float] = None
    # Tekil-SP önbellek/enstrümantasyon (yalnız baseline solver doldurur; reviewer (a)):
    #   single_sp_solves, single_sp_cache_hits, single_sp_distinct, single_sp_time_s,
    #   single_sp_status_dist (dict), dup_cuts_prevented.
    diag_stats: Dict[str, Any] = field(default_factory=dict)


def _decision(obj) -> dict:
    """Snapshot the full binary master decision from a MasterSolution or an Incumbent."""
    return {"y": dict(obj.y), "z": dict(obj.z), "q": dict(obj.q),
            "u_pre": dict(obj.u_pre), "u_post": dict(obj.u_post)}


def _select_sp(cfg):
    """Multi-cell subproblem engine. type-agg is required at |K|=150 (per-resource intractable)."""
    if cfg.sp_engine == "typeagg":
        return solve_typeagg_subproblem
    return solve_resource_subproblem


def _regime_of(sol, C):
    return {i: ("pre" if sol.u_pre.get(i, 0.0) > 0.5 else "post") for i in C}


def _gap(LB: float, UB: float) -> float:
    # UB sonsuz veya LB=-sonsuz iken aralık tanımsız (sonsuz). Kapalı [0,0] için 0 döner
    # (eski `UB==0 -> inf` yanlıştı); payda max(1,|UB|) sıfıra bölmeyi önler.
    if UB >= float("inf") or LB <= float("-inf"):
        return float("inf")
    return (UB - LB) / max(1.0, abs(UB))


def solve_lbbd(inst, cfg: Config, pre=None) -> LBBDResult:
    t0 = time.time()
    if pre is None:
        pre = preprocess_instance(inst, cfg)
    master = build_lbbd_master(inst, pre, cfg)
    sp_solve = _select_sp(cfg)

    LB, UB = float("-inf"), float("inf")
    incumbent: Optional[dict] = None
    sp_calls: Dict[str, int] = {s.value: 0 for s in SPStatus}
    cut_counts: Dict[str, int] = {}
    log: List[dict] = []
    streak = 0

    def count_cut(kind: str):
        cut_counts[kind] = cut_counts.get(kind, 0) + 1

    def pi_free(y):
        return sum(inst.pi[i] * (1 - y[i]) for i in inst.Nf)

    # ---------- initial incumbents (HEURISTIC ONLY) -------------------
    # Containment first: the only seed that is monolithic-FEASIBLE at scale (firebreak-sealed
    # burning core; most cells unburnt). The full-cascade seeds are gated out by the
    # mono-feasibility check on any grid where they ignite past the cap. The old per-cell-SP
    # `build_greedy_incumbent` is intentionally NOT used here — on large |Nf| it spends the whole
    # budget building a seed that is then gated out (full cascade); it is superseded by containment.
    candidates = [("containment", build_containment_incumbent(inst, pre, cfg)),
                  ("fallback", build_fallback_incumbent(inst, pre, cfg))]
    if getattr(cfg, "lbbd_fast_greedy_seed", True):
        candidates.append(("greedy_fast", build_greedy_incumbent_fast(inst, pre, cfg)))
    for source, cand in candidates:
        fwd = forward_pass(cand, inst)
        if fwd.status != FWD_OK:
            continue
        if _late_ignition_cells(fwd, inst, pre):
            continue                                 # monolithic-infeasible structure -> not a valid LB
        if not cand.C:
            val = pi_free(cand.y)                    # u ≡ 0 ⇒ recourse 0
        else:
            # A candidate carrying a closed-form assignment (containment) warm-starts the SP so it
            # returns a feasible recourse immediately even at scale (855 vehicles); symmetry-break
            # is disabled for that one call so the non-canonical warm start is accepted.
            wx = cand.start_vars.get("x") if getattr(cand, "start_vars", None) else None
            if wx is not None and cfg.sp_engine == "strong_resource":
                sp = sp_solve(fwd, cand.C, cand.regime, inst, pre, cfg,
                              time_budget=cfg.heur_budget, warm_x=wx, symmetry_break=False)
            else:
                sp = sp_solve(fwd, cand.C, cand.regime, inst, pre, cfg,
                              time_budget=cfg.heur_budget)
            sp_calls[sp.status.value] += 1
            if sp.status not in (SPStatus.OPTIMAL, SPStatus.FEASIBLE_NOT_PROVEN_OPTIMAL,
                                 SPStatus.TIME_LIMIT_WITH_INCUMBENT) or sp.obj is None:
                continue
            val = pi_free(cand.y) + sp.obj
        if val > LB:
            LB = val
            incumbent = {"source": source, "C": cand.C,
                         "regime": cand.regime, "value": val, **_decision(cand)}

    # ---------- main loop --------------------------------------------
    it = 0
    status = "TIME_LIMIT"
    while time.time() - t0 < cfg.total_budget and it < cfg.max_iterations:
        it += 1
        cuts_added = 0
        rec: Dict[str, Any] = {"iter": it}

        ms = master.solve(budget=cfg.master_budget)
        if ms.status == "INFEASIBLE":
            status = "INFEASIBLE"
            break
        if ms.obj_bound is not None:
            UB = min(UB, ms.obj_bound)
        rec.update(UB=UB, LB=LB, gap=_gap(LB, UB), master_status=ms.status)
        if UB - LB <= cfg.eps_abs or _gap(LB, UB) <= cfg.eps_rel:
            status = "OPTIMAL"
            log.append(rec)
            break

        fwd = forward_pass(ms, inst)
        rec["fwd_status"] = fwd.status

        if fwd.status == FWD_UNREACHABLE:
            for j in sorted(fwd.unreachable):
                cut_connectivity(fwd.unreachable, j, inst).apply(master)
                count_cut("connectivity"); cuts_added += 1
            rec["cuts"] = cuts_added; log.append(rec)
            continue

        if fwd.status == FWD_TIME_INCONSISTENT:
            for (i, j) in fwd.violating_arcs:
                cut_propagation(i, j, fwd).apply(master)
                count_cut("propagation"); cuts_added += 1
            rec["cuts"] = cuts_added; log.append(rec)
            continue

        # monolithic-feasibility gate: a burning cell igniting past burn_i·margin makes the
        # monolithic infeasible (constraint (5), forced v_ik=0). Forbid the offending ignition
        # path and reject this structure (no SP, no LB) — exact (see cut_ignition_time).
        late = _late_ignition_cells(fwd, inst, pre)
        if late:
            for i in late:
                cut_ignition_time(i, fwd).apply(master)
                count_cut("ignition_time"); cuts_added += 1
            rec.update(cuts=cuts_added, late_ignition=sorted(late)); log.append(rec)
            continue

        C = [i for i in inst.Nf if ms.u(i) > 0.5]
        regime = _regime_of(ms, C)
        rec["C_cells"] = list(C)          # diagnostic: which cells the master proposed

        # singleton pre-screen + per-cell p_solo (one single-cell solve serves both)
        p_solo: Dict[int, float] = {}
        dead: List[int] = []
        for i in C:
            feasible, ps = solve_single_cell(i, regime[i], fwd, inst, pre, cfg,
                                             time_budget=cfg.filter_budget)
            if not feasible:
                dead.append(i)
            else:
                p_solo[i] = ps
        if dead:
            for i in dead:
                cut_singleton_infeasible(i, ms, fwd, inst).apply(master)
                count_cut("singleton_infeasible"); cuts_added += 1
            rec.update(cuts=cuts_added, dead=dead); log.append(rec)
            continue

        # always-on per-cell optimality cuts (exact, DECISIVE): cap each ρ_i ≤ p_solo_i + π_i R({i})
        if cfg.use_percell_optimality_cuts:
            for i in C:
                if ms.rho.get(i, 0.0) > p_solo[i] + cfg.eps_abs:
                    cut_cell_optimality(i, p_solo[i], ms, fwd, inst).apply(master)
                    count_cut("cell_optimality"); cuts_added += 1

        # multi-cell subproblem (per-resource, or type-agg for large |K|). Warm-start the
        # per-resource SP with a closed-form assignment so it returns an incumbent fast at scale
        # (symmetry-break off for the warm-started call so the non-canonical start is accepted).
        if cfg.sp_engine == "strong_resource" and getattr(cfg, "sp_warm_start", True):
            wx = _closed_form_warm_x(C, regime, fwd, inst, pre, cfg)
            if wx is not None:
                sp = sp_solve(fwd, C, regime, inst, pre, cfg, time_budget=cfg.sp_budget,
                              warm_x=wx, symmetry_break=False)
            else:
                sp = sp_solve(fwd, C, regime, inst, pre, cfg, time_budget=cfg.sp_budget)
        else:
            sp = sp_solve(fwd, C, regime, inst, pre, cfg, time_budget=cfg.sp_budget)
        sp_calls[sp.status.value] += 1
        rec.update(sp_status=sp.status.value, sp_obj=sp.obj, C_size=len(C),
                   sp_runtime=sp.runtime, sp_bound=sp.obj_bound)

        if sp.status == SPStatus.INFEASIBLE_PROVEN:
            Cstar = extract_conflict(C, regime, fwd, inst, pre, cfg, sp_solve=sp_solve)
            cut_conflict_infeasible(Cstar, ms, fwd, inst).apply(master)
            count_cut("conflict_infeasible"); cuts_added += 1
            streak = 0
            rec.update(cuts=cuts_added, conflict=sorted(Cstar)); log.append(rec)
            continue

        if sp.status == SPStatus.OPTIMAL:
            val = pi_free(ms.y) + sp.obj
            if val > LB:
                LB = val
                incumbent = {"source": "master", "C": C, "regime": regime,
                             "value": val, "sp": sp.solution, **_decision(ms)}
            cut_aggregate_optimality(C, sp.obj, ms, fwd, inst).apply(master)
            count_cut("aggregate_optimality"); cuts_added += 1
            streak = 0
            rec.update(cuts=cuts_added, LB=LB, gap=_gap(LB, UB))
            log.append(rec)
            if UB - LB <= cfg.eps_abs or _gap(LB, UB) <= cfg.eps_rel:
                status = "OPTIMAL"
                break
            assert LB <= UB + cfg.compare_tol, f"invariant LB>UB: {LB} > {UB}"
            continue

        if sp.status in (SPStatus.FEASIBLE_NOT_PROVEN_OPTIMAL,
                         SPStatus.TIME_LIMIT_WITH_INCUMBENT):
            val = pi_free(ms.y) + sp.obj
            if val > LB:
                LB = val
                incumbent = {"source": "master_incumbent", "C": C, "regime": regime,
                             "value": val, **_decision(ms)}
            if cfg.use_dual_optimality_cut and sp.obj_bound is not None:
                cut_dual_bound_optimality(C, sp.obj_bound, ms, fwd, inst).apply(master)
                count_cut("dual_bound"); cuts_added += 1
            streak += 1
            rec.update(cuts=cuts_added, LB=LB, streak=streak); log.append(rec)
            if streak > cfg.max_inconclusive and cuts_added == 0:
                status = "INCONCLUSIVE"
                break
            continue

        # TIME_LIMIT_NO_INCUMBENT / NUMERICAL_FAILURE: no LB, no cut
        streak += 1
        rec.update(cuts=0, streak=streak); log.append(rec)
        if streak > cfg.max_inconclusive:
            status = "INCONCLUSIVE"
            break

        # stuck: no progress possible this iteration
        if cuts_added == 0:
            status = "INCONCLUSIVE"
            break

    runtime = time.time() - t0
    return LBBDResult(
        status=status, LB=LB, UB=UB, gap=_gap(LB, UB), incumbent=incumbent,
        iterations=it, sp_calls=sp_calls, cut_counts=cut_counts, runtime=runtime,
        iteration_log=log, cut_records=list(master.cut_records),
    )
