"""subproblem_weak.py — ZAYIF (Big-M'li) per-resource alt problem (F-SPWEAK faktörü).

Bu, DÜZELTİLMİŞ modelin (model_monolithic_bigmfix) alt probleme birebir kısıtlamasıdır: master
kararı ve ileri-geçiş zamanları sabitlenip özgün (2),(4)–(38) kısıtları Big-M'leriyle KORUNARAK
kurulur. Güçlü SP'nin (Lemma A) aksine v_ik, t_ik serbest değişkendir ve δ, ω^max/ω^min Big-M ile
tanımlanır; ω^max/ω^min için hmax/hmin ikili seçicileri eklenir. Bu yüzden LP gevşetmesi gevşektir
ve Gurobi optimalliği daha yavaş kanıtlar — ama Φ (=Σ_{i∈C} p_i) güçlü SP ile AYNIdır (ikisi de kesin).

Kritik: (4)/(5) deaktivasyon sabiti a_i·M_d'dir (M_i DEĞİL) — yani bu SP düzeltilmiş modelin
kısıtlamasıdır; geç-ateşleme sahte infizibilitesini taşımaz. (docs/asama0_rapor.md §4-C1.)

SPStatus / SPResult / _map_status / _optimize_capped GÜÇLÜ modülden içe aktarılır ki solver iki
motoru da tek biçimde ele alsın.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from lbbd_v2.config import Config
from lbbd_v2.data_loader import Instance
from lbbd_v2.preprocessing import Preprocessed
from lbbd_v2.benders_forward import ForwardResult
from lbbd_v2.lbbd_subproblem_resource import (
    SPStatus, SPResult, _map_status, _optimize_capped, _v_min, _tc_window)


def _build(cells, regime, fwd, inst, pre, cfg, K, integer, mip_gap=None):
    m = gp.Model("sp_weak")
    m.setParam("OutputFlag", 0)
    m.setParam("Seed", cfg.gurobi_seed)
    m.setParam("Threads", cfg.gurobi_threads)
    eff_gap = getattr(cfg, "sp_mip_gap", 0.0) if mip_gap is None else mip_gap
    m.setParam("MIPGap", eff_gap)   # 0.0=kanıt; pozitif=erken dur; per-çağrı override (güçlü SP ile
    # AYNI sözleşme: LB=ObjVal, kesme=ObjBound; kesinlik her toleransta korunur).
    K = list(K)
    midpoint = (cfg.omega_mode == "midpoint")
    Md = pre.M_d
    a, mu, pi, beta, d = inst.a, inst.mu, inst.pi, inst.beta, inst.d
    Mi, Ms = pre.M_i, pre.M_s
    Dwat, Dbuf = inst.delta_wat, inst.delta_buf
    vt = GRB.BINARY if integer else GRB.CONTINUOUS

    x = m.addVars(cells, K, vtype=vt, lb=0.0, ub=1.0, name="x")
    t = m.addVars(cells, K, lb=0.0, name="t")
    v = m.addVars(cells, K, lb=0.0, name="v")
    s = m.addVars(cells, K, lb=0.0, name="s")
    delta = m.addVars(cells, K, lb=0.0, name="delta")
    tc = m.addVars(cells, lb=0.0, name="tc")
    p = m.addVars(cells, lb=0.0, name="p")
    omega = m.addVars(cells, lb=0.0, name="omega")
    omega_max = m.addVars(cells, lb=0.0, name="omega_max")
    if midpoint:
        omega_min = m.addVars(cells, lb=0.0, name="omega_min")
        hmax = m.addVars(cells, K, vtype=GRB.BINARY, name="hmax")
        hmin = m.addVars(cells, K, vtype=GRB.BINARY, name="hmin")
    else:
        omega_min = hmax = hmin = None

    m.setObjective(gp.quicksum(p[i] for i in cells), GRB.MAXIMIZE)

    for i in cells:
        lo, hi = _tc_window(fwd, i, regime[i])       # (34),(35),(37),(38) rejim sabit
        tsi = fwd.ts[i]
        tsmin = fwd.ts_min.get(i, tsi)
        m.addConstr(tc[i] >= lo, f"tc_lo[{i}]")
        m.addConstr(tc[i] <= hi, f"tc_hi[{i}]")
        m.addConstr(p[i] <= pi[i] - beta[i] * (tc[i] - tsi), f"c2[{i}]")     # (2)
        m.addConstr(p[i] <= pi[i], f"c3[{i}]")                               # (3), u=1,y=1
        m.addConstr(gp.quicksum(x[i, k] for k in K) >= 1, f"must_serve[{i}]")  # (31)/(32), u=1

        for k in K:
            # (4),(5): deaktivasyon sabiti a_i·M_d (DÜZELTİLMİŞ)
            m.addConstr(delta[i, k] >= a[i] * (v[i, k] - tsi) - a[i] * Md * (1 - x[i, k]),
                        f"c4[{i},{k}]")
            m.addConstr(delta[i, k] <= a[i] * (v[i, k] - tsi) + a[i] * Md * (1 - x[i, k]),
                        f"c5[{i},{k}]")
            m.addConstr(delta[i, k] <= Mi[i] * x[i, k], f"c6[{i},{k}]")       # (6)
            m.addConstr(omega_max[i] >= delta[i, k] - Ms[i] * (1 - x[i, k]), f"c7[{i},{k}]")  # (7)
            # (21),(22): varış = sevk + seyahat
            m.addConstr(v[i, k] >= t[i, k] + d[(i, k)] - Md * (1 - x[i, k]), f"c21[{i},{k}]")
            m.addConstr(v[i, k] <= t[i, k] + d[(i, k)] + Md * (1 - x[i, k]), f"c22[{i},{k}]")
            m.addConstr(t[i, k] <= Md * x[i, k], f"c23t[{i},{k}]")            # (23)
            m.addConstr(s[i, k] <= Md * x[i, k], f"c23s[{i},{k}]")
            m.addConstr(v[i, k] <= Md * x[i, k], f"c23v[{i},{k}]")
            m.addConstr(v[i, k] + s[i, k] <= tc[i] + Md * (1 - x[i, k]), f"c24[{i},{k}]")  # (24)
            m.addConstr(v[i, k] >= tsi - Md * (1 - x[i, k]), f"c25[{i},{k}]")  # (25)
            m.addConstr(t[i, k] >= tsmin + Dbuf - Md * (1 - x[i, k]), f"c29[{i},{k}]")  # (29)

        if midpoint:
            for k in K:
                m.addConstr(omega_max[i] <= delta[i, k] + Ms[i] * (1 - hmax[i, k]), f"c8[{i},{k}]")
                m.addConstr(hmax[i, k] <= x[i, k], f"c9[{i},{k}]")
                m.addConstr(omega_min[i] <= delta[i, k] + Ms[i] * (1 - x[i, k]), f"c10[{i},{k}]")
                m.addConstr(omega_min[i] >= delta[i, k] - Ms[i] * (1 - hmin[i, k]), f"c11[{i},{k}]")
                m.addConstr(hmin[i, k] <= x[i, k], f"c12[{i},{k}]")
            m.addConstr(gp.quicksum(hmax[i, k] for k in K) == 1, f"c14[{i}]")  # u=1
            m.addConstr(gp.quicksum(hmin[i, k] for k in K) == 1, f"c15[{i}]")
            m.addConstr(2 * omega[i] == omega_max[i] + omega_min[i] + 2 * Dwat, f"c16[{i}]")  # u=1
        else:
            m.addConstr(omega[i] == omega_max[i] + Dwat, f"c16w[{i}]")        # u=1

        m.addConstr(omega[i] <= Ms[i], f"c17[{i}]")                          # (17), u=1
        m.addConstr(gp.quicksum(mu[k] * s[i, k] for k in K) >= omega[i], f"c18[{i}]")  # (18)/(19)

    for k in K:                                                              # (33) global rekabet
        m.addConstr(gp.quicksum(x[i, k] for i in cells) <= 1, f"c33[{k}]")

    handles = dict(x=x, tc=tc, p=p, omega=omega, delta=delta)
    return m, handles


def solve_weak_subproblem(fwd: ForwardResult, C: Iterable[int], regime: Dict[int, str],
                          inst: Instance, pre: Preprocessed, cfg: Config,
                          time_budget: Optional[float] = None,
                          warm_x=None, symmetry_break=None,
                          mip_gap: Optional[float] = None) -> SPResult:
    """Zayıf SP; imza güçlü SP ile aynı (warm_x/symmetry_break kabul edilir, KULLANILMAZ)."""
    cells = sorted(C)
    if not cells:
        return SPResult(SPStatus.OPTIMAL, 0.0, 0.0, 0.0, 0.0, {})
    m, h = _build(cells, regime, fwd, inst, pre, cfg, inst.K, integer=True, mip_gap=mip_gap)
    t0 = time.time()
    _optimize_capped(m, time_budget)
    if m.Status == GRB.INF_OR_UNBD:
        m.setParam("DualReductions", 0)
        rem = None if time_budget is None else max(1.0, time_budget - (time.time() - t0))
        _optimize_capped(m, rem)
    runtime = time.time() - t0

    status = _map_status(m)
    obj = m.ObjVal if m.SolCount > 0 else None
    obj_bound = (m.ObjBound if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED,
                                            GRB.SUBOPTIMAL) else None)
    try:
        gap = m.MIPGap if m.SolCount > 0 else None
    except Exception:  # noqa: BLE001
        gap = None
    solution: Dict = {}
    if m.SolCount > 0:
        solution["p"] = {i: h["p"][i].X for i in cells}
        solution["tc"] = {i: h["tc"][i].X for i in cells}
        solution["omega"] = {i: h["omega"][i].X for i in cells}
        solution["x"] = {(i, k): h["x"][i, k].X for i in cells for k in inst.K
                         if h["x"][i, k].X > 0.5}
    return SPResult(status, obj, obj_bound, gap, runtime, solution)


def solve_single_cell_weak(i: int, regime_i: str, fwd: ForwardResult, inst: Instance,
                           pre: Preprocessed, cfg: Config,
                           time_budget: Optional[float] = None, return_status: bool = False,
                           timing: Optional[dict] = None):
    """Rekabetsiz tek-hücre (tüm filo) zayıf çözüm -> (fizibil?, p_solo[, kind]). Güçlü sürümle
    AYNI sözleşme (return_status/timing: reviewer (a)/(b))."""
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
        # #3: K5 için p_solo GEÇERLİ ÜST SINIR olmalı (güçlü SP ile aynı sözleşme):
        # min(ObjBound, π_i). p.X (ObjVal) tolerans altında p_i(tam)'ın altında kalabilirdi.
        res = (True, min(m.ObjBound, inst.pi[i]), "FEASIBLE_VERIFIED")
    elif m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        res = (False, 0.0, "INFEASIBLE_PROVEN")
    else:
        res = (True, inst.pi[i], "UNKNOWN")   # zaman aşımı/sayısal: bilgi yok -> güvenli UB π_i
    return res if return_status else res[:2]
