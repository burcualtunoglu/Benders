"""model_monolithic.py — exact canonical (1)-(50) MILP reference.

Computational reference for parity/exactness (Gate B). Corrected canonical model under the
approved K1-K4 config; no hardcoded scalar water Big-M — every Big-M comes from preprocessing.
Supports omega_mode in {"midpoint","worstcase"} and the (26')/(28') variants via flags.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import gurobipy as gp
from gurobipy import GRB

from lbbd_v2.config import Config
from lbbd_v2.data_loader import Instance
from lbbd_v2.preprocessing import Preprocessed, preprocess_instance


@dataclass
class MonoResult:
    status: str
    obj_val: Optional[float]
    obj_bound: Optional[float]
    actual_mip_gap: Optional[float]
    runtime: float
    solution_count: int
    termination_reason: str
    proven_optimal: bool
    configuration: Dict[str, Any]
    solution: Dict[str, Any] = field(default_factory=dict)
    gurobi_version: str = ""


def build_monolithic_model(inst: Instance, pre: Preprocessed, cfg: Config) -> gp.Model:
    m = gp.Model("monolithic_canonical")
    Nf, K, Na = inst.Nf, inst.K, inst.Na
    nb = inst.neighbors
    Nplus = inst.Nplus
    arcs = inst.arcs
    alpha = inst.alpha
    lam, sig, a, pi, beta = inst.lam, inst.sig, inst.a, inst.pi, inst.beta
    mu, d = inst.mu, inst.d
    Md = pre.M_d
    Mi = pre.M_i
    Ms = pre.M_s
    Dwat, Dbuf = inst.delta_wat, inst.delta_buf
    midpoint = (cfg.omega_mode == "midpoint")

    # ---- variables -------------------------------------------------------
    p = m.addVars(Nf, lb=0.0, name="p")
    y = m.addVars(Nf, vtype=GRB.BINARY, name="y")
    ts = m.addVars(Nf, lb=0.0, name="ts")
    tm = m.addVars(Nf, lb=0.0, name="tm")
    te = m.addVars(Nf, lb=0.0, name="te")
    tc = m.addVars(Nf, lb=0.0, name="tc")
    u_pre = m.addVars(Nf, vtype=GRB.BINARY, name="u_pre")
    u_post = m.addVars(Nf, vtype=GRB.BINARY, name="u_post")
    x = m.addVars(Nf, K, vtype=GRB.BINARY, name="x")
    t = m.addVars(Nf, K, lb=0.0, name="t")
    s = m.addVars(Nf, K, lb=0.0, name="s")
    v = m.addVars(Nf, K, lb=0.0, name="v")
    delta = m.addVars(Nf, K, lb=0.0, name="delta")
    omega = m.addVars(Nf, lb=0.0, name="omega")
    omega_max = m.addVars(Nf, lb=0.0, name="omega_max")
    ts_min = m.addVars(Nf, lb=0.0, name="ts_min")
    z = m.addVars(arcs, vtype=GRB.BINARY, name="z")
    q = m.addVars(arcs, vtype=GRB.BINARY, name="q")
    bmin = m.addVars([(i, j) for i in Nf for j in Nplus[i]], vtype=GRB.BINARY, name="bmin")
    if midpoint:
        omega_min = m.addVars(Nf, lb=0.0, name="omega_min")
        hmax = m.addVars(Nf, K, vtype=GRB.BINARY, name="hmax")
        hmin = m.addVars(Nf, K, vtype=GRB.BINARY, name="hmin")
    else:
        omega_min = hmax = hmin = None

    def u(i):
        return u_pre[i] + u_post[i]

    # ---- objective (1) ---------------------------------------------------
    m.setObjective(gp.quicksum(p[i] for i in Nf), GRB.MAXIMIZE)

    # ---- reward (2),(3) --------------------------------------------------
    m.addConstrs((p[i] <= pi[i] - beta[i] * (tc[i] - ts[i]) for i in Nf), "c2")
    m.addConstrs((p[i] <= pi[i] * (u(i) + 1 - y[i]) for i in Nf), "c3")

    # ---- delay water δ (4),(5),(6) --------------------------------------
    m.addConstrs((delta[i, k] >= a[i] * (v[i, k] - ts[i]) - Mi[i] * (1 - x[i, k])
                  for i in Nf for k in K), "c4")
    m.addConstrs((delta[i, k] <= a[i] * (v[i, k] - ts[i]) + Mi[i] * (1 - x[i, k])
                  for i in Nf for k in K), "c5")
    m.addConstrs((delta[i, k] <= Mi[i] * x[i, k] for i in Nf for k in K), "c6")

    # ---- omega_max (7),(8),(9) ------------------------------------------
    m.addConstrs((omega_max[i] >= delta[i, k] - Ms[i] * (1 - x[i, k])
                  for i in Nf for k in K), "c7")
    if midpoint:
        m.addConstrs((omega_max[i] <= delta[i, k] + Ms[i] * (1 - hmax[i, k])
                      for i in Nf for k in K), "c8")
        m.addConstrs((hmax[i, k] <= x[i, k] for i in Nf for k in K), "c9")
        # omega_min (10),(11),(12)
        m.addConstrs((omega_min[i] <= delta[i, k] + Ms[i] * (1 - x[i, k])
                      for i in Nf for k in K), "c10")
        m.addConstrs((omega_min[i] >= delta[i, k] - Ms[i] * (1 - hmin[i, k])
                      for i in Nf for k in K), "c11")
        m.addConstrs((hmin[i, k] <= x[i, k] for i in Nf for k in K), "c12")
        # selectors (14),(15)
        m.addConstrs((gp.quicksum(hmax[i, k] for k in K) == u(i) for i in Nf), "c14")
        m.addConstrs((gp.quicksum(hmin[i, k] for k in K) == u(i) for i in Nf), "c15")
        # midpoint (16): 2ω = ω_max + ω_min + 2 Δ_wat u
        m.addConstrs((2 * omega[i] == omega_max[i] + omega_min[i] + 2 * Dwat * u(i)
                      for i in Nf), "c16")
    else:
        # worstcase (16w): ω = ω_max + Δ_wat u
        m.addConstrs((omega[i] == omega_max[i] + Dwat * u(i) for i in Nf), "c16w")

    # (17) omega ≤ M_s u
    m.addConstrs((omega[i] <= Ms[i] * u(i) for i in Nf), "c17")

    # ---- water supply (18),(19) -----------------------------------------
    m.addConstrs((gp.quicksum(mu[k] * s[i, k] for k in K) >= omega[i] - Ms[i] * (1 - u_pre[i])
                  for i in Nf), "c18")
    m.addConstrs((gp.quicksum(mu[k] * s[i, k] for k in K) >= omega[i] - Ms[i] * (1 - u_post[i])
                  for i in Nf), "c19")

    # ---- arrival/schedule (21),(22),(23),(24),(25) ----------------------
    m.addConstrs((v[i, k] >= t[i, k] + d[i, k] - Md * (1 - x[i, k]) for i in Nf for k in K), "c21")
    m.addConstrs((v[i, k] <= t[i, k] + d[i, k] + Md * (1 - x[i, k]) for i in Nf for k in K), "c22")
    m.addConstrs((t[i, k] <= Md * x[i, k] for i in Nf for k in K), "c23t")
    m.addConstrs((s[i, k] <= Md * x[i, k] for i in Nf for k in K), "c23s")
    m.addConstrs((v[i, k] <= Md * x[i, k] for i in Nf for k in K), "c23v")
    m.addConstrs((ts[i] <= Md * y[i] for i in Nf), "c23ts")
    m.addConstrs((v[i, k] + s[i, k] <= tc[i] + Md * (1 - x[i, k]) for i in Nf for k in K), "c24")
    m.addConstrs((v[i, k] >= ts[i] - Md * (1 - x[i, k]) for i in Nf for k in K), "c25")

    # ---- dispatch trigger t^{s,min} (26'/26),(27),(28),(28') ------------
    if cfg.constraint_26_variant == "26prime":  # K1
        m.addConstrs((ts_min[i] <= ts[j] + Md * (1 - y[j])
                      for i in Nf for j in Nplus[i]), "c26p")
    else:  # (26) plain — baseline variant
        m.addConstrs((ts_min[i] <= ts[j] for i in Nf for j in Nplus[i]), "c26")
    m.addConstrs((ts_min[i] >= ts[j] - Md * (1 - bmin[i, j])
                  for i in Nf for j in Nplus[i]), "c27")
    m.addConstrs((gp.quicksum(bmin[i, j] for j in Nplus[i]) == 1 for i in Nf), "c28")
    if cfg.use_28prime:  # K2
        m.addConstrs((bmin[i, j] <= y[j] + (1 - y[i]) for i in Nf for j in Nplus[i]), "c28p")
    m.addConstrs((t[i, k] >= ts_min[i] + Dbuf - Md * (1 - x[i, k]) for i in Nf for k in K), "c29")

    # ---- assignment/control bonds (30),(31),(32),(33) -------------------
    m.addConstrs((x[i, k] <= y[i] for i in Nf for k in K), "c30")
    m.addConstrs((u(i) >= x[i, k] for i in Nf for k in K), "c31")
    m.addConstrs((u(i) <= gp.quicksum(x[i, k] for k in K) for i in Nf), "c32")
    m.addConstrs((gp.quicksum(x[i, k] for i in Nf) <= 1 for k in K), "c33")

    # ---- control time / regimes (34)-(39) -------------------------------
    m.addConstrs((tc[i] <= te[i] for i in Nf), "c34")
    m.addConstrs((tc[i] >= ts[i] - Md * (1 - u(i)) for i in Nf), "c35")
    m.addConstrs((tc[i] <= Md * u(i) for i in Nf), "c36")
    m.addConstrs((tc[i] <= tm[i] + Md * (1 - u_pre[i]) for i in Nf), "c37")
    m.addConstrs((tc[i] >= tm[i] - Md * (1 - u_post[i]) for i in Nf), "c38")
    m.addConstrs((u(i) <= 1 for i in Nf), "c39")

    # ---- spread structure (40)-(43) -------------------------------------
    m.addConstrs((gp.quicksum(z[i, j] for j in nb[i]) == len(nb[i]) * (y[i] - u_pre[i])
                  for i in Nf), "c40")
    m.addConstrs((len(nb[j]) * y[j] >= gp.quicksum(z[i, j] for i in nb[j]) for j in Nf), "c41")
    m.addConstrs((q[i, j] <= z[i, j] for (i, j) in arcs), "c42")
    m.addConstrs((gp.quicksum(q[i, j] for i in nb[j]) == y[j] - inst.e[j] for j in Nf), "c43")

    # ---- time consistency (44),(45),(46) --------------------------------
    m.addConstrs((ts[j] >= tm[i] - Md * (1 - q[i, j]) for (i, j) in arcs), "c44")
    m.addConstrs((ts[j] <= tm[i] + Md * (1 - q[i, j]) for (i, j) in arcs), "c45")
    m.addConstrs((ts[j] <= tm[i] + Md * (1 - z[i, j]) for (i, j) in arcs), "c46")

    # ---- initial + dynamics (47),(48),(49),(50) -------------------------
    m.addConstr(gp.quicksum(y[i] for i in Na) == len(Na), "c47")
    m.addConstr(gp.quicksum(ts[i] for i in Na) == 0, "c48")
    m.addConstrs((tm[i] == ts[i] + alpha / lam[i] for i in Nf), "c49")
    m.addConstrs((te[i] == tm[i] + alpha / sig[i] for i in Nf), "c50")

    m._vars = dict(p=p, y=y, ts=ts, tm=tm, te=te, tc=tc, u_pre=u_pre, u_post=u_post,
                   x=x, t=t, s=s, v=v, delta=delta, omega=omega, omega_max=omega_max,
                   ts_min=ts_min, z=z, q=q, bmin=bmin,
                   omega_min=omega_min, hmax=hmax, hmin=hmin)
    return m


def apply_warm_start(m: gp.Model, start_vars: Dict[str, Dict]) -> int:
    """Seed a MIPStart from a closed-form greedy solution (SEED ONLY).

    Sets ``.Start`` on the integer/decision variables present in both ``start_vars`` and the
    model (``y, z, q, u_pre, u_post, x, bmin, hmax, hmin``). Unset variables are completed by
    Gurobi's MIPStart LP. A rejected start is harmless — it can never affect the proved bound
    or remove the optimum. Returns the number of variable values seeded.
    """
    mv = getattr(m, "_vars", {})
    seeded = 0
    for name, values in start_vars.items():
        var = mv.get(name)
        if var is None:
            continue
        for key, val in values.items():
            if key in var:
                var[key].Start = val
                seeded += 1
    m.update()
    return seeded


def _status_name(m: gp.Model) -> str:
    return {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.UNBOUNDED: "UNBOUNDED",
            GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INTERRUPTED: "INTERRUPTED"}.get(m.Status, f"STATUS_{m.Status}")


def solve_monolithic(inst: Instance, cfg: Config, pre: Optional[Preprocessed] = None,
                     mip_gap: Optional[float] = None, time_limit: Optional[float] = None,
                     extract_solution: bool = True, log_file: Optional[str] = None) -> MonoResult:
    if pre is None:
        pre = preprocess_instance(inst, cfg)
    m = build_monolithic_model(inst, pre, cfg)
    gap = cfg.mip_gap_reference if mip_gap is None else mip_gap
    m.setParam("MIPGap", gap)
    m.setParam("Seed", cfg.gurobi_seed)
    m.setParam("Threads", cfg.gurobi_threads)
    m.setParam("OutputFlag", 1 if cfg.gurobi_output else 0)
    if time_limit is not None:
        m.setParam("TimeLimit", time_limit)
    if log_file:
        m.setParam("LogFile", log_file)
    if getattr(cfg, "mono_mip_focus", 0):
        m.setParam("MIPFocus", int(cfg.mono_mip_focus))
    if getattr(cfg, "mono_no_rel_heur_time", 0.0) > 0.0:
        m.setParam("NoRelHeurTime", float(cfg.mono_no_rel_heur_time))

    # Primal warm start (seed only; never affects bounds/exactness). A rejected start is harmless.
    # Uses the CONTAINMENT incumbent (monolithic-feasible by construction); the old full-cascade
    # greedy is mono-INFEASIBLE and would be rejected, leaving no incumbent at scale.
    if getattr(cfg, "mono_warm_start", False):
        from lbbd_v2.lbbd_heuristics import build_containment_incumbent
        try:
            inc = build_containment_incumbent(inst, pre, cfg)
            if inc.start_vars:
                apply_warm_start(m, inc.start_vars)
        except Exception:  # noqa: BLE001  — a bad seed must never break the exact solve
            pass

    t0 = time.time()
    m.optimize()
    runtime = time.time() - t0

    status = _status_name(m)
    sol_count = m.SolCount
    obj_val = m.ObjVal if sol_count > 0 else None
    obj_bound = m.ObjBound if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) else None
    try:
        actual_gap = m.MIPGap if sol_count > 0 else None
    except Exception:
        actual_gap = None
    # OPTIMAL status with an incumbent IS the certificate; the residual ~1e-9 gap
    # is Gurobi's numerical feasibility, not a real optimality gap.
    proven_optimal = (m.Status == GRB.OPTIMAL and sol_count > 0)

    reason = status
    if status == "OPTIMAL":
        reason = f"OPTIMAL (gap={actual_gap})"
    elif status == "TIME_LIMIT":
        reason = "TIME_LIMIT" + (" with incumbent" if sol_count > 0 else " no incumbent")

    solution: Dict[str, Any] = {}
    if extract_solution and sol_count > 0:
        for name, var in m._vars.items():
            if var is None:
                continue
            solution[name] = {k: var[k].X for k in var.keys()}

    return MonoResult(
        status=status, obj_val=obj_val, obj_bound=obj_bound, actual_mip_gap=actual_gap,
        runtime=runtime, solution_count=sol_count, termination_reason=reason,
        proven_optimal=proven_optimal, configuration=cfg.to_dict(), solution=solution,
        gurobi_version=".".join(map(str, gp.gurobi.version())),
    )
