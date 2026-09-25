"""lbbd_master.py — LBBD master MILP (valid relaxation of the (y,z,q,u) projection).

Master carries ONLY the binary structure y, z, q, u^pre, u^post and the reward proxy ρ.
Time variables (t^s, t^{s,min}, b^min) are NOT in the master; they are reconstructed
deterministically by the forward pass (04 Theorem 2). Dropping them keeps the master a
valid relaxation (04 §4.3): every master constraint is an original constraint on these
vars, so any original-feasible point maps in with ρ_i := p_i, giving a valid UPPER bound.

Constraints (05 §3):
  (47) roots burn; (40)-(43) spread structure; regime u≤y, u≤1; reward proxy ρ_i ≤ π_i u_i;
  C0/C0' variable fixing; C3 tightened reward bound; optional static single-commodity
  flow connectivity (04 §4.4, PROVEN EXACT) forbidding floating q-cycles structurally.
Cuts 1-6 are added at runtime via add_cut.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from lbbd_v2.config import Config
from lbbd_v2.data_loader import Instance
from lbbd_v2.preprocessing import Preprocessed


@dataclass
class MasterSolution:
    status: str
    obj_val: Optional[float]
    obj_bound: Optional[float]        # valid UB on OPT (dual bound)
    runtime: float
    solution_count: int = 0           # Gurobi SolCount: 0 => fizibil inkümbent yok
    y: Dict[int, float] = field(default_factory=dict)
    z: Dict[Tuple[int, int], float] = field(default_factory=dict)
    q: Dict[Tuple[int, int], float] = field(default_factory=dict)
    u_pre: Dict[int, float] = field(default_factory=dict)
    u_post: Dict[int, float] = field(default_factory=dict)
    rho: Dict[int, float] = field(default_factory=dict)

    def u(self, i) -> float:
        return self.u_pre.get(i, 0.0) + self.u_post.get(i, 0.0)


class MasterModel:
    def __init__(self, inst: Instance, pre: Preprocessed, cfg: Config):
        self.inst = inst
        self.pre = pre
        self.cfg = cfg
        self.m = gp.Model("lbbd_master")
        self.m.setParam("OutputFlag", 1 if cfg.gurobi_output else 0)
        self.m.setParam("Seed", cfg.gurobi_seed)
        self.m.setParam("Threads", cfg.gurobi_threads)
        self.cut_records: List[dict] = []
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        inst, pre, cfg = self.inst, self.pre, self.cfg
        m = self.m
        Nf, Na, Nplus = inst.Nf, inst.Na, inst.Nplus
        nb = inst.neighbors
        arcs = inst.arcs
        pi, beta, e = inst.pi, inst.beta, inst.e

        y = m.addVars(Nf, vtype=GRB.BINARY, name="y")
        z = m.addVars(arcs, vtype=GRB.BINARY, name="z")
        q = m.addVars(arcs, vtype=GRB.BINARY, name="q")
        u_pre = m.addVars(Nf, vtype=GRB.BINARY, name="u_pre")
        u_post = m.addVars(Nf, vtype=GRB.BINARY, name="u_post")
        rho = m.addVars(Nf, lb=0.0, name="rho")

        def u(i):
            return u_pre[i] + u_post[i]

        # objective: max Σ π_i(1-y_i) + Σ ρ_i
        m.setObjective(gp.quicksum(pi[i] * (1 - y[i]) for i in Nf)
                       + gp.quicksum(rho[i] for i in Nf), GRB.MAXIMIZE)

        # (47) all roots burn
        m.addConstr(gp.quicksum(y[i] for i in Na) == len(Na), "c47")

        # regime: u ≤ y, u ≤ 1  (both hold in the original; valid)
        m.addConstrs((u(i) <= y[i] for i in Nf), "u_le_y")
        m.addConstrs((u(i) <= 1 for i in Nf), "c39")

        # reward proxy ρ_i ≤ π_i u_i
        m.addConstrs((rho[i] <= pi[i] * u(i) for i in Nf), "rho_proxy")

        # C0 / C0' variable fixing (PROVEN EXACT)
        if cfg.enable_C0:
            for i in Nf:
                if not pre.controllable[i]:
                    m.addConstr(u(i) == 0, f"C0[{i}]")
        if cfg.enable_C0prime:
            for i in Nf:
                if not pre.pre_possible[i]:
                    m.addConstr(u_pre[i] == 0, f"C0p[{i}]")

        # C3 tightened reward bound ρ_i ≤ (π_i − β_i Δ^min_i) u_i (roots tight)
        if cfg.enable_C3:
            for i in Nf:
                m.addConstr(rho[i] <= (pi[i] - beta[i] * pre.delta_min[i]) * u(i), f"C3[{i}]")

        # (40)-(43) spread structure
        m.addConstrs((gp.quicksum(z[i, j] for j in nb[i]) == len(nb[i]) * (y[i] - u_pre[i])
                      for i in Nf), "c40")
        m.addConstrs((len(nb[j]) * y[j] >= gp.quicksum(z[i, j] for i in nb[j]) for j in Nf), "c41")
        m.addConstrs((q[i, j] <= z[i, j] for (i, j) in arcs), "c42")
        m.addConstrs((gp.quicksum(q[i, j] for i in nb[j]) == y[j] - e[j] for j in Nf), "c43")

        # C2 (optional, vacuous): Σ u_i ≤ |K|
        if cfg.enable_C2:
            m.addConstr(gp.quicksum(u(i) for i in Nf) <= len(inst.K), "C2")

        # static single-commodity flow connectivity (04 §4.4): each burning non-root pulls
        # one unit of flow from a root along q-arcs; forbids floating cycles structurally.
        f = None
        if cfg.enable_flow_connectivity:
            f = m.addVars(arcs, lb=0.0, name="flow")
            cap = float(len(Nf))
            m.addConstrs((f[i, j] <= cap * q[i, j] for (i, j) in arcs), "flow_cap")
            for i in Nf:
                if i in Na:
                    continue  # roots are free sources
                inflow = gp.quicksum(f[j, i] for j in nb[i])
                outflow = gp.quicksum(f[i, j] for j in nb[i])
                m.addConstr(inflow - outflow == y[i], f"flow_cons[{i}]")

        # time-consistency layer: reconstruct t^s in the master and enforce z-arc consistency (46)
        # so the master proposes only time-CONSISTENT (y,z,q) — collapsing the propagation churn.
        ts = None
        if getattr(cfg, "enable_master_time_consistency", False):
            Md = pre.M_d
            alpha, lam = inst.alpha, inst.lam
            ts = m.addVars(Nf, lb=0.0, name="ts")
            m.addConstr(gp.quicksum(ts[r] for r in Na) == 0, "mc48")   # roots ignite at 0
            m.addConstrs((ts[i] <= Md * y[i] for i in Nf), "mc23ts")   # unburnt ⇒ t^s=0
            for (i, j) in arcs:
                tmi = ts[i] + alpha / lam[i]                           # t^m_i = t^s_i + α/λ_i (49)
                m.addConstr(ts[j] >= tmi - Md * (1 - q[i, j]), f"mc44[{i},{j}]")
                m.addConstr(ts[j] <= tmi + Md * (1 - q[i, j]), f"mc45[{i},{j}]")
                m.addConstr(ts[j] <= tmi + Md * (1 - z[i, j]), f"mc46[{i},{j}]")
            # monolithic-feasibility cap in the master: a burning cell must ignite by burn_i·margin
            # (constraint (5) with forced v_ik=0). VALID (every feasible solution satisfies it) and
            # it stops the master proposing late-ignition structures — collapsing the ignition_time
            # cut churn the same way (46) collapses the propagation churn.
            for i in Nf:
                if inst.a[i] > 1e-12:
                    cap_i = pre.M_i[i] / inst.a[i]
                    m.addConstr(ts[i] <= cap_i + Md * (1 - y[i]), f"mccap[{i}]")

        self.vars = dict(y=y, z=z, q=q, u_pre=u_pre, u_post=u_post, rho=rho, flow=f, ts=ts)

    # ------------------------------------------------------------------
    def add_cut(self, lhs: gp.LinExpr, sense: str, rhs: float,
                name: str = "cut", record: Optional[dict] = None) -> None:
        """Add a runtime cut. `sense` in {'<=','>=','=='}. Records for cuts.jsonl / re-eval."""
        if sense == "<=":
            self.m.addConstr(lhs <= rhs, name=name)
        elif sense == ">=":
            self.m.addConstr(lhs >= rhs, name=name)
        elif sense == "==":
            self.m.addConstr(lhs == rhs, name=name)
        else:
            raise ValueError(f"bad sense {sense!r}")
        if record is not None:
            self.cut_records.append(record)

    # ------------------------------------------------------------------
    def solve(self, budget: Optional[float] = None, mip_gap: float = 0.0) -> MasterSolution:
        m = self.m
        m.setParam("MIPGap", mip_gap)
        if budget is not None:
            m.setParam("TimeLimit", budget)
        t0 = time.time()
        m.optimize()
        runtime = time.time() - t0

        st = m.Status
        status = {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
                  GRB.INF_OR_UNBD: "INFEASIBLE", GRB.TIME_LIMIT: "TIME_LIMIT",
                  GRB.UNBOUNDED: "UNBOUNDED"}.get(st, f"STATUS_{st}")

        sol = MasterSolution(status=status, obj_val=None, obj_bound=None, runtime=runtime)
        if m.SolCount > 0:
            v = self.vars
            sol.obj_val = m.ObjVal
            sol.y = {i: v["y"][i].X for i in v["y"].keys()}
            sol.z = {a: v["z"][a].X for a in v["z"].keys()}
            sol.q = {a: v["q"][a].X for a in v["q"].keys()}
            sol.u_pre = {i: v["u_pre"][i].X for i in v["u_pre"].keys()}
            sol.u_post = {i: v["u_post"][i].X for i in v["u_post"].keys()}
            sol.rho = {i: v["rho"][i].X for i in v["rho"].keys()}
        if st in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            sol.obj_bound = m.ObjBound
        return sol


def build_lbbd_master(inst: Instance, pre: Preprocessed, cfg: Config) -> MasterModel:
    return MasterModel(inst, pre, cfg)
