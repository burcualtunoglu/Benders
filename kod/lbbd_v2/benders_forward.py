"""benders_forward.py — deterministic time reconstruction + Controls A/B (04 §11, Theorem 2).

Given a master decision (ȳ, z̄, q̄), rebuild t^s, t^m, t^e, t^{s,min} from (48),(49),(50) and
the q̄-forest (44),(45). Two structural checks:
  * Control A (reachability): a burning cell not reachable from a root ⇒ FWD_UNREACHABLE
    (floating q-cycle / disconnected component) ⇒ triggers connectivity cut (CC).
  * Control B (time consistency): a z̄-arc with t^s_j > t^m_i ⇒ FWD_TIME_INCONSISTENT ⇒
    triggers propagation cut (46).

Theorem 2: when Control A passes, q̄ is a root-forest, so (44),(45),(48)-(50) have this as
their UNIQUE solution — the forward pass is exactly equivalent to those constraints.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from lbbd_v2.data_loader import Instance

FWD_OK = "FWD_OK"
FWD_UNREACHABLE = "FWD_UNREACHABLE"
FWD_TIME_INCONSISTENT = "FWD_TIME_INCONSISTENT"

Arc = Tuple[int, int]


@dataclass
class ForwardResult:
    status: str
    ts: Dict[int, float] = field(default_factory=dict)
    tm: Dict[int, float] = field(default_factory=dict)
    te: Dict[int, float] = field(default_factory=dict)
    ts_min: Dict[int, float] = field(default_factory=dict)
    q_path: Dict[int, List[Arc]] = field(default_factory=dict)
    root_of: Dict[int, int] = field(default_factory=dict)
    reached: Set[int] = field(default_factory=set)
    unreachable: Set[int] = field(default_factory=set)
    violating_arcs: List[Arc] = field(default_factory=list)


def _burning(sol_y: Dict[int, float], i: int, tol: float = 0.5) -> bool:
    return sol_y.get(i, 0.0) > tol


def forward_pass(sol, inst: Instance, pre=None, tol: float = 1e-6) -> ForwardResult:
    """`sol` provides .y (dict), .z (dict arc->val), .q (dict arc->val)."""
    y, z, q = sol.y, sol.z, sol.q
    alpha, lam, sig = inst.alpha, inst.lam, inst.sig
    Na, Nf = inst.Na, inst.Nf
    Nplus = inst.Nplus

    res = ForwardResult(status=FWD_OK)

    # 1. roots
    for r in Na:
        res.ts[r] = 0.0
        res.root_of[r] = r
        res.q_path[r] = []
        res.reached.add(r)

    # 2. BFS along q̄ (guard against cycles by not revisiting)
    dq = deque(Na)
    while dq:
        i = dq.popleft()
        for j in inst.neighbors[i]:
            if q.get((i, j), 0.0) > 0.5 and j not in res.reached:
                res.ts[j] = res.ts[i] + alpha / lam[i]
                res.root_of[j] = res.root_of[i]
                res.q_path[j] = res.q_path[i] + [(i, j)]
                res.reached.add(j)
                dq.append(j)

    # 3. tm, te for reached cells
    for i in res.reached:
        res.tm[i] = res.ts[i] + alpha / lam[i]
        res.te[i] = res.tm[i] + alpha / sig[i]

    # 4. Control A — reachability of every burning cell
    unreachable = {i for i in Nf if _burning(y, i) and i not in res.reached}
    if unreachable:
        res.status = FWD_UNREACHABLE
        res.unreachable = unreachable
        return res

    # 5. t^{s,min}_i = min{ t^s_j : j∈N^+(i), ȳ_j=1 }  (26')+(27)+(28)+(28')
    for i in Nf:
        cand = [res.ts[j] for j in Nplus[i] if _burning(y, j)]
        if cand:
            res.ts_min[i] = min(cand)

    # 6. Control B — time consistency on z̄-arcs
    for (i, j) in inst.arcs:
        if z.get((i, j), 0.0) > 0.5 and i in res.ts and j in res.ts:
            if res.ts[j] > res.tm[i] + tol:
                res.violating_arcs.append((i, j))
    if res.violating_arcs:
        res.status = FWD_TIME_INCONSISTENT
        return res

    return res
