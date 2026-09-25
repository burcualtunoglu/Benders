"""lbbd_conflicts.py — irreducible conflict extraction via deletion filter (Saken et al. 2023).

Given an INFEASIBLE control set C, shrink it to a (locally) minimal infeasible subset C* by
removing cells one at a time and keeping only those whose removal restores/maintains
infeasibility. R(C*) ≥ 1 is a stronger feasibility cut than R(C) ≥ 1 (antitone: any superset
of an infeasible set is infeasible, 04 §8.4).

Safety rule: a sub-solve that returns UNKNOWN (timeout / numeric) is NOT proof of feasibility,
so that cell is KEPT in the conflict set (05 §12). The returned set is always a valid, still
provably-infeasible superset if the filter cannot complete.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Optional, Set

from lbbd_v2.benders_forward import ForwardResult
from lbbd_v2.lbbd_subproblem_resource import (
    solve_resource_subproblem, SPStatus)


def extract_conflict(C: Iterable[int], regime: Dict[int, str], fwd: ForwardResult,
                     inst, pre, cfg, time_budget: float | None = None,
                     sp_solve: Optional[Callable] = None) -> Set[int]:
    sp_solve = sp_solve or solve_resource_subproblem
    budget = time_budget if time_budget is not None else cfg.filter_budget
    deadline = time.time() + budget          # filtrenin TOPLAM bütçesi (deadline)
    keep: List[int] = sorted(C)
    for cand in list(keep):
        if len(keep) <= 1:
            break
        rem = deadline - time.time()          # her alt-çözümden ÖNCE kalan süreyi yeniden hesapla
        if rem <= 0.0:
            break                             # filtre bütçesi bitti -> mevcut (geçerli) küme ile dur
        trial = [i for i in keep if i != cand]
        sp = sp_solve(fwd, trial, {i: regime[i] for i in trial},
                      inst, pre, cfg, time_budget=max(1.0, rem))
        if sp.status == SPStatus.INFEASIBLE_PROVEN:
            keep = trial            # cand not needed for infeasibility -> drop it
        # else (feasible OR unknown): cand is required -> keep it
    return set(keep)
