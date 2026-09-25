"""subproblem.py — alt problem motoru seçici (güçlü | zayıf).

Her iki motor da DÜZELTİLMİŞ modelin kesin kısıtlamasıdır ve aynı Φ'yi verir; fark yalnız
formülasyon sıkılığı/hızıdır (docs/asama0_rapor.md, F-SPWEAK). solver bu seçiciyle tek biçimde
çağırır: (multi_cell_solver, single_cell_solver).
"""
from __future__ import annotations

from typing import Callable, Tuple

from lbbd_v2.config import Config
from lbbd_v2.lbbd_subproblem_resource import solve_resource_subproblem, solve_single_cell
from lbbd_v2.baseline.config import SP_STRONG, SP_WEAK, SP_GROUPAGG
from lbbd_v2.baseline.subproblem_weak import solve_weak_subproblem, solve_single_cell_weak
from lbbd_v2.baseline.subproblem_groupagg import solve_groupagg_subproblem


def select_sp(cfg: Config) -> Tuple[Callable, Callable]:
    """(çok-hücre SP, tek-hücre SP) çiftini cfg.sp_engine'e göre döndürür."""
    engine = getattr(cfg, "sp_engine", SP_STRONG)
    if engine == SP_WEAK:
        return solve_weak_subproblem, solve_single_cell_weak
    if engine == SP_STRONG:
        return solve_resource_subproblem, solve_single_cell
    if engine == SP_GROUPAGG:
        # çok-hücre SP grup-toplulaştırılmış; tek-hücre ön-eleme bireysel (değişmedi)
        return solve_groupagg_subproblem, solve_single_cell
    raise ValueError(f"unknown sp_engine {engine!r}")
