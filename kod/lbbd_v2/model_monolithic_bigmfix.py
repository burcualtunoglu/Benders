"""model_monolithic_bigmfix.py — deaktivasyon Big-M düzeltmesi (ORİJİNAL DOSYALAR DEĞİŞMEZ).

Sorun (bkz. tartışma): (4)/(5) gecikme-suyu kısıtlarında kullanılan M_i = a_i·burn_i·marj,
δ_ik'yı SINIRLAMAK için doğru ama x_ik=0 (araç atanmamış, dolayısıyla (23) ile v_ik=0) iken
kısıtı DEVRE DIŞI bırakmak için yetersizdir: (5) o durumda 0 ≤ M_i − a_i·t^s_i, yani her yanan
hücrede t^s_i ≤ M_i/a_i = burn_i·marj olmasını zorlar. Kökten uzak, geç tutuşan hücreler bunu
aşar ve model SAHTE biçimde infizibil olur ("her şey yanar" durumu bile reddedilir).

Düzeltme: (4)/(5)'teki deaktivasyon Big-M'ini a_i·M_d yap. M_d ufuk olduğundan M_d ≥ t^s ve
v ≤ M_d; böylece x=0 iken (5): 0 ≤ a_i(M_d − t^s_i) daima sağlanır. x=1 iken (1−x)=0 terimi
kaybolur, δ = a_i(v−t^s) AYNEN kalır. Böylece (4)/(5) AMAÇLANAN modeli doğru gösterir.
(2026-09-25 anlatım düzeltmesi) Bu, eski modelin optimumunu korumak DEĞİLDİR: eski model yanan
hücrelerde t^s_i ≤ burn_i koşulunu da dayattığından fizibil kümesi daha dardır; eski model fizibil
olsa bile düzeltilmiş modelin optimumu ondan büyük olabilir. Referans yalnız düzeltilmiş modeldir.
Bedeli daha gevşek bir LP gevşetmesidir.

Kullanım:
    from lbbd_v2.model_monolithic_bigmfix import solve_monolithic_bigmfix
    res = solve_monolithic_bigmfix(inst, cfg, time_limit=1800)
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import gurobipy as gp
from gurobipy import GRB

from lbbd_v2.config import Config
from lbbd_v2.data_loader import Instance
from lbbd_v2.preprocessing import Preprocessed, preprocess_instance
from lbbd_v2.model_monolithic import (
    build_monolithic_model, apply_warm_start, _status_name, MonoResult)


def build_monolithic_model_bigmfix(inst: Instance, pre: Preprocessed, cfg: Config) -> gp.Model:
    """Kanonik modeli kurar, ardından (4)/(5)'i deaktivasyon Big-M'i a_i·M_d ile YENİDEN kurar."""
    m = build_monolithic_model(inst, pre, cfg)
    m.update()
    # Orijinal c4/c5'i (delta McCormick, küçük deaktivasyon M_i) kaldır.
    for c in list(m.getConstrs()):
        nm = c.ConstrName
        if nm.startswith("c4[") or nm.startswith("c5["):
            m.remove(c)
    m.update()

    vv = m._vars
    x, delta, ts, v = vv["x"], vv["delta"], vv["ts"], vv["v"]
    a, Md = inst.a, pre.M_d
    Nf, K = inst.Nf, inst.K
    # Deaktivasyon Big-M = a_i·M_d (M_d ≥ t^s_ub ≥ t^s ve v ≤ M_d): x=0 iken (4)/(5) gerçekten boş.
    m.addConstrs((delta[i, k] >= a[i] * (v[i, k] - ts[i]) - a[i] * Md * (1 - x[i, k])
                  for i in Nf for k in K), "c4")
    m.addConstrs((delta[i, k] <= a[i] * (v[i, k] - ts[i]) + a[i] * Md * (1 - x[i, k])
                  for i in Nf for k in K), "c5")
    m.update()
    return m


def solve_monolithic_bigmfix(inst: Instance, cfg: Config, pre: Optional[Preprocessed] = None,
                             mip_gap: Optional[float] = None, time_limit: Optional[float] = None,
                             extract_solution: bool = True, log_file: Optional[str] = None) -> MonoResult:
    """solve_monolithic ile aynı; yalnız düzeltilmiş modeli kurar."""
    if pre is None:
        pre = preprocess_instance(inst, cfg)
    m = build_monolithic_model_bigmfix(inst, pre, cfg)
    gap = cfg.mip_gap_reference if mip_gap is None else mip_gap
    m.setParam("MIPGap", gap)
    # Büyük deaktivasyon Big-M'i (a_i·M_d) sayısal kondisyonu bozar; sıkı tolerans, optimumun
    # gevşek-tolerans gürültüsünü (~1e-6) önler. (4x4/6x6_4'te optimum böylece BİREBİR korunur.)
    m.setParam("FeasibilityTol", 1e-9)
    m.setParam("IntFeasTol", 1e-9)
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
    if getattr(cfg, "mono_warm_start", False):
        from lbbd_v2.lbbd_heuristics import build_containment_incumbent
        try:
            inc = build_containment_incumbent(inst, pre, cfg)
            if inc.start_vars:
                apply_warm_start(m, inc.start_vars)
        except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
        actual_gap = None
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
