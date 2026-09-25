"""run_mono_bigmfix.py — Big-M düzeltmesini bir örnekte dener (ORİJİNAL DOSYALAR DEĞİŞMEZ).

İki şeyi gösterir:
  (1) ORİJİNAL monolitik model örneği INF_OR_UNBD/infizibil bulur mu,
  (2) DÜZELTİLMİŞ model (a_i·M_d deaktivasyon Big-M'i) aynı örneğe bir sonuç verir mi.
Düzeltilmiş modele, kesin fizibil "her şey yanar, kontrol yok" çözümünü MIPStart olarak verir;
böylece fizibilite ANINDA görülür (orijinalde bu çözüm c5 yüzünden infizibildi).

  cd ~/wildfire_hacettepe && source .venv/bin/activate
  python -m lbbd_v2.run_mono_bigmfix inputs/inputs_10x10_v_low.xlsx [mono_TL]
"""
from __future__ import annotations

import sys
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from lbbd_v2.config import canonical_config
from lbbd_v2.data_loader import load_instance
from lbbd_v2.preprocessing import preprocess_instance
from lbbd_v2.model_monolithic import build_monolithic_model, apply_warm_start
from lbbd_v2.model_monolithic_bigmfix import build_monolithic_model_bigmfix
from lbbd_v2.lbbd_heuristics import _full_cascade, Incumbent
from lbbd_v2.benders_forward import forward_pass, FWD_OK


def build_full_burn_start(inst, pre, cfg):
    """'Her şey yanar, kontrol yok' (u≡0) çözümü — düzeltilmiş modelde fizibil, tam MIPStart."""
    midpoint = (cfg.omega_mode == "midpoint")
    y, z, q, _ = _full_cascade(inst)
    u_pre = {i: 0.0 for i in inst.Nf}
    u_post = {i: 0.0 for i in inst.Nf}
    inc = Incumbent(y=y, z=z, q=q, u_pre=u_pre, u_post=u_post)
    fwd = forward_pass(inc, inst)
    if fwd.status != FWD_OK:
        return None
    Nf, K = inst.Nf, inst.K

    def tsval(j):
        return fwd.ts.get(j, 0.0)

    bmin, ts_min = {}, {}
    for i in Nf:
        bn = [j for j in inst.Nplus[i] if y[j] > 0.5]
        jstar = min(bn, key=lambda j: (tsval(j), j)) if bn else i
        ts_min[i] = tsval(jstar)
        for j in inst.Nplus[i]:
            bmin[(i, j)] = 1.0 if j == jstar else 0.0

    ts_v, tm_v, te_v, tc_v, p_v = {}, {}, {}, {}, {}
    for i in Nf:
        if y[i] > 0.5:
            ts_v[i], tm_v[i], te_v[i] = fwd.ts[i], fwd.tm[i], fwd.te[i]
            p_v[i] = 0.0                       # yanan, kontrolsüz -> p=0
        else:
            ts_v[i] = 0.0
            tm_v[i] = inst.alpha / inst.lam[i]
            te_v[i] = tm_v[i] + inst.alpha / inst.sig[i]
            p_v[i] = inst.pi[i]                # yanmayan -> tam ödül
        tc_v[i] = 0.0

    zero_ik = {(i, k): 0.0 for i in Nf for k in K}
    zero_i = {i: 0.0 for i in Nf}
    start = {"y": dict(y), "z": dict(z), "q": dict(q), "u_pre": u_pre, "u_post": u_post,
             "x": dict(zero_ik), "t": dict(zero_ik), "s": dict(zero_ik), "v": dict(zero_ik),
             "delta": dict(zero_ik), "omega": dict(zero_i), "omega_max": dict(zero_i),
             "ts": ts_v, "tm": tm_v, "te": te_v, "tc": tc_v, "p": p_v,
             "ts_min": ts_min, "bmin": bmin}
    if midpoint:
        start["omega_min"] = dict(zero_i)
        start["hmax"] = dict(zero_ik)
        start["hmin"] = dict(zero_ik)
    return start


def _solve(m, tl):
    m.setParam("OutputFlag", 0)
    m.setParam("DualReductions", 0)
    m.setParam("TimeLimit", tl)
    m.optimize()
    st = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 9: "TIME_LIMIT"}.get(m.Status, str(m.Status))
    obj = "%.2f" % m.ObjVal if m.SolCount > 0 else "None"
    bnd = "%.2f" % m.ObjBound if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) else "None"
    return st, obj, bnd


def main(argv):
    path = argv[0] if argv else "inputs/inputs_10x10_v_low.xlsx"
    tl = float(argv[1]) if len(argv) > 1 else 120.0
    cfg = canonical_config()
    inst = load_instance(path)
    pre = preprocess_instance(inst, cfg)
    print(f"{Path(path).name}: |Nf|={len(inst.Nf)} |K|={len(inst.K)} roots={len(inst.Na)}")

    # (1) ORİJİNAL model
    m0 = build_monolithic_model(inst, pre, cfg)
    st0, obj0, bnd0 = _solve(m0, min(tl, 120))
    print(f"  ORİJİNAL   : status={st0} obj={obj0} bound={bnd0}")

    # (2) DÜZELTİLMİŞ model + 'her şey yanar' MIPStart
    m1 = build_monolithic_model_bigmfix(inst, pre, cfg)
    fb = build_full_burn_start(inst, pre, cfg)
    seeded = apply_warm_start(m1, fb) if fb else 0
    st1, obj1, bnd1 = _solve(m1, tl)
    print(f"  DÜZELTİLMİŞ: status={st1} obj={obj1} bound={bnd1}  (MIPStart {seeded} değişken)")
    print("  -> " + ("Düzeltme SONUÇ VERİYOR (orijinal infizibil, düzeltilmiş fizibil)."
                     if fb and obj1 != "None" else
                     "beklenen: orijinal INF_OR_UNBD, düzeltilmiş obj != None"))


if __name__ == "__main__":
    main(sys.argv[1:])
