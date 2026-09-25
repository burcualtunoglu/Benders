"""full_solution.py — LBBD çözüm tanığından özgün (düzeltilmiş) tam modele dönüşüm ve ÇÖZÜCÜSÜZ denetim.

2026-09-25 (düzeltme sürümü v2). Üç ayrı adım, üç ayrı sorumluluk:

  (1) GAGG → bireysel güçlü SP: ``subproblem_groupagg.disaggregate_solution`` + ``check_individual``
      (yalnız bireysel güçlü-SP kısıtları; bu modül bu adımı YAPMAZ).
  (2) Bireysel güçlü SP tanığı → özgün tam model değişkenleri: ``build_full_solution``.
  (3) Özgün modelin BÜTÜN kısıtlarının bağımsız denetimi: ``check_full_solution``. Denetim, özgün
      (düzeltilmiş) modeli kuran kodun kendisini (``model_monolithic_bigmfix``) kullanır, modeli
      ÇÖZMEZ; her kısıt satırını verilen değerlerle hesaplar, değişken sınırlarını ve ikililiği denetler,
      aile başına en büyük ihlali ve yerini raporlar.

Çözüm tanığı (witness) alanları — bağımsız doğrulama için GEREKLİ olanlar:
  * ana problem kararı: y, u_pre, u_post (bütün N_f), z ve q (değeri 1 olan yaylar; eksik = 0)
  * kontrol edilen her hücre i için SP tanığı: x (atanan (i,k) çiftleri), tc_i, p_i
  * kayıtlı amaç değeri ``value`` (= Σ π_i(1−y_i) + Σ_{i∈C} p_i olmalı)
Bilgi amaçlı (denetimde kullanılmaz, karşılaştırılır): SP'nin ω_i ve s_ik değerleri, ileri geçiş zamanları.

Dönüşüm kuralları (``build_full_solution``):
  * Zamanlar karar vektöründen ileri geçişle yeniden hesaplanır; yanmayan hücrede t^s=0, t^m=α/λ, t^e=t^m+α/σ.
  * t^{s,min}_i ve b_ij: j* = argmin{t^s_j : j∈N⁺(i), y_j=1} (yoksa j*=i, t^{s,min}_i=0).
  * Atanan araç: v_ik = v^min_ik = max(t^{s,min}_i+Δ_buf+d_ik, t^s_i) (Lemma A), t_ik = v_ik−d_ik,
    δ_ik = D_ik = a_i(v_ik−t^s_i), s_ik = t^c_i − v_ik (varıştan kontrol anına kadar en uzun servis).
  * ω^max_i = max_k D_ik, ω^min_i = min_k D_ik (ATANAN araçların GERÇEK değerleri), h^max/h^min bu araçlarda;
    ω_i iş yükü moduna göre yeniden kurulur. SP'nin ω değişkenleri yalnız alt sınırla bağlı olduğundan
    (ω^max ≥ D x; ω^min atananların dışbükey bileşimi) SP tanığındaki ω, gerçek değerden BÜYÜK olabilir;
    ω burada kopyalanmaz, yeniden kurulur.
  * Atanmayan araç: x=t=s=v=δ=h=0. Kontrol edilmeyen hücre: t^c=0, ω=0, p_i = π_i(1−y_i).
Fizibiliteyi koruma gerekçesi docs/cozum_tanigi_dogrulama.md dosyasındadır; bu modül yine de her çözümü
kısıt kısıt denetler ve gerekçeye güvenmez.
"""
from __future__ import annotations

import json
import math
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lbbd_v2.benders_forward import forward_pass, FWD_OK

VAR_NAMES = ("p", "y", "ts", "tm", "te", "tc", "u_pre", "u_post", "x", "t", "s", "v", "delta", "omega",
             "omega_max", "ts_min", "z", "q", "bmin", "omega_min", "hmax", "hmin")


# ----------------------------------------------------------------------------- tanık okuma
def _pair(k) -> Tuple[int, int]:
    if isinstance(k, (tuple, list)):
        return int(k[0]), int(k[1])
    a, b = str(k).split(",")
    return int(a), int(b)


def witness_from_incumbent(inc: Dict[str, Any]) -> Dict[str, Any]:
    """Çözücünün bellek içi inkümbent sözlüğünden tanık (anahtarlar zaten int / (int,int))."""
    sp = inc.get("sp") or {}
    return {"source": inc.get("source"), "value": inc.get("value"),
            "y": dict(inc.get("y") or {}), "u_pre": dict(inc.get("u_pre") or {}),
            "u_post": dict(inc.get("u_post") or {}),
            "z": {k: v for k, v in (inc.get("z") or {}).items() if v > 0.5},
            "q": {k: v for k, v in (inc.get("q") or {}).items() if v > 0.5},
            "x": {k: 1.0 for k, v in (sp.get("x") or {}).items() if v > 0.5},
            "tc": dict(sp.get("tc") or {}), "p": dict(sp.get("p") or {}),
            "omega_sp": dict(sp.get("omega") or {}), "s_sp": dict(sp.get("s") or {})}


def witness_from_json(js: Dict[str, Any]) -> Dict[str, Any]:
    """runner'ın ``incumbent_solution`` kaydından tanık ("i,j" anahtarları çiftlere çevrilir).
    Eski kayıtlar da okunur (sp_s, zamanlar yoksa boş kalır)."""
    def cell(d):
        return {int(k): float(v) for k, v in (d or {}).items()}

    def pairs(d):
        return {_pair(k): float(v) for k, v in (d or {}).items()}
    return {"source": js.get("source"), "value": js.get("value"),
            "y": cell(js.get("y")), "u_pre": cell(js.get("u_pre")), "u_post": cell(js.get("u_post")),
            "z": pairs(js.get("z")), "q": pairs(js.get("q")),
            "x": {k: 1.0 for k, v in pairs(js.get("sp_x")).items() if v > 0.5},
            "tc": cell(js.get("sp_tc")), "p": cell(js.get("sp_p")),
            "omega_sp": cell(js.get("sp_omega")), "s_sp": pairs(js.get("sp_s"))}


def witness_completeness(w: Dict[str, Any], Nf: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    """Tanığın bağımsız doğrulamaya YETERLİ olup olmadığı ve eksik alanlar."""
    missing: List[str] = []
    if not w.get("y"):
        missing.append("y")
    for f in ("u_pre", "u_post"):
        if w.get(f) is None:
            missing.append(f)
    C = [i for i, v in (w.get("y") or {}).items()
         if (w.get("u_pre", {}).get(i, 0.0) + w.get("u_post", {}).get(i, 0.0)) > 0.5]
    if C:
        if not w.get("x"):
            missing.append("sp_x")
        if any(i not in (w.get("tc") or {}) for i in C):
            missing.append("sp_tc")
        if any(i not in (w.get("p") or {}) for i in C):
            missing.append("sp_p")
        served = {i for (i, _k) in (w.get("x") or {})}
        if w.get("x") and any(i not in served for i in C):
            missing.append("sp_x (bazı kontrol edilen hücrelerde atama yok)")
    if w.get("value") is None:
        missing.append("value")
    return {"complete": not missing, "missing": missing, "n_controlled": len(C)}


# ----------------------------------------------------------------------------- (2) dönüşüm
def build_full_solution(inst, pre, cfg, w: Dict[str, Any]) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
    """Tanıktan özgün modelin bütün değişkenlerini kur (yeniden optimizasyon YOK)."""
    Nf, K = list(inst.Nf), list(inst.K)
    midpoint = (cfg.omega_mode == "midpoint")
    y = {i: 1.0 if w["y"].get(i, 0.0) > 0.5 else 0.0 for i in Nf}
    u_pre = {i: 1.0 if w["u_pre"].get(i, 0.0) > 0.5 else 0.0 for i in Nf}
    u_post = {i: 1.0 if w["u_post"].get(i, 0.0) > 0.5 else 0.0 for i in Nf}
    z = {a: 1.0 if w["z"].get(a, 0.0) > 0.5 else 0.0 for a in inst.arcs}
    q = {a: 1.0 if w["q"].get(a, 0.0) > 0.5 else 0.0 for a in inst.arcs}
    fwd = forward_pass(SimpleNamespace(y=y, z=z, q=q, u_pre=u_pre, u_post=u_post), inst)
    info: Dict[str, Any] = {"fwd_status": fwd.status}
    if fwd.status != FWD_OK:
        return {}, info

    ts, tm, te = {}, {}, {}
    for i in Nf:
        if y[i] > 0.5:
            ts[i], tm[i], te[i] = fwd.ts[i], fwd.tm[i], fwd.te[i]
        else:
            ts[i] = 0.0
            tm[i] = inst.alpha / inst.lam[i]
            te[i] = tm[i] + inst.alpha / inst.sig[i]
    ts_min, bmin = {}, {}
    for i in Nf:
        burning = [j for j in inst.Nplus[i] if y[j] > 0.5]
        jstar = min(burning, key=lambda j: (ts[j], j)) if burning else i
        ts_min[i] = ts[jstar] if burning else 0.0
        for j in inst.Nplus[i]:
            bmin[(i, j)] = 1.0 if j == jstar else 0.0

    zero_ik = lambda: {(i, k): 0.0 for i in Nf for k in K}  # noqa: E731
    x, t, s, v, delta = zero_ik(), zero_ik(), zero_ik(), zero_ik(), zero_ik()
    hmax, hmin = zero_ik(), zero_ik()
    tc = {i: 0.0 for i in Nf}
    p = {i: (inst.pi[i] if y[i] < 0.5 else 0.0) for i in Nf}
    omega = {i: 0.0 for i in Nf}
    omax = {i: 0.0 for i in Nf}
    omin = {i: 0.0 for i in Nf}
    assigned: Dict[int, List[int]] = {}
    for (i, k) in w.get("x", {}):
        assigned.setdefault(i, []).append(k)
    omega_slack = 0.0
    s_gain_min = math.inf
    for i in Nf:
        if u_pre[i] + u_post[i] < 0.5:
            continue
        tc[i] = float(w["tc"][i])
        p[i] = float(w["p"][i])
        A = sorted(assigned.get(i, []))
        D = {}
        for k in A:
            vm = max(ts_min[i] + inst.delta_buf + inst.d[(i, k)], ts[i])
            x[(i, k)] = 1.0
            v[(i, k)] = vm
            t[(i, k)] = vm - inst.d[(i, k)]
            D[k] = inst.a[i] * (vm - ts[i])
            delta[(i, k)] = D[k]
            s[(i, k)] = tc[i] - vm
            if (i, k) in w.get("s_sp", {}):
                s_gain_min = min(s_gain_min, s[(i, k)] - w["s_sp"][(i, k)])
        if A:
            kmax = max(A, key=lambda k: (D[k], -k))
            kmin = min(A, key=lambda k: (D[k], k))
            omax[i], omin[i] = D[kmax], D[kmin]
            hmax[(i, kmax)] = 1.0
            hmin[(i, kmin)] = 1.0
            if midpoint:
                omega[i] = 0.5 * (omax[i] + omin[i]) + inst.delta_wat
            else:
                omega[i] = omax[i] + inst.delta_wat
            if i in w.get("omega_sp", {}):
                omega_slack = max(omega_slack, w["omega_sp"][i] - omega[i])
    full = {"p": p, "y": y, "ts": ts, "tm": tm, "te": te, "tc": tc, "u_pre": u_pre, "u_post": u_post,
            "x": x, "t": t, "s": s, "v": v, "delta": delta, "omega": omega, "omega_max": omax,
            "ts_min": ts_min, "z": z, "q": q, "bmin": bmin}
    if midpoint:
        full.update(omega_min=omin, hmax=hmax, hmin=hmin)
    info.update(
        # SP tanığındaki ω'nın yeniden kurulan gerçek değerden ne kadar büyük olduğu (≥0 beklenir; bilgi)
        omega_sp_minus_exact_max=omega_slack,
        # yeniden kurulan servis süresi − SP servis süresi (≥0 beklenir; SP s kayıtlıysa)
        s_rebuilt_minus_sp_min=(None if s_gain_min == math.inf else s_gain_min),
        ts_min_vs_forward_max_abs=max((abs(ts_min[i] - fwd.ts_min.get(i, ts_min[i])) for i in Nf), default=0.0))
    return full, info


# ----------------------------------------------------------------------------- (3) denetim
def check_full_solution(inst, pre, cfg, full: Dict[str, Dict], tol: float = 1e-6) -> Dict[str, Any]:
    """Özgün DÜZELTİLMİŞ modelin (model_monolithic_bigmfix) her kısıt satırını verilen değerlerle hesapla.
    Model ÇÖZÜLMEZ. Döndürür: ok, max_violation, aile başına en büyük ihlal ve yeri, sınır/ikililik ihlalleri,
    yeniden hesaplanan amaç."""
    import gurobipy as gp
    from gurobipy import GRB
    from lbbd_v2.model_monolithic_bigmfix import build_monolithic_model_bigmfix

    t0 = time.time()
    m = None
    try:
        m = build_monolithic_model_bigmfix(inst, pre, cfg)
        m.update()
        mv = m._vars
        val = [0.0] * m.NumVars
        missing_vals = 0
        for name, var in mv.items():
            if var is None:
                continue
            src = full.get(name)
            for key in var.keys():
                if src is None or key not in src:
                    missing_vals += 1
                    continue
                val[var[key].index] = float(src[key])
        fam: Dict[str, Tuple[float, str]] = {}
        fam_sc: Dict[str, Tuple[float, str]] = {}

        def bump(family, viol, where, scaled=None):
            if family not in fam or viol > fam[family][0]:
                fam[family] = (viol, where)
            sv = viol if scaled is None else scaled
            if family not in fam_sc or sv > fam_sc[family][0]:
                fam_sc[family] = (sv, where)
        for c in m.getConstrs():
            row = m.getRow(c)
            act = row.getConstant()
            mag = abs(row.getConstant())
            for kk in range(row.size()):
                term = row.getCoeff(kk) * val[row.getVar(kk).index]
                act += term
                mag = max(mag, abs(term))
            rhs, sense = c.RHS, c.Sense
            if sense == GRB.LESS_EQUAL:
                viol = act - rhs
            elif sense == GRB.GREATER_EQUAL:
                viol = rhs - act
            else:
                viol = abs(act - rhs)
            # ölçekli ihlal: satırın büyüklüğüne göre (max(1, |sağ taraf|, en büyük |katsayı·değer|))
            bump(c.ConstrName.split("[")[0], viol, c.ConstrName, viol / max(1.0, abs(rhs), mag))
        for vv in m.getVars():
            xv = val[vv.index]
            bump("sinir_alt", vv.LB - xv, vv.VarName)
            if vv.UB < GRB.INFINITY:
                bump("sinir_ust", xv - vv.UB, vv.VarName)
            if vv.VType == GRB.BINARY:
                bump("ikililik", min(abs(xv), abs(xv - 1.0)), vv.VarName)
        obj = sum(full["p"].values())
        worst = max(fam.items(), key=lambda kv: kv[1][0]) if fam else ("", (0.0, ""))
        mx = worst[1][0]
        worst_sc = max(fam_sc.items(), key=lambda kv: kv[1][0]) if fam_sc else ("", (0.0, ""))
        return {"ok": (mx <= tol and missing_vals == 0), "tol": tol, "max_violation": mx,
                "ok_scaled": (worst_sc[1][0] <= tol and missing_vals == 0),
                "max_scaled_violation": worst_sc[1][0], "worst_scaled_constraint": worst_sc[1][1],
                "worst_family": worst[0], "worst_constraint": worst[1][1],
                "family_max": {f: v for f, (v, _w) in sorted(fam.items())},
                "violations": {f: {"max": v, "where": wh} for f, (v, wh) in sorted(fam.items()) if v > tol},
                "n_constraints": m.NumConstrs, "n_vars": m.NumVars, "missing_values": missing_vals,
                "objective": obj, "check_s": time.time() - t0}
    finally:
        if m is not None:
            m.dispose()


def verify_witness(inst, pre, cfg, w: Dict[str, Any], tol: float = 1e-6) -> Dict[str, Any]:
    """Tanık → tam çözüm → özgün model denetimi → kayıtlı amaçla karşılaştırma."""
    comp = witness_completeness(w, inst.Nf)
    rep: Dict[str, Any] = {"source": w.get("source"), "completeness": comp}
    if not comp["complete"]:
        rep.update(status="EKSIK_TANIK", ok=False)
        return rep
    full, info = build_full_solution(inst, pre, cfg, w)
    rep["build"] = info
    if not full:
        rep.update(status="ILERI_GECIS_" + str(info.get("fwd_status")), ok=False)
        return rep
    chk = check_full_solution(inst, pre, cfg, full, tol=tol)
    rep["check"] = chk
    rec_val = w.get("value")
    diff = None if rec_val is None else chk["objective"] - float(rec_val)
    rep["objective_recomputed"] = chk["objective"]
    rep["objective_recorded"] = rec_val
    rep["objective_diff"] = diff
    obj_ok = diff is not None and abs(diff) <= max(tol, 1e-9 * max(1.0, abs(float(rec_val))))
    rep["ok"] = bool(chk["ok"] and obj_ok)
    rep["ok_scaled"] = bool(chk["ok_scaled"] and obj_ok)
    # DOGRULANDI: bütün satırlar mutlak tolerans içinde. DOGRULANDI_OLCEKLI: mutlak ihlal > tol ama satır
    # büyüklüğüne göre ölçekli ihlal ≤ tol (tipik neden: SP çözücüsünün FeasibilityTol payının μ_k ile
    # büyümesi). KISIT_IHLALI: ölçekli ihlal de > tol. İki ölçü de raporlanır; hiçbiri gizlenmez.
    if not obj_ok:
        rep["status"] = "AMAC_UYUSMAZ"
    elif chk["ok"]:
        rep["status"] = "DOGRULANDI"
    elif chk["ok_scaled"]:
        rep["status"] = "DOGRULANDI_OLCEKLI"
    else:
        rep["status"] = "KISIT_IHLALI"
    return rep


# ----------------------------------------------------------------------------- komut satırı
def verify_run_json(path: str, inputs_dir: Optional[str] = None, tol: float = 1e-6) -> Dict[str, Any]:
    """Bir LBBD koşu JSON'unun ``incumbent_solution`` kaydını doğrula (koşunun yapılandırmasıyla)."""
    from pathlib import Path
    from lbbd_v2.data_loader import load_instance
    from lbbd_v2.baseline.config import BaselineConfig
    from lbbd_v2.baseline.preprocessing import prepare

    js = json.loads(Path(path).read_text())
    inc = js.get("incumbent_solution")
    if not inc:
        return {"status": "TANIK_ALANI_YOK" if "incumbent_solution" not in js else "INKUMBENT_YOK", "ok": False}
    base = Path(inputs_dir) if inputs_dir else Path(__file__).resolve().parents[2] / "inputs"
    ipath = base / f"{js['instance']}.xlsx"
    if not ipath.exists():
        return {"status": "GIRDI_YOK", "ok": False, "input": str(ipath.name)}
    cfg = BaselineConfig()
    for k, v in (js.get("config") or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    inst = load_instance(ipath)
    pre = prepare(inst, cfg)
    # girdi uyumu (dolaylı): kayıttaki boyut ve ön işleme değerleri bu girdiden yeniden hesaplananla aynı mı?
    mt = js.get("metrics") or {}
    meta_cmp = {"n_Nf": (mt.get("n_Nf"), len(inst.Nf)), "n_K": (mt.get("n_K"), len(inst.K)),
                "M_d": (mt.get("M_d"), pre.M_d), "ts_ub": (mt.get("ts_ub"), pre.ts_ub)}
    uyum = all(a is None or (abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))) for a, b in meta_cmp.values())
    rep = verify_witness(inst, pre, cfg, witness_from_json(inc), tol=tol)
    rep["input_meta_match"] = uyum
    rep["input_meta"] = {k: {"kayit": a, "girdi": b} for k, (a, b) in meta_cmp.items()}
    return rep


if __name__ == "__main__":  # python -m lbbd_v2.baseline.full_solution <koşu.json> [girdi_klasörü]
    r = verify_run_json(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
    sys.exit(0 if r.get("ok") else 1)
