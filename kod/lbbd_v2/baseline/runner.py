"""runner.py — tek koşu (varyant×örnek) -> metrikler -> tidy CSV (long) + JSON. SLURM giriş noktası.

Kullanım (yerel doğrulama veya SLURM içinden):
    .venv/bin/python -m lbbd_v2.baseline.runner --instance inputs_4x4 --factors BASELINE \\
        --omega midpoint --threads 1 --seed 0 --total 3600 --master 600 --sp 300 \\
        --reference --outdir result_lbbd/ablation

- "--factors BASELINE" saf çekirdek; "--factors K5,S11" gibi virgülle faktör ekler (registry adları).
- "--reference" bigmfix monolitiğini de çözüp geçerlilik izleyicilerini (max LB−OPT, max OPT−UB) ve
  eşleşmeyi kaydeder. Referans yalnız optimalliği KANITLANIRSA kullanılır.
- Sonuç, örnek/varyant bittikçe long-format tidy CSV'ye EKLENİR (satır=varyant×örnek×metrik); iş
  kesilse bile veri kalır. Ayrıca zaman-damgalı tam JSON yazılır (uydurma yok; her sayı gerçek koşudan).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from lbbd_v2.config import Config
from lbbd_v2.data_loader import load_instance
from lbbd_v2.model_monolithic_bigmfix import solve_monolithic_bigmfix

from lbbd_v2.baseline.config import make_config
from lbbd_v2.baseline.preprocessing import prepare
from lbbd_v2.baseline.solver import solve_baseline_lbbd

REPO = Path(__file__).resolve().parents[2]
GAP_THRESHOLDS = (0.10, 0.05, 0.03, 0.01)


def _time_to_gap(log: List[dict], thr: float) -> Optional[float]:
    """UB-tarafı gap trajesi: gap ≤ thr olan İLK yinelemenin geçen süresi (yoksa None)."""
    for r in log:
        ub, lb, t = r.get("UB"), r.get("LB"), r.get("t_elapsed")
        if ub is None or lb is None or t is None:
            continue
        if ub >= float("inf") or lb <= float("-inf"):
            continue
        gap = (ub - lb) / max(1.0, abs(ub))
        if gap <= thr:
            return t
    return None


def _time_shares(log: List[dict]) -> Dict[str, float]:
    mt = sum(r.get("master_runtime", 0.0) or 0.0 for r in log)
    st = sum(r.get("sp_runtime", 0.0) or 0.0 for r in log)
    tot = mt + st
    return {"master_time_s": mt, "sp_time_s": st,
            "master_time_share": (mt / tot) if tot > 0 else 0.0,
            "sp_time_share": (st / tot) if tot > 0 else 0.0}


def run_one(instance: str, factors: List[str], *, omega: str = "midpoint", seed: int = 0,
            threads: int = 1, total: float = 3600.0, master: float = 600.0, sp: float = 300.0,
            reference: bool = False, ref_time_limit: float = 1800.0,
            outdir: str = "result_lbbd/ablation", excel: bool = False,
            sp_gap: float = 0.0, sp_tighten: bool = False,
            sp_tight_gap: float = 0.0, singleton_cache: bool = True,
            seed_budget: float = 120.0, bigm_threshold: int = 18, bigm_margin: float = 1.00,
            lp_bound: bool = False, seed_budget_frac: float = 0.0,
            heur_budget: Optional[float] = None, repair_budget: float = 60.0,
            apriori_budget: float = 300.0, label: str = "",
            sp_recur: bool = False, verify_witness: bool = True,
            cut_records: str = "ozet") -> Dict[str, Any]:
    inst_path = REPO / "inputs" / f"{instance}.xlsx"
    inst = load_instance(inst_path)
    # v2 (2026-09-19): tohum bütçesi TOPLAM'ın oranı olarak verilebilir (ölçekle büyür); aday
    # doğrulama tavanı (heur_budget) tohum bütçesiyle ölçeklenir (eski 30 s, 120 s tohumda aynı).
    if seed_budget_frac and seed_budget_frac > 0:
        seed_budget = seed_budget_frac * total
    if heur_budget is None:
        heur_budget = max(30.0, seed_budget / 4.0)

    variant = "BASELINE" if (not factors or factors == ["BASELINE"]) else "+".join(factors)
    fac = [] if variant == "BASELINE" else factors
    cfg = make_config(fac, omega_mode=omega, gurobi_seed=seed, gurobi_threads=threads,
                      total_budget=total, master_budget=master, sp_budget=sp,
                      sp_mip_gap=sp_gap, sp_gap_tighten_on_stall=sp_tighten,
                      sp_tight_mip_gap=sp_tight_gap, sp_singleton_cache=singleton_cache,
                      seed_budget=seed_budget, bigm_pathlen_max_nodes=bigm_threshold,
                      big_m_margin=bigm_margin, heur_budget=heur_budget,
                      repair_budget=repair_budget, apriori_budget=apriori_budget,
                      sp_recur_policy=sp_recur)

    # reviewer §10: TOPLAM bütçe ön-işlemeyi de kapsasın -> saat prepare'den ÖNCE başlar ve solve'a
    # t_start olarak geçirilir (ön işleme + model kurulumu + başlangıç çözümü + doğrulama hepsi 1800s içinde).
    t_run_start = time.time(); c_run_start = time.process_time()
    pre = prepare(inst, cfg)
    res = solve_baseline_lbbd(inst, cfg, pre=pre, t_start=t_run_start)
    wall = time.time() - t_run_start
    cpu_wall = time.process_time() - c_run_start

    # --- AYRI TEŞHİS (reviewer §4): ana problem KÖK LP-gevşetme UB'si. ARAMA SONRASI, AYRI bir master
    # kopyasında; süresi lp_relax_time_s ile raporlanır ve LBBD 1800s bütçesine DAHİL DEĞİLDİR. Sonucu
    # ana aramaya/başlangıç çözümüne/kesmelere AKTARILMAZ. LP-optimumu (kanıtlıysa) OPT'un geçerli UB'si;
    # MIP'in kök-kesme/dallanma sonrası dual sınırından FARKLIDIR. ---
    lp_diag = None
    if lp_bound:
        from lbbd_v2.baseline.master import build_baseline_master
        lp_diag = build_baseline_master(inst, pre, cfg).lp_relaxation_bound(budget=min(master, 300.0))

    # --- metrikler ---
    m: Dict[str, Any] = {
        "status": res.status, "LB": res.LB, "UB": res.UB, "gap": res.gap,
        "iterations": res.iterations, "runtime_s": res.runtime, "wall_s": wall, "cpu_s": cpu_wall,
        "n_sp_calls": sum(res.sp_calls.values()),
        "incumbent_value": (res.incumbent or {}).get("value"),
        "certified": 1 if res.status == "OPTIMAL" else 0,
        "n_Nf": len(inst.Nf), "n_K": len(inst.K), "sp_engine": cfg.sp_engine,
        "sp_mip_gap": sp_gap, "sp_tighten": int(sp_tighten), "sp_tight_gap": sp_tight_gap,
        # seed deneyi: hangi tohum(lar) + ayrılan bütçe (JSON config'te de var, kolay erişim için)
        "seed_localsearch": int(getattr(cfg, "lbbd_localsearch_seed", False)),
        "seed_budget_s": getattr(cfg, "seed_budget", None),
        "seed_budget_frac": seed_budget_frac, "heur_budget_s": heur_budget,
        "repair_budget_s": repair_budget, "apriori_budget_s": apriori_budget,
        "incumbent_source": (res.incumbent or {}).get("source"),
        "termination_reason": res.status,
        # kontrol yapısı + kullanılan araç sayısı (inkümbent SP atamasından; yoksa None)
        "inc_n_C": len((res.incumbent or {}).get("C") or []) if res.incumbent else None,
        "inc_n_pre": sum(1 for r in ((res.incumbent or {}).get("regime") or {}).values() if r == "pre") if res.incumbent else None,
        "inc_n_post": sum(1 for r in ((res.incumbent or {}).get("regime") or {}).values() if r == "post") if res.incumbent else None,
        "inc_n_vehicles_used": (len({k for (_i, k), v in (((res.incumbent or {}).get("sp") or {}).get("x") or {}).items() if v > 0.5})
                                if res.incumbent and ((res.incumbent.get("sp") or {}).get("x")) else None),
        "label": label, "sp_recur_policy": int(sp_recur),
        # ETKİN bileşenler (2026-09-25): faktör listesi + çözücü/arama bayraklarının koşuda fiilen aldığı değerler
        "effective_factors": "+".join(fac) if fac else "BASELINE",
        "eff_sp_engine": cfg.sp_engine, "eff_enable_C2": int(getattr(cfg, "enable_C2", False)),
        "eff_singleton_delta_cuts": int(getattr(cfg, "use_singleton_delta_cuts", False)),
        "eff_sdc_timed": int(getattr(cfg, "sdc_timed", False)), "eff_sdc_probe_eta": getattr(cfg, "sdc_probe_eta", None),
        "eff_sp_recur_policy": int(getattr(cfg, "sp_recur_policy", False)),
        "eff_branch_and_check": int(getattr(cfg, "use_branch_and_check", False)),
        "eff_gurobi_threads": cfg.gurobi_threads, "eff_gurobi_seed": cfg.gurobi_seed,
        "eff_total_budget": cfg.total_budget, "eff_master_budget": cfg.master_budget, "eff_sp_budget": cfg.sp_budget,
        "eff_seed_budget": getattr(cfg, "seed_budget", None), "eff_heur_budget": getattr(cfg, "heur_budget", None),
        "eff_sp_symmetry_break_cfg": int(getattr(cfg, "sp_symmetry_break", False)),
        "eff_sp_warm_start": int(getattr(cfg, "sp_warm_start", False)),
        # Big-M eşik/pay deneyi (reviewer §11) + ön-işleme tanılaması (preprocessing.bigm_aux)
        "bigm_threshold": bigm_threshold, "bigm_margin": bigm_margin,
        "bigm_method": pre.bigm_aux.get("bigm_method"),
        "bigm_nodes_in_dp": pre.bigm_aux.get("bigm_nodes_in_dp"),
        "bigm_dp_completed": pre.bigm_aux.get("bigm_dp_completed"),
        "M_d": pre.M_d, "ts_ub": pre.ts_ub, "te_ub": pre.te_ub,
    }
    # AYRI teşhis: kök LP-gevşetme (LBBD bütçesine DAHİL DEĞİL). initial_lp_ub YALNIZ LP optimalse dolu;
    # zaman aşımında geçerli dual sınır ayrı alanda + durum kodu.
    if lp_diag is not None:
        m["initial_lp_ub"] = lp_diag["lp_ub"]                 # yalnız lp_relax_optimal=1 iken anlamlı
        m["initial_lp_dualbound"] = lp_diag["dual_bound"]     # yalnız timeout'ta (LP kanıtlanmadı)
        m["lp_relax_time_s"] = lp_diag["runtime"]
        m["lp_relax_optimal"] = int(lp_diag["optimal"])
        m["lp_relax_status"] = lp_diag["status"]
    # --- seed metrikleri: başlangıç LB + ona ulaşma süresi (solver'ın 'seed' log kaydından) ---
    seed_rec = next((r for r in res.iteration_log if r.get("seed")), None)
    if seed_rec is not None:
        m["seed_LB"] = seed_rec.get("LB")
        m["seed_time_s"] = seed_rec.get("t_elapsed")
        m["seed_source"] = seed_rec.get("seed_source")
    for k, v in res.sp_calls.items():
        if v:
            m[f"spcall_{k}"] = v
    for k, v in res.cut_counts.items():
        m[f"cut_{k}"] = v
    # reviewer (a): tekil-SP önbellek/enstrümantasyon metrikleri (varsa)
    for k, v in (getattr(res, "diag_stats", None) or {}).items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                m[f"{k}_{kk}"] = vv
        else:
            m[k] = v
    m.update(_time_shares(res.iteration_log))
    # --- 2026-09-25 (v2): çözüm tanığının bağımsız denetimi. Arama BİTTİKTEN sonra, LBBD bütçesi DIŞINDA;
    # model ÇÖZÜLMEZ, özgün düzeltilmiş modelin kısıt satırları tanıktan kurulan tam çözümle hesaplanır
    # (lbbd_v2/baseline/full_solution.py). Sonuç yalnız raporlanır; LB/UB/durumu DEĞİŞTİRMEZ.
    m.update(_witness_metrics(res, inst, pre, cfg, verify_witness))
    for thr in GAP_THRESHOLDS:
        m[f"t_to_gap_{int(thr*100)}pct"] = _time_to_gap(res.iteration_log, thr)

    # --- referans (bigmfix, canlı) --- referans hatası LBBD sonucunu KAYBETTİRMEMELİ.
    # ADALET: MILP referansı da LBBD ile AYNI threads/seed kullanır (eşit-bütçeli kıyas için şart).
    if reference:
        ref_cfg = Config(omega_mode=omega, gurobi_threads=threads, gurobi_seed=seed)
        ref_cfg.validate()
        try:
            ref = solve_monolithic_bigmfix(inst, ref_cfg, mip_gap=0.0,
                                           time_limit=ref_time_limit, extract_solution=False)
        except Exception as ex:  # noqa: BLE001
            m["ref_error"] = repr(ex)
            ref = None
    if reference and ref is not None:
        m["ref_status"] = ref.status
        m["ref_proven_optimal"] = int(bool(ref.proven_optimal))
        m["ref_obj"] = ref.obj_val
        m["ref_bound"] = ref.obj_bound
        # --- MILP head-to-head sütunları (LBBD ile aynı tanımlarla) ---
        m["milp_obj"] = ref.obj_val
        m["milp_bound"] = ref.obj_bound
        m["milp_runtime_s"] = ref.runtime
        m["milp_budget_s"] = ref_time_limit
        m["milp_certified"] = int(bool(ref.proven_optimal))
        if ref.obj_val is not None and ref.obj_bound is not None:
            # LBBD ile AYNI gap tanımı: (UB-LB)/max(1,|UB|), UB=ObjBound, LB=ObjVal
            m["milp_gap"] = (ref.obj_bound - ref.obj_val) / max(1.0, abs(ref.obj_bound))
        # --- HEAD-TO-HEAD: LBBD en-iyi vs MILP (eşit bütçe/threads koşulduğunda anlamlı) ---
        lb = res.LB
        if ref.obj_val is not None and lb > float("-inf"):
            m["h2h_obj_delta"] = lb - ref.obj_val          # >0: LBBD daha yüksek fizibil ödül (daha iyi çözüm)
        if "milp_gap" in m and res.gap < float("inf"):
            m["h2h_gap_delta"] = m["milp_gap"] - res.gap    # >0: LBBD daha sıkı aralık
        lc = 1 if res.status == "OPTIMAL" else 0
        mc = int(m.get("milp_certified", 0))
        if lc != mc:
            win = "LBBD" if lc > mc else "MILP"
        elif ref.obj_val is not None and lb > float("-inf") and abs(lb - ref.obj_val) > 1e-4:
            win = "LBBD" if lb > ref.obj_val else "MILP"
        elif "milp_gap" in m and res.gap < float("inf") and abs(m["milp_gap"] - res.gap) > 1e-4:
            win = "LBBD" if (m["milp_gap"] - res.gap) > 0 else "MILP"
        else:
            win = "tie"
        m["h2h_winner"] = win
        if ref.proven_optimal and ref.obj_val is not None:
            opt = ref.obj_val
            mlo = max((r["LB"] - opt for r in res.iteration_log
                       if r.get("LB", float("-inf")) > float("-inf")), default=float("-inf"))
            mou = max((opt - r["UB"] for r in res.iteration_log
                       if r.get("UB", float("inf")) < float("inf")), default=float("-inf"))
            m["max_LB_over_opt"] = mlo
            m["max_opt_over_UB"] = mou
            m["exactness_ok"] = int(mlo <= 1e-6 and mou <= 1e-6)
            m["match_opt"] = int(res.status == "OPTIMAL" and abs(res.LB - opt) <= 1e-4
                                 and abs(res.UB - opt) <= 1e-4)

    # --- yaz: JSON (tam) + long CSV (ekle) ---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        import gurobipy as _gp
        gver = ".".join(str(x) for x in _gp.gurobi.version())
    except Exception:
        gver = ""
    out = REPO / outdir
    out.mkdir(parents=True, exist_ok=True)
    meta = {"variant": variant, "factors": fac, "instance": instance, "omega": omega,
            "seed": seed, "threads": threads,
            "budgets": {"total": total, "master": master, "sp": sp},
            "timestamp": stamp, "gurobi_version": gver,
            "config": cfg.to_dict(), "metrics": m,
            "iteration_log": res.iteration_log, "cut_counts": res.cut_counts,
            "sp_calls": res.sp_calls}
    # Dosya adına yapılandırma eki (SP-tolerans/tighten/önbellek) + BENZERSİZ koşu kimliği: aynı
    # (variant,instance,seed) için farklı config'ler (ör. head-to-head'in 3 config'i, önbellek A/B)
    # ÇAKIŞMASIN, ad kendini belgelesin ve aynı saniyede başlayan koşular birbirine karışmasın.
    import os as _os
    runid = _os.environ.get("SLURM_ARRAY_JOB_ID", "") + \
        (("-" + _os.environ["SLURM_ARRAY_TASK_ID"]) if _os.environ.get("SLURM_ARRAY_TASK_ID") else "")
    if not runid:
        import uuid as _uuid
        runid = _uuid.uuid4().hex[:8]
    meta["run_id"] = runid
    meta["label"] = label
    cfgtag = f"__g{sp_gap:g}__t{int(sp_tighten)}__c{int(singleton_cache)}" + (f"__{label}" if label else "")
    jname = out / f"{variant}__{instance}__{omega}__seed{seed}{cfgtag}__{stamp}__{runid}.json"
    meta["incumbent_solution"] = _sanitize_incumbent(res.incumbent)   # çözüm tanığı (JSON + Excel)
    # kesme kayıtları (v2): "yok" = yazılmaz (eski davranış), "ozet" = tür/yön/sağ taraf/üst veri (katsayısız),
    # "tam" = katsayılar dahil. Eski JSON'larda bu alan YOKTUR.
    meta["cut_records_mode"] = cut_records
    if cut_records != "yok":
        meta["cut_records"] = [({k: v for k, v in r.items() if k != "terms"} if cut_records == "ozet" else r)
                               for r in (res.cut_records or [])]
    jname.write_text(json.dumps(meta, indent=2, default=_json_default))

    csv_path = out / "ablation_tidy.csv"
    _append_long_csv(csv_path, variant, instance, omega, seed, stamp, m)

    if excel:
        xname = out / f"{variant}__{instance}__{omega}__seed{seed}{cfgtag}__{stamp}__{runid}.xlsx"
        try:
            _write_excel(xname, meta, res)
            print(f"    excel -> {xname.name}")
        except Exception as ex:  # noqa: BLE001  — excel yazımı ana sonucu bozmamalı
            print(f"    excel YAZILAMADI: {ex!r}")

    print(f"[{variant} | {instance}] status={m['status']} LB={m['LB']:.6g} UB={m['UB']:.6g} "
          f"gap={m['gap']:.3g} iters={m['iterations']} runtime={m['runtime_s']:.1f}s "
          f"-> {jname.name}")
    if reference and "exactness_ok" in m:
        print(f"    referans OPT={m['ref_obj']:.6g}  exactness_ok={m['exactness_ok']}  "
              f"match_opt={m['match_opt']}  max(LB-OPT)={m['max_LB_over_opt']:.2e}  "
              f"max(OPT-UB)={m['max_opt_over_UB']:.2e}")
    return meta



def _witness_state(inc) -> Dict[str, Any]:
    from lbbd_v2.baseline.full_solution import witness_from_incumbent, witness_completeness
    return witness_completeness(witness_from_incumbent(inc))


def _witness_metrics(res, inst, pre, cfg, enabled: bool) -> Dict[str, Any]:
    """Koşu sonu tanık denetimi metrikleri (witness_*)."""
    out: Dict[str, Any] = {"witness_check": "kapali" if not enabled else None}
    if not enabled:
        return out
    if not res.incumbent:
        out["witness_check"] = "inkumbent_yok"
        return out
    from lbbd_v2.baseline.full_solution import witness_from_incumbent, verify_witness
    if len(inst.Nf) * len(inst.K) > 250_000:
        out["witness_check"] = "atlandi_boyut"
        return out
    t0 = time.time()
    try:
        r = verify_witness(inst, pre, cfg, witness_from_incumbent(res.incumbent))
    except Exception as ex:  # noqa: BLE001  — denetim hatası sonucu kaybettirmemeli
        return {"witness_check": "hata", "witness_error": repr(ex), "witness_check_s": time.time() - t0}
    chk = r.get("check") or {}
    out.update(witness_check=r.get("status"), witness_ok=int(bool(r.get("ok"))),
               witness_missing=",".join(r.get("completeness", {}).get("missing", [])) or None,
               witness_max_violation=chk.get("max_violation"), witness_worst=chk.get("worst_constraint"),
               witness_obj_recomputed=r.get("objective_recomputed"), witness_obj_diff=r.get("objective_diff"),
               witness_omega_sp_minus_exact=(r.get("build") or {}).get("omega_sp_minus_exact_max"),
               witness_check_s=time.time() - t0)
    return out


def _sanitize_incumbent(inc):
    """İnkümbentin çözüm tanığını JSON-uyumlu hale getir (tuple anahtarlar -> "i,j").
    Çözüm davranışını DEĞİŞTİRMEZ. Bağımsız doğrulama için gerekli alanlar: y, u_pre, u_post, z, q (1 olanlar),
    kontrol edilen hücrelerde sp_x, sp_tc, sp_p ve value (lbbd_v2/baseline/full_solution.py). Diğerleri bilgidir.
    Eski (v1) kayıtlarda witness_version/witness/sp_s/times alanları YOKTUR; master_incumbent kaynaklı v1
    kayıtlarında sp_* alanları BOŞTUR (tanık eksik)."""
    if not inc:
        return None
    def keyed(d):
        out = {}
        for k, v in (d or {}).items():
            kk = ",".join(str(x) for x in k) if isinstance(k, tuple) else str(k)
            out[kk] = v
        return out
    sp = inc.get("sp") or {}
    return {"source": inc.get("source"), "value": inc.get("value"),
            "C": list(inc.get("C") or []), "regime": keyed(inc.get("regime") or {}),
            "y": keyed(inc.get("y")), "u_pre": keyed(inc.get("u_pre")), "u_post": keyed(inc.get("u_post")),
            "z": keyed({k: v for k, v in (inc.get("z") or {}).items() if v > 0.5}),
            "q": keyed({k: v for k, v in (inc.get("q") or {}).items() if v > 0.5}),
            "sp_x": keyed(sp.get("x")), "sp_tc": keyed(sp.get("tc")),
            "sp_omega": keyed(sp.get("omega")), "sp_p": keyed(sp.get("p")),
            # v2 (2026-09-25): servis süreleri ve ileri-geçiş zamanları; tanık bütünlüğü
            "sp_s": keyed(sp.get("s")),
            "times": {k: keyed(v) for k, v in (inc.get("times") or {}).items()},
            "witness_version": 2,
            "witness": _witness_state(inc)}

def _json_default(o):
    if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
        return str(o)
    return str(o)


def _append_long_csv(path: Path, variant, instance, omega, seed, stamp, metrics: Dict[str, Any]):
    """Long format: satır = varyant×örnek×metrik. Dosya yoksa başlık yaz, sonra EKLE."""
    exists = path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(["timestamp", "variant", "instance", "omega", "seed", "metric", "value"])
        for k, v in metrics.items():
            if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                v = str(v)
            w.writerow([stamp, variant, instance, omega, seed, k, v])


def _san(v):
    """Excel/openpyxl inf/nan yazamaz -> string'e çevir."""
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return str(v)
    return v


def _write_excel(path: Path, meta: Dict[str, Any], res) -> None:
    """LBBD+kesme koşusunu Excel'e yaz: run/metrics/cut_counts/sp_calls/iterations/config sayfaları."""
    import pandas as pd
    m = {k: _san(v) for k, v in meta["metrics"].items()}
    run = {"variant": meta["variant"], "instance": meta["instance"], "omega": meta["omega"],
           "seed": meta["seed"], "threads": meta["threads"], **meta["budgets"]}
    itlog = [{k: _san(v) for k, v in r.items()} for r in res.iteration_log]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame([run]).to_excel(xw, sheet_name="run", index=False)
        pd.DataFrame(sorted(m.items()), columns=["metric", "value"]).to_excel(
            xw, sheet_name="metrics", index=False)
        pd.DataFrame(sorted(res.cut_counts.items()), columns=["cut_family", "count"]).to_excel(
            xw, sheet_name="cut_counts", index=False)
        pd.DataFrame([(k, v) for k, v in res.sp_calls.items() if v],
                     columns=["sp_status", "count"]).to_excel(xw, sheet_name="sp_calls", index=False)
        if itlog:
            pd.DataFrame(itlog).to_excel(xw, sheet_name="iterations", index=False)
        pd.DataFrame(sorted(meta["config"].items()), columns=["flag", "value"]).to_excel(
            xw, sheet_name="config", index=False)
        # v2: çözüm tanığı sayfaları (JSON'daki incumbent_solution ile aynı içerik)
        inc = meta.get("incumbent_solution") or {}
        if inc:
            cells = sorted({int(k) for k in (inc.get("y") or {})})
            tim = inc.get("times") or {}
            rows = [{"hucre": i, "y": inc["y"].get(str(i)), "u_pre": (inc.get("u_pre") or {}).get(str(i)),
                     "u_post": (inc.get("u_post") or {}).get(str(i)),
                     "tc": (inc.get("sp_tc") or {}).get(str(i)), "p": (inc.get("sp_p") or {}).get(str(i)),
                     "omega_sp": (inc.get("sp_omega") or {}).get(str(i)),
                     **{f: (tim.get(f) or {}).get(str(i)) for f in ("ts", "tm", "te", "ts_min")}}
                    for i in cells]
            pd.DataFrame(rows).to_excel(xw, sheet_name="cozum_hucreler", index=False)
            xs = inc.get("sp_x") or {}
            pd.DataFrame([{"hucre_arac": k, "x": v, "s": (inc.get("sp_s") or {}).get(k)} for k, v in sorted(xs.items())],
                         columns=["hucre_arac", "x", "s"]).to_excel(xw, sheet_name="cozum_atamalar", index=False)
            pd.DataFrame([{"alan": k, "deger": _san(v) if not isinstance(v, (dict, list)) else json.dumps(v)}
                          for k, v in inc.items() if k in ("source", "value", "C", "witness_version", "witness")]
                         ).to_excel(xw, sheet_name="cozum_ozet", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="baseline/variant tek koşu -> metrik CSV/JSON")
    ap.add_argument("--instance", required=True, help="inputs/ altındaki dosya adı (uzantısız)")
    ap.add_argument("--factors", default="BASELINE",
                    help="'BASELINE' ya da virgülle faktör listesi (ör. K5,S11)")
    ap.add_argument("--omega", default="midpoint", choices=["midpoint", "worstcase"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--total", type=float, default=3600.0)
    ap.add_argument("--master", type=float, default=600.0)
    ap.add_argument("--sp", type=float, default=300.0)
    ap.add_argument("--sp-gap", type=float, default=0.0,
                    help="alt-problem iç MIPGap (0.0=kanıt; pozitif=erken dur; KESİNLİĞİ ETKİLEMEZ)")
    ap.add_argument("--sp-tighten", action="store_true",
                    help="duraklamada tekrarlayan TAM SP girdisini sıkı toleransla yeniden çöz")
    ap.add_argument("--sp-tight-gap", type=float, default=0.0,
                    help="uyarlamalı sıkılaştırmada kullanılacak sıkı MIPGap (0.0=tam kanıt)")
    ap.add_argument("--singleton-cache", type=int, default=1, choices=[0, 1],
                    help="tekil-SP sonucu önbelleği (1=açık varsayılan, 0=kapalı; kontrollü A/B için)")
    ap.add_argument("--seed-budget", type=float, default=120.0,
                    help="geliştirilmiş tohuma (SEEDLS) ayrılan süre (s); TOPLAM bütçeden düşülür")
    ap.add_argument("--bigm-threshold", type=int, default=18,
                    help="tam-yol Held-Karp DP düğüm eşiği (bigm_pathlen_max_nodes; MDxx deneyi)")
    ap.add_argument("--bigm-margin", type=float, default=1.00,
                    help="Big-M çarpanı (big_m_margin; M1yy deneyi; 1.00=pay yok)")
    ap.add_argument("--seed-budget-frac", type=float, default=0.0,
                    help="v2: tohum bütçesi = frac × TOPLAM (>0 ise --seed-budget'ı geçersiz kılar)")
    ap.add_argument("--heur-budget", type=float, default=None,
                    help="v2: tohum adayı doğrulama SP tavanı (s); None => max(30, seed_budget/4)")
    ap.add_argument("--repair-budget", type=float, default=60.0,
                    help="v2 REPAIR: onarım SP'si başına tavan (s)")
    ap.add_argument("--apriori-budget", type=float, default=300.0,
                    help="v2 APRI: a-priori tekil tarama tavanı (s)")
    ap.add_argument("--sp-recur", action="store_true",
                    help="tekrarlayan-imza SP çağrı politikası (config.sp_recur_policy; 2026-09-24 pilot)")
    ap.add_argument("--label", default="",
                    help="kampanya etiketi: dosya adına ve meta.label'a eklenir (eski kampanyalardan ayırt etmek için)")
    ap.add_argument("--lp-bound", action="store_true",
                    help="AYRI teşhis: ana problem kök LP-gevşetme UB'sini hesapla (LBBD bütçesine dahil değil)")
    ap.add_argument("--reference", action="store_true", help="bigmfix monolitiğini de çöz (geçerlilik)")
    ap.add_argument("--ref-time-limit", type=float, default=1800.0)
    ap.add_argument("--outdir", default="result_lbbd/ablation")
    ap.add_argument("--excel", action="store_true", help="LBBD+kesme sonucunu .xlsx olarak da yaz")
    ap.add_argument("--no-verify-witness", action="store_true",
                    help="koşu sonu çözüm tanığı denetimini kapat (v2; varsayılan açık, bütçe dışında)")
    ap.add_argument("--cut-records", default="ozet", choices=["yok", "ozet", "tam"],
                    help="kesme kayıtlarını JSON'a yaz: yok | ozet (katsayısız, varsayılan) | tam (katsayılar dahil)")
    a = ap.parse_args()
    factors = [f.strip() for f in a.factors.split(",") if f.strip()]
    run_one(a.instance, factors, omega=a.omega, seed=a.seed, threads=a.threads,
            total=a.total, master=a.master, sp=a.sp, reference=a.reference,
            ref_time_limit=a.ref_time_limit, outdir=a.outdir, excel=a.excel, sp_gap=a.sp_gap,
            sp_tighten=a.sp_tighten, sp_tight_gap=a.sp_tight_gap,
            singleton_cache=bool(a.singleton_cache), seed_budget=a.seed_budget,
            bigm_threshold=a.bigm_threshold, bigm_margin=a.bigm_margin, lp_bound=a.lp_bound,
            seed_budget_frac=a.seed_budget_frac, heur_budget=a.heur_budget,
            repair_budget=a.repair_budget, apriori_budget=a.apriori_budget, label=a.label,
            sp_recur=a.sp_recur, verify_witness=not a.no_verify_witness, cut_records=a.cut_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
