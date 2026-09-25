"""run_bigmfix_excel.py — monolitik MILP referans çalıştırıcısı + karar değişkeni Excel dökümü.

İki kullanım:
  * both/orig : ORİJİNAL vs DÜZELTİLMİŞ Big-M karşılaştırması (Big-M düzeltme gösterimi).
  * fix       : YALNIZ düzeltilmiş (γ=1.00 nihai) model — NİHAİ MILP REFERANS KOŞUSU (BATCH-3).
                Tek çözümden HEM özet (T4/T5: LB/UB/gap/süre/durum) HEM detaylı Excel üretilir;
                ikisi AYNI benzersiz run_id'ye bağlanır. İkinci bir optimizasyon çözülmez.

Nihai protokol (fix): ön işleme + model kurulumu 1800 s yöntem bütçesine DAHİL (çözücü TimeLimit'i
kalan süreyle daraltılır); Excel dışa aktarımı süresi AYRI kaydedilir ve modeli yeniden çözmez;
inkümbent yoksa durum+sınır kaydedilir, karar değişkeni sayfaları boş kalır (çözüm bulunamadı).

Çıktılar (fix):
  result_DUZELTILMIS_<örnek>_seed<seed>_<tag>.xlsx   (karar değişkenleri; meta'da run_id)
  milp_<örnek>_seed<seed>_<tag>.json                 (özet + config + sürüm; aynı run_id)

Kullanım:
  python -m lbbd_v2.run_bigmfix_excel <örnek.xlsx> [TL] [threads] [out_dir] [both|fix|orig] [seed] [run_id]
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import gurobipy as gp

from lbbd_v2.config import canonical_config
from lbbd_v2.data_loader import load_instance
from lbbd_v2.preprocessing import preprocess_instance
from lbbd_v2.model_monolithic import build_monolithic_model, apply_warm_start
from lbbd_v2.model_monolithic_bigmfix import build_monolithic_model_bigmfix
from lbbd_v2.run_mono_bigmfix import build_full_burn_start
from lbbd_v2.export_excel import (write_solution_xlsx, write_comparison_xlsx,
                                  make_gap_time_callback, hit_to_meta)

_STAT = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 9: "TIME_LIMIT", 13: "SUBOPTIMAL"}


def _solve(m, tl, threads, seed=0, mip_gap=0.0, mip_gap_abs=0.0):
    m.setParam("OutputFlag", 1)
    m.setParam("DualReductions", 0)
    m.setParam("TimeLimit", max(1.0, tl))
    m.setParam("Threads", threads)
    m.setParam("Seed", int(seed))
    # DÜZELTME (2026-09-16): MIPGap/MIPGapAbs modele AÇIKÇA uygulanır. Önceden yalnız
    # build sırasında set ediliyordu ve _solve override'ında yeniden yazılmadığından etkin
    # değer Gurobi varsayılanı (1e-4) kalıyordu; JSON ise istenen 0.0'ı raporluyordu.
    m.setParam("MIPGap", mip_gap)
    m.setParam("MIPGapAbs", mip_gap_abs)
    # UYGULANAN (etkin) parametreleri optimize ÖNCESİ modelden oku (istenen değil):
    eff = {"MIPGap": float(m.Params.MIPGap), "MIPGapAbs": float(m.Params.MIPGapAbs),
           "Threads": int(m.Params.Threads), "Seed": int(m.Params.Seed),
           "TimeLimit": float(m.Params.TimeLimit)}
    cb, hit = make_gap_time_callback()      # gap %10/%5/%3/%1 ilk-iniş süresi
    m.optimize(cb)
    return m, hit, eff


def _milp_summary(m, inst, path, cfg, seed, run_id, tag, opt_time_s, excel_time_s,
                  method_setup_s, tl_effective, eff_params):
    """Tek MILP koşusundan T4/T5 özeti + config + sürüm (aynı run_id). MAKS problem:
    LB = inkümbent (m.ObjVal), UB = m.ObjBound; gap = (UB-LB)/max(1,|UB|).
    solver_status (Gurobi durumu) ile certified_common (ORTAK LB-UB ölçütü) AYRILIR:
    Status=OPTIMAL olması tek başına ORTAK sertifika sayılmaz."""
    has_inc = m.SolCount > 0
    lb = m.ObjVal if has_inc else None
    ub = m.ObjBound if m.Status in (2, 9, 13) else None
    gap = None
    if lb is not None and ub is not None:
        try:
            gap = (ub - lb) / max(1.0, abs(ub))
        except (TypeError, ZeroDivisionError):
            gap = None
    # ORTAK sertifika ölçütü (LBBD ile AYNI): θ=max(eps_abs, eps_rel|UB|); UB-LB ≤ θ.
    eps_abs = getattr(cfg, "eps_abs", 1e-6); eps_rel = getattr(cfg, "eps_rel", 1e-9)
    certified_common = 0; theta = None
    if has_inc and ub is not None:
        theta = max(eps_abs, eps_rel * abs(ub))
        certified_common = int((ub - lb) <= theta)
    return {
        "kind": "milp_reference",
        "run_id": run_id,
        "instance": Path(path).stem,
        "seed": int(seed),
        "timestamp": tag,
        "gurobi_version": ".".join(str(x) for x in gp.gurobi.version()),
        "model": "monolithic_bigmfix",
        "protocol": {
            "total_budget_s": tl_effective + method_setup_s,   # yöntem bütçesi (ön işleme+kurulum+çözüm)
            "solver_time_limit_s": round(tl_effective, 3),      # ön işleme+kurulum düşülmüş çözücü TL
            "method_setup_s": round(method_setup_s, 3),         # ön işleme+model kurulumu (bütçeye dahil)
            "threads": int(cfg.gurobi_threads if hasattr(cfg, "gurobi_threads") else 1),
            "big_m_margin": cfg.big_m_margin,
            "bigm_pathlen_max_nodes": cfg.bigm_pathlen_max_nodes,
            "omega_mode": cfg.omega_mode,
            "mip_gap_requested": cfg.mip_gap_reference,      # İSTENEN
            "effective_params": eff_params,                  # UYGULANAN (modelden okundu)
            "eps_abs": eps_abs, "eps_rel": eps_rel,
        },
        "metrics": {
            "solver_status": _STAT.get(m.Status, str(m.Status)),   # Gurobi durumu (AYRI)
            "status": _STAT.get(m.Status, str(m.Status)),          # geriye-uyumluluk
            "status_code": m.Status,
            "has_incumbent": bool(has_inc),
            "LB": lb, "UB": ub, "gap": gap,
            "milp_obj": lb, "milp_bound": ub,               # aggregate_final.py uyumlu
            "milp_gap": (m.MIPGap if has_inc else None),
            "certified_common": certified_common,           # ORTAK LB-UB ölçütü (θ)
            "common_theta": theta,
            "solver_optimal": int(m.Status == 2),           # yalnız çözücü etiketi; ORTAK sertifika DEĞİL
            "runtime_s": round(m.Runtime, 3),               # çözücü süresi
            "opt_time_s": round(opt_time_s, 3),             # optimize() duvar süresi
            "excel_time_s": round(excel_time_s, 3),         # Excel dışa aktarım süresi (AYRI)
            "sol_count": m.SolCount,
            "n_Nf": len(inst.Nf), "n_K": len(inst.K),
        },
    }


def main(argv):
    path = argv[0] if argv else "inputs/inputs_8x8.xlsx"
    tl = float(argv[1]) if len(argv) > 1 else 600.0
    threads = int(argv[2]) if len(argv) > 2 else 24
    out_dir = Path(argv[3]) if len(argv) > 3 else Path("sonuclar")
    # 5. argüman: hangi model(ler) çözülsün — both|fix|orig (varsayılan both, geriye-uyumlu).
    #  fix  : yalnız DÜZELTİLMİŞ (γ=1.00 nihai model) — NİHAİ MILP referans koşusu (özet JSON + Excel).
    #  orig : yalnız ORİJİNAL. both : ikisi + karşılaştırma.
    only = (argv[4] if len(argv) > 4 else "both").strip().lower()
    if only not in ("both", "fix", "orig"):
        raise SystemExit(f"5. argüman 'both'|'fix'|'orig' olmalı, gelen: {only!r}")
    seed = int(argv[5]) if len(argv) > 5 else 0
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(path).stem
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = argv[6] if len(argv) > 6 else f"{stem}_seed{seed}_{tag}"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cfg = canonical_config()
    cfg.gurobi_threads = threads
    cfg.gurobi_seed = seed
    # NİHAİ protokol: ön işleme + model kurulumu yöntem bütçesine DAHİL -> saat prepare'den ÖNCE.
    t_method0 = time.time()
    inst = load_instance(path)
    pre = preprocess_instance(inst, cfg)
    print(f"[{stamp}] {Path(path).name}: |Nf|={len(inst.Nf)} |K|={len(inst.K)} "
          f"roots={len(inst.Na)}  TL={tl:.0f}s threads={threads} seed={seed} only={only} run_id={run_id}")

    m0 = m1 = e0 = e1 = None
    if only in ("both", "orig"):
        print("\n########## ORİJİNAL (DÜZELTME YOK) ##########")
        m0, hit0, _eff0 = _solve(build_monolithic_model(inst, pre, cfg), tl, threads, seed,
                                 mip_gap=cfg.mip_gap_reference)
        e0 = hit_to_meta(hit0)
        f0 = out_dir / f"result_ORIJINAL_{stem}_{tag}.xlsx"
        write_solution_xlsx(m0, inst, path, str(f0), extra_meta=e0)
        print(f"  status={_STAT.get(m0.Status)} obj={m0.ObjVal if m0.SolCount else None} -> {f0}")

    if only in ("both", "fix"):
        print("\n########## DÜZELTİLMİŞ (DÜZELTME VAR) ##########")
        m1 = build_monolithic_model_bigmfix(inst, pre, cfg)
        fb = build_full_burn_start(inst, pre, cfg)
        apply_warm_start(m1, fb) if fb else 0
        # ön işleme+kurulum yöntem bütçesinden düşülür; çözücü TL kalan süredir.
        setup_s = time.time() - t_method0
        tl_eff = max(1.0, tl - setup_s)
        t_opt0 = time.time()
        m1, hit1, eff1 = _solve(m1, tl_eff, threads, seed, mip_gap=cfg.mip_gap_reference)
        opt_s = time.time() - t_opt0
        e1 = hit_to_meta(hit1)
        # run_id + seed + sürüm meta sayfasına da yazılsın (özetle aynı kimlik).
        e1.update({"run_id": run_id, "seed": seed,
                   "gamma": cfg.big_m_margin, "dp_threshold": cfg.bigm_pathlen_max_nodes})
        suffix = f"{stem}_seed{seed}_{tag}" if only == "fix" else f"{stem}_{tag}"
        f1 = out_dir / f"result_DUZELTILMIS_{suffix}.xlsx"
        t_xl0 = time.time()
        write_solution_xlsx(m1, inst, path, str(f1), extra_meta=e1)   # inkümbent yoksa sayfalar boş
        excel_s = time.time() - t_xl0
        print(f"  status={_STAT.get(m1.Status)} obj={m1.ObjVal if m1.SolCount else None} "
              f"solcount={m1.SolCount} -> {f1}  (excel {excel_s:.2f}s AYRI)")
        if only == "fix":
            summ = _milp_summary(m1, inst, path, cfg, seed, run_id, tag,
                                 opt_time_s=opt_s, excel_time_s=excel_s,
                                 method_setup_s=setup_s, tl_effective=tl_eff, eff_params=eff1)
            summ["excel_file"] = f1.name
            fj = out_dir / f"milp_{suffix}.json"
            fj.write_text(json.dumps(summ, indent=2))
            mm = summ["metrics"]
            print(f"  ÖZET -> {fj}  LB={mm['LB']} UB={mm['UB']} gap={mm['gap']} "
                  f"solver_status={mm['solver_status']} certified_common={mm['certified_common']} "
                  f"eff_MIPGap={summ['protocol']['effective_params']['MIPGap']}")

    if only == "both":
        fc = out_dir / f"karsilastirma_{stem}_{tag}.xlsx"
        write_comparison_xlsx(inst, m0, m1, path, str(fc), extra0=e0, extra1=e1)
        print(f"\nKarşılaştırma -> {fc}")

    o0 = m0.ObjVal if (m0 is not None and m0.SolCount) else None
    o1 = m1.ObjVal if (m1 is not None and m1.SolCount) else None
    print("\n" + "=" * 60)
    if m0 is not None:
        print(f"ÖZET  ORİJİNAL={_STAT.get(m0.Status)} obj={o0}")
    if m1 is not None:
        print(f"      DÜZELTİLMİŞ={_STAT.get(m1.Status)} obj={o1}")
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1:])
