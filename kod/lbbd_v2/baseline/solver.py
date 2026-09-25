"""solver.py — DÜZELTİLMİŞ model üzerinde SADE, kesin LBBD döngüsü.

lbbd_v2.lbbd_solver'dan iki temel farkı vardır (docs/asama0_rapor.md §4-C1):
  1. `_late_ignition_cells` GEÇİDİ YOK ve `cut_ignition_time` (Kesme~8) ÜRETİLMEZ — bunlar
     düzeltilmemiş modelin geç-ateşleme tavanını dayatır; düzeltilmiş modelde geçersizdir.
  2. Tüm hızlandırıcılar (LB-sezgiseli, tekil ön-eleme, K5, K7, deletion filter, SP warm-start)
     BAYRAKLARLA kontrol edilir; minimal baseline'da hepsi KAPALIDIR.

Değişmez kesinlik sözleşmesi (KESİNLİK-GEREKLİ çekirdek her zaman açık):
  * UB yalnız master dual sınırından; LB yalnız doğrulanmış inkümbentten.
  * Kesme yalnız kanıtlanmış SP durumundan (INFEASIBLE_PROVEN => fizibilite; OPTIMAL => optimalite).
  * Kontrol A başarısızlığı => bağlantılılık kesmesi; Kontrol B => yayılım kesmesi (tembel, çekirdek).
  * LB ≤ OPT ≤ UB her yineleme; OPTIMAL yalnız UB−LB ≤ ε iken ilan edilir.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

from lbbd_v2.config import Config
from lbbd_v2.benders_forward import (
    forward_pass, FWD_OK, FWD_UNREACHABLE, FWD_TIME_INCONSISTENT, ForwardResult)
from lbbd_v2.lbbd_subproblem_resource import SPStatus, SPResult
from lbbd_v2.lbbd_conflicts import extract_conflict
from lbbd_v2.lbbd_cuts import (
    cut_connectivity, cut_propagation, cut_singleton_infeasible,
    cut_conflict_infeasible, cut_cell_optimality, cut_aggregate_optimality,
    cut_dual_bound_optimality, cut_singleton_predecessor, cut_singleton_predecessor_timed,
)
from lbbd_v2.lbbd_heuristics import (
    build_fallback_incumbent, build_greedy_incumbent_fast, build_containment_incumbent)
from lbbd_v2.lbbd_seed_localsearch import build_localsearch_incumbent, _structure_from_firebreaks
from lbbd_v2.lbbd_solver import LBBDResult, _decision, _regime_of, _gap, _closed_form_warm_x

from lbbd_v2.baseline.config import SP_STRONG
from lbbd_v2.baseline.preprocessing import prepare
from lbbd_v2.baseline.master import build_baseline_master
from lbbd_v2.baseline.subproblem import select_sp
from lbbd_v2.baseline.bch import BchController


def _terminal_status(LB: float, UB: float, cfg: Config):
    """Sınır durumu politikası (reviewer §1): sadece POZİTİF-ve-kapalı aralık optimalliktir.
      * gap = UB − LB;  opt_thr = max(eps_abs, eps_rel·max(1,|UB|)).
      * num_tol = 1e-9·max(1,|UB|): yalnız BİT düzeyi ters crossing (eps_abs'ten çok küçük) tolere edilir.
    Dönüş: "OPTIMAL" | "INCONSISTENT_BOUNDS" (büyük terslik, dur) | "NUMERICAL" (küçük terslik,
    optimallik DEĞİL, devam) | None (aralık henüz kapanmadı).
    """
    if UB >= float("inf") or LB <= float("-inf"):
        return None
    gap = UB - LB
    opt_thr = max(cfg.eps_abs, cfg.eps_rel * max(1.0, abs(UB)))
    num_tol = 1e-9 * max(1.0, abs(UB))
    if gap < -cfg.compare_tol:
        return "INCONSISTENT_BOUNDS"
    if gap < -num_tol:
        return "NUMERICAL"          # makine-üstü ters aralık: sayısal tutarsızlık, OPTIMAL DEĞİL
    if gap <= opt_thr:              # -num_tol ≤ gap ≤ opt_thr
        return "OPTIMAL"
    return None


def _sp_full_sig(C, regime, fwd):
    """Uyarlamalı sıkılaştırmanın TEKRAR ölçütü: TAM SP girdisi imzası.
    Yalnız kontrol kümesi C DEĞİL; rejimler VE ileri-geçişten gelen zamanlar (ts/tm/te/ts_min) da
    dahildir. Böylece 'aynı C ama farklı zamanlar' FARKLI SP sayılır ve yanlışlıkla sıkılaştırma
    tetiklenmez (reviewer: aynı C tek başına aynı SP anlamına gelmez). Zamanlar 9 basamağa
    yuvarlanır (deterministik ileri-geçiş; kayan-nokta gürültüsüne karşı kararlı imza)."""
    cells = sorted(C)
    return (tuple(cells),
            tuple((i, regime[i]) for i in cells),
            tuple((round(fwd.ts[i], 9), round(fwd.tm[i], 9), round(fwd.te[i], 9),
                   round(fwd.ts_min.get(i, 0.0), 9)) for i in cells))


def _times(fw) -> Dict[str, Dict[int, float]]:
    """Çözüm tanığı (2026-09-25): inkümbentin yapısına ait ileri-geçiş zamanları (yalnız izlenebilirlik;
    bağımsız doğrulama bu zamanları karar vektöründen YENİDEN hesaplayıp özgün kısıtlarla denetler)."""
    if fw is None:
        return {}
    return {"ts": dict(fw.ts), "tm": dict(fw.tm), "te": dict(fw.te), "ts_min": dict(fw.ts_min)}


def _sdc_eps_link(master, pre, i: int, a: int, others, tau: float, regime_i: str, diag: Dict[str, Any]):
    """SDC-t yardımcı ikilileri ε_(i,a,j,τ) ve bağlantı kısıtları (docs/singleton_delta_cuts.md §3b, §3f).

    ε_(i,a,j,τ) = 1  ⇒  y_j = 1  ve  t^s_j ≤ t^s_a − τ     (ε ≤ y_j;  t^s_j − t^s_a + Mε ≤ M − τ,  M = M_d + τ + 1)
    A3 düzeltmesi (2026-09-25): anahtar TAM τ ile kurulur. v1 anahtarı (i, a, j, round(τ, 6)) idi ve rejim
    içermiyordu; bağlantı kısıtı ise ilk oluşturmadaki yuvarlanmamış τ ile kuruluyordu. Aynı (i, a, j) için
    1e-6'dan yakın ama farklı iki eşik aynı ε'yu paylaşabiliyor, ilk eşik büyükse sonraki kesme gereğinden
    güçlü bağlantıya dayanıyordu. ε'nun anlamı yalnız (a, j, τ)'ya bağlıdır; aynı TAM τ'da paylaşım güvenlidir.
    Döndürür: kesmede kullanılacak anahtar listesi (others sırasıyla)."""
    keys = [(i, a, j, tau) for j in others]
    Mbig = pre.M_d + tau + 1.0
    for j, k in zip(others, keys):
        v, new = master.ensure_binary("eps", k, f"eps[{i},{a},{j},{tau!r}]")
        if new:
            master.add_cut(v - master.vars["y"][j], "<=", 0.0, f"eps_y[{i},{a},{j}]",
                           {"kind": "sdc_eps_link_y", "key": list(k)})
            master.add_cut(master.vars["ts"][j] - master.vars["ts"][a] + Mbig * v,
                           "<=", Mbig - tau, f"eps_ts[{i},{a},{j}]",
                           {"kind": "sdc_eps_link_ts", "key": list(k), "tau": tau, "M": Mbig, "regime": regime_i})
            diag["sdc_eps_vars"] = diag.get("sdc_eps_vars", 0) + 1
        else:
            diag["sdc_eps_reused"] = diag.get("sdc_eps_reused", 0) + 1
    return keys


def _sig_digest(sig) -> str:
    """Tam SP imzasının kararlı (process-bağımsız) kısa özeti — iteration_log'a yazılır ki
    tekrar/sıkılaştırma davranışı sonradan doğrulanabilsin (reviewer §4)."""
    return hashlib.blake2b(repr(sig).encode("utf-8"), digest_size=8).hexdigest()


def _update_sp_cache(cache: dict, sig, sp, tightened: bool) -> None:
    """Tam SP girdisi başına GEÇERLİ sınırları biriktir: en iyi (max) doğrulanmış fizibil ObjVal ve
    en sıkı (min) geçerli ObjBound. Böylece daha önce (sıkı) çözülmüş bir SP gevşek toleransla yeniden
    ÇÖZÜLMEZ; saklanan geçerli değerler yeniden kullanılır (reviewer §3). 'tightened' bir kez True
    olduysa KALICIDIR (sonraki ziyaretlerde gevşeğe DÖNMEZ)."""
    prev = cache.get(sig)
    best_obj = sp.obj
    if prev and prev.get("obj") is not None:
        best_obj = prev["obj"] if best_obj is None else max(best_obj, prev["obj"])
    best_bound = sp.obj_bound
    if prev and prev.get("bound") is not None:
        best_bound = prev["bound"] if best_bound is None else min(best_bound, prev["bound"])
    # çözüm (inkümbent yeniden kurulumu için): en iyi ObjVal'i vereni sakla
    keep_new_sol = sp.obj is not None and (prev is None or prev.get("obj") is None
                                           or sp.obj >= prev["obj"])
    best_sol = sp.solution if keep_new_sol else (prev.get("solution") if prev else {})
    tightened_flag = tightened or (prev.get("tightened") if prev else False)
    proven = (best_obj is not None and best_bound is not None
              and abs(best_bound - best_obj) <= 1e-6 * max(1.0, abs(best_bound)))
    # Yeniden-kullanım durumu (reviewer F2): en iyi ObjVal YOKSA (inkümbentsiz), sonucu
    # TIME_LIMIT_WITH_INCUMBENT diye ETİKETLEME — aksi halde reuse yolunda `pi_free + sp.obj(None)`
    # çöker ve D4 (inkümbentsiz timeout => LB/kesme YOK) ihlal edilir. best_obj None ise
    # TIME_LIMIT_NO_INCUMBENT (solver bunu atlar, LB/kesme üretmez). Kanıtlıysa OPTIMAL; aksi halde
    # inkümbentli-ama-kanıtsız (dış opt İMA ETMEZ).
    if best_obj is None:
        status = SPStatus.TIME_LIMIT_NO_INCUMBENT
    elif proven:
        status = SPStatus.OPTIMAL
    else:
        status = SPStatus.TIME_LIMIT_WITH_INCUMBENT
    cache[sig] = dict(obj=best_obj, bound=best_bound, solution=best_sol,
                      tightened=tightened_flag, status=status, proven=proven)


def _recur_update(cache: dict, sig, sp, budget: float):
    """Tekrarlayan-imza politikası (sp_recur_policy) önbelleği: _update_sp_cache'in geçerli birleşimi
    (max doğrulanmış ObjVal + o çözüm, min geçerli ObjBound) + çağrı sayacı/son bütçe/toplam süre.
    Döndürülen SPResult BİRLEŞİK durumdur: yeni çağrı daha kötü/inkümbentsiz dönse de önceki en iyi
    fizibil çözüm ve geçerli sınır KAYBOLMAZ. INFEASIBLE_PROVEN etiketi korunur (kanıtlı)."""
    prev = cache.get(sig)
    esc = (prev["escalations"] + 1) if prev else 0
    calls = (prev["calls"] + 1) if prev else 1
    tsum = (prev["time_s"] + (sp.runtime or 0.0)) if prev else (sp.runtime or 0.0)
    conflict = prev.get("conflict") if prev else None
    _update_sp_cache(cache, sig, sp, tightened=False)
    e = cache[sig]
    if sp.status == SPStatus.INFEASIBLE_PROVEN or (prev and prev.get("status") == SPStatus.INFEASIBLE_PROVEN):
        e["status"] = SPStatus.INFEASIBLE_PROVEN; e["proven"] = True
    e.update(escalations=esc, last_budget=float(budget), calls=calls, time_s=tsum, conflict=conflict)
    return SPResult(e["status"], e["obj"], e["bound"], sp.mip_gap, sp.runtime, e.get("solution") or {})


def solve_baseline_lbbd(inst, cfg: Config, pre=None, t_start=None) -> LBBDResult:
    # t_start verilirse toplam bütçe O ANDAN sayılır (reviewer §10: ön işleme + model kurulumu +
    # başlangıç çözümü + doğrulama TOPLAM bütçeye DAHİL). Runner, prepare'den ÖNCEki anı geçirir;
    # böylece bu çağrıda `pre` verilse bile ön-işleme süresi bütçeye yansır (verilmezse burada yapılır).
    t0 = t_start if t_start is not None else time.time()
    if pre is None:
        pre = prepare(inst, cfg)
    master = build_baseline_master(inst, pre, cfg)
    sp_solve, single_cell = select_sp(cfg)

    LB, UB = float("-inf"), float("inf")
    incumbent: Optional[dict] = None
    sp_calls: Dict[str, int] = {s.value: 0 for s in SPStatus}
    cut_counts: Dict[str, int] = {}
    log: List[dict] = []
    streak = 0
    no_inc = 0                      # master inkümbentsiz yineleme sayısı (termination ayrımı)
    last_sig = None                 # #7: aynı kararın ilerlemesiz tekrarını yakalamak için
    last_UB = float("inf")
    last_LB = float("-inf")
    repeat = 0
    # Uyarlamalı tolerans (yalnız cfg.sp_gap_tighten_on_stall açıkken): TAM SP girdisi tekrar
    # geldiğinde bir kez daha SIKI toleransla çözülür (bkz. config.sp_gap_tighten_on_stall).
    seen_sp_sigs: set = set()
    sp_cache: Dict[Any, dict] = {}          # tam SP girdisi -> saklanan geçerli sınırlar (reviewer §3)
    tighten_on_stall = (getattr(cfg, "sp_gap_tighten_on_stall", False)
                        and getattr(cfg, "sp_mip_gap", 0.0) > 0.0)
    # 2026-09-24 pilot: tekrarlayan-imza SP çağrı politikası (config.sp_recur_policy) + her iki kolda
    # AYNI ölçüm: imza başına gerçek çağrı sayısı ve duvar süresi (model kurma dahil).
    recur_policy = bool(getattr(cfg, "sp_recur_policy", False))
    recur_cache: Dict[Any, dict] = {}
    sig_calls: Dict[str, int] = {}
    sig_time: Dict[str, float] = {}
    # --- reviewer (a): tekil-SP sonucu önbelleği + kesme dedup + enstrümantasyon ---
    sc_cache: Dict[Any, dict] = {}          # tekil-SP: TAM girdi -> {feasible,p_solo,kind,solve_s}
    sc_distinct: set = set()                # farklı tam girdi sayısı (önbellek KAPALI iken de ölçülür)
    seen_cut_sigs: set = set()              # kesme imzası (kind,sense,rhs,birleşik katsayılar) dedup
    use_sc_cache = getattr(cfg, "sp_singleton_cache", True)   # reviewer (b): kontrollü açık/kapalı
    diag: Dict[str, Any] = {
        "singleton_cache_enabled": int(use_sc_cache),
        "single_sp_requests": 0,            # önbelleğe BAKILMADAN önceki toplam istek
        "single_sp_solves": 0,              # gerçek çözüm sayısı
        "single_sp_cache_hits": 0,          # önbellekten karşılanan
        "single_sp_build_time_s": 0.0,      # model kurma (yalnız gerçek çözümler)
        "single_sp_solve_time_s": 0.0,      # çözme (yalnız gerçek çözümler)
        "single_sp_time_s": 0.0,            # kurma+çözme toplamı
        "cache_lookup_time_s": 0.0,         # anahtar oluşturma + sözlük arama
        "est_prevented_solve_time_s": 0.0,  # TAHMİNİ: isabetlerin saklanan çözme süreleri toplamı
        "single_sp_status": {"FEASIBLE_VERIFIED": 0, "INFEASIBLE_PROVEN": 0, "UNKNOWN": 0},
        "dup_cuts_prevented": 0,
    }
    diag.update({"sp_recur_policy": int(recur_policy), "sp_recur_fresh": 0, "sp_recur_escalations": 0,
                 "sp_recur_reuses": 0, "sp_sig_distinct": 0, "sp_sig_max_calls": 0,
                 "sp_sig_max_time_s": 0.0, "sp_wall_total_s": 0.0})

    def count_cut(kind: str):
        cut_counts[kind] = cut_counts.get(kind, 0) + 1

    def _cut_sig(cut):
        return (cut.kind, cut.sense, round(cut.rhs, 9),
                tuple(sorted((vn, tuple(key) if isinstance(key, (list, tuple)) else key,
                              round(coef, 9)) for vn, key, coef in cut.terms)))

    def emit_cut(cut, kind: str) -> bool:
        """Kesmeyi uygula; AMA birleşik katsayı/yön/rhs imzası daha önce eklendiyse ATLA (mükerrer
        kesme = matematiksel no-op; ana problem fizibil kümesi değişmez). Eklendiyse True döner."""
        sig = _cut_sig(cut)
        if sig in seen_cut_sigs:
            diag["dup_cuts_prevented"] += 1
            return False
        seen_cut_sigs.add(sig)
        cut.apply(master); count_cut(kind)
        return True

    cpu0 = time.process_time()

    def emit(rec: Dict[str, Any]):
        """Kaydı EKLEMEDEN önce LB/UB/gap'i O ANKİ değerlerle tutarlı yaz (reviewer §5: gap–zaman
        eğrisi için her satırın açıklığı, satırın LB/UB'siyle birebir uyuşmalı)."""
        rec["LB"] = LB
        rec["UB"] = UB
        rec["gap"] = _gap(LB, UB)
        rec["cpu_elapsed"] = time.process_time() - cpu0   # süreç CPU süresi: uyku/yük teşhisi (duvar ≫ CPU ise makine durdu)
        log.append(rec)

    def pi_free(y):
        return sum(inst.pi[i] * (1 - y[i]) for i in inst.Nf)

    warm_ok = (cfg.sp_engine == SP_STRONG and getattr(cfg, "sp_warm_start", False))
    OKST = (SPStatus.OPTIMAL, SPStatus.FEASIBLE_NOT_PROVEN_OPTIMAL,
            SPStatus.TIME_LIMIT_WITH_INCUMBENT)
    use_apri = getattr(cfg, "use_apriori_singleton", False)
    use_repair = getattr(cfg, "use_candidate_repair", False)
    use_sdc = getattr(cfg, "use_singleton_delta_cuts", False)
    # Ölçüm (HER İKİ kolda): aynı (hücre, rejim, Δ) tekil olgusunun tekrar sayısı (farklı imzalar altında)
    fact_seen: Dict[Any, int] = {}
    diag.update({"sdc_enabled": int(use_sdc), "sdc_pred_cuts": 0, "sdc_fallback_sig_cuts": 0,
                 "sdc_dominance_hits": 0, "sdc_fixed_pre": 0, "sdc_fixed_post": 0, "sdc_apriori_screened": 0,
                 "sdc_apriori_time_s": 0.0, "sdc_cut_time_s": 0.0, "sdc_eps_vars": 0, "sdc_eps_reused": 0, "sdc_probe_solves": 0, "sdc_probe_proven": 0,
                 "dead_fact_events": 0, "dead_fact_distinct": 0, "dead_fact_repeats": 0, "dead_fact_max_repeat": 0})
    if use_apri or use_repair:
        diag.update({"apriori_enabled": int(use_apri), "apriori_screened": 0,
                     "apriori_fixed_pre": 0, "apriori_fixed_post": 0, "apriori_time_s": 0.0,
                     "apriori_dominance_hits": 0,
                     "apriori_fixed_cells_pre": [], "apriori_fixed_cells_post": [],
                     "apriori_fixed_both": 0,          # iki rejim de elendi => u_i = 0
                     "repair_enabled": int(use_repair), "repair_attempts": 0,
                     "repair_verified": 0, "repair_improved": 0, "repair_sp_calls": 0,
                     "repair_time_s": 0.0})
    # (i, rejim) -> kanıtlanmış infizibilite Δ-eşiği (Δ_i := t^s_i − t^{s,min}_i). Lemma (config.py
    # F-APRI): tekil-SP fizibilitesi Δ'da monoton artan => Δ_i ≤ eşik ise çözmeden infizibil.
    dinf: Dict[Any, float] = {}

    def _delta_of(fw: ForwardResult, i: int) -> float:
        return fw.ts.get(i, 0.0) - fw.ts_min.get(i, 0.0)

    # ---------- initial incumbents (yalnız F-SEED açıkken) ----------------
    # ORTAK BAŞLANGIÇ BÜTÇESİ (reviewer §1): TÜM başlangıç fazı — tohum kurulumu, yerel arama, tekil
    # ve çok-hücre SP çağrıları, tam-çözüm doğrulaması — cfg.seed_budget saniyeyle sınırlıdır ve TOPLAM
    # bütçenin İÇİNDEDİR. Süre dolunca YENİ SP BAŞLATILMAZ (kalan≤0); kalan süreden büyük çağrı bütçesi
    # verilmez. Kullanılmayan süre ana aramaya kalır (ana döngü t0'dan sayar).
    if getattr(cfg, "use_lb_heuristics", False):
        seed_deadline = t0 + max(0.0, min(getattr(cfg, "seed_budget", 120.0), cfg.total_budget))

        def _admit(source: str, cand) -> None:
            """Bir adayı başlangıç LB'sine kabul et. DOĞRULANMIŞ aday (localsearch: SP çözümünü
            zaten üretmiş) YENİDEN ÇÖZÜLMEZ — reviewer §2/§3: korunmuş çözüm doğrudan kabul edilir,
            aday-puanı ile doğrulanmış değer ayrılır. Doğrulanmamış aday, kalan ortak bütçe içinde
            (yoksa hiç) SP ile doğrulanır. Yeni SP yalnız kalan süre>0 iken başlatılır."""
            nonlocal LB, incumbent
            fwd = forward_pass(cand, inst)
            if fwd.status != FWD_OK:
                return
            # #1: C'yi u'dan türet (cand.C'ye güvenme); firebreak u_pre=1 ama C=[] yapısı geçersiz
            # (fazla) LB verirdi -> SP ile doğrularız.
            C_u = [i for i in inst.Nf if cand.u_pre.get(i, 0.0) + cand.u_post.get(i, 0.0) > 0.5]
            regime_u = {i: ("pre" if cand.u_pre.get(i, 0.0) > 0.5 else "post") for i in C_u}
            sol: Dict[str, Any] = {}
            if getattr(cand, "validated", False) and cand.value is not None:
                # reviewer §2/§3: localsearch bu yapıyı ÇOK-HÜCRE SP ile zaten çözüp fizibil çözümü
                # (atama+zaman) sakladı. Yeniden çözmek (küçük kalan bütçede) korunmuş çözümün yerine
                # GEÇMEZ; doğrulanmış değeri doğrudan kabul et, çözümü koru.
                val = cand.value
                sol = getattr(cand, "sp_solution", {}) or {}
            elif not C_u:
                val = pi_free(cand.y)                # gerçekten u≡0 -> recourse 0, geçerli LB
            else:
                remaining = seed_deadline - time.time()
                if remaining <= 0:
                    return                           # ortak bütçe doldu -> YENİ SP başlatma
                budget = min(cfg.heur_budget, remaining)   # kalandan büyük çağrı bütçesi verme
                wx = cand.start_vars.get("x") if getattr(cand, "start_vars", None) else None
                if wx is not None and warm_ok:
                    sp = sp_solve(fwd, C_u, regime_u, inst, pre, cfg,
                                  time_budget=budget, warm_x=wx, symmetry_break=False)
                else:
                    sp = sp_solve(fwd, C_u, regime_u, inst, pre, cfg, time_budget=budget)
                sp_calls[sp.status.value] += 1
                # SP doğrulamadıysa (INFEASIBLE_PROVEN / UNKNOWN / inkümbentsiz timeout) adayı REDDET
                # -> geçersiz LB girmez (bu, aday-puanının reddi; kanıtlanmış fizibilsizlik iddiası değil).
                if sp.status not in OKST or sp.obj is None:
                    return
                val = pi_free(cand.y) + sp.obj
                sol = getattr(sp, "solution", {}) or {}
            if val > LB:
                LB = val
                incumbent = {"source": source, "C": C_u, "regime": regime_u,
                             "value": val, "sp": sol, "times": _times(fwd), **_decision(cand)}

        # (a) ucuz kapalı-form adaylar (hızlı kurulur), ortak bütçe içinde SP ile doğrulanır
        base_cands = [("containment", build_containment_incumbent(inst, pre, cfg)),
                      ("fallback", build_fallback_incumbent(inst, pre, cfg))]
        if getattr(cfg, "lbbd_fast_greedy_seed", False):
            base_cands.append(("greedy_fast", build_greedy_incumbent_fast(inst, pre, cfg)))
        for source, cand in base_cands:
            if time.time() >= seed_deadline:
                break
            _admit(source, cand)

        # (b) GELİŞTİRİLMİŞ tohum: KALAN ortak bütçeyle koşar (kendi içinde SP-doğrulamalı, çözümü
        # korur). Döndürdüğü incumbent DOĞRULANMIŞSA doğrudan kabul edilir (yeniden çözme yok).
        if getattr(cfg, "lbbd_localsearch_seed", False) and time.time() < seed_deadline:
            ls_budget = seed_deadline - time.time()
            ls = build_localsearch_incumbent(inst, pre, cfg, budget=ls_budget,
                                             sp_solve=sp_solve, single_cell=single_cell)
            if ls is not None:
                _admit("localsearch", ls)

    # Başlangıç (tohum) kaydı: başlangıç LB'yi ve ONA ULAŞMA süresini (initial-incumbent döngüsü
    # t0'dan sonra koştuğu için t_elapsed = tohumun harcadığı toplam süre) kalıcı olarak logla —
    # seed deneyinin raporlaması (başlangıç LB, ona ulaşma süresi) buradan okunur. LB henüz -inf ise
    # (hiç sezgisel açık değil) kayıt atlanır.
    if LB > float("-inf"):
        log.append({"iter": 0, "seed": True,
                    "seed_source": (incumbent or {}).get("source"),
                    "LB": LB, "UB": UB, "gap": _gap(LB, UB),
                    "t_elapsed": time.time() - t0})

    need_single = (getattr(cfg, "use_singleton_prescreen", False)
                   or getattr(cfg, "use_percell_optimality_cuts", False))

    def _screen(Cs: List[int], reg: Dict[int, str], fw: ForwardResult):
        """Tekil ön-eleme + p_solo (önbellekli). Dönüş (p_solo, dead). Davranış, eski satır-içi
        döngüyle birebir aynıdır; APRI açıkken ek olarak Δ-baskınlık kısayolu uygulanır."""
        p_s: Dict[int, float] = {}
        dd: List[int] = []
        for i in Cs:
            diag["single_sp_requests"] += 1
            # Önbellek anahtarı = tekil-SP'yi belirleyen TAM girdi (hücre + rejim + ileri-geçiş
            # zamanları). Aynı girdi tekrar gelirse (önbellek AÇIKken) yeniden ÇÖZME.
            _kt = time.perf_counter()
            sc_key = (i, reg[i], round(fw.ts.get(i, 0.0), 9), round(fw.tm.get(i, 0.0), 9),
                      round(fw.te.get(i, 0.0), 9), round(fw.ts_min.get(i, 0.0), 9))
            cached = sc_cache.get(sc_key) if use_sc_cache else None
            diag["cache_lookup_time_s"] += time.perf_counter() - _kt
            if cached is not None:
                diag["single_sp_cache_hits"] += 1
                diag["est_prevented_solve_time_s"] += cached.get("solve_s", 0.0)
                feasible, ps = cached["feasible"], cached["p_solo"]
            elif (use_apri or use_sdc) and (i, reg[i]) in dinf and _delta_of(fw, i) <= dinf[(i, reg[i])] + 1e-9:
                # Δ-baskınlık (Lemma F-APRI): daha büyük/eşit Δ'da kanıtlanmış infizibil => burada da.
                if use_apri:
                    diag["apriori_dominance_hits"] += 1
                if use_sdc:
                    diag["sdc_dominance_hits"] += 1
                feasible, ps = False, 0.0
            else:
                timing: Dict[str, float] = {}
                feasible, ps, kind = single_cell(
                    i, reg[i], fw, inst, pre, cfg,
                    time_budget=max(1.0, min(cfg.filter_budget,
                                             cfg.total_budget - (time.time() - t0))),
                    return_status=True, timing=timing)
                diag["single_sp_solves"] += 1
                b_s = timing.get("build_s", 0.0); s_s = timing.get("solve_s", 0.0)
                diag["single_sp_build_time_s"] += b_s
                diag["single_sp_solve_time_s"] += s_s
                diag["single_sp_time_s"] += b_s + s_s
                diag["single_sp_status"][kind] = diag["single_sp_status"].get(kind, 0) + 1
                sc_distinct.add(sc_key)      # farklı girdi sayısı (önbellek kapalıyken de)
                if use_sc_cache:
                    sc_cache[sc_key] = {"feasible": feasible, "p_solo": ps, "kind": kind,
                                        "solve_s": s_s}
                if (use_apri or use_sdc) and kind == "INFEASIBLE_PROVEN":
                    d_i = _delta_of(fw, i)
                    if d_i > dinf.get((i, reg[i]), float("-inf")):
                        dinf[(i, reg[i])] = d_i
                    if use_sdc and getattr(cfg, "sdc_timed", True):
                        # SDC-t sonda: Δ̄ + η'da da kanıtlı fizibilsiz mi? (τ ≥ η > 0 ⇒ t^s_j = t^s_a eşitlikleri kesilir;
                        # geçerlilik Δ̄+η'daki kanıta dayanır — docs/singleton_delta_cuts.md §3b)
                        eta = float(getattr(cfg, "sdc_probe_eta", 1e-3))
                        target = d_i + eta
                        if target > dinf[(i, reg[i])] + 1e-12 and cfg.total_budget - (time.time() - t0) > 2.0:
                            ff = ForwardResult(status=FWD_OK)
                            ff.ts[i] = target; ff.ts_min[i] = 0.0
                            ff.tm[i] = target + inst.alpha / inst.lam[i]
                            ff.te[i] = ff.tm[i] + inst.alpha / inst.sig[i]
                            _f2, _p2, k2 = single_cell(i, reg[i], ff, inst, pre, cfg,
                                                       time_budget=max(1.0, min(cfg.filter_budget,
                                                                                cfg.total_budget - (time.time() - t0))),
                                                       return_status=True)
                            diag["sdc_probe_solves"] += 1
                            if k2 == "INFEASIBLE_PROVEN":
                                dinf[(i, reg[i])] = target; diag["sdc_probe_proven"] += 1
            if not feasible:
                dd.append(i)
            else:
                p_s[i] = ps
        return p_s, dd

    # ---------- APRI: a-priori tekil-infizibilite (döngü ÖNCESİ, Lemma F-APRI) ----------
    # En gevşek durum: Δ_max = ts_ub (t^s_i ≤ ts_ub her fizibil çözümde; t^{s,min}_i ≥ 0). Kökler için
    # zamanlar sabittir (t^s=0, Δ=0) => onların tekil-SP'si zaten zaman-bağımsızdır.
    if use_apri and need_single:
        _ta = time.time()
        a_budget = max(0.0, min(getattr(cfg, "apriori_budget", 300.0),
                                cfg.total_budget - (time.time() - t0)))
        a_deadline = time.time() + a_budget
        Na_set = set(inst.Na)
        ffake = ForwardResult(status=FWD_OK)
        for i in sorted(inst.Nf, key=lambda c: (-inst.pi[c], c)):
            if time.time() >= a_deadline:
                break
            d_max = 0.0 if i in Na_set else float(pre.ts_ub)
            ffake.ts[i] = d_max; ffake.ts_min[i] = 0.0
            ffake.tm[i] = d_max + inst.alpha / inst.lam[i]
            ffake.te[i] = ffake.tm[i] + inst.alpha / inst.sig[i]
            for reg_i in ("pre", "post"):
                if time.time() >= a_deadline:
                    break
                if reg_i == "pre" and getattr(cfg, "enable_C0prime", False) and not pre.pre_possible[i]:
                    continue                      # C0' zaten u_pre=0
                if getattr(cfg, "enable_C0", False) and not pre.controllable[i]:
                    continue                      # C0 zaten u=0
                _feas, _ps, kind = single_cell(
                    i, reg_i, ffake, inst, pre, cfg,
                    time_budget=max(1.0, min(cfg.filter_budget, a_deadline - time.time())),
                    return_status=True)
                diag["apriori_screened"] += 1
                if kind == "INFEASIBLE_PROVEN":
                    dinf[(i, reg_i)] = max(dinf.get((i, reg_i), float("-inf")), d_max)
                    vn = "u_pre" if reg_i == "pre" else "u_post"
                    master.fix_zero(vn, i, f"APRI_{reg_i}[{i}]",
                                    record={"kind": "apriori_singleton_infeasible", "cell": i,
                                            "regime": reg_i, "delta_max": d_max})
                    diag[f"apriori_fixed_{reg_i}"] += 1
                    diag[f"apriori_fixed_cells_{reg_i}"].append(i)
        diag["apriori_fixed_both"] = len(set(diag["apriori_fixed_cells_pre"])
                                         & set(diag["apriori_fixed_cells_post"]))
        diag["apriori_time_s"] = time.time() - _ta

    # ---------- SDC a-priori: Δ^max_i = ts_ub (kökte 0) ile tekil tarama; kanıtlı ⇒ u^r_i = 0 ----------
    # Geçerlilik: Δ_i = t^s_i − t^{s,min}_i ≤ t^s_i ≤ ts_ub (t^{s,min} ≥ 0; ön işleme üst sınırı); köklerde
    # t^s = t^{s,min} = 0 ⇒ Δ = 0. Δ-monotonluk ⇒ Δ^max'ta fizibilsiz olan (i, r) hiçbir yapıda fizibil değil ⇒ u^r_i = 0.
    # (APRI ile aynı sınır; docs/singleton_delta_cuts.md §4. α/λ tabanlı sınır GEÇERSİZDİR: öncül yalnız alt sınır verir.)
    if use_sdc and need_single and getattr(cfg, "sdc_apriori", True):
        _ta = time.time()
        a_budget = max(0.0, min(getattr(cfg, "sdc_apriori_budget", 60.0), cfg.total_budget - (time.time() - t0)))
        a_deadline = time.time() + a_budget
        Na_set = set(inst.Na)
        ffake = ForwardResult(status=FWD_OK)
        for i in sorted(inst.Nf, key=lambda c: (-inst.pi[c], c)):
            if time.time() >= a_deadline:
                break
            d_max = 0.0 if i in Na_set else float(pre.ts_ub)
            ffake.ts[i] = d_max; ffake.ts_min[i] = 0.0
            ffake.tm[i] = d_max + inst.alpha / inst.lam[i]
            ffake.te[i] = ffake.tm[i] + inst.alpha / inst.sig[i]
            for reg_i in ("pre", "post"):
                if time.time() >= a_deadline:
                    break
                if reg_i == "pre" and getattr(cfg, "enable_C0prime", False) and not pre.pre_possible[i]:
                    continue
                if getattr(cfg, "enable_C0", False) and not pre.controllable[i]:
                    continue
                _feas, _ps, kind = single_cell(i, reg_i, ffake, inst, pre, cfg,
                                               time_budget=max(1.0, min(cfg.filter_budget, a_deadline - time.time())),
                                               return_status=True)
                diag["sdc_apriori_screened"] += 1
                if kind == "INFEASIBLE_PROVEN":
                    dinf[(i, reg_i)] = max(dinf.get((i, reg_i), float("-inf")), d_max)
                    vn = "u_pre" if reg_i == "pre" else "u_post"
                    master.fix_zero(vn, i, f"SDC_{reg_i}[{i}]",
                                    record={"kind": "sdc_apriori_fix", "cell": i, "regime": reg_i, "delta_max": d_max})
                    diag[f"sdc_fixed_{reg_i}"] += 1
        diag["sdc_apriori_time_s"] = time.time() - _ta

    # ---------- REPAIR: elenen adayı onar -> doğrulanmış fizibil çözüm (YALNIZ LB) ----------
    repair_seen: set = set()

    def _repair(C0: List[int], regime0: Dict[int, str], dead0: List[int]) -> Dict[str, Any]:
        """C0 \\ dead0 ile yapı kur (firebreak Dijkstra ormanı), yeniden tekil-ele, ÇOK-HÜCRE SP ile
        doğrula; doğrulanan değer > LB ise inkümbent güncellenir. Kesme/UB üretmez."""
        nonlocal LB, incumbent
        t_r = time.time()
        diag["repair_attempts"] += 1
        dead_set = set(dead0)
        F = {i for i in C0 if regime0[i] == "pre" and i not in dead_set}
        P = {i for i in C0 if regime0[i] == "post" and i not in dead_set}
        out: Dict[str, Any] = {"status": "none"}
        for _round in range(4):
            remaining = cfg.total_budget - (time.time() - t0)
            if remaining <= 1.0:
                out["status"] = "no_time"; break
            inc, Feff, y = _structure_from_firebreaks(F, inst, pre)
            fw = forward_pass(inc, inst)
            if fw.status != FWD_OK:
                out["status"] = "fwd_" + fw.status; break
            post = sorted(i for i in P if y.get(i, 0.0) > 0.5 and i not in Feff
                          and pre.controllable.get(i, True))
            for i in post:
                inc.u_post[i] = 1.0
            Cr = sorted(set(Feff) | set(post))
            reg = {i: ("pre" if i in Feff else "post") for i in Cr}
            key = (frozenset(Feff), frozenset(post))
            if key in repair_seen:
                out["status"] = "seen"; break
            _p, dd = _screen(Cr, reg, fw)
            if dd:
                F.difference_update(dd); P.difference_update(dd)
                out["status"] = "rescreen"; out["rounds"] = _round + 1
                continue
            repair_seen.add(key)
            sol: Dict[str, Any] = {}
            if not Cr:
                val = pi_free(y)                       # kontrolsüz yapı: recourse 0, geçerli LB
                st = "EMPTY"
            else:
                budget = max(1.0, min(getattr(cfg, "repair_budget", 60.0), remaining))
                sp = sp_solve(fw, Cr, reg, inst, pre, cfg, time_budget=budget)
                sp_calls[sp.status.value] += 1
                diag["repair_sp_calls"] += 1
                st = sp.status.value
                if sp.status not in OKST or sp.obj is None:
                    out.update(status="sp_" + st, C_size=len(Cr)); break
                val = pi_free(y) + sp.obj
                sol = getattr(sp, "solution", {}) or {}
            diag["repair_verified"] += 1
            out.update(status="verified", sp_status=st, value=val, C_size=len(Cr), improved=False)
            if val > LB:
                LB = val
                incumbent = {"source": "repair", "C": Cr, "regime": reg, "value": val,
                             "sp": sol, "times": _times(fw), **_decision(inc)}
                diag["repair_improved"] += 1
                out["improved"] = True
            break
        diag["repair_time_s"] += time.time() - t_r
        return out

    # ---------- BCH: branch-and-check denetleyicisi (F-BCH; lbbd_v2/baseline/bch.py) ----------
    use_bch = getattr(cfg, "use_branch_and_check", False)
    bch = None
    if use_bch:
        def _bch_validated(val: float, C_v, regime_v, sp_sol, ms_v, source: str) -> None:
            """LB yalnız DOĞRULANMIŞ tam çözümden (callback içi SP OPTIMAL / fizibil inkümbent)."""
            nonlocal LB, incumbent
            if val > LB:
                LB = val
                fw_v = forward_pass(ms_v, inst)
                incumbent = {"source": source, "C": list(C_v), "regime": dict(regime_v),
                             "value": val, "sp": sp_sol,
                             "times": _times(fw_v) if fw_v.status == FWD_OK else {}, **_decision(ms_v)}
                log.append({"iter": bch_iter[0], "bch_lb_update": True, "source": source,
                            "LB": LB, "UB": UB, "gap": _gap(LB, UB), "t_elapsed": time.time() - t0})

        def _count_sp(st: str) -> None:
            sp_calls[st] = sp_calls.get(st, 0) + 1

        bch_iter = [0]
        last_stop = [False]     # bir önceki master çözümü güvenli durdurmayla bitti mi (yeniden başlatma etiketi)
        bch = BchController(
            inst, pre, cfg, master, sp_solve=sp_solve, screen=_screen, regime_of=_regime_of,
            pi_free=pi_free, sp_full_sig=_sp_full_sig, t0=t0, diag=diag, count_sp=_count_sp,
            on_validated=_bch_validated, extract_conflict=extract_conflict,
            warm_x=(lambda C_, r_, f_: _closed_form_warm_x(C_, r_, f_, inst, pre, cfg)) if warm_ok else None,
            need_single=need_single)

    # ---------- ana döngü -------------------------------------------------
    it = 0
    status = "TIME_LIMIT"
    while time.time() - t0 < cfg.total_budget and it < cfg.max_iterations:
        it += 1
        cuts_added = 0
        rec: Dict[str, Any] = {"iter": it}

        # #5: alt çağrılara kalan TOPLAM bütçeyi uygula (yerel bütçe kalanı aşamaz).
        remaining = cfg.total_budget - (time.time() - t0)
        if remaining <= 0:
            status = "TIME_LIMIT"; break
        if bch is not None:
            bch.reset(); bch_iter[0] = it
            diag["bch_master_solves"] += 1
            ms = master.solve(budget=min(cfg.master_budget, remaining), callback=bch.callback)
            persisted = 0
            for cut, kind in bch.state["pending"]:
                if emit_cut(cut, kind):          # kalıcılaştırmada dedup (lazy uygulamada dedup YOK)
                    persisted += 1
            diag["bch_persisted_cuts"] += persisted
            cuts_added += persisted
            # TANI (2329 gözlemi): master'ın döndürdüğü çözüm callback'te görüldü mü? (a) imzası SP önbelleğinde mi,
            # (b) bu çözümde lazy eklenen kesmelerden ihlal edilen var mı (Gurobi lazy kısıtı ihlal eden bir inkümbent
            # döndürdü mü?). Dış döngü zaten yeniden doğrular; bu sayaçlar davranışı belgeler.
            if ms.solution_count > 0:
                _pt = {"y": ms.y, "z": ms.z, "q": ms.q, "u_pre": ms.u_pre, "u_post": ms.u_post, "rho": ms.rho}
                _viol = [kind for cut, kind in bch.state["pending"] if not cut.satisfied(_pt)]
                try:
                    _fwd_chk = forward_pass(ms, inst)
                    _C_chk = [i for i in inst.Nf if ms.u(i) > 0.5]
                    _in_cache = int(_fwd_chk.status == FWD_OK and bool(_C_chk)
                                    and _sp_full_sig(_C_chk, _regime_of(ms, _C_chk), _fwd_chk) in bch.sp_cache)
                except Exception:
                    _in_cache = -1
                rec.update(bch_final_violates_lazy=len(_viol), bch_final_sig_in_cache=_in_cache)
                diag["bch_final_violates_lazy_total"] = diag.get("bch_final_violates_lazy_total", 0) + len(_viol)
                diag["bch_final_unseen"] = diag.get("bch_final_unseen", 0) + int(_in_cache == 0)
            rec.update(bch_callbacks=bch.state["n_cb"], bch_lazy=bch.state["n_lazy"],
                       bch_accepted=bch.state["n_accepted"], bch_unproven=bch.state["n_unproven"],
                       bch_sp_calls=bch.state["n_sp"], bch_persisted=persisted,
                       bch_restart=int(diag["bch_restarts"] > 0 and it > 1 and bool(last_stop[0])))
            last_stop[0] = bch.state["inconclusive"] is not None
        else:
            ms = master.solve(budget=min(cfg.master_budget, remaining))
        if ms.status == "INFEASIBLE":
            status = "INFEASIBLE"
            break
        if bch is not None and bch.state["inconclusive"] is not None:
            # GÜVENLİ DURDURMA: callback doğrulaması tamamlanamadı, geçerli kesme yok => durdurulan master'ın
            # ÇÖZÜMÜ KULLANILMAZ (Gurobi durdurulan adayı iç inkümbent yapmış olabilir; LB'ye alınmaz, sertifika
            # ilan edilmez). UB: ObjBound, lazy kesmelerle güçlendirilmiş geçerli gevşetmenin dal-sınır dual
            # sınırıdır; iç inkümbent master-fizibil olduğundan onunla budanan düğümler master optimumunu
            # içeremez => ObjBound ≥ master-opt ≥ OPT (geçerli UB). Sertifika YALNIZ kesintisiz bir master
            # çözümünden sonraki dış LB–UB kapanışıyla ilan edilir. Kesme eklendiyse ilerleme var; yoksa streak.
            reason = bch.state["inconclusive"]
            # UB KAYNAĞI (madde 1): iki geçerli aday sınır — (a) callback'te durdurma ÖNCESİ okunan küresel
            # MIPSOL_OBJBND, (b) kesinti sonrası ObjBound. İkisi de lazy kesmelerle güçlendirilmiş geçerli
            # gevşetmenin dal-sınır dual sınırıdır (her master-fizibil nokta ya açık bir düğümdedir [sınır ≥ değeri],
            # ya iç inkümbentle budanmıştır [değeri ≤ inkümbent ≤ sınır], ya da inkümbentle kapanmıştır; iç
            # inkümbentin doğrulanmamış olması bunu etkilemez, yalnız master-fizibil olması yeter) => ≥ master-opt
            # ≥ OPT. Durdurulan adayın amaç değeri (stopped_obj) ASLA UB/LB olarak kullanılmaz. Hiçbiri yoksa UB
            # değişmez (önceki geçerli UB ya da +inf).
            cands = []
            if bch.state["last_cb_bound"] is not None:
                cands.append(("cb_objbnd", bch.state["last_cb_bound"]))
            if ms.obj_bound is not None:
                cands.append(("objbound_after_stop", ms.obj_bound))
            ub_source = "none"
            if cands:
                src, val = min(cands, key=lambda kv: kv[1])
                if val < UB:
                    UB = val; ub_source = src
                else:
                    ub_source = "previous_ub"
            master.discard_solution()                      # doğrulanmamış iç inkümbent SONRAKİ çözüme taşınmaz
            diag["bch_restarts"] += 1                      # bir sonraki master.solve yeni bir optimize() = yeniden başlatma
            rec.update(UB=UB, LB=LB, gap=_gap(LB, UB), master_status=ms.status,
                       master_runtime=ms.runtime, master_solcount=ms.solution_count,
                       t_elapsed=time.time() - t0, bch_stop=reason, bch_unproven=bch.state["n_unproven"],
                       ub_source=ub_source, stopped_obj=bch.state["stopped_obj"],
                       cb_objbnd=bch.state["last_cb_bound"], objbound_after_stop=ms.obj_bound)
            if _terminal_status(LB, UB, cfg) == "INCONSISTENT_BOUNDS":
                status = "INCONSISTENT_BOUNDS"; rec["inconsistent_bounds"] = True; emit(rec); break
            if reason.startswith("no_time") or cfg.total_budget - (time.time() - t0) <= 0:
                status = "TIME_LIMIT"; emit(rec); break
            if persisted == 0:
                streak += 1
            rec["streak"] = streak
            emit(rec)
            if streak > cfg.max_inconclusive:
                status = "INCONCLUSIVE"; break
            continue
        if ms.obj_bound is not None:
            UB = min(UB, ms.obj_bound)
        rec.update(UB=UB, LB=LB, gap=_gap(LB, UB), master_status=ms.status,
                   master_runtime=ms.runtime, t_elapsed=time.time() - t0)
        # #2: OPTIMAL ilanından ÖNCE geçerlilik. LB, UB'yi ANLAMLI aşıyorsa optimallik İLAN ETME
        # (geçersiz LB / sayısal çelişki durumunu sessizce "optimal" saymayı önler).
        term = _terminal_status(LB, UB, cfg)
        if term == "INCONSISTENT_BOUNDS":
            status = term; rec["inconsistent_bounds"] = True; emit(rec); break
        if term == "NUMERICAL":
            rec["numerical_inconsistency"] = True    # optimallik DEĞİL; devam (kesme eklenirse ilerler)
        elif term == "OPTIMAL":
            status = "OPTIMAL"; emit(rec); break
        # #4: master fizibil çözüm bulamadıysa (inkümbent yok -> boş karar) ileri geçiş/SP ÇALIŞTIRMA;
        # geçerli UB (obj_bound'dan) korunur, KeyError önlenir.
        if ms.solution_count == 0:
            no_inc += 1
            rec["master_no_incumbent"] = True; emit(rec)
            if no_inc > cfg.max_inconclusive:
                status = "MASTER_NO_INCUMBENT"; break   # termination nedeni AÇIK (zaman aşımı değil)
            continue

        fwd = forward_pass(ms, inst)
        rec["fwd_status"] = fwd.status

        if fwd.status == FWD_UNREACHABLE:
            for j in sorted(fwd.unreachable):
                if emit_cut(cut_connectivity(fwd.unreachable, j, inst), "connectivity"):
                    cuts_added += 1
            rec["cuts"] = cuts_added; emit(rec); continue

        if fwd.status == FWD_TIME_INCONSISTENT:
            for (i, j) in fwd.violating_arcs:
                if emit_cut(cut_propagation(i, j, fwd), "propagation"):
                    cuts_added += 1
            rec["cuts"] = cuts_added; emit(rec); continue

        # NOT: geç-ateşleme geçidi YOK (düzeltilmiş model).

        C = [i for i in inst.Nf if ms.u(i) > 0.5]
        regime = _regime_of(ms, C)
        rec["C_cells"] = list(C)

        # #5: kalan toplam süre bittiyse tek-hücre/SP çözümü BAŞLATMA (yeni Gurobi çağrısı açma).
        if cfg.total_budget - (time.time() - t0) <= 0:
            status = "TIME_LIMIT"; emit(rec); break

        # #7 (reviewer §2): İLERLEME = aynı karar DEĞİL, ya da UB düştü, ya da LB yükseldi. Yalnız
        # "aynı karar + iki sınır da değişmedi" durumunda ilerlemesizlik sayılır (LB iyileşirse RESET).
        sig = (frozenset(C), tuple(sorted((i, regime[i]) for i in C)))
        num = 1e-9 * max(1.0, abs(UB) if UB < float("inf") else 1.0)
        no_prog = (sig == last_sig and UB >= last_UB - num and LB <= last_LB + num)
        repeat = repeat + 1 if no_prog else 0
        last_sig, last_UB, last_LB = sig, UB, LB
        if repeat > cfg.max_inconclusive:
            status = "INCONCLUSIVE_NO_PROGRESS"; rec["no_progress"] = True
            emit(rec); break

        # tekil ön-eleme + p_solo (yalnız PRE veya K5 açıkken)
        p_solo: Dict[int, float] = {}
        dead: List[int] = []
        if need_single:
            p_solo, dead = _screen(C, regime, fwd)
            if dead and getattr(cfg, "use_singleton_prescreen", False):
                for i in dead:
                    # ölçüm (her iki kol): aynı (hücre, rejim, Δ) olgusu daha önce ölü bulundu mu?
                    fkey = (i, regime[i], round(_delta_of(fwd, i), 6))
                    fact_seen[fkey] = fact_seen.get(fkey, 0) + 1
                    diag["dead_fact_events"] += 1
                    if fact_seen[fkey] > 1:
                        diag["dead_fact_repeats"] += 1
                    pred_cuts_current = False
                    if use_sdc:
                        _tc = time.perf_counter()
                        dbar = dinf.get((i, regime[i]))
                        has_ts = master.vars.get("ts") is not None
                        if dbar is not None:
                            point = _decision(ms)
                            for a in inst.neighbors[i]:
                                if i not in inst.neighbors[a]:
                                    continue                       # simetri şart (a ∈ N⁺(i))
                                aol = inst.alpha / inst.lam[a]
                                if aol > dbar + 1e-9:
                                    continue
                                if has_ts and getattr(cfg, "sdc_timed", True):
                                    # SDC-t: zaman koşullu kesme (MTC t^s + yardımcı ε ikilileri; docs §3b)
                                    tau = max(0.0, dbar - aol)
                                    others = [j for j in inst.neighbors[i] if j != a]
                                    keys = _sdc_eps_link(master, pre, i, a, others, tau, regime[i], diag)
                                    cut = cut_singleton_predecessor_timed(i, regime[i], a, tau, others, keys,
                                          {"delta_bar": dbar, "alpha_over_lam": aol, "delta_now": _delta_of(fwd, i)})
                                    # mevcut noktayı kesiyor mu: u=1, q_ai=1 ve τ kadar erken yanan komşu yok
                                    cur_q = point["q"].get((a, i), 0.0) > 0.5 and a in fwd.ts
                                    early = cur_q and any(point["y"].get(j, 0.0) > 0.5 and j in fwd.ts
                                                          and fwd.ts[j] <= fwd.ts[a] - tau + 1e-9 for j in others)
                                    if cur_q and not early:
                                        pred_cuts_current = True
                                    if emit_cut(cut, "singleton_pred_t"):
                                        cuts_added += 1; diag["sdc_pred_cuts"] += 1
                                else:
                                    cut = cut_singleton_predecessor(i, regime[i], a, inst,
                                          {"delta_bar": dbar, "alpha_over_lam": aol, "delta_now": _delta_of(fwd, i)})
                                    if not cut.satisfied(point):
                                        pred_cuts_current = True   # mevcut master noktasını kesiyor
                                    if emit_cut(cut, "singleton_pred"):
                                        cuts_added += 1; diag["sdc_pred_cuts"] += 1
                        diag["sdc_cut_time_s"] += time.perf_counter() - _tc
                    if not pred_cuts_current:
                        # kesme sözleşmesi: mevcut nokta MUTLAKA kesilir (klasik imza kesmesi)
                        if emit_cut(cut_singleton_infeasible(i, ms, fwd, inst), "singleton_infeasible"):
                            cuts_added += 1
                            if use_sdc:
                                diag["sdc_fallback_sig_cuts"] += 1
                if use_repair and cfg.total_budget - (time.time() - t0) > 1.0:
                    rec["repair"] = _repair(C, regime, dead)
                    rec["t_elapsed"] = time.time() - t0
                rec.update(cuts=cuts_added, dead=dead); emit(rec); continue

        # K5: her yineleme hücre optimalite kesmesi (PRE ile eşleşik => p_solo hazır)
        if getattr(cfg, "use_percell_optimality_cuts", False):
            for i in C:
                if i in p_solo and ms.rho.get(i, 0.0) > p_solo[i] + cfg.eps_abs:
                    if emit_cut(cut_cell_optimality(i, p_solo[i], ms, fwd, inst), "cell_optimality"):
                        cuts_added += 1

        # çok-hücre SP (#5: kalan toplam bütçeyle sınırlı)
        sp_budget = max(1.0, min(cfg.sp_budget, cfg.total_budget - (time.time() - t0)))
        # TAM SP girdisi imzası — HER çok-hücre SP için hesaplanır: tanılama loglaması (sp_sig) +
        # (tighten açıkken) tekrar/önbellek/sıkılaştırma kararı. Aynı C ama farklı zaman => farklı imza.
        sp_full_sig = _sp_full_sig(C, regime, fwd)
        rec["sp_sig"] = _sig_digest(sp_full_sig)

        eff_gap = None
        reuse = None
        if tighten_on_stall:
            cached = sp_cache.get(sp_full_sig)
            recurring = sp_full_sig in seen_sp_sigs
            seen_sp_sigs.add(sp_full_sig)
            if recurring and cached is not None:
                if not cached["tightened"]:
                    # DURAKLAMA: bu tam girdi tekrar geldi, henüz SIKI çözülmedi -> bir kez SIKI çöz.
                    eff_gap = getattr(cfg, "sp_tight_mip_gap", 0.0)
                    rec["sp_tightened"] = True
                else:
                    # zaten SIKI çözülmüş/timeout: SAKLANAN geçerli sınırları YENİDEN KULLAN, gevşek
                    # toleransla yeniden ÇÖZME. Timeout SP'nin master kararı ELENMEZ (fizibilsizlik
                    # kanıtı değil); yalnız sonuçsuz tekrar-hesap önlenir (reviewer §3).
                    reuse = cached

        # 2026-09-24 pilot — tekrarlayan-imza SP çağrı politikası (yalnız ÇAĞRI; model/kesme aynı):
        #   fresh    : bu tam girdi ilk kez -> mevcut yol (kapalı-biçim ılık başlangıç, sp_budget)
        #   escalate : tekrar + kanıtsız + hak var -> bütçe ×growth (kalan TOPLAM bütçeyi aşmaz),
        #              MIP start = önceki SP inkümbentinin x'i (yoksa kapalı-biçim)
        #   reuse    : kanıtlı (OPTIMAL/INFEASIBLE) ya da hak bitti ya da bütçe yok -> saklanan
        #              geçerli sınırlar/çözüm; yeni Gurobi çağrısı YOK
        recur_mode = None
        recur_ent = recur_cache.get(sp_full_sig) if recur_policy else None
        if recur_policy and reuse is None:
            if recur_ent is None:
                recur_mode = "fresh"
            elif recur_ent["proven"] or recur_ent["status"] == SPStatus.INFEASIBLE_PROVEN:
                recur_mode = "reuse"
            elif recur_ent["escalations"] >= int(getattr(cfg, "sp_recur_max_escalations", 2)):
                recur_mode = "reuse"
            else:
                remaining = cfg.total_budget - (time.time() - t0)
                want = float(getattr(cfg, "sp_recur_growth", 2.0)) * recur_ent["last_budget"]
                esc_budget = min(want, remaining - 0.5)        # son çağrı kalan TOPLAM bütçeyi AŞMAZ
                if esc_budget < 1.0:
                    recur_mode = "reuse"
                else:
                    recur_mode = "escalate"; sp_budget = esc_budget
            rec["sp_policy"] = recur_mode

        t_sp_wall = time.time(); c_sp_cpu = time.process_time()
        if reuse is None and bch is not None and sp_full_sig in bch.sp_cache:
            # callback bu TAM girdiyi zaten çözdü (aynı zamanlar, aynı rejimler) => yeniden çözme
            sp = bch.sp_cache[sp_full_sig]
            rec["sp_reused_bch"] = True
        elif reuse is not None:
            sp = SPResult(reuse["status"], reuse["obj"], reuse["bound"], None, 0.0,
                          reuse.get("solution") or {})
            rec["sp_reused"] = True
        elif recur_mode == "reuse":
            sp = SPResult(recur_ent["status"], recur_ent["obj"], recur_ent["bound"], None, 0.0,
                          recur_ent.get("solution") or {})
            rec["sp_reused_recur"] = True
            diag["sp_recur_reuses"] += 1
        else:
            wx = None
            if recur_mode == "escalate":
                wx = (recur_ent.get("solution") or {}).get("x") or None   # önceki SP inkümbenti
                rec["sp_escalation"] = recur_ent["escalations"] + 1
                rec["sp_warm_from_prev"] = int(wx is not None)
                diag["sp_recur_escalations"] += 1
            if wx is None and warm_ok:
                wx = _closed_form_warm_x(C, regime, fwd, inst, pre, cfg)
            rec["sp_budget_given"] = sp_budget
            if wx is not None:
                # ılık başlangıç varken S11 (simetri kırma) ÇAĞRI DÜZEYİNDE kapatılır: kapalı-biçim/önceki
                # atama kanonik olmayabilir, simetri kısıtlarıyla çelişirdi (her iki pilot kolunda aynı).
                sp = sp_solve(fwd, C, regime, inst, pre, cfg, time_budget=sp_budget,
                              warm_x=wx, symmetry_break=False, mip_gap=eff_gap)
            else:
                sp = sp_solve(fwd, C, regime, inst, pre, cfg, time_budget=sp_budget, mip_gap=eff_gap)
            sp_calls[sp.status.value] += 1
            rec["sp_raw_status"] = sp.status.value; rec["sp_raw_obj"] = sp.obj; rec["sp_raw_bound"] = sp.obj_bound
            if tighten_on_stall:
                _update_sp_cache(sp_cache, sp_full_sig, sp, tightened=(eff_gap is not None))
            if recur_policy:
                sp = _recur_update(recur_cache, sp_full_sig, sp, sp_budget)   # birleşik (kayıpsız) sonuç
                recur_ent = recur_cache[sp_full_sig]
                if recur_mode == "fresh":
                    diag["sp_recur_fresh"] += 1
        sp_wall = time.time() - t_sp_wall
        rec["sp_wall"] = sp_wall
        rec["sp_cpu"] = time.process_time() - c_sp_cpu
        real_call = not (rec.get("sp_reused_recur") or rec.get("sp_reused") or rec.get("sp_reused_bch"))
        rec["sp_real_call"] = int(real_call)
        if real_call:
            dg = rec["sp_sig"]
            sig_calls[dg] = sig_calls.get(dg, 0) + 1
            sig_time[dg] = sig_time.get(dg, 0.0) + sp_wall
            diag["sp_wall_total_s"] += sp_wall
        rec.update(sp_status=sp.status.value, sp_obj=sp.obj, C_size=len(C),
                   sp_runtime=sp.runtime, sp_bound=sp.obj_bound)

        if sp.status == SPStatus.INFEASIBLE_PROVEN:
            if recur_mode == "reuse" and recur_ent is not None and recur_ent.get("conflict"):
                Cstar = set(recur_ent["conflict"])           # aynı girdi için çakışma zaten çıkarıldı
            elif getattr(cfg, "use_deletion_filter", False):
                Cstar = extract_conflict(C, regime, fwd, inst, pre, cfg, sp_solve=sp_solve,
                                         time_budget=max(1.0, min(cfg.filter_budget,
                                             cfg.total_budget - (time.time() - t0))))
            else:
                Cstar = set(C)
            if recur_policy and sp_full_sig in recur_cache:
                recur_cache[sp_full_sig]["conflict"] = sorted(Cstar)
            if emit_cut(cut_conflict_infeasible(Cstar, ms, fwd, inst), "conflict_infeasible"):
                cuts_added += 1
            streak = 0
            rec.update(cuts=cuts_added, conflict=sorted(Cstar)); emit(rec); continue

        if sp.status == SPStatus.OPTIMAL:
            # LB: DOĞRULANMIŞ fizibil değerden (sp.obj = ObjVal). Kesme RHS: Φ üzerinde GEÇERLİ üst
            # sınırdan (sp.obj_bound = ObjBound). #3: MIPGap toleransıyla (varsayılan 1e-4) sp.obj,
            # gerçek Φ'nin ALTINDA olabilir; kesmede kullanmak fizibil ödülü silebilirdi. ObjBound ≥ Φ
            # her toleransta geçerli üst sınırdır.
            val = pi_free(ms.y) + sp.obj
            if val > LB:
                LB = val
                incumbent = {"source": "master", "C": C, "regime": regime,
                             "value": val, "sp": sp.solution, "times": _times(fwd), **_decision(ms)}
            cut_phi = sp.obj_bound if sp.obj_bound is not None else sp.obj
            if emit_cut(cut_aggregate_optimality(C, cut_phi, ms, fwd, inst), "aggregate_optimality"):
                cuts_added += 1
            streak = 0
            rec.update(cuts=cuts_added, LB=LB, gap=_gap(LB, UB),
                       t_elapsed=time.time() - t0)  # #6: LB'nin iyileştiği GERÇEK zaman
            emit(rec)
            term = _terminal_status(LB, UB, cfg)
            if term == "INCONSISTENT_BOUNDS":
                status = term; break
            if term == "OPTIMAL":
                status = "OPTIMAL"; break
            # term == "NUMERICAL" veya None: optimallik DEĞİL -> döngü devam
            continue

        if sp.status in (SPStatus.FEASIBLE_NOT_PROVEN_OPTIMAL,
                         SPStatus.TIME_LIMIT_WITH_INCUMBENT):
            val = pi_free(ms.y) + sp.obj
            if val > LB:
                LB = val
                # A1 (2026-09-25): önceki sürüm burada SP çözümünü SAKLAMIYORDU (değer doğru, tanık eksik).
                # Amaç değeri ve tanık aynı SPResult'tan (sp.obj ile sp.solution) gelir.
                incumbent = {"source": "master_incumbent", "C": C, "regime": regime,
                             "value": val, "sp": sp.solution, "times": _times(fwd), **_decision(ms)}
            if getattr(cfg, "use_dual_optimality_cut", False) and sp.obj_bound is not None:
                if emit_cut(cut_dual_bound_optimality(C, sp.obj_bound, ms, fwd, inst), "dual_bound"):
                    cuts_added += 1
            streak += 1
            rec.update(cuts=cuts_added, LB=LB, streak=streak,
                       t_elapsed=time.time() - t0)  # #6: LB'nin iyileştiği gerçek zaman
            emit(rec)
            if streak > cfg.max_inconclusive and cuts_added == 0:
                status = "INCONCLUSIVE"; break
            continue

        # TIME_LIMIT_NO_INCUMBENT / NUMERICAL_FAILURE: LB yok, kesme yok
        streak += 1
        rec.update(cuts=0, streak=streak); emit(rec)
        if streak > cfg.max_inconclusive:
            status = "INCONCLUSIVE"; break
        if cuts_added == 0:
            status = "INCONCLUSIVE"; break

    runtime = time.time() - t0
    diag["cpu_total_s"] = time.process_time() - cpu0
    diag["single_sp_distinct"] = len(sc_distinct)
    diag["sp_sig_distinct"] = len(sig_calls)
    diag["dead_fact_distinct"] = len(fact_seen)
    diag["dead_fact_max_repeat"] = max(fact_seen.values()) if fact_seen else 0
    diag["sp_sig_max_calls"] = max(sig_calls.values()) if sig_calls else 0
    diag["sp_sig_max_time_s"] = max(sig_time.values()) if sig_time else 0.0
    return LBBDResult(
        status=status, LB=LB, UB=UB, gap=_gap(LB, UB), incumbent=incumbent,
        iterations=it, sp_calls=sp_calls, cut_counts=cut_counts, runtime=runtime,
        iteration_log=log, cut_records=list(master.cut_records), diag_stats=diag,
    )
