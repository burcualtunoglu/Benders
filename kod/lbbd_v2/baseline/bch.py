"""bch.py — F-BCH: branch-and-check denetleyicisi (2026-09-23; rev. 2, kullanıcı maddeleri 1–7).

Tam branch-and-check: ana problemin dal-sınır ağacındaki HER tamsayı inkümbent (MIPSOL) callback içinde
klasik LBBD yinelemesinin doğrulama zinciriyle AYNI şekilde işlenir:
    ileri geçiş -> Kontrol A/B -> tekil ön-eleme -> (K5) hücre ödül vekili -> BİRLEŞİK SP -> ödül vekili
Sabit yapıdaki alt problem için L_SP = ObjVal (doğrulanmış fizibil), U_SP = ObjBound (geçerli üst sınır),
Φ* = SP optimumu, L_SP ≤ Φ* ≤ U_SP; R = Σ_{i∈C} ρ_i adayın ödül vekili. Karar:
    R ≤ L_SP        => KABUL (aynı yapıda doğrulanmış tam çözüm adayın ödülünü karşılar; SP optimalliği şart değil).
    R > U_SP        => imzaya bağlı, adayı gerçekten ihlal eden optimalite kesmesi (Φ = U_SP; OPTIMAL'de
                       toplulaştırılmış kesme, aksi hâlde K7 dual-bound kesmesi).
    L_SP < R ≤ U_SP => DOĞRULANMAMIŞ. Mevcut ailelerden ihlal edilen geçerli kesme yoksa kalan toplam bütçede SP
                       doğrulaması SÜRDÜRÜLÜR (bütçe ×2, sıkı tolerans, ılık başlangıç; en çok
                       cfg.bch_sp_escalations kez); yine sonuçlanmazsa GÜVENLİ DURDURMA. Sessiz kabul YOK.
SP'nin fizibil çözümü (tahsis SP'de, zamanlar ileri geçişte doğrulanmış) dış LB'yi gerçek amaç değeriyle günceller;
bu, adayın kabulünden AYRI bir işlemdir. Fizibilite kesmeleri yalnız INFEASIBLE_PROVEN'dan; optimalite kesmeleri yalnız
geçerli üst sınırdan; sezgisel amaç değeri asla üst sınır olarak kullanılmaz.
Güvenli durdurma: `model.terminate()` + `state["inconclusive"]`. terminate()'in etkisi VARSAYILMAZ: durdurulan
adayın Gurobi iç inkümbenti olabileceği (test_bch_solver_behaviour) bilinir; dış döngü durdurulmuş master'ın
çözümünü KULLANMAZ (LB'ye almaz, sertifika ilan etmez), yalnız ObjBound'u geçerli UB olarak alır (gerekçe:
ObjBound, lazy kesmelerle güçlendirilmiş geçerli gevşetmenin dal-sınır dual sınırıdır; iç inkümbent master-fizibil
olduğundan onunla budanan düğümler master optimumunu içeremez => ObjBound ≥ master-opt ≥ OPT).
Formülasyon, kesme aileleri, tohum, toleranslar ve bütçeler DEĞİŞMEZ; callback süreleri toplam bütçeye dahildir.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from gurobipy import GRB

from lbbd_v2.benders_forward import forward_pass, FWD_UNREACHABLE, FWD_TIME_INCONSISTENT
from lbbd_v2.lbbd_subproblem_resource import SPStatus
from lbbd_v2.lbbd_cuts import (
    cut_connectivity, cut_propagation, cut_singleton_infeasible, cut_conflict_infeasible,
    cut_cell_optimality, cut_aggregate_optimality, cut_dual_bound_optimality)

CutList = List[Tuple[Any, str]]


class BchController:
    def __init__(self, inst, pre, cfg, master, *, sp_solve: Callable, screen: Callable,
                 regime_of: Callable, pi_free: Callable, sp_full_sig: Callable, t0: float,
                 diag: Dict[str, Any], count_sp: Callable[[str], None],
                 on_validated: Callable, extract_conflict: Optional[Callable] = None,
                 warm_x: Optional[Callable] = None, need_single: bool = True):
        self.inst, self.pre, self.cfg, self.master = inst, pre, cfg, master
        self.sp_solve, self.screen, self.regime_of = sp_solve, screen, regime_of
        self.pi_free, self.sp_full_sig, self.t0 = pi_free, sp_full_sig, t0
        self.diag, self.count_sp, self.on_validated = diag, count_sp, on_validated
        self.extract_conflict, self.warm_x, self.need_single = extract_conflict, warm_x, need_single
        # tam SP imzası -> {"sp": SPResult|None, "attempts": n, "budget": s}  (doğrulama durumu, iterasyonlar arası)
        self.sp_cache: Dict[Any, Dict[str, Any]] = {}
        self._sig_counts: Dict[Any, int] = {}
        self.state: Dict[str, Any] = {}
        diag.update({"bch_enabled": 1, "bch_master_solves": 0, "bch_restarts": 0, "bch_callbacks": 0,
                     "bch_sp_time_s": 0.0, "bch_sig_repeats": {}, "bch_sig_max_repeat": 0,
                     "bch_lazy_cuts": 0, "bch_lazy_by_kind": {}, "bch_accepted": 0,
                     "bch_accepted_by_optimal": 0, "bch_accepted_by_incumbent": 0,
                     "bch_accepted_unproven": 0,           # tasarım gereği HER ZAMAN 0 (tek başına kanıt değil)
                     "bch_sp_calls": 0, "bch_sp_status": {}, "bch_sp_escalations": 0,
                     "bch_sp_cache_hits": 0, "bch_unproven_events": 0, "bch_persisted_cuts": 0,
                     "bch_cb_time_s": 0.0, "bch_inconclusive_stops": 0, "bch_stop_reasons": {},
                     "bch_budget_stops": 0, "bch_callback_errors": 0})
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.state = {"pending": [], "n_cb": 0, "n_lazy": 0, "n_accepted": 0, "n_sp": 0,
                      "n_unproven": 0, "inconclusive": None,
                      "last_cb_bound": None,      # durdurma ÖNCESİ callback'te okunan son küresel sınır (MIPSOL_OBJBND)
                      "stopped_obj": None}        # durdurulan adayın master amaç değeri (YALNIZ raporlama; UB/LB için KULLANILMAZ)

    def remaining(self) -> float:
        return self.cfg.total_budget - (time.time() - self.t0)

    def _stop(self, model, reason: str) -> None:
        """Güvenli durdurma: geçerli kesme üretilemedi ve doğrulama tamamlanamadı => master durdurulur;
        aday KABUL EDİLMEZ (LB'ye alınmaz, sertifika ilan edilmez)."""
        self.state["inconclusive"] = reason
        self.diag["bch_inconclusive_stops"] += 1
        self.diag["bch_stop_reasons"][reason] = self.diag["bch_stop_reasons"].get(reason, 0) + 1
        if reason.startswith("no_time"):
            self.diag["bch_budget_stops"] += 1
        if model is not None:
            model.terminate()

    # ------------------------------------------------------------------
    def callback(self, model, where) -> None:
        if where != GRB.Callback.MIPSOL:
            return
        _tc = time.perf_counter()
        self.state["n_cb"] += 1
        self.diag["bch_callbacks"] += 1
        try:
            if self.state["inconclusive"] is not None:
                model.terminate(); return          # durdurma istendi; yeni aday işleme
            # durdurma ÖNCESİ küresel sınırı kaydet (MIPSOL_OBJBND): geçerli gevşetmenin o anki dal-sınır dual
            # sınırı; durdurulan adayın amaç değeri DEĞİL. Güvenli durdurmada UB kaynağı olarak kullanılabilir.
            b = self.master.cb_bound(model)
            if b is not None:
                lb_ = self.state["last_cb_bound"]
                self.state["last_cb_bound"] = b if lb_ is None else min(lb_, b)
            try:
                self.state["stopped_obj"] = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
            except Exception:
                self.state["stopped_obj"] = None
            if self.remaining() <= 1.0:
                self._stop(model, "no_time"); return
            try:
                ms = self.master.cb_solution(model)
                cuts = self.check(ms, model)
            except Exception as exc:                      # doğrulama tamamlanamadı => sessizce KABUL ETME
                self.diag["bch_callback_errors"] += 1
                self._stop(model, "cb_error:" + type(exc).__name__)
                return
            self.apply(model, cuts)
        finally:
            self.diag["bch_cb_time_s"] += time.perf_counter() - _tc

    def apply(self, model, cuts: CutList) -> None:
        """Kesmeleri cbLazy ile ekle (DEDUP YOK: ihlal edilen kesme her zaman uygulanır; kalıcılaştırmadaki
        dedup ayrı) ve kalıcılaştırma listesine al. Kesme yoksa ve durdurma yoksa aday KABUL edilmiştir."""
        if not cuts:
            if self.state["inconclusive"] is None:
                self.state["n_accepted"] += 1
                self.diag["bch_accepted"] += 1
            return
        for cut, kind in cuts:
            expr = cut.to_linexpr(self.master.vars)
            if cut.sense == "<=":
                model.cbLazy(expr <= cut.rhs)
            elif cut.sense == ">=":
                model.cbLazy(expr >= cut.rhs)
            else:
                model.cbLazy(expr == cut.rhs)
            self.state["n_lazy"] += 1
            self.diag["bch_lazy_cuts"] += 1
            self.diag["bch_lazy_by_kind"][kind] = self.diag["bch_lazy_by_kind"].get(kind, 0) + 1
            self.state["pending"].append((cut, kind))

    # ------------------------------------------------------------------
    def _solve_sp(self, ent: Dict[str, Any], C, regime, fwd) -> Any:
        inst, pre, cfg = self.inst, self.pre, self.cfg
        remaining = self.remaining()
        budget = max(1.0, min(ent["budget"], remaining))
        gap = None if ent["attempts"] == 0 else getattr(cfg, "sp_tight_mip_gap", 0.0)
        wx = None
        prev = ent.get("prev")
        if prev is not None and isinstance(getattr(prev, "solution", None), dict) and prev.solution.get("x"):
            wx = prev.solution["x"]
        elif self.warm_x is not None:
            wx = self.warm_x(C, regime, fwd)
        if wx is not None:
            sp = self.sp_solve(fwd, C, regime, inst, pre, cfg, time_budget=budget,
                               warm_x=wx, symmetry_break=False, mip_gap=gap)
        else:
            sp = self.sp_solve(fwd, C, regime, inst, pre, cfg, time_budget=budget, mip_gap=gap)
        st = sp.status.value if hasattr(sp.status, "value") else str(sp.status)
        self.count_sp(st)
        self.state["n_sp"] += 1
        self.diag["bch_sp_calls"] += 1
        self.diag["bch_sp_time_s"] += float(getattr(sp, "runtime", 0.0) or 0.0)
        self.diag["bch_sp_status"][st] = self.diag["bch_sp_status"].get(st, 0) + 1
        if ent["attempts"] > 0:
            self.diag["bch_sp_escalations"] += 1
        ent["attempts"] += 1
        ent["sp"] = sp
        ent["budget"] = budget
        return sp

    def check(self, ms, model=None) -> CutList:
        """Adayı doğrula; reddedilecekse kesme listesi (boş liste = kabul VEYA güvenli durdurma; ayrım
        state['inconclusive'] ile)."""
        inst, cfg = self.inst, self.cfg
        fwd = forward_pass(ms, inst)
        if fwd.status == FWD_UNREACHABLE:
            return [(cut_connectivity(fwd.unreachable, j, inst), "connectivity")
                    for j in sorted(fwd.unreachable)]
        if fwd.status == FWD_TIME_INCONSISTENT:
            return [(cut_propagation(i, j, fwd), "propagation") for (i, j) in fwd.violating_arcs]

        C = [i for i in inst.Nf if ms.u(i) > 0.5]
        if not C:
            # kontrolsüz yapı: telafi 0, Σρ ≤ π·u = 0 => doğrulanmış (recourse'suz) tam çözüm
            self.on_validated(self.pi_free(ms.y), [], {}, {}, ms, "bch_empty")
            return []
        regime = self.regime_of(ms, C)
        p_solo: Dict[int, float] = {}
        dead: List[int] = []
        if self.need_single:
            p_solo, dead = self.screen(C, regime, fwd)
        if dead:
            return [(cut_singleton_infeasible(i, ms, fwd, inst), "singleton_infeasible") for i in dead]

        # K5: hücre ödül vekili kanıtlı üst sınırı (tekil-SP optimumu p_solo) aşıyorsa reddet — SP'ye gerek yok
        cuts: CutList = []
        if getattr(cfg, "use_percell_optimality_cuts", False):
            for i in C:
                if i in p_solo and ms.rho.get(i, 0.0) > p_solo[i] + cfg.eps_abs:
                    cuts.append((cut_cell_optimality(i, p_solo[i], ms, fwd, inst), "cell_optimality"))
        if cuts:
            return cuts

        # ---- BİRLEŞİK SP: (L_SP, U_SP, R) sınıflandırması, doğrulamayı sürdürme, güvenli durdurma ----
        R = sum(ms.rho.get(i, 0.0) for i in C)
        sig = self.sp_full_sig(C, regime, fwd)
        # aynı tam SP imzasının tekrar sayısı (raporlama: BCH aynı adayı doğrulamaya çalışırken bütçe tüketiyor mu?)
        rep = self._sig_counts
        rep[sig] = rep.get(sig, 0) + 1
        self.diag["bch_sig_max_repeat"] = max(self.diag["bch_sig_max_repeat"], rep[sig])
        # histogram {tekrar sayısı: imza sayısı} (imza başına sözlük runner'da binlerce metrik anahtarına dönüşüyordu)
        hist: Dict[str, int] = {}
        for c in rep.values():
            hist[str(c)] = hist.get(str(c), 0) + 1
        self.diag["bch_sig_repeats"] = hist
        ent = self.sp_cache.get(sig)
        if ent is None:
            ent = {"sp": None, "attempts": 0, "budget": min(cfg.sp_budget, max(1.0, self.remaining())),
                   "prev": None}
            self.sp_cache[sig] = ent
        max_att = 1 + int(getattr(cfg, "bch_sp_escalations", 2))
        while True:
            if ent["sp"] is None:
                if self.remaining() <= 1.0:
                    self._stop(model, "no_time"); return []
                sp = self._solve_sp(ent, C, regime, fwd)
            else:
                sp = ent["sp"]                             # aynı TAM girdi: kanıtlı sonuç yeniden kullanılır
                self.diag["bch_sp_cache_hits"] += 1
            st = sp.status.value if hasattr(sp.status, "value") else str(sp.status)

            if sp.status == SPStatus.INFEASIBLE_PROVEN:
                Cstar = set(C)
                if getattr(cfg, "use_deletion_filter", False) and self.extract_conflict is not None:
                    Cstar = self.extract_conflict(
                        C, regime, fwd, inst, self.pre, cfg, sp_solve=self.sp_solve,
                        time_budget=max(1.0, min(cfg.filter_budget, self.remaining())))
                return [(cut_conflict_infeasible(Cstar, ms, fwd, inst), "conflict_infeasible")]

            L_sp = sp.obj if isinstance(sp.obj, (int, float)) else None
            U_sp = sp.obj_bound if isinstance(sp.obj_bound, (int, float)) else None
            if L_sp is not None:
                # (4) SP'nin fizibil çözümü özgün modelin tüm kısıtlarını sağlar (zamanlar ileri geçişte, tahsis
                # SP'de doğrulandı) => dış LB gerçek amaç değeriyle güncellenebilir. Adayın kabulünden AYRIDIR.
                self.on_validated(self.pi_free(ms.y) + L_sp, C, regime, sp.solution or {}, ms,
                                  "bch" if sp.status == SPStatus.OPTIMAL else "bch_incumbent")
            if L_sp is not None and R <= L_sp + cfg.eps_abs:
                key = "bch_accepted_by_optimal" if sp.status == SPStatus.OPTIMAL else "bch_accepted_by_incumbent"
                self.diag[key] += 1
                return []                                   # (1) KABUL: R ≤ L_SP ≤ Φ*
            if U_sp is not None and R > U_sp + cfg.eps_abs:
                # (2) geçerli üst sınırla, imzaya bağlı, adayı ihlal eden optimalite kesmesi
                if sp.status == SPStatus.OPTIMAL:
                    return [(cut_aggregate_optimality(C, U_sp, ms, fwd, inst), "aggregate_optimality")]
                if getattr(cfg, "use_dual_optimality_cut", False):
                    return [(cut_dual_bound_optimality(C, U_sp, ms, fwd, inst), "dual_bound")]
                # K7 ailesi kapalı => bu aile kullanılamaz; doğrulanmamış gibi sürdür
            # (3) L_SP < R ≤ U_SP (ya da sınır yok): DOĞRULANMAMIŞ. K5 zaten denetlendi; başka aile yok.
            self.state["n_unproven"] += 1
            self.diag["bch_unproven_events"] += 1
            if (ent["attempts"] >= max_att or self.remaining() <= 2.0
                    or sp.status == SPStatus.NUMERICAL_FAILURE):
                self._stop(model, "sp_unproven:" + st); return []
            ent["prev"] = sp
            ent["sp"] = None                                # doğrulamayı sürdür: bütçe ×2, sıkı tolerans, ılık başlangıç
            ent["budget"] = max(1.0, min(2.0 * ent["budget"], self.remaining() - 1.0))
