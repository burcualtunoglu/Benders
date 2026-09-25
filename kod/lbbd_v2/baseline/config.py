"""config.py — BaselineConfig (minimal-kesin çekirdek) + ablasyon FAKTÖR registry.

BaselineConfig, mevcut ``lbbd_v2.config.Config``'ten TÜRETİLİR; böylece yeniden kullanılan modüller
(preprocessing, güçlü SP, cuts, heuristics) hiçbir değişiklik olmadan aynı öznitelikleri okur.
Fark: TÜM performans-only bayraklar VARSAYILAN olarak KAPALIDIR — yani ``BaselineConfig()`` doğrudan
minimal-kesin çekirdektir (docs/asama0_rapor.md §2). K1–K4 model kararları (26'/28'/midpoint/
cell_specific) düzeltilmiş modelin TANIMIdır, ablasyon faktörü değildir; kanonik kalırlar.

Bir "varyant" = baseline + tek bir faktör AÇIK. Faktörler ``FACTORS`` registry'sinde; ``make_config``
bir faktör listesini uygulayarak varyant üretir. Kod tekrarı yok: her varyant tek bir Config nesnesidir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from lbbd_v2.config import Config


# Alt problem motoru etiketleri (baseline.subproblem._select tarafından tanınır)
SP_STRONG = "strong_resource"   # Big-M'siz güçlü SP (Lemma A) — onaylı baseline referansı
SP_WEAK = "weak_resource"       # Big-M'li zayıf SP — düzeltilmiş modelin doğrudan kısıtlaması
SP_GROUPAGG = "group_agg"       # 2026-09-24 Geliştirme 1: grup-toplulaştırılmış güçlü SP (eşdeğer gösterim)


@dataclass
class BaselineConfig(Config):
    # ---- K1–K4: MODEL tanımı (ablasyon değil; kanonik/sabit) --------------
    # constraint_26_variant="26prime", use_28prime=True, omega_mode="midpoint",
    # big_m_mode="cell_specific" — Config varsayılanlarından miras; değiştirilmez.

    # ---- Minimal-kesin çekirdek: TÜM performans-only bayraklar KAPALI -----
    enable_C0: bool = False
    enable_C0prime: bool = False
    enable_C3: bool = False
    enable_C2: bool = False
    enable_flow_connectivity: bool = False
    enable_master_time_consistency: bool = False

    sp_engine: str = SP_STRONG          # onaylı baseline: güçlü SP açık (karar #1)
    sp_symmetry_break: bool = False     # F-S11 kapalı
    sp_warm_start: bool = False         # F-WARM kapalı

    use_percell_optimality_cuts: bool = False   # F-K5 kapalı
    use_dual_optimality_cut: bool = False        # F-K7 kapalı

    # ---- baseline'a özgü YENİ bayraklar (mevcut Config'te yoktu) ---------
    use_singleton_prescreen: bool = False   # F-PRE: tek-hücre fizibilite ön-elemesi (K3 üretir)
    use_deletion_filter: bool = False       # F-FILT: çakışma çıkarımı (Kesme~4'ü güçlendirir)
    tighten_md: bool = False                # F-MD: sıkılaştırılmış M_d (kapalı => gevşek küresel-toplam)
    use_lb_heuristics: bool = False         # F-SEED: LB-sezgisel tohumları (containment/greedy_fast)

    # LB-sezgisel tohumu ana bayrağa bağlanır (mevcut Config alanı; tutarlılık için kapat)
    lbbd_fast_greedy_seed: bool = False
    lbbd_localsearch_seed: bool = False     # F-SEEDLS: açgözlü + yerel arama tohumu (kapalı)
    mono_warm_start: bool = False           # yalnız monolitik; LBBD baseline'da gereksiz

    # ---- 2026-09-24 yerel pilot: tekrarlayan-imza SP ÇAĞRI politikası (yalnız alt problem çağrısı;
    # model, başlangıç ana problemi ve kesme aileleri AYNI). Aynı TAM SP girdisi (C, rejim, zamanlar)
    # tekrar geldiğinde: kanıtlı ise saklanan sonucu yeniden kullan; değilse kalan TOPLAM bütçe içinde
    # süreyi ×sp_recur_growth büyüt ve önceki SP inkümbentini MIP start ver (en çok
    # sp_recur_max_escalations kez); sonra yalnız yeniden kullan. Saklanan en iyi fizibil ObjVal ve en
    # sıkı geçerli ObjBound hiçbir çağrıda kaybolmaz. Kapalı (False) = mevcut çağrı politikası.
    # ---- 2026-09-24 Geliştirme 2: Δ-tabanlı (yol-bağımsız) tekil fizibilsizlik kesmeleri (faktör SDC, PRE'ye bağımlı).
    # Tekil-SP (i, r) Δ̄'da kanıtlı fizibilsizse her öncül a ∈ N(i), α/λ_a ≤ Δ̄ için u^r_i + q_ai ≤ 1 (Lemma F-APRI +
    # Δ_i ≤ α/λ_a). Mevcut noktayı kesen öncül kesmesi yoksa klasik imza kesmesine (R({i}) ≥ 1) geri düşülür.
    # sdc_apriori: döngü öncesi Δ^max_i = max_{a∈N(i)} α/λ_a (kökte 0) ile tekil tarama; kanıtlı ⇒ u^r_i = 0.
    use_singleton_delta_cuts: bool = False
    sdc_apriori: bool = True
    sdc_probe_eta: float = 1e-3     # SDC-t: fizibilsizlik eşiğini Δ̄+η'da yeniden kanıtla (τ ≥ η ⇒ eşitlik kesilir)
    sdc_timed: bool = True          # SDC-t: MTC t^s varsa zaman koşullu kesme (ε yardımcı ikilileri); yoksa sözdizimsel kesme
    sdc_apriori_budget: float = 60.0
    sp_recur_policy: bool = False
    sp_recur_max_escalations: int = 2
    sp_recur_growth: float = 2.0

    # ---- v2 iyileştirmeleri (2026-09-19; docs/lbbd_v2_iyilestirme.md) ----------------------
    # F-REPAIR: tekil-infizibil hücreler yüzünden elenen ana-problem adayını ATMAK yerine o hücreleri
    # düşüp kalan (fizibil) kümeyle ÇOK-HÜCRE SP'yi çöz -> her yinelemede doğrulanmış fizibil çözüm
    # (YALNIZ LB; ne kesme ne UB; kesinlik sözleşmesi aynı). PRE'ye bağımlı (dead kümesi oradan).
    use_candidate_repair: bool = False
    repair_budget: float = 60.0             # onarım SP'si başına tavan (s); kalan TOPLAM ile sınırlı
    # F-APRI: a-priori tekil-infizibilite. Δ_i := t^s_i − t^{s,min}_i için tekil-SP fizibilitesi Δ'da
    # MONOTON ARTANDIR (Lemma: Δ büyüdükçe v_min küçülür => D_ik, ω_i küçülür, servis penceresi
    # büyür; diğer kısıtlar Δ'dan bağımsız). Dolayısıyla EN GEVŞEK Δ_max = ts_ub'de INFEASIBLE_PROVEN
    # => her fizibil çözümde u^r_i = 0 (döngü ÖNCESİ sabitlenir). Ayrıca döngü içinde Δ_i ≤ Δ_inf(i,r)
    # (daha önce kanıtlanmış infizibil eşik) => çözmeden 'dead' (Δ-baskınlık; aynı lemma).
    use_apriori_singleton: bool = False
    apriori_budget: float = 300.0           # a-priori tarama tavanı (s); TOPLAM içinde, aşılırsa kalan hücreler taranmaz
    # F-BCH (2026-09-23): branch-and-check (Thorsteinsson 2001; Beck 2010) — KARMA düzen. Ana problem
    # FORMÜLASYONU ve kesme aileleri DEĞİŞMEZ; yalnız arama düzeni: master'ın dal-sınır ağacında her
    # tamsayı inkümbent (MIPSOL callback) ileri geçiş + tekil ön-elemeden geçer ve ihlal varsa ilgili
    # ÇEKİRDEK kesme (bağlantılılık / yayılım / tekil-infizibilite, Kesme 1) LAZY olarak eklenir
    # (cbLazy) => master her yinelemede sıfırdan çözülmez, yineleme başına onlarca kesme. Birleşik SP,
    # deletion filter ve optimallik kesmeleri (K5/K6/K7) DIŞ döngüde kalır (Beck 2010: pahalı SP
    # klasik döngüde). Lazy kesmeler çözüm sonunda kalıcı kısıt olarak da eklenir (dedup ile).
    # UB = master ObjBound (lazy kesmeler geçerli => sınır geçerli); LB yalnız doğrulanmış çözümden.
    use_branch_and_check: bool = False
    bch_sp_escalations: int = 2             # BCH: doğrulanmamış adayda SP doğrulamasını sürdürme sayısı (bütçe ×2, sıkı tolerans)

    # ------------------------------------------------------------------
    def validate(self) -> None:
        assert self.constraint_26_variant in ("26prime", "26plain"), self.constraint_26_variant
        assert self.omega_mode in ("midpoint", "worstcase"), self.omega_mode
        assert self.big_m_mode == "cell_specific", self.big_m_mode
        assert self.big_m_margin >= 1.0, self.big_m_margin
        assert self.sp_engine in (SP_STRONG, SP_WEAK, SP_GROUPAGG), (
            f"unknown sp_engine {self.sp_engine!r}; expected {SP_STRONG!r}, {SP_WEAK!r} or {SP_GROUPAGG!r}")
        assert not self.enable_C1, "enable_C1 NOT implemented"
        assert not self.allow_unserved_post, "allow_unserved_post NOT implemented"
        # K5, tek-hücre p_solo'ya bağımlıdır -> PRE olmadan K5 tutarsız olur (rapor §5.6).
        if self.use_candidate_repair and not self.use_singleton_prescreen:
            raise ValueError("use_candidate_repair=True requires use_singleton_prescreen=True "
                             "(FACTORS['REPAIR'] kurar)")
        if self.use_apriori_singleton and not self.use_singleton_prescreen:
            raise ValueError("use_apriori_singleton=True requires use_singleton_prescreen=True "
                             "(FACTORS['APRI'] kurar)")
        if self.use_branch_and_check and not self.use_singleton_prescreen:
            raise ValueError("use_branch_and_check=True requires use_singleton_prescreen=True "
                             "(FACTORS['BCH'] kurar)")
        if self.use_percell_optimality_cuts and not self.use_singleton_prescreen:
            raise ValueError(
                "use_percell_optimality_cuts=True, use_singleton_prescreen=False: hücre "
                "optimalite kesmesi (K5) tek-hücre p_solo'ya bağımlı; PRE de açık olmalı "
                "(FACTORS['K5'] bu bağımlılığı otomatik kurar).")

    def profile_name(self) -> str:
        on = self.active_factors()
        return "BASELINE" if not on else "BASELINE+" + "+".join(on)

    def active_factors(self) -> List[str]:
        """Baseline'a göre AÇIK olan faktörlerin adları (etiketleme/CSV için)."""
        base = BaselineConfig()
        out: List[str] = []
        for name, overrides in FACTORS.items():
            if all(getattr(self, k) == v for k, v in overrides.items()) and \
               any(getattr(base, k) != v for k, v in overrides.items()):
                out.append(name)
        return out


# =====================================================================
# FAKTÖR registry — her giriş, baseline üzerine uygulanacak TAM bayrak kümesi.
# Bağımlılıklar giriş içinde AÇIKÇA ifade edilir (rapor §5.6): ör. K5 -> PRE de açar.
# =====================================================================
FACTORS: Dict[str, Dict[str, Any]] = {
    # --- Ana problem gevşetme sıkılaştırmaları ---
    "C0":    {"enable_C0": True},
    "C0p":   {"enable_C0prime": True},
    "C3":    {"enable_C3": True},
    "C2":    {"enable_C2": True},        # Σ_i(u_pre+u_post) ≤ |K| (geçerli; ana problemi sıkılaştırır)
    "FLOW":  {"enable_flow_connectivity": True},
    "MTC":   {"enable_master_time_consistency": True},
    # --- Alt problem ---
    "SPWEAK": {"sp_engine": SP_WEAK},           # baseline strong => bu varyant zayıf SP
    "GAGG":   {"sp_engine": SP_GROUPAGG},       # 2026-09-24: grup-toplulaştırılmış SP (Geliştirme 1; üretim dışı)
    "S11":   {"sp_symmetry_break": True},
    "WARM":  {"sp_warm_start": True},
    # --- Kesme aileleri / ön-elemeler ---
    "PRE":   {"use_singleton_prescreen": True},
    "K5":    {"use_percell_optimality_cuts": True, "use_singleton_prescreen": True},  # K5 PRE'ye bağımlı
    "K7":    {"use_dual_optimality_cut": True},
    "FILT":  {"use_deletion_filter": True},
    # --- Ön işleme / primal ---
    "MD":    {"tighten_md": True},
    "SEED":  {"use_lb_heuristics": True, "lbbd_fast_greedy_seed": True},
    # SEEDLS = SEED + geliştirilmiş açgözlü/yerel-arama tohumu (ek aday; tek-değişkenli deney).
    # use_lb_heuristics ŞART (solver başlangıç-incumbent döngüsü yalnız o açıkken koşar).
    "SEEDLS": {"use_lb_heuristics": True, "lbbd_fast_greedy_seed": True,
               "lbbd_localsearch_seed": True},
    # --- v2 (2026-09-19): aday onarımı (LB) + a-priori tekil-infizibilite (Δ-monoton lemma) ---
    "REPAIR": {"use_candidate_repair": True, "use_singleton_prescreen": True},   # PRE'ye bağımlı
    "APRI":   {"use_apriori_singleton": True, "use_singleton_prescreen": True},  # PRE'ye bağımlı
    # --- 2026-09-23: branch-and-check (arama düzeni; formülasyon/kesme aileleri aynı) ---
    "BCH":    {"use_branch_and_check": True, "use_singleton_prescreen": True},   # PRE'ye bağımlı
    # --- 2026-09-24 Geliştirme 2: Δ-tabanlı tekil fizibilsizlik kesmeleri (kesme ailesi; kanıt docs/singleton_delta_cuts.md) ---
    "SDC":    {"use_singleton_delta_cuts": True, "use_singleton_prescreen": True}, # PRE'ye bağımlı
}


def make_config(factors: Iterable[str] = (), **overrides: Any) -> BaselineConfig:
    """Baseline + verilen faktörler AÇIK bir BaselineConfig üretir.

    ``factors`` FACTORS anahtarlarından oluşur; her biri kendi bayrak kümesini uygular.
    ``overrides`` (ör. total_budget=..., gurobi_threads=...) doğrudan set edilir.
    Bilinmeyen faktör adı hata verir (sessiz yanlış varyant üretmemek için).
    """
    cfg = BaselineConfig()
    for f in factors:
        if f not in FACTORS:
            raise KeyError(f"unknown factor {f!r}; known: {sorted(FACTORS)}")
        for k, v in FACTORS[f].items():
            setattr(cfg, k, v)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config field {k!r}")
        setattr(cfg, k, v)
    cfg.validate()
    return cfg


def baseline_config(**overrides: Any) -> BaselineConfig:
    """Saf minimal-kesin çekirdek (hiç faktör açık değil)."""
    return make_config((), **overrides)


def all_single_factor_variants(**overrides: Any) -> Dict[str, BaselineConfig]:
    """{faktör_adı: baseline+o_faktör} — Aşama 2 tek-tek ablasyon matrisi için."""
    out = {"BASELINE": baseline_config(**overrides)}
    for f in FACTORS:
        out[f] = make_config([f], **overrides)
    return out
