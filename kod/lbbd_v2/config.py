"""config.py — central configuration for the exact lbbd_v2 solver.

All K1–K4 canonical decisions live here as runtime flags so both variants of every
decision can be exercised in tests. Defaults follow the APPROVED canonical config
(PROMPT §2, which sets midpoint as the default ω-mode):

    K1 = (26') dispatch trigger           -> constraint_26_variant = "26prime"
    K2 = (28') selector restriction       -> use_28prime = True
    K3 = midpoint default (+worstcase)    -> omega_mode = "midpoint"
    K4 = cell-specific M^s_i (derived)    -> big_m_mode = "cell_specific"

The faulty scalar water Big-M of the old baseline is intentionally UNREPRESENTABLE in
this production path: `big_m_mode` accepts only "cell_specific". The documented legacy
counterexample (validation-plan Test 8a) is reproduced exclusively in test code, so the
literal legacy scalar never appears in any production module.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


# Fixed physical constants of the formulation (parameters, not tunables).
DELTA_WAT: float = 500.0   # Δ_wat — base water per controlled cell
DELTA_BUF: float = 5.0     # Δ_buf — dispatch buffer after detection


@dataclass
class Config:
    # ---- K1–K4 canonical decisions (APPROVED defaults) -------------------
    constraint_26_variant: str = "26prime"   # "26prime" (K1=yes) | "26plain"
    use_28prime: bool = True                 # K2
    omega_mode: str = "midpoint"             # K3: "midpoint" | "worstcase"
    big_m_mode: str = "cell_specific"        # K4: only "cell_specific" in v2

    # ---- Big-M çarpanı (NİHAİ VARSAYILAN 1.00 = pay YOK) ------------------
    # M_d = te_UB+Δ_buf+max d_ik, M_i = a_i·burn_i, M_s = M_i+Δ_wat türetimleri kullanım yerlerinde
    # GEÇERLİ üst sınırlardır; ek payın GEREKLİ olduğuna dair bir dayanak gösterilmemiştir (18/%2
    # deneyi: pay LBBD performansını/geçerliliğini değiştirmedi; M_s≥sup 1.00'da eşitlikle sağlanır).
    # Bu yüzden nihai varsayılan 1.00'dır. YALNIZ M_d ve M_i'ye BİR KEZ uygulanır; M_s türetilmiş
    # olduğundan pay İKİNCİ KEZ uygulanmaz. Duyarlılık için 1.02 hâlâ verilebilir (ör. --bigm-margin 1.02).
    big_m_margin: float = 1.00
    # Tam-yol (en-uzun-ateşleme-yolu) Held-Karp DP düğüm eşiği: |Nf| ≤ bu ise EXACT DP (sıkı t^s_UB),
    # aşarsa KÜRESEL TOPLAM fallback'i (geçerli ama gevşek). DP maliyeti O(2^n · n) — eşik bu maliyeti
    # sınırlar. Kontrollü eşik deneyi (MD16/MD18/MD20) bunu değiştirir. Varsayılan 18 (mevcut davranış).
    bigm_pathlen_max_nodes: int = 18

    # ---- Master relaxation strengthenings (baseline scope, 04 §15) -------
    enable_C0: bool = True        # controllable_i == False => u_i = 0
    enable_C0prime: bool = True   # pre_possible_i == False => u_pre_i = 0
    enable_C3: bool = True        # ρ_i <= (π_i − β_i Δ^min_i) u_i (roots tight)
    # NOT IMPLEMENTED: no C1 (fractional Hall matching) block exists in lbbd_master.py.
    # The flag is kept so the idea stays on record; validate() refuses True so that a
    # future experimenter cannot silently believe C1 is active.
    enable_C1: bool = False       # fractional Hall matching -- NOT IMPLEMENTED
    enable_C2: bool = False       # Σ u_i <= |K| (implemented in lbbd_master; vacuous)
    # static single-commodity flow connectivity in master (04 §4.4, PROVEN EXACT):
    # prevents floating q-cycles structurally instead of only via lazy (CC) cuts.
    # Lazy (CC) cut remains as a safety net either way.
    enable_flow_connectivity: bool = True
    # Master time-consistency layer (04 §11 / Theorem 2 constraints (44),(45),(46),(48) with t^s
    # variables). Reconstructs t^s inside the master and enforces the z-arc consistency (46) that
    # the forward pass otherwise checks lazily. VALID tightening (original constraints on t^s ⇒
    # UB stays valid) that makes the master propose only time-CONSISTENT (y,z,q), collapsing the
    # propagation-cut churn (9x9: ~112/114 iters were FWD_TIME_INCONSISTENT). Exact either way —
    # the lazy propagation cut remains the safety net.
    enable_master_time_consistency: bool = True

    # ---- Subproblem options (P2/P3) -------------------------------------
    # sp_engine: EXACTLY TWO values are recognised by lbbd_solver._select_sp:
    #   "strong_resource" (Big-M-free individual SP, Lemma A — DEFAULT, 4×4 cert V2)
    #   "typeagg"         (type-aggregated SP; OUT OF SCOPE for the individual study)
    # Anything else used to fall through to the individual SP SILENTLY — in particular
    # the historical label "baseline_typeagg" selected the INDIVIDUAL engine, the exact
    # opposite of its name. validate() now rejects unknown values.
    sp_engine: str = "strong_resource"
    # Symmetry-breaking in the INDIVIDUAL per-resource SP: within each interchangeability group
    # (identical d_ik and µ_k) fix a canonical assignment (used vehicles = lowest-indexed members,
    # ordered by ascending cell rank). VALID (same-group vehicles are interchangeable → any
    # solution relabels to canonical with identical objective) and removes the vehicle-permutation
    # symmetry that makes the individual SP slow to PROVE optimal. Keeps x_ik individual (no
    # aggregation). Default ON; test code that fixes x to a non-canonical assignment turns it off.
    sp_symmetry_break: bool = True
    # Warm-start the multi-cell SP in the main loop with a closed-form assignment so it returns an
    # incumbent fast at scale (855 vehicles). Symmetry-break is turned off for the warm-started
    # call so the non-canonical start is accepted; exactness is unaffected (the SP's optimal Φ and
    # its dual bound are both symmetry-independent). SEED ONLY — never derives a bound or cut.
    sp_warm_start: bool = True
    # NOT IMPLEMENTED: the per-resource SP hard-codes  Σ_k x_ik ≥ 1  ("must_serve",
    # lbbd_subproblem_resource.py), so this flag has no effect whatsoever. Kept for the
    # record; validate() refuses True rather than letting it be silently ignored.
    allow_unserved_post: bool = False        # D10 baseline: False -- NOT IMPLEMENTED
    # Per-cell optimality cut generated ALWAYS (ACCELERATION, exact, DECISIVE for 4×4):
    #   ρ_i ≤ p_solo_i + π_i R({i})  — uses only the single-cell competition-free reward.
    use_percell_optimality_cuts: bool = True
    # Dual-bound optimality cut (ACCELERATION, exact safety net): on an SP timeout with an
    # incumbent, the SP's ObjBound is a VALID upper bound on Φ, so Σρ ≤ ObjBound + Θ R(C)
    # is a valid (weaker) cut. Using the incumbent would be INVALID; the dual bound is valid.
    use_dual_optimality_cut: bool = True

    # ---- Tolerances (06 §0) ---------------------------------------------
    eps_abs: float = 1e-6
    eps_rel: float = 1e-9
    compare_tol: float = 1e-4                 # cross-model numeric comparison
    # UNUSED: no module reads feas_tol. Solver feasibility tolerances are left at the
    # Gurobi defaults, except model_monolithic_bigmfix.py which hard-codes 1e-9 because
    # the enlarged deactivation constant a_i·M_d degrades numerical conditioning.
    feas_tol: float = 1e-6                    # UNUSED (kept for the record)

    # ---- Solver / reproducibility ---------------------------------------
    gurobi_seed: int = 0
    gurobi_threads: int = 1
    gurobi_output: bool = False
    mip_gap_reference: float = 0.0            # monolithic reference must prove optimality
    # ---- Alt-problem iç MIPGap (SADECE hız; KESİNLİĞİ ETKİLEMEZ) ----------
    # SP'yi bu bağıl gap'te durdurur. Varsayılan 0.0 = mevcut davranış (SP kendi optimumunu
    # kanıtlar). Pozitif değer SP'yi erken durdurur: dış LB yine DOĞRULANMIŞ fizibil ObjVal'den,
    # optimalite kesmesi yine GEÇERLİ ObjBound'dan (≥Φ, her toleransta geçerli) alınır -> kesinlik
    # korunur. Tek etki: kesme gevşer; dış aralık kapanmazsa ilerleme koruması INCONCLUSIVE üretir
    # (SP'nin OPTIMAL etiketi dış optimalliği İMA ETMEZ). Bkz. baseline/solver.py satır 250-273.
    sp_mip_gap: float = 0.0
    # ---- Uyarlamalı tolerans: ilerleme durunca tekrarlayan SP'yi SIKI çöz (SADECE hız/sertifika
    # denemesi; KESİNLİĞİ ETKİLEMEZ). Kapalıyken (varsayılan) mevcut davranış birebir korunur.
    # TAM SP girdisi (C + rejim + ileri-geçiş zamanları) tekrar geldiğinde, o girdi bir kez daha
    # `sp_tight_mip_gap` ile çözülür: doğrulanmış ObjVal LB'yi, geçerli ObjBound kesmeyi günceller;
    # ana problem yeniden çözülür (global UB = ana problem ObjBound). Sertifikayı GARANTİ ETMEZ
    # (ana problem başka karara geçebilir); yalnız kayıtlarda görülen tekrarı ele alır.
    sp_gap_tighten_on_stall: bool = False
    sp_tight_mip_gap: float = 0.0            # duraklamada kullanılacak sıkı tolerans (0.0 = tam kanıt)
    # ---- Koşullu tekil-SP sonucu önbelleği (reviewer (a)/(b)): SADECE tekrarlı hesabı azaltır;
    # KESİNLİĞİ/kesme geçerliliğini ETKİLEMEZ (kesme mevcut master imzasından kurulur, dedup ayrı).
    # Kontrollü açık/kapalı deneyi için bayrak. Varsayılan AÇIK.
    sp_singleton_cache: bool = True

    # ---- Primal warm start (SEED ONLY — never affects bounds / exactness) ----
    # Feed a closed-form greedy feasible solution (build_greedy_incumbent_fast) to Gurobi as a
    # MIPStart so the monolithic finds an incumbent immediately at scale. A rejected start is
    # harmless (Gurobi simply discards it); it can never remove the optimum or tighten a bound.
    mono_warm_start: bool = True
    # Gurobi feasibility emphasis for the monolithic (0 => leave solver default; 1 => MIPFocus=1,
    # find feasible solutions fast — recommended when no incumbent is the failure mode at scale).
    mono_mip_focus: int = 0
    # Seconds of NoRel heuristic before branch-and-bound (0 => off). Helps find a first incumbent
    # on very large instances. Pure primal heuristic; does not affect the proved bound.
    mono_no_rel_heur_time: float = 0.0
    # Also seed the fast greedy incumbent into the LBBD initial-incumbent candidates (validated by
    # the SP exactly like the other candidates before any LB is admitted).
    lbbd_fast_greedy_seed: bool = True
    # GELİŞTİRİLMİŞ tohum (lbbd_seed_localsearch): firebreak (ön-kontrol mühürleme) kümesi üzerinde
    # açgözlü + yerel arama. Yalnız BİR ADAY DAHA ekler; LB = adayların max'ı olduğundan mevcut
    # tohumdan asla daha kötü olamaz. SEED ONLY — solver yine forward_pass+SP ile doğrular; ne
    # kesme ne üst sınır üretir (kesinlik solver'ın mevcut doğrulama kapısına bağlı). Varsayılan
    # KAPALI (tek-değişkenli deney için bayrak).
    lbbd_localsearch_seed: bool = False
    # Yerel-arama tohumuna ayrılan AYRI süre bütçesi (saniye). Bu süre TOPLAM bütçeden düşülür
    # (initial-incumbent döngüsü t0'dan sonra sayılır, ana döngü kalan süreyle koşar). Reviewer
    # kararı: 120 s.
    seed_budget: float = 120.0

    # ---- Time budgets (seconds) -----------------------------------------
    total_budget: float = 3600.0
    master_budget: float = 600.0
    sp_budget: float = 300.0
    heur_budget: float = 30.0
    filter_budget: float = 60.0

    # ---- Loop control ----------------------------------------------------
    max_iterations: int = 1000
    max_inconclusive: int = 5
    # UNUSED: lbbd_solver never grows a budget. On an SP timeout the same sp_budget is
    # reused every iteration; escalating it is an open idea, not implemented behaviour.
    budget_growth: float = 2.0                # UNUSED (kept for the record)

    # ---- Derived constants (do not override) ----------------------------
    delta_wat: float = DELTA_WAT
    delta_buf: float = DELTA_BUF

    def validate(self) -> None:
        assert self.constraint_26_variant in ("26prime", "26plain"), self.constraint_26_variant
        assert self.omega_mode in ("midpoint", "worstcase"), self.omega_mode
        # v2 production path derives M^s per cell; the faulty scalar is not selectable here.
        assert self.big_m_mode == "cell_specific", (
            f"big_m_mode must be 'cell_specific' in lbbd_v2 (got {self.big_m_mode!r})")
        assert self.big_m_margin >= 1.0, self.big_m_margin
        # Only these two engine labels are recognised by lbbd_solver._select_sp; every
        # other string used to select the INDIVIDUAL engine silently.
        assert self.sp_engine in ("strong_resource", "typeagg"), (
            f"unknown sp_engine {self.sp_engine!r}; expected 'strong_resource' or "
            f"'typeagg' (an unrecognised value would silently run the individual SP)")
        # Flags that are declared but not implemented: fail loudly instead of pretending.
        assert not self.enable_C1, (
            "enable_C1=True but the C1 (fractional Hall matching) cut is NOT implemented "
            "in lbbd_master.py; setting it would have no effect")
        assert not self.allow_unserved_post, (
            "allow_unserved_post=True but the per-resource SP hard-codes must_serve "
            "(Σ_k x_ik ≥ 1); setting it would have no effect")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def profile_name(self) -> str:
        """Human-readable experimental profile label for result files.

        A run must not be labelled REFERENCE unless it really is the canonical
        configuration. The old version only tracked K1/K2/K3, so an ablation with
        (say) use_percell_optimality_cuts=False — the cut the 4×4 certification
        depends on — still wrote "REFERENCE_MIDPOINT" into its result file. Every
        flag below materially changes the master relaxation, the cut families or the
        subproblem engine, so any deviation now shows up in the label.
        """
        base = "REFERENCE_MIDPOINT" if self.omega_mode == "midpoint" else "REFERENCE_WORSTCASE"
        noncanonical = (
            self.constraint_26_variant != "26prime"        # K1
            or not self.use_28prime                        # K2
            or self.big_m_margin != 1.00
            or not self.enable_C0
            or not self.enable_C0prime
            or not self.enable_C3
            or not self.enable_flow_connectivity
            or not self.enable_master_time_consistency
            or self.sp_engine != "strong_resource"
            or not self.sp_symmetry_break
            or not self.use_percell_optimality_cuts
            or not self.use_dual_optimality_cut
        )
        if noncanonical:
            base += "_NONCANONICAL"
        return base


def canonical_config() -> Config:
    """The approved K1–K4 canonical configuration (midpoint default)."""
    c = Config()
    c.validate()
    return c


def worstcase_config() -> Config:
    c = Config(omega_mode="worstcase")
    c.validate()
    return c
