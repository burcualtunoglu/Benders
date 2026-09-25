"""lbbd_cuts.py — R(C) signature + the six cut families (04 §8), all PROVEN EXACT.

Every cut is an affine inequality over the master binaries (y, z, q, u_pre, u_post) and the
reward proxy ρ. A Cut stores its terms so it can be (a) added to the master and (b) re-evaluated
at any point — used by the cut-safety test to prove no cut removes the monolithic optimum.

R(C) (04 §8.4, Lemma 2): R(C)=0 ⟺ the data (t^s,t^m,t^e,t^{s,min},regime) of every i∈C is
preserved, so a cut keyed on R(C) is inactive exactly on signature-preserving master decisions
and vacuous otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gurobipy as gp

Term = Tuple[str, Any, float]     # (varname, key, coef)


# ---------------------------------------------------------------------------
@dataclass
class Cut:
    kind: str
    terms: List[Term]
    sense: str                    # '<=', '>=', '=='
    rhs: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_linexpr(self, master_vars: Dict[str, Any]) -> gp.LinExpr:
        expr = gp.LinExpr()
        for vn, key, coef in self.terms:
            expr.addTerms(coef, master_vars[vn][key])
        return expr

    def apply(self, master, name: Optional[str] = None) -> None:
        name = name or self.kind
        master.add_cut(self.to_linexpr(master.vars), self.sense, self.rhs, name, self.to_json())

    def lhs_value(self, point: Dict[str, Dict[Any, float]]) -> float:
        return sum(coef * point.get(vn, {}).get(key, 0.0) for vn, key, coef in self.terms)

    def satisfied(self, point: Dict[str, Dict[Any, float]], tol: float = 1e-6) -> bool:
        lhs = self.lhs_value(point)
        if self.sense == "<=":
            return lhs <= self.rhs + tol
        if self.sense == ">=":
            return lhs >= self.rhs - tol
        return abs(lhs - self.rhs) <= tol

    def to_json(self) -> dict:
        return {"kind": self.kind, "sense": self.sense, "rhs": self.rhs,
                "terms": [[vn, list(key) if isinstance(key, tuple) else key, coef]
                          for vn, key, coef in self.terms],
                "meta": self.meta}


def _accumulate(raw: List[Term]) -> List[Term]:
    acc: Dict[Tuple[str, Any], float] = {}
    for vn, key, coef in raw:
        acc[(vn, key)] = acc.get((vn, key), 0.0) + coef
    return [(vn, key, c) for (vn, key), c in acc.items() if abs(c) > 1e-12]


# ---------------------------------------------------------------------------
def r_affine(C: Iterable[int], sol, fwd, inst) -> Tuple[List[Term], float]:
    """R(C) as (terms, const):  R(C) = const + Σ coef·var  (04 §8.4)."""
    raw: List[Term] = []
    const = 0.0
    for i in C:
        # (1 - u_i)
        const += 1.0
        raw += [("u_pre", i, -1.0), ("u_post", i, -1.0)]
        # (1 - u^•_i): the active regime indicator
        active = "u_pre" if sol.u_pre.get(i, 0.0) > 0.5 else "u_post"
        const += 1.0
        raw.append((active, i, -1.0))
        # Σ_{(a,b)∈P(i)} (1 - q_ab)
        for (a, b) in fwd.q_path.get(i, []):
            const += 1.0
            raw.append(("q", (a, b), -1.0))
        # neighbours j ∈ N^+(i)
        for j in inst.Nplus[i]:
            if sol.y.get(j, 0.0) > 0.5:
                const += 1.0
                raw.append(("y", j, -1.0))
                for (a, b) in fwd.q_path.get(j, []):
                    const += 1.0
                    raw.append(("q", (a, b), -1.0))
            else:
                raw.append(("y", j, 1.0))
    return _accumulate(raw), const


def r_value(C, sol, fwd, inst, point: Dict[str, Dict[Any, float]]) -> float:
    terms, const = r_affine(C, sol, fwd, inst)
    return const + sum(c * point.get(vn, {}).get(key, 0.0) for vn, key, c in terms)


# ---------------------------------------------------------------------------
# Feasibility cuts (R(C) ≥ 1)
def _r_ge_one(kind: str, C, sol, fwd, inst, meta) -> Cut:
    terms, const = r_affine(C, sol, fwd, inst)
    return Cut(kind, terms, ">=", 1.0 - const, meta)


def cut_singleton_infeasible(i, sol, fwd, inst) -> Cut:
    return _r_ge_one("singleton_infeasible", [i], sol, fwd, inst, {"cell": i})


def cut_singleton_predecessor(i: int, regime_i: str, a: int, inst, meta: Optional[Dict[str, Any]] = None) -> Cut:
    """YOL-BAĞIMSIZ tekil fizibilsizlik kesmesi (2026-09-24, Geliştirme 2; docs/singleton_delta_cuts.md):
        u^r_i + q_ai − Σ_{j∈N(i)\\{a}} (y_j − q_ij − q_aj) ≤ 1        (q_aj terimi yalnız (a,j) yay ise)

    Δ_i = t^s_i − min_{j∈N⁺(i), y_j=1} t^s_j = max_j (t^s_i − t^s_j). q_ai = 1 ⇒ t^s_i = t^s_a + α/λ_a ⇒ Δ_i ≥ α/λ_a;
    ayrıca her diğer komşu j ya yanmıyorsa ya da i'den veya a'dan tutuşuyorsa (t^s_j ≥ t^s_a) ⇒ Δ_i = α/λ_a TAM olarak.
    y_j − q_ij − q_aj ∈ {0,1} (giriş derecesi ≤ 1): 1 ⇔ j yanıyor ve i/a dışından tutuşuyor ("erken olabilir").
    (i, r) bir Δ̄ ≥ α/λ_a değerinde INFEASIBLE_PROVEN ise (Δ-monotonluk) örüntü {u^r_i=1, q_ai=1, tüm j geç} fizibilsiz ⇒
    kesme yalnız fizibilsiz noktaları kaldırır. Bir komşu "erken olabilir" ise kesme gevşek kalır (kanıt yok)."""
    vn = "u_pre" if regime_i == "pre" else "u_post"
    arcs = set(inst.arcs)
    terms: List[Term] = [(vn, i, 1.0), ("q", (a, i), 1.0)]
    for j in inst.neighbors[i]:
        if j == a:
            continue
        terms.append(("y", j, -1.0))
        if (i, j) in arcs:
            terms.append(("q", (i, j), 1.0))
        if (a, j) in arcs:
            terms.append(("q", (a, j), 1.0))
    m = {"cell": i, "regime": regime_i, "pred": a}
    if meta:
        m.update(meta)
    return Cut("singleton_pred", _accumulate(terms), "<=", 1.0, m)


def cut_singleton_predecessor_timed(i: int, regime_i: str, a: int, tau: float, others: List[int],
                                    eps_keys: List[Any], meta: Optional[Dict[str, Any]] = None) -> Cut:
    """ZAMANLI yol-bağımsız tekil fizibilsizlik kesmesi (SDC-t, 2026-09-24; docs/singleton_delta_cuts.md §3b):
        u^r_i + q_ai − Σ_{j∈N(i)\\{a}} ε_j ≤ 1,   ε_j ≤ y_j,   t^s_j ≤ t^s_a − τ + M(1 − ε_j)   (τ = Δ̄ − α/λ_a ≥ 0)
    ε_j = 1 yalnız j yanıyor ve a'dan en az τ erken tutuşmuşsa mümkündür (ana problem MTC t^s değişkenleri; ε yardımcı ikili).
    Geçerlilik: u^r_i = 1, q_ai = 1 olan her fizibil çözümde Δ_i > Δ̄ ⇒ ∃ yanan j: t^s_i − t^s_j > Δ̄ ⇒ t^s_j < t^s_a − τ ⇒ ε_j = 1
    seçilebilir ⇒ kesme sağlanır. Hiçbir komşu τ kadar erken değilse Δ_i ≤ α/λ_a + τ = Δ̄ ⇒ fizibilsiz ⇒ kesilir (LHS = 2).
    Bağlantı kısıtları çözücü tarafında (master.ensure_binary + add_cut) eklenir; bu nesne yalnız ana eşitsizliktir."""
    vn = "u_pre" if regime_i == "pre" else "u_post"
    terms: List[Term] = [(vn, i, 1.0), ("q", (a, i), 1.0)] + [("eps", k, -1.0) for k in eps_keys]
    m = {"cell": i, "regime": regime_i, "pred": a, "tau": tau, "others": list(others), "eps_keys": [list(k) for k in eps_keys]}
    if meta:
        m.update(meta)
    return Cut("singleton_pred_t", _accumulate(terms), "<=", 1.0, m)


def cut_conflict_infeasible(Cstar, sol, fwd, inst) -> Cut:
    return _r_ge_one("conflict_infeasible", sorted(Cstar), sol, fwd, inst,
                     {"cells": sorted(Cstar)})


# Optimality cuts
def cut_cell_optimality(i, p_bar, sol, fwd, inst) -> Cut:
    """ρ_i ≤ p_bar + π_i R({i})  ⇔  ρ_i − π_i·(Σc·var) ≤ p_bar + π_i·const."""
    terms, const = r_affine([i], sol, fwd, inst)
    pi_i = inst.pi[i]
    out: List[Term] = [("rho", i, 1.0)]
    out += [(vn, key, -pi_i * c) for vn, key, c in terms]
    return Cut("cell_optimality", _accumulate(out), "<=", p_bar + pi_i * const,
               {"cell": i, "p_bar": p_bar})


def cut_aggregate_optimality(C, Phi, sol, fwd, inst, theta: Optional[float] = None,
                             kind: str = "aggregate_optimality") -> Cut:
    """Σ_{i∈C} ρ_i ≤ Φ + Θ R(C),  Θ = Σ_{i∈C} π_i + 1 (04 §8.5)."""
    C = sorted(C)
    if theta is None:
        theta = sum(inst.pi[i] for i in C) + 1.0
    terms, const = r_affine(C, sol, fwd, inst)
    out: List[Term] = [("rho", i, 1.0) for i in C]
    out += [(vn, key, -theta * c) for vn, key, c in terms]
    return Cut(kind, _accumulate(out), "<=", Phi + theta * const,
               {"cells": C, "Phi": Phi, "theta": theta})


def cut_dual_bound_optimality(C, obj_bound, sol, fwd, inst, theta: Optional[float] = None) -> Cut:
    """Σρ ≤ ObjBound + Θ R(C): on an SP timeout the dual bound is a valid UB on Φ (exact)."""
    return cut_aggregate_optimality(C, obj_bound, sol, fwd, inst, theta, kind="dual_bound")


# Structural cuts
def cut_propagation(i, j, fwd) -> Cut:
    """Geçerli yayılım tutarlılığı kesmesi — İKİ uç hücrenin ateşleme yollarını korur:
        Σ_{(a,b)∈P(i)∪P(j)} q_ab + z_ij ≤ |P(i)∪P(j)|.
    Tetik: Kontrol B'de z̄_ij=1 iken t^s_j > t^m_i ihlali (04 §8.1).

    Neden P(i) DE gerekir: t^m_i = t^s_i + α_i/λ_i, yani P(i)'ye bağlıdır; yalnız P(j)'yi
    sabitlemek t^m_i'yi sabitlemez. Bu yüzden ESKİ yalnız-P(j) biçimi
    (Σ_{P(j)} q + z_ij ≤ |P(j)|), P(j)'yi koruyup P(i)'yi UZATAN (t^m_i büyüyen, dolayısıyla
    t^s_j ≤ t^m_i sağlanan) FİZİBİL bir çözümü yanlışlıkla siler → GENEL DURUMDA GEÇERSİZ.
    Birleşim biçimi yalnız "her iki yol da korunuyor + z_ij=1" birlikteliğini yasaklar; o
    birliktelikte t^s_i, t^s_j (dolayısıyla t^m_i) referansa sabitlenir ve ihlal yeniden doğar
    (Teorem: ileri geçiş) → kesme yalnız infizibil noktaları kaldırır, referansı da keser.

    Özel durum: P(i) ⊆ P(j) iken birleşim P(j)'ye iner ve kesme eski
    Σ_{P(j)} q + z_ij ≤ |P(j)| biçimini alır (ör. i bir kök: P(i)=∅). Yol kısaltma için geçerli
    koşul on:lift'te. q_path'e DOĞRUDAN erişilir (kökler ileri geçişte açıkça [] alır); eksik bir
    yol sessizce kök sanılmasın diye .get(...) kullanılmaz."""
    path_i = fwd.q_path[i]
    path_j = fwd.q_path[j]
    support = sorted(set(path_i) | set(path_j))
    terms: List[Term] = [("q", (a, b), 1.0) for (a, b) in support]
    terms.append(("z", (i, j), 1.0))
    return Cut("propagation", _accumulate(terms), "<=", float(len(support)),
               {"arc": [i, j], "path_i_len": len(path_i), "path_j_len": len(path_j),
                "support_len": len(support)})


def cut_ignition_time(i, fwd) -> Cut:
    """Σ_{(a,b)∈P(i)} q_ab ≤ |P(i)| − 1 : forbid the ignition PATH that makes cell i ignite too
    late to be monolithic-feasible.

    Exactness (mirror of cut_propagation): with the q-forest fixed, forward-pass gives
    t^s_i = Σ_{(a,b)∈P(i)} α/λ_a. If that exceeds M^i_i/a_i = burn_i·margin, then for EVERY
    unassigned vehicle (x_ik=0 ⇒ v_ik=0 by (23v)) constraint (5) forces δ_ik ≤ M^i_i − a_i t^s_i
    < 0 against δ_ik ≥ 0 — the monolithic model is infeasible for this (y,z,q). Cell i therefore
    cannot burn along THIS path in any feasible solution, so forbidding the full path removes only
    infeasible points (a shorter path, or i not burning, keeps the cut slack). Roots (P(i)=∅,
    t^s=0) never trigger it. Cells with a_i=0 are exempt (δ_ik≡0, no cap)."""
    path = fwd.q_path.get(i, [])
    terms: List[Term] = [("q", (a, b), 1.0) for (a, b) in path]
    return Cut("ignition_time", _accumulate(terms), "<=", float(len(path)) - 1.0,
               {"cell": i, "path_len": len(path)})


def cut_connectivity(S, j, inst) -> Cut:
    """Σ_{j'∈S} Σ_{i∈N(j')\\S} q_{ij'} ≥ y_j  (04 §4.4 / §8.3)."""
    S = set(S)
    terms: List[Term] = []
    for jp in S:
        for i in inst.neighbors[jp]:
            if i not in S:
                terms.append(("q", (i, jp), 1.0))
    terms.append(("y", j, -1.0))
    return Cut("connectivity", _accumulate(terms), ">=", 0.0,
               {"S": sorted(S), "cell": j})
