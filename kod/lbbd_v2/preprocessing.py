"""preprocessing.py — derived bounds, Big-M, Δ^min, resource groups.

Sources: 03 §6, 04 §5 (Δ^min), §7.1 (resource groups), §12 (Big-M).

All quantities here are VALID bounds (preprocessing), not part of the model. Every Big-M
is DERIVED, never hardcoded. The faulty legacy scalar water Big-M is not representable here
(K4): M^s is always cell-specific.

Big-M validity (04 §12):
    M_d   = (ts_ub + max_i burn_i + Δ_buf + max d_ik) · margin           -- valid time horizon
    M^i_i = a_i · burn_i · margin                                        -- δ_ik ≤ a_i·burn_i
    M^s_i = M^i_i + Δ_wat                                                -- ω_i ≤ ω^max_i + Δ_wat

NOTE on M_d (TIGHTENED): t^s_ub is the maximum-weight simple ignition path from the root set
(node weight α/λ_i), computed exactly by `longest_ignition_path_ub`, NOT the loose global sum
Σ_i α/λ_i. Validity: q-arcs form a forest ((43)), roots have t^s=0 ((48)), and on an active
q-arc t^s_j = t^s_i + α/λ_i ((44)/(45)); hence t^s_j is the α/λ sum over the simple root→j
q-path, so max_j t^s_j ≤ the longest such path. This is a rigorous per-instance horizon and
MAY be below the historical Excel `Mtime` scalar (a looser precomputed constant) — that is
valid, so the old "M_d must dominate Mtime" guard is dropped. For |Nf| beyond the DP guard the
always-valid global sum is used as a safe fallback.
"""
from __future__ import annotations

import heapq
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from lbbd_v2.config import Config
from lbbd_v2.data_loader import Instance


class PreprocessError(Exception):
    pass


@dataclass
class ResourceGroup:
    gid: int
    members: Tuple[int, ...]       # vehicle ids k
    mu: float                      # common µ
    d: Dict[int, float]            # d_g[i] = d_ik for any k in group
    size: int


@dataclass
class Preprocessed:
    burn: Dict[int, float]
    ts_ub: float
    te_ub: float
    tm_ub: Dict[int, float]
    M_d: float
    M_i: Dict[int, float]
    M_s: Dict[int, float]
    tau_lb: Dict[int, float]        # earliest ignition time lower bound
    delta_min: Dict[int, float]     # reward-delay lower bound (04 §5)
    delta_min_pre: Dict[int, float]
    controllable: Dict[int, bool]   # C0
    pre_possible: Dict[int, bool]   # C0'
    groups: List[ResourceGroup]
    group_of: Dict[int, int]        # k -> gid
    excel_mtime: float = 0.0
    ts_ub_exact: bool = True         # True if ts_ub is the exact longest-path (not the fallback sum)
    cfg_snapshot: Dict = field(default_factory=dict)
    bigm_aux: Dict = field(default_factory=dict)   # compute_big_m tanılaması (eşik/pay deneyi, reviewer §2)


# ---------------------------------------------------------------------------
def _burn(inst: Instance) -> Dict[int, float]:
    return {i: inst.alpha / inst.lam[i] + inst.alpha / inst.sig[i] for i in inst.Nf}


def longest_ignition_path_ub(inst, max_nodes: int = 18) -> Tuple[float, bool]:
    """Valid UPPER bound on max_i t^s_i: max-weight simple ignition path from the root set,
    node weight α/λ_i (endpoint included → conservative). Exact via Held-Karp DP over
    (visited-set, endpoint); returns (value, exact_flag). Falls back to the always-valid
    global sum Σ α/λ_i (looser) when |Nf| > max_nodes or there is no in-graph root.

    Validity: q-arcs form a forest ((43)), roots have t^s=0 ((48)), and on an active q-arc
    t^s_j = t^s_i + α/λ_i ((44)/(45)); hence t^s_j equals the α/λ sum over the simple root→j
    q-path, so max_j t^s_j ≤ the max-weight simple path from a root. Summing the endpoint too
    only over-estimates → still valid, and it never exceeds the global sum.
    """
    Nf = list(inst.Nf)
    n = len(Nf)
    w = {i: inst.alpha / inst.lam[i] for i in Nf}
    global_sum = sum(w.values())
    if n == 0:
        return 0.0, True
    if n > max_nodes:
        return global_sum, False
    idx = {i: b for b, i in enumerate(Nf)}
    nb = {i: [j for j in inst.neighbors[i] if j in idx] for i in Nf}
    roots = [r for r in inst.Na if r in idx]
    if not roots:
        return global_sum, False
    NEG = float("-inf")
    dp = [dict() for _ in range(1 << n)]
    for r in roots:
        dp[1 << idx[r]][idx[r]] = w[r]
    best = max(w[r] for r in roots)
    for mask in range(1 << n):
        layer = dp[mask]
        if not layer:
            continue
        for v, val in layer.items():
            if val > best:
                best = val
            for u in nb[Nf[v]]:
                bu = idx[u]
                if mask & (1 << bu):
                    continue
                nmask = mask | (1 << bu)
                nval = val + w[u]
                if nval > dp[nmask].get(bu, NEG):
                    dp[nmask][bu] = nval
    return min(best, global_sum), True   # clamp to the always-valid global sum


def compute_big_m(inst: Instance, cfg: Config
                  ) -> Tuple[float, Dict[int, float], Dict[int, float], Dict[str, object]]:
    burn = _burn(inst)
    margin = cfg.big_m_margin
    max_nodes = int(getattr(cfg, "bigm_pathlen_max_nodes", 18))
    # TIGHTENED horizon: longest-weighted-simple ignition path from roots (valid, ≤ Σ α/λ_i).
    _t_path0 = time.perf_counter()
    ts_ub, ts_ub_exact = longest_ignition_path_ub(inst, max_nodes=max_nodes)
    path_time_s = time.perf_counter() - _t_path0
    # --- tanılama: hangi yöntem, neden fallback, DP'ye giren düğüm sayısı, tamamlandı mı ---
    n_nf = len(inst.Nf)
    roots_in_graph = [r for r in inst.Na if r in set(inst.Nf)]
    if ts_ub_exact:
        bigm_method, fallback_reason, nodes_in_dp = "exact_dp", None, n_nf
    else:
        bigm_method = "global_sum"
        fallback_reason = ("n_gt_threshold" if n_nf > max_nodes
                           else ("no_in_graph_root" if not roots_in_graph else "n_zero_or_other"))
        nodes_in_dp = 0
    if not ts_ub_exact:
        warnings.warn(
            f"t^s_UB fell back to the global sum (|Nf|={len(inst.Nf)} exceeds the "
            f"Held-Karp guard, or no in-graph root): M_d is the LOOSE horizon, not the "
            f"tightened longest-ignition-path bound. Still valid, only weaker.",
            RuntimeWarning, stacklevel=2)
    te_ub = ts_ub + max(burn.values())
    max_d = max(inst.d[(i, k)] for i in inst.Nf for k in inst.K)
    M_d = (te_ub + inst.delta_buf + max_d) * margin

    M_i = {i: inst.a[i] * burn[i] * margin for i in inst.Nf}
    M_s = {i: M_i[i] + inst.delta_wat for i in inst.Nf}
    # M^s must dominate the burn-window UPPER BOUND of ω_i, not its own definition.
    # ω_i = (ω^max_i + ω^min_i)/2 + Δ_wat  and  ω^max_i, ω^min_i ≤ max_k δ_ik ≤ a_i·burn_i
    # (from v_ik ≤ t^c_i ≤ t^e_i = t^s_i + burn_i).  Hence ω_i ≤ a_i·burn_i + Δ_wat.
    # (2026-09-25: this is a valid upper bound; that it is attained under the other constraints is
    # NOT shown, so it is not called a supremum or the smallest valid bound.)
    # NOTE: comparing M_s against M_i + Δ_wat would be a TAUTOLOGY (that is how M_s is
    # defined two lines above) and would test nothing; the margin-free bound is the
    # quantity that actually has to be dominated.
    for i in inst.Nf:
        omega_sup = inst.a[i] * burn[i] + inst.delta_wat
        if M_s[i] < omega_sup - 1e-9:
            raise PreprocessError(
                f"M_s[{i}]={M_s[i]} below the burn-window upper bound of ω_i ({omega_sup}); "
                f"big_m_margin={margin} must be ≥ 1")

    if not math.isfinite(M_d):
        raise PreprocessError("M_d not finite")
    return M_d, M_i, M_s, {
        "ts_ub": ts_ub, "te_ub": te_ub, "burn": burn,
        "ts_ub_exact": ts_ub_exact,
        # --- kontrollü Big-M deneyi tanılaması (reviewer §2) ---
        "n_Nf": n_nf,                          # |N_f|
        "bigm_pathlen_max_nodes": max_nodes,   # istenen eşik
        "bigm_method": bigm_method,            # "exact_dp" | "global_sum"
        "bigm_fallback_reason": fallback_reason,
        "bigm_nodes_in_dp": nodes_in_dp,       # DP'ye gerçekten giren düğüm sayısı
        "bigm_dp_completed": bool(ts_ub_exact),
        "bigm_path_time_s": path_time_s,       # yalnız yol hesabı süresi
        "big_m_margin": margin,                # uygulanan pay
        # pay UYGULAMASI: M_d = (te_ub+Δ_buf+max_d)·margin ; M_i = a_i·burn_i·margin ;
        # M_s = M_i + Δ_wat  (pay İKİNCİ KEZ UYGULANMAZ — türetilmiş sabit).
        "margin_applied_to": ("M_d", "M_i"),
        "margin_not_reapplied_to": ("M_s",),
        "max_d": max_d,
    }


def compute_tau_lb(inst: Instance) -> Dict[int, float]:
    """Earliest ignition time lower bound: shortest ignition chain from roots.

    On q-arc (i->j): t^s_j = t^m_i = t^s_i + α/λ_i. Dijkstra from roots (ts=0),
    edge (i->j) weight = α/λ_i.
    """
    INF = float("inf")
    dist = {i: INF for i in inst.Nf}
    pq: List[Tuple[float, int]] = []
    for r in inst.Na:
        dist[r] = 0.0
        heapq.heappush(pq, (0.0, r))
    while pq:
        du, u = heapq.heappop(pq)
        if du > dist[u]:
            continue
        w = inst.alpha / inst.lam[u]
        for j in inst.neighbors[u]:
            nd = du + w
            if nd < dist[j]:
                dist[j] = nd
                heapq.heappush(pq, (nd, j))
    return dist


def _root_delta_min(inst: Instance, r: int, cfg: Config) -> Tuple[float, float]:
    """Δ^min_r and Δ^min,pre_r via optimistic capacity W_r(T;S) (04 §5.2).

    W_r(T;S) = Σ_{k∈S} µ_k max(0, T - Δ_buf - d_rk) ≥ ω^LB_r.
    Root r has t^s_r = 0, t^{s,min}_r = 0. Competition ((33)) ignored -> optimistic.
    Level (a) SAFE (ω^LB = Δ_wat), valid in BOTH ω-modes (04 §5.5).
    """
    dbuf = inst.delta_buf
    omega_lb = inst.delta_wat  # level (a) safe

    def W(T: float) -> float:
        return sum(inst.mu[k] * max(0.0, T - dbuf - inst.d[(r, k)]) for k in inst.K)

    lo, hi = dbuf, dbuf + 1.0
    it = 0
    while W(hi) < omega_lb:
        hi *= 2.0
        it += 1
        if it > 200:
            raise PreprocessError(f"Δ^min root {r}: capacity never reaches ω^LB")
    # Değişmez: W(lo) < ω^LB ≤ W(hi), yani gerçek eşik T* ∈ (lo, hi].
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if W(mid) >= omega_lb:
            hi = mid
        else:
            lo = mid
    # Δ^min AŞAĞIDAN geçerli olmalı (u_i=1 ⇒ t^c−t^s ≥ Δ^min): feasible T ≥ T* > lo, dolayısıyla
    # lo GEÇERLİ bir alt sınırdır. hi ≥ T* kullanmak FAZLA-tahmindir ve C0/C3'ü geçersizleştirebilir
    # (fizibil kontrolü yanlışlıkla eleyebilir). Bu yüzden lo döndürülür.
    delta_min = lo  # T = t^c_r; delay = t^c_r − t^s_r = T (t^s_r = 0)
    return delta_min, delta_min


def compute_delta_min(inst: Instance, cfg: Config
                      ) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Δ^min_i (04 §5). Roots: tight optimistic bound. Non-roots: universal weak
    bound Δ_wat/Σµ_k (04 §5.4) — practically negligible but valid."""
    delta_min: Dict[int, float] = {}
    delta_min_pre: Dict[int, float] = {}
    universal = inst.delta_wat / sum(inst.mu[k] for k in inst.K)
    for i in inst.Nf:
        if i in inst.Na:
            dm, dmp = _root_delta_min(inst, i, cfg)
            delta_min[i] = dm
            delta_min_pre[i] = dmp
        else:
            delta_min[i] = universal
            delta_min_pre[i] = universal
    return delta_min, delta_min_pre


def compute_resource_groups(inst: Instance) -> Tuple[List[ResourceGroup], Dict[int, int]]:
    """True interchangeability classes from the mathematical signature (04 §7.1):
        k ~ k'  <=>  (d_ik = d_ik' ∀ i∈Nf) ∧ (µ_k = µ_k').
    (base,type) is NOT assumed — computed from data (05 §6)."""
    # S11 geçerliliği araçların MODEL BAKIMINDAN gerçek değiştirilebilirliğini gerektirir; yuvarlama
    # ile "benzer" bulunan araçlar bunu SAĞLAMAZ. Aynı üs+tip+hız araçları d_ik ve µ_k'yı bit-birebir
    # aynı hesaplar (data_loader), bu yüzden TAM eşitlik doğru gruplamayı verir (yuvarlama YOK).
    sig_map: Dict[Tuple, List[int]] = {}
    for k in inst.K:
        sig = (tuple(inst.d[(i, k)] for i in inst.Nf), inst.mu[k])
        sig_map.setdefault(sig, []).append(k)
    groups: List[ResourceGroup] = []
    group_of: Dict[int, int] = {}
    for gid, (sig, members) in enumerate(sorted(sig_map.items(), key=lambda kv: kv[1][0])):
        members_t = tuple(sorted(members))
        k0 = members_t[0]
        dg = {i: inst.d[(i, k0)] for i in inst.Nf}
        grp = ResourceGroup(gid=gid, members=members_t, mu=inst.mu[k0], d=dg, size=len(members_t))
        groups.append(grp)
        for k in members_t:
            group_of[k] = gid
    return groups, group_of


def preprocess_instance(inst: Instance, cfg: Config) -> Preprocessed:
    cfg.validate()
    M_d, M_i, M_s, aux = compute_big_m(inst, cfg)
    burn = aux["burn"]
    ts_ub = aux["ts_ub"]
    te_ub = aux["te_ub"]
    tm_ub = {i: ts_ub + inst.alpha / inst.lam[i] for i in inst.Nf}

    tau_lb = compute_tau_lb(inst)
    delta_min, delta_min_pre = compute_delta_min(inst, cfg)

    controllable: Dict[int, bool] = {}
    pre_possible: Dict[int, bool] = {}
    for i in inst.Nf:
        # C0: Δ^min_i > burn_i  =>  cannot control at all
        controllable[i] = delta_min[i] <= burn[i] + 1e-9
        # C0': Δ^min,pre_i > α/λ_i  =>  cannot control before spread
        pre_possible[i] = delta_min_pre[i] <= inst.alpha / inst.lam[i] + 1e-9

    groups, group_of = compute_resource_groups(inst)

    # Excel Mtime is recorded for reference ONLY. The derived M_d is now a rigorous per-instance
    # horizon (longest ignition path) and MAY legitimately be below the historical Excel scalar,
    # so the old "M_d must dominate Mtime" guard is intentionally dropped (see module docstring).
    P = {str(k).strip(): v for k, v in inst.raw_params.items()}
    excel_mtime = 0.0
    try:
        excel_mtime = float(str(P.get("Mtime")).replace(",", "."))
    except (TypeError, ValueError):
        excel_mtime = 0.0

    # Sanity: the tightened horizon must still dominate the earliest-ignition lower bound
    # (longest path ≥ shortest path) — a cheap guard against an accidentally-too-small ts_ub.
    # Only FINITE τ_lb entries count; cells unreachable from roots (τ_lb=inf) cannot ignite via
    # a valid q-path and so never contribute to any t^s.
    finite_tau = [t for t in tau_lb.values() if math.isfinite(t)]
    if finite_tau and ts_ub + 1e-9 < max(finite_tau):
        raise PreprocessError(
            f"ts_ub ({ts_ub}) below max finite earliest-ignition τ_lb ({max(finite_tau)})")

    return Preprocessed(
        burn=burn, ts_ub=ts_ub, te_ub=te_ub, tm_ub=tm_ub,
        M_d=M_d, M_i=M_i, M_s=M_s, tau_lb=tau_lb,
        delta_min=delta_min, delta_min_pre=delta_min_pre,
        controllable=controllable, pre_possible=pre_possible,
        groups=groups, group_of=group_of,
        excel_mtime=excel_mtime, ts_ub_exact=bool(aux.get("ts_ub_exact", True)),
        cfg_snapshot=cfg.to_dict(),
        bigm_aux={k: v for k, v in aux.items() if k != "burn"},   # burn hariç (büyük dict)
    )
