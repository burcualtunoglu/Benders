"""lbbd_seed_localsearch.py — GELİŞTİRİLMİŞ başlangıç sezgiseli (açgözlü + yerel arama).

BAĞLAM (10x10_v_full teşhisi, gerçek server diag'ı): MILP'in fizibil çözümü 18 hücreyi kontrol eder
(16 PRE + 2 post); ödülün ~%70'i PRE-control'den gelir. Mevcut tohumlar bu kaldıracı kaçırır:
  * ``build_greedy_incumbent_fast``: yalnız POST regime (u^pre≡0, tam kaskad) -> düşük tohum.
  * Kapalı-form araç ataması PRE-control fizibilitesini SİSTEMATİK hafife alır (MILP 16 pre-hücre
    atarken kapalı-form çoğunu reddeder). Bu yüzden fizibilite/ödül DOĞRUDAN alt-problemden okunur.

TASARIM (iki faz, ``budget`` saniye içinde):
  FAZ 1 — kapalı-form firebreak yerel araması: bir ÖNERİ pre-control kümesi üretir (SP-suz, hızlı).
  FAZ 2 — SP-güdümlü: tekil-SP p_solo (rekabetsiz değer sinyali) ile hücreleri sırala; artımlı
    açgözlü, her adımda ÇOK-HÜCRE SP ile jointly-fizibiliteyi doğrular (must_serve nedeniyle aşırı-
    talep kümeler SP'de fizibilsiz döner -> o hücre atlanır). Kaldıraç budur (ölçekte).

KESİNLİK / REVIEWER SÖZLEŞMESİ:
  * Bu YALNIZ bir TOHUM'dur; ne kesme ne UB üretir.
  * DÖNEN değer YALNIZ SP-DOĞRULANMIŞ çözümlerden gelir (``validated=True``); bulunan fizibil çözüm
    (kaynak ATAMALARI + zamanlar, ``sp_solution``) korunur ve doğrudan LB'ye kabul edilir — solver
    aynı yapıyı YENİDEN çözmez (reviewer §2/§3). Kapalı-form "puan" yalnız iç sıralamadır; kaynak
    fizibilitesi doğrulanmamış yüksek puanlı bir öneri, doğrulanmış çözümü ELEMEZ.
  * Hiç doğrulanmış çözüm bulunamazsa dışarıda-doğrulanmış bir taban (containment/greedy_fast/fallback)
    ÖNERİ olarak döner (validated=False) ve solver onu doğrular -> mevcut tohumdan asla kötü değil.
  * Süre (deadline) sözleşmesi: deadline dolduğunda YENİ SP başlatılmaz; çağrı bütçesi kalan süreden
    büyük verilmez. Determinizm: sorted() + tohumlanmış RNG (ama süre-bütçeli SP wall-clock'a duyarlı).
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set

from lbbd_v2.benders_forward import forward_pass, FWD_OK
from lbbd_v2.lbbd_heuristics import (
    Incumbent, build_containment_incumbent, build_greedy_incumbent_fast,
    build_fallback_incumbent, _assign_cell)


@dataclass
class _Score:
    """Kapalı-form firebreak kümesi değerlendirmesi (YALNIZ FAZ 1 arama sıralaması için).
    ``feasible`` True ise kapalı-form bir yapı kuruldu ve ``value`` onun (kapalı-form) amaç değeridir;
    False ise ``score`` fizibiliteye+korumaya doğru tepe-tırmanışı için CEZALI kılavuzdur. Bu bir
    ÖNERİ puanıdır — LB'ye kabul için değil (kabul yalnız SP doğrulamasıyla). ``feasible=False``,
    KANITLANMIŞ fizibilsizlik değildir; yalnız kapalı-formun bir çözüm kuramadığı anlamına gelir."""
    feasible: bool
    score: float
    value: float
    y: Dict[int, float]


def _score_firebreaks(F, inst, pre, cfg, midpoint: bool, penalty: float,
                      max_vehicles_per_cell: int = 64) -> _Score:
    """F firebreak (ön-kontrol) kümesinden KAPALI-FORM bir aday yapı kur ve puanla (FAZ 1 arama
    kılavuzu). Araç ataması kapalı-formdur (``_assign_cell``); bu ataманın başarısızlığı KANITLANMIŞ
    fizibilsizlik DEĞİLDİR — yalnız kapalı-formun atayamadığını gösteren bir arama cezasıdır (gerçek
    fizibilite SP ile belirlenir). Düzeltilmiş model için: eski t^s≤M_i/a_i (geç-tutuşma) sezgisel
    filtresi UYGULANMAZ (düzeltilmiş model gerektirmez); model kısıtları (M3 vb.) değiştirilmez."""
    Na, Nf, nb = set(inst.Na), inst.Nf, inst.neighbors
    F = {i for i in F if pre.pre_possible.get(i, False)}

    # ---- yanan küme B: köklerden yayıl; F yaymaz (u^pre=1 => (40) TÜM komşulara yayılım yok) ----
    B: Set[int] = set(Na)
    stack: List[int] = list(Na)
    while stack:
        c = stack.pop()
        if c in F:
            continue
        for j in nb[c]:
            if j not in B:
                B.add(j)
                stack.append(j)
    Feff = {i for i in F if i in B}
    y = {i: (1.0 if i in B else 0.0) for i in Nf}
    pi_free_intended = sum(inst.pi[i] * (1 - y[i]) for i in Nf)

    def _guide(nviol: int) -> _Score:
        # ARAMA KILAVUZU cezası (kanıtlanmış fizibilsizlik DEĞİL): koruma (pi_free) yüksekse iyi, ama
        # ihlal başına ağır ceza -> her kapalı-form fizibil öneri her cezalıdan yüksek skorlu; ihlal
        # azaldıkça skor artar (fizibiliteye doğru tırman).
        return _Score(False, pi_free_intended - penalty * max(1, nviol), float("-inf"), y)

    # ---- q-ormanı (B içinde firebreak OLMAYAN hücrelerden Dijkstra), z, forward pass ----
    alpha, lam = inst.alpha, inst.lam
    INF = float("inf")
    dist: Dict[int, float] = {r: 0.0 for r in Na}
    parent: Dict[int, int] = {}
    pq = [(0.0, r) for r in Na]
    heapq.heapify(pq)
    expanded: Set[int] = set()
    while pq:
        du, u = heapq.heappop(pq)
        if u in expanded:
            continue
        expanded.add(u)
        if u in Feff:
            continue
        w = alpha / lam[u]
        for j in nb[u]:
            if j not in B:
                continue
            nd = du + w
            if nd < dist.get(j, INF) - 1e-12:
                dist[j] = nd
                parent[j] = u
                heapq.heappush(pq, (nd, j))

    q = {arc: 0.0 for arc in inst.arcs}
    for j, i in parent.items():
        q[(i, j)] = 1.0
    z = {arc: 0.0 for arc in inst.arcs}
    for (i, j) in inst.arcs:
        if i in B and j in B and i not in Feff:
            z[(i, j)] = 1.0
    inc = Incumbent(y=y, z=z, q=q,
                    u_pre={i: (1.0 if i in Feff else 0.0) for i in Nf},
                    u_post={i: 0.0 for i in Nf})
    fwd = forward_pass(inc, inst)
    if fwd.status != FWD_OK:
        return _guide(len(fwd.unreachable) + len(fwd.violating_arcs))

    # ---- kapalı-form araç ataması: firebreak (pre) sonra post bonus ----
    pool: Set[int] = set(inst.K)
    p_of: Dict[int, float] = {}
    unassigned = 0
    for c in sorted(Feff, key=lambda i: -inst.pi[i]):
        rec = _assign_cell(c, fwd.ts[c], fwd.tm[c], fwd.ts[c], fwd.ts_min.get(c, fwd.ts[c]),
                           pool, inst, pre, midpoint, max_vehicles_per_cell)
        if rec is None:
            unassigned += 1                       # kapalı-form atayamadı (SP atayabilir) -> kılavuz cezası
            continue
        p_of[c] = rec["p"]
        pool.difference_update(rec["S"])
    if unassigned:
        return _guide(unassigned)
    for c in sorted((i for i in B if i not in Feff), key=lambda i: -inst.pi[i]):
        if not pre.controllable[c]:
            continue
        rec = _assign_cell(c, fwd.tm[c], fwd.te[c], fwd.ts[c], fwd.ts_min.get(c, fwd.ts[c]),
                           pool, inst, pre, midpoint, max_vehicles_per_cell)
        if rec is None:
            continue
        p_of[c] = rec["p"]
        pool.difference_update(rec["S"])

    value = pi_free_intended + sum(p_of.values())
    return _Score(True, value, value, y)


def _structure_from_firebreaks(F, inst, pre):
    """F ön-kontrol (firebreak) kümesinden model yapısı (y,z,q,u_pre) kur — araç ataması YAPMADAN
    (fizibilite/ödül SP'ye bırakılır). Dönüş: (inc, Feff, y)."""
    Na, nb = set(inst.Na), inst.neighbors
    alpha, lam = inst.alpha, inst.lam
    INF = float("inf")
    F = {i for i in F if pre.pre_possible.get(i, False)}
    B = set(Na)
    stack = list(Na)
    while stack:
        c = stack.pop()
        if c in F:
            continue
        for j in nb[c]:
            if j not in B:
                B.add(j)
                stack.append(j)
    Feff = {i for i in F if i in B}
    dist = {r: 0.0 for r in Na}
    parent = {}
    pq = [(0.0, r) for r in Na]
    heapq.heapify(pq)
    ex = set()
    while pq:
        du, u = heapq.heappop(pq)
        if u in ex:
            continue
        ex.add(u)
        if u in Feff:
            continue
        w = alpha / lam[u]
        for j in nb[u]:
            if j not in B:
                continue
            nd = du + w
            if nd < dist.get(j, INF) - 1e-12:
                dist[j] = nd
                parent[j] = u
                heapq.heappush(pq, (nd, j))
    y = {i: (1.0 if i in B else 0.0) for i in inst.Nf}
    q = {arc: 0.0 for arc in inst.arcs}
    for j, i in parent.items():
        q[(i, j)] = 1.0
    z = {arc: 0.0 for arc in inst.arcs}
    for (i, j) in inst.arcs:
        if i in B and j in B and i not in Feff:
            z[(i, j)] = 1.0
    inc = Incumbent(y=y, z=z, q=q,
                    u_pre={i: (1.0 if i in Feff else 0.0) for i in inst.Nf},
                    u_post={i: 0.0 for i in inst.Nf})
    return inc, Feff, y


def _closed_form_search(inst, pre, cfg, midpoint: bool, penalty: float,
                        deadline: float) -> FrozenSet[int]:
    """FAZ 1: kapalı-form ceza-güdümlü firebreak yerel araması -> en iyi ÖNERİ pre-control kümesi.
    SP ÇAĞIRMAZ; yalnız öneri üretir (değer sonradan SP ile doğrulanır). ``deadline``a uyar."""
    nb = inst.neighbors
    cache: Dict[FrozenSet[int], _Score] = {}
    best_F: FrozenSet[int] = frozenset()
    best_score = float("-inf")

    def score(F: FrozenSet[int]) -> _Score:
        nonlocal best_F, best_score
        sc = cache.get(F)
        if sc is None:
            sc = _score_firebreaks(F, inst, pre, cfg, midpoint, penalty)
            cache[F] = sc
        if sc.feasible and sc.value > best_score + 1e-9:
            best_score, best_F = sc.value, F
        return sc

    def local_opt(F_start: FrozenSet[int]) -> None:
        cur = score(F_start)
        cur_F: Set[int] = set(F_start)
        while time.time() < deadline:
            B = {i for i, v in cur.y.items() if v > 0.5}
            neigh = [frozenset(cur_F - {f}) for f in cur_F]
            for i in B:
                if i in cur_F or not pre.pre_possible.get(i, False):
                    continue
                neigh.append(frozenset(cur_F | {i}))
            best_move: Optional[FrozenSet[int]] = None
            best_move_score = cur.score
            for Fn in neigh:
                if time.time() >= deadline:
                    break
                sc = score(Fn)
                if sc.score > best_move_score + 1e-9:
                    best_move_score, best_move = sc.score, Fn
            if best_move is None:
                return
            cur_F, cur = set(best_move), cache[best_move]

    con = build_containment_incumbent(inst, pre, cfg)
    F_con = frozenset(i for i in inst.Nf if con.u_pre.get(i, 0.0) > 0.5)
    for Fs in (F_con, frozenset()):
        if time.time() >= deadline:
            break
        local_opt(Fs)
    return best_F


def _inner_caps(budget: float) -> Dict[str, float]:
    """SEEDLS iç çağrı tavanları TOHUM BÜTÇESİYLE ölçeklenir (v2, 2026-09-19). budget=120 s'de eski
    değerlerle BİREBİR aynı (FAZ1 18 s, tekil 3 s, doğrulama 8 s); büyük bütçede orantılı büyür —
    aksi hâlde 3 saatlik koşuda 8 s'lik SP doğrulaması 282+ araçta inkümbent bulamayıp tohum
    'greedy_fast'a düşüyordu (job 2312 teşhisi)."""
    return {"cf": 0.15 * budget,                    # FAZ1 kapalı-form arama (eski: min(0.15b, 20))
            "single": max(3.0, budget / 200.0),     # FAZ2 tekil-SP çağrısı (eski: 3)
            "validate": max(8.0, 0.05 * budget)}    # FAZ2 çok-hücre doğrulama (eski: 8)


def build_localsearch_incumbent(inst, pre, cfg, budget: float,
                                sp_solve=None, single_cell=None, verbose: int = 0) -> Incumbent:
    """Geliştirilmiş TOHUM. YALNIZ SP-DOĞRULANMIŞ çözüm döndürür (``validated=True``, ``sp_solution``
    korunur); hiç doğrulanmış çözüm yoksa bir ÖNERİ (validated=False) döner ve solver doğrular ->
    mevcut tohumdan asla kötü değil. Tüm SP çağrıları ``budget`` (deadline) içindedir; deadline sonrası
    YENİ SP başlatılmaz, çağrı bütçesi kalandan büyük verilmez."""
    from lbbd_v2.lbbd_subproblem_resource import SPStatus
    OK = (SPStatus.OPTIMAL, SPStatus.FEASIBLE_NOT_PROVEN_OPTIMAL, SPStatus.TIME_LIMIT_WITH_INCUMBENT)
    t0 = time.time()
    deadline = t0 + budget
    midpoint = (cfg.omega_mode == "midpoint")
    penalty = sum(inst.pi[i] for i in inst.Nf) + 1.0

    best_inc: Optional[Incumbent] = None
    best_val = float("-inf")                     # YALNIZ SP-doğrulanmış değerler
    best_found_dt = float("nan")                 # en iyi doğrulanmış çözümün bulunma zamanı (s, t0'dan)
    sp_cache: Dict[FrozenSet[int], Any] = {}     # aynı pre-kümeyi iki kez SP-çözme
    caps = _inner_caps(budget)

    def validate_pre_set(F, val_budget: float):
        """F pre-control kümesini ÇOK-HÜCRE SP ile doğrula. Fizibilse doğrulanmış değeri + çözümü
        (atama+zaman) KORU ve en iyiyi izle. Yeni SP yalnız kalan süre>0 iken başlatılır."""
        nonlocal best_inc, best_val, best_found_dt
        if sp_solve is None:
            return None
        key = frozenset(F)
        if key in sp_cache:
            return sp_cache[key]
        inc, Feff, y = _structure_from_firebreaks(key, inst, pre)
        fwd = forward_pass(inc, inst)
        if not Feff or fwd.status != FWD_OK:
            sp_cache[key] = None
            return None
        remaining = deadline - time.time()
        if remaining <= 0:                       # deadline doldu -> YENİ SP yok
            return None
        C = sorted(Feff)
        regime = {c: "pre" for c in C}
        sp = sp_solve(fwd, C, regime, inst, pre, cfg, time_budget=min(val_budget, remaining))
        if sp.status in OK and sp.obj is not None:
            val = sum(inst.pi[k] * (1 - y[k]) for k in inst.Nf) + sp.obj
            inc.C = C
            inc.regime = regime
            inc.value = val
            inc.validated = True                 # SP-DOĞRULANMIŞ (dinamik öznitelik)
            inc.sp_solution = getattr(sp, "solution", {}) or {}   # atama+zaman KORUNUR
            sp_cache[key] = (inc, val)
            if val > best_val + 1e-9:
                best_inc, best_val = inc, val
                best_found_dt = time.time() - t0
            return (inc, val)
        sp_cache[key] = None                     # kapalı-form/SP doğrulayamadı (kanıtlı fizibilsizlik değil)
        return None

    # ---- FAZ 1: kapalı-form öneri kümesi -> SP ile doğrula ----
    if sp_solve is not None and time.time() < deadline:
        cf_deadline = min(deadline, t0 + caps["cf"])
        F_cf = _closed_form_search(inst, pre, cfg, midpoint, penalty, cf_deadline)
        if F_cf and time.time() < deadline:
            validate_pre_set(F_cf, min(cfg.heur_budget, deadline - time.time()))

    # ---- FAZ 2: tekil-SP p_solo sıralaması + artımlı açgözlü (SP-doğrulamalı) ----
    if sp_solve is not None and single_cell is not None and time.time() < deadline:
        base = build_greedy_incumbent_fast(inst, pre, cfg)
        fwd0 = forward_pass(base, inst)
        if fwd0.status == FWD_OK:
            cells = [i for i in inst.Nf if base.y[i] > 0.5 and pre.pre_possible.get(i, False)]
            psolo: Dict[int, float] = {}
            for i in cells:
                if time.time() >= deadline:
                    break
                rem = deadline - time.time()
                feas, ps, kind = single_cell(i, "pre", fwd0, inst, pre, cfg,
                                             time_budget=max(0.5, min(caps["single"], rem)),
                                             return_status=True)
                if feas and ps > 1e-6:
                    psolo[i] = ps
            ranked = sorted(psolo, key=lambda i: (-psolo[i], i))    # determinist sıralama
            F: List[int] = []
            for i in ranked:
                if time.time() >= deadline:
                    break
                res = validate_pre_set(F + [i], min(caps["validate"], deadline - time.time()))
                if res is not None:              # F ∪ {i} jointly-fizibil -> kümede tut
                    F = F + [i]

    # ---- fallback: hiç DOĞRULANMIŞ çözüm yok -> ÖNERİ döndür (solver doğrular) ----
    if best_inc is None:
        con = build_containment_incumbent(inst, pre, cfg)
        gf = build_greedy_incumbent_fast(inst, pre, cfg)
        for c in (con, gf):
            if c.value is not None and forward_pass(c, inst).status == FWD_OK:
                best_inc = c
                break
        if best_inc is None:
            fb = build_fallback_incumbent(inst, pre, cfg)
            fb.value = sum(inst.pi[i] * (1 - fb.y.get(i, 0.0)) for i in inst.Nf)
            best_inc = fb
    best_inc.found_dt = best_found_dt            # en iyi doğrulanmış çözümün bulunma zamanı (raporlama)
    if verbose:
        nfb = sum(1 for i in inst.Nf if best_inc.u_pre.get(i, 0.0) > 0.5)
        print(f"  [localsearch] validated={getattr(best_inc, 'validated', False)} "
              f"val={best_inc.value:.6g} pre={nfb} found_dt={best_found_dt:.1f}s "
              f"elapsed={time.time()-t0:.1f}s budget={budget:.0f}s")
    return best_inc
