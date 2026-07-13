"""
benders_lbbd.py — Logic-Based Benders Decomposition (rapor §4.6).

Klasik varyanttan farkı: ATAMA değişkeni $x_{ik}$ master'da DEĞİL, alt problemdedir.
Master yalnız ağaç+rejim ikililerine (y,z,q,u,upre,upost,w,bmin) + hücre ödülü ρ_i'ye
karar verir; 150 araçlık atama binary'si master'dan tamamen düşer -> master çok küçülür.

Mimari: klasik LBBD iteratif döngüsü (master her turda optimale çözülür):
  1. Master'ı çöz -> (ağaç, rejim, ρ̄), üst sınır UB = Σπ(1-y)+Σρ.
  2. İleri geçiş; (46) ihlali -> kombinatoryal kesme, 1'e dön.
  3. Kontrol edilen hücreler C için ATAMA+ÇİZELGELEME MILP alt problemini çöz (Φ).
     - uygunsuz -> logic-based uygunluk kesmesi (C+ağaç+rejim değişmeli).
     - Σ_{C}ρ̄ > Φ -> logic-based optimalite kesmesi (ρ toplamı Φ ile bağlanır).
  4. UB ≈ LB ise dur.

Kesmeler dual yerine no-good/gevşetilmiş biçimdedir (alt problem MILP olduğundan);
yerel imza (ateşleme yolu + yanan komşuluk + rejim) değişince gevşerler -> geçerli.
"""

from __future__ import annotations

import time
from typing import Dict, List

import gurobipy as gp
from gurobipy import GRB

from data_loader import Instance
from config import Config
from preprocessing import compute_bigM
from benders_forward import forward_pass
from benders_lbbd_subproblem import solve_assignment_subproblem, cell_feasible, cell_solo_reward

EPS = 1e-5


def build_lbbd_master(inst: Instance, cfg: Config):
    """Atama içermeyen master: yalnız ağaç+rejim ikilileri + ρ_i."""
    if not inst.Md:
        compute_bigM(inst, cfg.model)
    Nf, Na = inst.Nf, inst.Na
    Nb, e = inst.neighbors, inst.e
    edges = [(i, j) for i in Nf for j in Nb[i]]

    m = gp.Model(f"lbbd_master_{inst.name}")
    m.setParam("OutputFlag", 0)

    y = m.addVars(Nf, vtype=GRB.BINARY, name="y")
    u_pre = m.addVars(Nf, vtype=GRB.BINARY, name="u_pre")
    u_post = m.addVars(Nf, vtype=GRB.BINARY, name="u_post")
    u = m.addVars(Nf, vtype=GRB.BINARY, name="u")
    z = m.addVars(edges, vtype=GRB.BINARY, name="z")
    q = m.addVars(edges, vtype=GRB.BINARY, name="q")
    bmin = m.addVars([(i, j) for i in Nf for j in inst.Nplus(i)], vtype=GRB.BINARY, name="bmin")
    w = m.addVars(Na, Nf, vtype=GRB.BINARY, name="w")
    rho = m.addVars(Nf, lb=0.0, name="rho")

    m.addConstrs((u[i] == u_pre[i] + u_post[i] for i in Nf), "c13")                                    # (13)
    m.addConstrs((u_pre[i] + u_post[i] <= 1 for i in Nf), "c39")                                        # (39)
    m.addConstrs((u[i] <= y[i] for i in Nf), "u_le_y")             # kontrol ancak yanan hücrede
    m.addConstrs((gp.quicksum(z[i, j] for j in Nb[i]) == len(Nb[i]) * (y[i] - u_pre[i]) for i in Nf), "c40")  # (40)
    m.addConstrs((len(Nb[j]) * y[j] >= gp.quicksum(z[i, j] for i in Nb[j]) for j in Nf), "c41")        # (41)
    m.addConstrs((q[i, j] <= z[i, j] for (i, j) in edges), "c42")                                      # (42)
    m.addConstrs((gp.quicksum(q[i, j] for i in Nb[j]) == y[j] - e[j] for j in Nf), "c43")              # (43)
    m.addConstr(gp.quicksum(y[i] for i in Na) == len(Na), "c47")                                       # (47)
    for i in Nf:
        for j in inst.Nplus(i):
            m.addConstr(bmin[i, j] <= y[j] + (1 - y[i]), f"c28p[{i},{j}]")                             # (28')
        m.addConstr(gp.quicksum(bmin[i, j] for j in inst.Nplus(i)) == 1, f"c28[{i}]")                  # (28)
    # kök etiketleme (1)-(3)
    m.addConstrs((gp.quicksum(w[r, i] for r in Na) == y[i] for i in Nf), "w1")                         # (1)
    for r in Na:
        m.addConstr(w[r, r] == 1, f"w2s[{r}]")
        for r2 in Na:
            if r2 != r:
                m.addConstr(w[r, r2] == 0, f"w2c[{r},{r2}]")                                           # (2)
    for r in Na:
        for (i, j) in edges:
            m.addConstr(w[r, j] - w[r, i] <= 1 - q[i, j], f"w3a[{r},{i},{j}]")                         # (3)
            m.addConstr(w[r, i] - w[r, j] <= 1 - q[i, j], f"w3b[{r},{i},{j}]")

    # yapısal per-hücre ödül sınırı (klasik varyanttaki gibi -> güçlü master-LB)
    m.addConstrs((rho[i] <= inst.pi[i] * u[i] for i in Nf), "rho_struct")

    # --- alt problem GEVŞETMESİ: ÇEKİŞME (eşleme/kapasite) — rapor yol haritası -----
    # (C0) kontrol edilemez hücre: hiçbir araç Δwat'ı zamanında veremez -> u_i = 0.
    for i in Nf:
        if not inst.controllable.get(i, True):
            m.addConstr(u[i] == 0, f"C0_uncontrollable[{i}]")

    # (C1) EŞLEME/KAPASİTE gevşetmesi: kesirli atama f_ik ile, MANDATORY (pre) hücreler
    #  kappa_i adet AYRIK araca eşlenebilmelidir (araç tekliği kesirli). Bu, "az sayıda
    #  hızlı araç için yarışan çok hücre" (Hall ihlali) önerilerini master'da anında keser.
    fpairs = [(i, k) for i in Nf for k in inst.A.get(i, [])]
    f = m.addVars(fpairs, lb=0.0, name="f")
    m.addConstrs((gp.quicksum(f[i, k] for k in inst.A[i]) >= inst.kappa[i] * u_pre[i]
                  for i in Nf if inst.A.get(i)), "C1_match_demand")     # pre -> kappa_i araç
    # hiçbir araç tm_UB'den önce yetişemiyorsa (A_i boş) pre-kontrol imkânsız
    for i in Nf:
        if not inst.A.get(i):
            m.addConstr(u_pre[i] == 0, f"C1_no_pre[{i}]")
    kof = {}
    for (i, k) in fpairs:
        kof.setdefault(k, []).append(i)
    m.addConstrs((gp.quicksum(f[i, k] for i in kof[k]) <= 1 for k in kof), "C1_veh_unique")  # araç ≤1
    # (C2) araç bütçesi (aggregate): kontrol edilen hücre sayısı araç sayısını aşamaz.
    m.addConstr(gp.quicksum(u[i] for i in Nf) <= len(inst.K), "C2_veh_budget")

    # NOT (P1 negatif bulgu, 2026-07-12): tip-agregasyonlu STATİK taşıma-kapasite gevşetmesi
    # (Σ_t cap_it·g_it ≥ Δwat·u_i, Σ_i g_it ≤ N_t) denendi ve GEÇERLİ olduğu doğrulandı
    # (sentetik gap 0 korundu), ANCAK 4x4 master UB'sini HİÇ sıkmadı (639.24→639.24): 32 araç
    # / 2 kontrol hücresi olduğundan kapasite bağlayıcı değil — 4x4 gap'i TEMPORALDİR (zamanlama).
    # Statik kapasite yalnız iterasyon dinamiğini bozdu (8x8 yavaşladı). Bu yüzden kaldırıldı;
    # doğru P1 ZAMAN-ÇÖZÜNÜRLÜKLÜ (aralık) kapasite ya da P2 (ağaca-bağlı kesme güçlendirme).

    m.setObjective(gp.quicksum(inst.pi[i] * (1 - y[i]) for i in Nf) + gp.quicksum(rho[i] for i in Nf),
                   GRB.MAXIMIZE)
    V = dict(y=y, u_pre=u_pre, u_post=u_post, u=u, z=z, q=q, bmin=bmin, w=w, rho=rho)
    return m, V


def _path_arcs(i, parent, n):
    arcs, cur, steps = [], i, 0
    while cur in parent and steps <= n:
        arcs.append((parent[cur], cur)); cur = parent[cur]; steps += 1
    return arcs


def _local_sig(inst, i, ybar, parent, q, y):
    """i'nin ts_i (yol) + ts_min_i (yanan komşuluk) imzası; toplam=0 <=> imza aynı."""
    terms = [(1 - q[a, b]) for (a, b) in _path_arcs(i, parent, len(inst.Nf))]
    for j in inst.Nplus(i):
        if ybar[j] > 0.5:
            terms.append(1 - y[j])
            terms += [(1 - q[a, b]) for (a, b) in _path_arcs(j, parent, len(inst.Nf))]
        else:
            terms.append(y[j])
    return terms


def _fallback_incumbent(inst, cfg, sub_solver, tl=None):
    """
    HER ZAMAN geçerli, kendi-tutarlı bir başlangıç inkümbenti:
      TAM YAYILIM (hiç pre-kontrol yok → yangın köklerden erişebildiği her yere yayılır)
      + POST-salvage (yanan hücreleri, fizibıl olduğunca, kontrol edip kısmi ödül topla).
    Post opsiyonel olduğundan alt problem her zaman fizibıldır → geçerli alt sınır.
    Master büyük/birleşik-infeasible C önerip inkümbent üretemese bile LBBD elinde
    anlamlı bir LB tutar (aksi halde 5x5/6x6'da 'inkümbent yok').
    """
    import heapq
    dist = {i: float("inf") for i in inst.Nf}
    parent = {}
    pq = []
    for r in inst.Na:
        dist[r] = 0.0; heapq.heappush(pq, (0.0, r))
    while pq:
        d, i = heapq.heappop(pq)
        if d > dist[i] + 1e-12:
            continue
        wi = inst.alpha[i] / inst.lam[i]              # i yanınca komşuya yayılma süresi
        for j in inst.neighbors[i]:
            nd = d + wi
            if nd < dist[j] - 1e-12:
                dist[j] = nd; parent[j] = i; heapq.heappush(pq, (nd, j))
    reach = set(i for i in inst.Nf if dist[i] < float("inf"))
    ybar = {i: (1.0 if i in reach else 0.0) for i in inst.Nf}
    zbar, qbar = {}, {}
    for i in inst.Nf:
        for j in inst.neighbors[i]:
            zbar[i, j] = 1.0 if ybar[i] > 0.5 else 0.0
            qbar[i, j] = 1.0 if parent.get(j) == i else 0.0
    fp = forward_pass(inst, ybar, zbar, qbar)
    nonburn = sum(inst.pi[i] * (1 - ybar[i]) for i in inst.Nf)
    if fp.feas_cuts:                                  # tutarsız tohum (olmamalı) → salvage'siz LB
        return dict(obj=nonburn, y=ybar, u_pre={i: 0.0 for i in inst.Nf},
                    u_post={i: 0.0 for i in inst.Nf}, p_cell={})
    C = [i for i in inst.Nf if i in reach and inst.e[i] == 0]   # yanan (kök-olmayan) → salvage
    prebar = {i: 0.0 for i in inst.Nf}
    postbar = {i: (1.0 if i in set(C) else 0.0) for i in inst.Nf}
    sub = sub_solver(inst, cfg, C, fp, prebar, postbar, tl=(min(30.0, tl) if tl else None))
    phi = sub.obj if sub.status == "optimal" else 0.0
    p_cell = sub.p_cell if sub.status == "optimal" else {}
    return dict(obj=nonburn + phi, y=ybar, u_pre=prebar, u_post=postbar, p_cell=p_cell)


def _spread_from(inst, blocked):
    """Köklerden min-zamanlı yayılım; `blocked` hücreleri YANAR ama yaymaz (u_pre).
    Döndürür: (reach kümesi, parent, ts-benzeri dist)."""
    import heapq
    dist = {i: float("inf") for i in inst.Nf}
    parent = {}
    pq = []
    for r in inst.Na:
        dist[r] = 0.0; heapq.heappush(pq, (0.0, r))
    while pq:
        d, i = heapq.heappop(pq)
        if d > dist[i] + 1e-12:
            continue
        if i in blocked:                 # pre-kontrol → i yayılmaz
            continue
        wi = inst.alpha[i] / inst.lam[i]
        for j in inst.neighbors[i]:
            nd = d + wi
            if nd < dist[j] - 1e-12:
                dist[j] = nd; parent[j] = i; heapq.heappush(pq, (nd, j))
    return set(i for i in inst.Nf if dist[i] < float("inf")), parent


def _build_config(inst, blocked):
    """`blocked` (pre-kontrol) verildiğinde tutarlı (ybar,zbar,qbar) kur."""
    reach, parent = _spread_from(inst, blocked)
    ybar = {i: (1.0 if i in reach else 0.0) for i in inst.Nf}
    zbar, qbar = {}, {}
    for i in inst.Nf:
        spreads = (ybar[i] > 0.5) and (i not in blocked)
        for j in inst.neighbors[i]:
            zbar[i, j] = 1.0 if spreads else 0.0
            qbar[i, j] = 1.0 if parent.get(j) == i else 0.0
    return reach, ybar, zbar, qbar


def _greedy_incumbent(inst, cfg, sub_solver, tl=None):
    """
    Açgözlü PRE-kontrol (chokepoint) sezgiseli: yüksek alt-ağaç-değerli hücreleri, fizibıl
    oldukça, pre-kontrol edip yayılımı erken durdur; kalan yananları post-salvage et.
    Her aday config `forward_pass` + alt problem ile DOĞRULANIR → geçerli inkümbent.
    Taban: hiç pre-kontrol (tam-yayılım+salvage). Zaman-korumalı.
    """
    import time as _t
    t0 = _t.time()

    def evaluate(blocked):
        """(obj, config) veya None (infeasible)."""
        reach, ybar, zbar, qbar = _build_config(inst, blocked)
        fp = forward_pass(inst, ybar, zbar, qbar)
        if fp.feas_cuts:
            return None
        pre = [i for i in blocked if ybar[i] > 0.5]
        post = [i for i in reach if inst.e[i] == 0 and i not in blocked]
        prebar = {i: (1.0 if i in set(pre) else 0.0) for i in inst.Nf}
        postbar = {i: (1.0 if i in set(post) else 0.0) for i in inst.Nf}
        C = pre + post
        rem = None if tl is None else max(1.0, min(20.0, tl - (_t.time() - t0)))
        sub = sub_solver(inst, cfg, C, fp, prebar, postbar, tl=rem)
        if sub.status != "optimal":
            return None
        nonburn = sum(inst.pi[i] * (1 - ybar[i]) for i in inst.Nf)
        return (nonburn + sub.obj,
                dict(obj=nonburn + sub.obj, y=ybar, u_pre=prebar, u_post=postbar, p_cell=sub.p_cell))

    # taban: pre-kontrol yok
    base = evaluate(set())
    best_obj, best = (base if base else (float("-inf"), None))
    # aday chokepoint'ler: tam-yayılım ağacında alt-ağaç-değeri yüksek hücreler
    reach0, parent0 = _spread_from(inst, set())
    children = {i: [] for i in inst.Nf}
    for j, pi in parent0.items():
        children[pi].append(j)
    subval = {}
    def _sv(i):
        if i in subval: return subval[i]
        v = inst.pi[i] + sum(_sv(c) for c in children[i])
        subval[i] = v; return v
    for i in reach0:
        if i not in inst.Na:            # kökler pre-kontrol edilmez (zaten yanıyor, yaymayı durdurmak mantıksız kökte)
            _sv(i)
    cands = sorted((i for i in reach0 if i not in inst.Na), key=lambda i: -subval.get(i, 0.0))

    blocked = set()
    for i in cands:
        if tl is not None and _t.time() - t0 > tl:
            break
        if any(a in blocked for a in _ancestors(i, parent0)):
            continue                     # zaten üst-akıştaki bir pre-kontrol tarafından budandı
        trial = blocked | {i}
        res = evaluate(trial)
        if res and res[0] > best_obj + 1e-6:
            best_obj, best = res
            blocked = trial              # kabul: daha iyi geçerli inkümbent
    return best


def _ancestors(i, parent):
    a, cur = [], i
    while cur in parent:
        cur = parent[cur]; a.append(cur)
    return a


def _deletion_filter(inst, cfg, cells, fp, prebar, postbar, stats, t0=None, tl=None,
                     sub_solver=solve_assignment_subproblem):
    """
    P3 deletion-filter (Karlsson-Rönnberg 2021, Saken 2023): uygunsuz kontrol kümesinden
    başlayıp her hücreyi tek tek çıkararak IRREDUCIBLE (minimal) çakışma alt kümesi bulur.
    Her yeniden-çözüm daha kısıtlı (küçük) olduğundan MILP-IIS'ten ucuzdur.
    En çok-çekişen (küçük A_i) hücreleri önce dener -> daha hızlı küçülme.
    Süre korumalı: bütçe biterse eldeki (hâlâ geçerli) çakışmayı döndürür.
    """
    conflict = sorted(cells, key=lambda i: len(inst.A.get(i, [])))   # az ulaşılabilir önce
    i = 0
    while i < len(conflict):
        if tl is not None and t0 is not None and time.time() - t0 > tl:
            break                     # bütçe bitti -> mevcut (üst-küme, geçerli) çakışma
        trial = conflict[:i] + conflict[i + 1:]
        stats["sub_solves"] += 1
        if not trial:
            i += 1; continue
        sub_tl = None if tl is None else min(20.0, max(1.0, tl - (time.time() - t0)))
        sub = sub_solver(inst, cfg, trial, fp, prebar, postbar, tl=sub_tl)
        if sub.status == "infeasible":
            conflict = trial          # çıkarılan hücre uygunsuzluk için gereksiz
        else:                         # feasible VEYA unknown -> hücreyi tut (güvenli)
            i += 1
    return conflict


def solve_lbbd(inst: Instance, cfg: Config, verbose: bool = True, max_iter: int = 100000,
               sub_solver=solve_assignment_subproblem):
    m, V = build_lbbd_master(inst, cfg)
    y, u_pre, u_post, u = V["y"], V["u_pre"], V["u_post"], V["u"]
    z, q, w, rho = V["z"], V["q"], V["w"], V["rho"]
    Nf = inst.Nf
    tl = cfg.solver.time_limit
    t0 = time.time()

    stats = {"comb_cuts": 0, "opt_cuts": 0, "feas_cuts": 0, "iters": 0, "sub_solves": 0}
    best_lb, best = float("-inf"), None
    ub = float("inf")

    # --- başlangıç inkümbenti: açgözlü chokepoint sezgiseli (taban: tam-yayılım+salvage).
    #     Master feasible C üretemese de LBBD elinde GÜÇLÜ bir LB tutar (çok-araçlı 5x5/6x6). ---
    fb_tl = None if tl is None else min(tl * 0.5, 60.0)     # inkümbent için süre bütçesi
    fb = _greedy_incumbent(inst, cfg, sub_solver, tl=fb_tl)
    if fb is not None and fb["obj"] > best_lb:
        best_lb, best = fb["obj"], dict(**fb)
        stats["greedy_lb"] = fb["obj"]

    while stats["iters"] < max_iter:
        if tl is not None and time.time() - t0 > tl:
            break
        stats["iters"] += 1
        if tl is not None:
            m.setParam("TimeLimit", max(1.0, tl - (time.time() - t0)))
        m.optimize()
        if m.SolCount == 0:
            break
        ub = m.ObjVal
        ybar = {i: y[i].X for i in Nf}
        zbar = {k: z[k].X for k in z.keys()}
        qbar = {k: q[k].X for k in q.keys()}
        prebar = {i: u_pre[i].X for i in Nf}
        postbar = {i: u_post[i].X for i in Nf}
        rhobar = {i: rho[i].X for i in Nf}

        fp = forward_pass(inst, ybar, zbar, qbar)

        # (46) kombinatoryal uygunluk
        if fp.feas_cuts:
            for arcs in fp.feas_cuts:
                m.addConstr(gp.quicksum(q[a, b] for (a, b) in arcs[:-1]) + z[arcs[-1]] <= len(arcs) - 1)
                stats["comb_cuts"] += 1
            continue

        C = [i for i in Nf if prebar[i] + postbar[i] > 0.5]

        # --- gevşetme imzası: verilen hücre kümesinin kontrol/rejim/yerel-imza değişimi
        def relax_expr(cells):
            rel = gp.LinExpr()
            for i in cells:
                rel += (1 - u[i])
                rel += (1 - u_pre[i]) if prebar[i] > 0.5 else (1 - u_post[i])
                for term in _local_sig(inst, i, ybar, fp.parent, q, y):
                    rel += term
            return rel

        # (1) ÖLÜMCÜL hücreler: yalnız PRE (zorunlu) hücreler için — bireysel olarak
        # tc ≤ tm sağlanamıyorsa ağaç geçersiz -> singleton uygunluk kesmesi.
        # (POST hücre servis edilemezse alt problem onu p=0 bırakır, kesme gerekmez.)
        fatal = [i for i in C if prebar[i] > 0.5
                 and not cell_feasible(inst, cfg, i, fp, prebar[i], postbar[i])]
        if fatal:
            for i in fatal:                      # her biri singleton uygunluk kesmesi
                m.addConstr(relax_expr([i]) >= 1)
                stats["feas_cuts"] += 1
            continue

        # (2) araç-çekişmesi: global atama MILP'i (kalan-süre limitiyle -> askıda kalmaz)
        stats["sub_solves"] += 1
        sub_tl = None if tl is None else min(30.0, max(1.0, tl - (time.time() - t0)))
        sub = sub_solver(inst, cfg, C, fp, prebar, postbar, tl=sub_tl)
        if sub.status == "unknown":
            stats["timeout"] = stats.get("timeout", 0) + 1
            break                              # alt problem kanıtlanamadı -> eldeki inkümbentle dur
        if sub.status == "infeasible":
            # P3: minimal çakışma alt kümesini DELETION-FILTER ile bul (Karlsson-Rönnberg
            # 2021, Saken 2023) — pahalı MILP-IIS yerine; her yeniden-çözüm DAHA KÜÇÜK.
            conflict = _deletion_filter(inst, cfg, C, fp, prebar, postbar, stats, t0, tl, sub_solver)
            m.addConstr(relax_expr(conflict) >= 1)
            stats["feas_cuts"] += 1
            continue

        Phi = sub.obj
        nonburn = sum(inst.pi[i] * (1 - ybar[i]) for i in Nf)
        lb = nonburn + Phi
        if lb > best_lb:
            best_lb = lb
            best = dict(y=ybar, u_pre=prebar, u_post=postbar, p_cell=sub.p_cell,
                        x_assign=sub.x_assign, obj=lb)

        # (a) PER-HÜCRE üst sınır: ρ_i ≤ p_i^solo (çekişmesiz max) + π_i·(imza değişimi).
        #     Master-UB'yi hücre bazında sıkar; imzaya göre tekrar kullanılabilir.
        added_cell = False
        for i in C:
            if rhobar[i] > EPS:
                solo = cell_solo_reward(inst, cfg, i, fp, prebar[i], postbar[i])
                if rhobar[i] > solo + EPS + 1e-6 * abs(solo):
                    m.addConstr(rho[i] <= solo + inst.pi[i] * relax_expr([i]))
                    stats["opt_cuts"] += 1
                    added_cell = True

        sum_rho_C = sum(rhobar[i] for i in C)
        if sum_rho_C <= Phi + EPS + 1e-6 * abs(Phi):
            break                              # ρ toplamı Φ ile tutarlı -> UB=LB -> optimal
        if added_cell:
            continue                           # önce per-hücre kesmeler etkisini göstersin

        # (b) çekişme düzeltmesi: Σ_{i∈C} ρ_i ≤ Φ + Θ·(imza değişimi)
        Theta = sum(inst.pi[i] for i in C) + 1.0
        m.addConstr(gp.quicksum(rho[i] for i in C) <= Phi + Theta * relax_expr(C))
        stats["opt_cuts"] += 1

    wall = time.time() - t0
    gap = (None if best is None or ub == float("inf")
           else abs(ub - best_lb) / (abs(best_lb) + 1e-9))
    result = {
        "instance": inst.name, "method": "lbbd",
        "obj": (best["obj"] if best else None),
        "bound": ub if ub != float("inf") else None,
        "gap": gap, "wall": wall,
        "master_vars": m.NumVars, "master_constrs": m.NumConstrs,
        **{f"cuts_{k}": v for k, v in stats.items() if k.endswith("cuts")},
        "iters": stats["iters"], "sub_solves": stats["sub_solves"],
        "cuts_total": stats["comb_cuts"] + stats["opt_cuts"] + stats["feas_cuts"],
    }
    if verbose and best:
        print(f">>> {inst.name} [lbbd] obj={result['obj']:.4f} gap={gap:.4%} "
              f"time={wall:.2f}s iters={stats['iters']} "
              f"cuts(c/o/f)={stats['comb_cuts']}/{stats['opt_cuts']}/{stats['feas_cuts']}")
    return m, V, result
