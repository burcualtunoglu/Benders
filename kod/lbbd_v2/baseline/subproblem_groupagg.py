"""subproblem_groupagg.py — GRUP-TOPLULAŞTIRILMIŞ güçlü alt problem (2026-09-24, Geliştirme 1).

Bireysel güçlü SP'nin (lbbd_subproblem_resource._build; Lemma A: varış = v^min sabit) araç
değiştirilebilirlik grupları üzerinden EŞDEĞER gösterimi. Grup g = {k : d_ik = d_ik' ∀i∈N_f, µ_k = µ_k'}
(preprocessing.compute_resource_groups; TAM eşitlik, yuvarlama yok).

Değişkenler (i∈C, g∈G):
  n_ig ∈ {0..N_g}  hücre i'ye atanan g-grubu araç sayısı        (= Σ_{k∈g} x_ik)
  u_ig ∈ {0,1}     grup g hücre i'de kullanıldı (n_ig ≥ 1)       (n_ig ≤ N_g u_ig, u_ig ≤ n_ig)
  W_ig ≥ 0         g-grubu araçlarının i'deki TOPLAM hizmet süresi (= Σ_{k∈g} s_ik)
  win_ig ≥ 0       hizmet penceresi tc_i − v^min_ig (kullanılmıyorsa 0)
  β_igb ∈ {0,1}    n_ig'nin ikili açılımı (n_ig = Σ_b 2^b β_igb)
  γ_igb ≥ 0        β_igb · win_ig (KESİN doğrusallaştırma: γ ≤ win, γ ≤ Wcap·β)  → W_ig ≤ Σ_b 2^b γ_igb = n_ig·win_ig
  w_ig ∈ [0,1]     ω^min seçicisi (midpoint)                    (= Σ_{k∈g} w_ik)
  tc_i, p_i, ω_i, ω^max_i, ω^min_i  hücre değişkenleri (bireysel modelle aynı)

Kısıtlar bireysel güçlü SP'nin birebir grup karşılığıdır:
  ω^max_i ≥ D_ig u_ig            (bireysel: ω^max_i ≥ D_ik x_ik, D_ig = a_i (v^min_ig − t^s_i) sabit)
  win_ig ≤ tc_i − v^min_ig·u_ig ; win_ig ≤ Wcap_ig u_ig   (bireysel: s_ik ≤ tc_i − v^min_ik + M(1−x_ik), s_ik ≤ M x_ik)
  W_ig ≤ n_ig·win_ig (ikili açılım + McCormick üst sınırı; n tamsayı olduğundan kesin)
  Σ_g µ_g W_ig ≥ ω_i             (bireysel: Σ_k µ_k s_ik ≥ ω_i)
  ω^min_i = Σ_g D_ig w_ig, Σ_g w_ig = 1, w_ig ≤ u_ig ; 2ω_i = ω^max_i + ω^min_i + 2δ_wat ; ω_i ≤ M^s_i
  Σ_g n_ig ≥ 1 ; lo_i ≤ tc_i ≤ hi_i ; p_i ≤ π_i − β_i (tc_i − t^s_i) ; p_i ≤ π_i ; Σ_i n_ig ≤ N_g

Eşdeğerlik (fizibilite + amaç koruyan dönüşümler; docs/groupagg_sp.md):
  bireysel → toplulaştırılmış: n_ig = Σ_{k∈g} x_ik, W_ig = Σ_{k∈g} s_ik, w_ig = Σ_{k∈g} w_ik, u_ig = [n_ig ≥ 1],
     win_ig = (tc_i − v^min_ig) u_ig, β = bits(n), γ_b = β_b win; hücre değişkenleri aynı. Her bireysel kısıt grup
     kısıtını doğurur (s_ik ≤ tc − v^min_ig ∀ atanan k ⇒ W ≤ n·win). Amaç aynı (Σ p).
  toplulaştırılmış → bireysel: grubun ilk n_ig boş üyesine x_ik = 1, s_ik = W_ig/n_ig (≤ win_ig), w_ik = w_ig/n_ig;
     Σ_i n_ig ≤ N_g farklı araç seçilebilmesini garanti eder; ω^max ≥ D_ig u_ig = D_ik x_ik; su, seçici ve pencere
     kısıtları toplam korunduğu için sağlanır. Amaç aynı.
  ⇒ Φ_agg = Φ_ind (aynı optimum değer). LP gevşetmeleri ve süre sınırlı çözücü sonuçları FARKLI olabilir.

Bu modül özgün model/ana problem değişikliği DEĞİLDİR; yalnız SP gösterimi. Üretim varsayılanı: strong_resource.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from lbbd_v2.config import Config
from lbbd_v2.data_loader import Instance
from lbbd_v2.preprocessing import Preprocessed
from lbbd_v2.benders_forward import ForwardResult
from lbbd_v2.lbbd_subproblem_resource import (SPResult, SPStatus, _v_min, _tc_window,
                                              _optimize_capped, _map_status)

SP_GROUPAGG = "group_agg"


# ----------------------------------------------------------------------------- veri
def pair_data(cells: List[int], regime: Dict[int, str], fwd: ForwardResult,
              inst: Instance, pre: Preprocessed) -> Dict[str, Any]:
    """(i,g) çifti sabitleri: v^min_ig, D_ig, pencere [lo_i, hi_i], Wcap_ig, fizibil çift mi."""
    groups = list(pre.groups)
    gmap = {g.gid: g for g in groups}
    vmin, D, Wcap, feas = {}, {}, {}, {}
    lo, hi = {}, {}
    for i in cells:
        lo[i], hi[i] = _tc_window(fwd, i, regime[i])
        for g in groups:
            k0 = g.members[0]
            v = _v_min(fwd, inst, i, k0)                 # grup içinde d_ik eşit ⇒ v^min eşit
            vmin[i, g.gid] = v
            D[i, g.gid] = inst.a[i] * (v - fwd.ts[i])
            Wcap[i, g.gid] = max(0.0, hi[i] - v)
            feas[i, g.gid] = (v <= hi[i] + 1e-12)        # v^min > hi ⇒ atanamaz (tc ≥ v^min > hi çelişki)
    return dict(groups=groups, gmap=gmap, vmin=vmin, D=D, Wcap=Wcap, feas=feas, lo=lo, hi=hi)


# ----------------------------------------------------------------------------- model
def build_groupagg(cells: List[int], regime: Dict[int, str], fwd: ForwardResult,
                   inst: Instance, pre: Preprocessed, cfg: Config,
                   integer: bool = True, mip_gap: Optional[float] = None,
                   prune_infeasible_pairs: bool = True) -> Tuple[gp.Model, dict]:
    m = gp.Model("sp_groupagg")
    m.setParam("OutputFlag", 0)
    m.setParam("Seed", cfg.gurobi_seed)
    m.setParam("Threads", cfg.gurobi_threads)
    eff_gap = getattr(cfg, "sp_mip_gap", 0.0) if mip_gap is None else mip_gap
    m.setParam("MIPGap", eff_gap)
    midpoint = (cfg.omega_mode == "midpoint")
    pd = pair_data(cells, regime, fwd, inst, pre)
    groups, gmap = pd["groups"], pd["gmap"]
    gids = [g.gid for g in groups]
    pairs = [(i, gid) for i in cells for gid in gids]
    ivt = GRB.INTEGER if integer else GRB.CONTINUOUS
    bvt = GRB.BINARY if integer else GRB.CONTINUOUS
    pi, beta = inst.pi, inst.beta

    n = m.addVars(pairs, vtype=ivt, lb=0.0, name="n")
    u = m.addVars(pairs, vtype=bvt, lb=0.0, ub=1.0, name="u")
    W = m.addVars(pairs, lb=0.0, name="W")
    win = m.addVars(pairs, lb=0.0, name="win")
    tc = m.addVars(cells, lb=0.0, name="tc")
    p = m.addVars(cells, lb=0.0, name="p")
    omega = m.addVars(cells, lb=0.0, name="omega")
    omega_max = m.addVars(cells, lb=0.0, name="omega_max")
    if midpoint:
        omega_min = m.addVars(cells, lb=0.0, name="omega_min")
        w = m.addVars(pairs, lb=0.0, ub=1.0, name="w")
    else:
        omega_min = w = None
    bits: Dict[Tuple[int, int], List[gp.Var]] = {}
    gam: Dict[Tuple[int, int], List[gp.Var]] = {}

    m.setObjective(gp.quicksum(p[i] for i in cells), GRB.MAXIMIZE)

    for (i, gid) in pairs:
        Ng = gmap[gid].size
        n[i, gid].UB = Ng
        if prune_infeasible_pairs and not pd["feas"][i, gid]:
            n[i, gid].UB = 0.0; u[i, gid].UB = 0.0; W[i, gid].UB = 0.0; win[i, gid].UB = 0.0
            if midpoint:
                w[i, gid].UB = 0.0
            bits[i, gid] = []; gam[i, gid] = []
            m.addConstr(n[i, gid] == 0, f"pruned[{i},{gid}]")
            continue
        Bg = max(1, math.ceil(math.log2(Ng + 1)))
        bl, gl = [], []
        for b in range(Bg):
            bb = m.addVar(vtype=bvt, lb=0.0, ub=1.0, name=f"beta[{i},{gid},{b}]")
            gg = m.addVar(lb=0.0, name=f"gam[{i},{gid},{b}]")
            bl.append(bb); gl.append(gg)
            m.addConstr(gg <= win[i, gid], f"gam_win[{i},{gid},{b}]")
            m.addConstr(gg <= pd["Wcap"][i, gid] * bb, f"gam_bit[{i},{gid},{b}]")
        bits[i, gid] = bl; gam[i, gid] = gl
        m.addConstr(n[i, gid] == gp.quicksum((2 ** b) * bl[b] for b in range(Bg)), f"bits[{i},{gid}]")
        m.addConstr(n[i, gid] <= Ng * u[i, gid], f"n_le_Nu[{i},{gid}]")
        m.addConstr(u[i, gid] <= n[i, gid], f"u_le_n[{i},{gid}]")
        # pencere: kullanılıyorsa tc − v^min (Big-M'siz: u=0 ⇒ win ≤ tc, ayrıca win ≤ Wcap·u = 0)
        m.addConstr(win[i, gid] <= tc[i] - pd["vmin"][i, gid] * u[i, gid], f"win_tc[{i},{gid}]")
        m.addConstr(win[i, gid] <= pd["Wcap"][i, gid] * u[i, gid], f"win_cap[{i},{gid}]")
        # toplam hizmet ≤ n·win (kesin: n tamsayı, β ikili)
        m.addConstr(W[i, gid] <= gp.quicksum((2 ** b) * gl[b] for b in range(Bg)), f"W_le_nwin[{i},{gid}]")
        # ω^max ≥ D_ig u_ig
        m.addConstr(omega_max[i] >= pd["D"][i, gid] * u[i, gid], f"omax[{i},{gid}]")
        if midpoint:
            m.addConstr(w[i, gid] <= u[i, gid], f"omin_le_u[{i},{gid}]")

    for i in cells:
        m.addConstr(tc[i] >= pd["lo"][i], f"tc_lo[{i}]")
        m.addConstr(tc[i] <= pd["hi"][i], f"tc_hi[{i}]")
        m.addConstr(p[i] <= pi[i] - beta[i] * (tc[i] - fwd.ts[i]), f"p_reward[{i}]")
        m.addConstr(p[i] <= pi[i], f"p_cap[{i}]")
        m.addConstr(gp.quicksum(n[i, gid] for gid in gids) >= 1, f"must_serve[{i}]")
        if midpoint:
            m.addConstr(omega_min[i] == gp.quicksum(pd["D"][i, gid] * w[i, gid] for gid in gids), f"omin[{i}]")
            m.addConstr(gp.quicksum(w[i, gid] for gid in gids) == 1, f"omin_sel[{i}]")
            m.addConstr(2 * omega[i] == omega_max[i] + omega_min[i] + 2 * inst.delta_wat, f"omega_mid[{i}]")
        else:
            m.addConstr(omega[i] == omega_max[i] + inst.delta_wat, f"omega_wc[{i}]")
        m.addConstr(omega[i] <= pre.M_s[i], f"omega_cap[{i}]")
        m.addConstr(gp.quicksum(gmap[gid].mu * W[i, gid] for gid in gids) >= omega[i], f"water[{i}]")

    for g in groups:
        m.addConstr(gp.quicksum(n[i, g.gid] for i in cells) <= g.size, f"cap[{g.gid}]")

    handles = dict(n=n, u=u, W=W, win=win, w=w, bits=bits, gam=gam, tc=tc, p=p, omega=omega,
                   omega_max=omega_max, omega_min=omega_min, pairs=pairs, gids=gids, pd=pd, cells=cells)
    return m, handles


# ----------------------------------------------------------------------------- dönüşümler
def aggregate_solution(sol_ind: Dict[str, Any], pre: Preprocessed, cells: List[int]) -> Dict[str, Any]:
    """Bireysel çözüm {x:{(i,k)}, s:{(i,k)}, w:{(i,k)}, tc,p,omega,omega_max,omega_min:{i}} → grup çözümü."""
    gof = pre.group_of
    n: Dict[Tuple[int, int], int] = {}
    W: Dict[Tuple[int, int], float] = {}
    w: Dict[Tuple[int, int], float] = {}
    for (i, k), v in (sol_ind.get("x") or {}).items():
        if v > 0.5:
            n[i, gof[k]] = n.get((i, gof[k]), 0) + 1
    for (i, k), v in (sol_ind.get("s") or {}).items():
        W[i, gof[k]] = W.get((i, gof[k]), 0.0) + v
    for (i, k), v in (sol_ind.get("w") or {}).items():
        w[i, gof[k]] = w.get((i, gof[k]), 0.0) + v
    out = {k: dict(v) for k, v in sol_ind.items() if k in ("tc", "p", "omega", "omega_max", "omega_min")}
    out.update(n=n, W=W, w=w)
    return out


def disaggregate_solution(sol_agg: Dict[str, Any], pre: Preprocessed, cells: List[int],
                          K: Iterable[int]) -> Dict[str, Any]:
    """Grup çözümü → bireysel çözüm: grubun (indeks sırasıyla) ilk n_ig boş üyesi; s = W/n, w = w_g/n.
    Σ_i n_ig ≤ N_g olduğundan her araç en çok bir hücre alır (c33)."""
    Kset = set(K)
    x: Dict[Tuple[int, int], float] = {}
    s: Dict[Tuple[int, int], float] = {}
    w: Dict[Tuple[int, int], float] = {}
    ptr = {g.gid: 0 for g in pre.groups}
    members = {g.gid: [k for k in g.members if k in Kset] for g in pre.groups}
    for i in sorted(cells):
        for g in pre.groups:
            cnt = int(round(sol_agg.get("n", {}).get((i, g.gid), 0)))
            if cnt <= 0:
                continue
            Wig = sol_agg.get("W", {}).get((i, g.gid), 0.0)
            wig = sol_agg.get("w", {}).get((i, g.gid), 0.0)
            for _ in range(cnt):
                if ptr[g.gid] >= len(members[g.gid]):
                    raise ValueError(f"grup {g.gid} kapasitesi aşıldı (Σ_i n_ig > N_g)")
                k = members[g.gid][ptr[g.gid]]; ptr[g.gid] += 1
                x[i, k] = 1.0; s[i, k] = Wig / cnt; w[i, k] = wig / cnt
    out = {k: dict(v) for k, v in sol_agg.items() if k in ("tc", "p", "omega", "omega_max", "omega_min")}
    out.update(x=x, s=s, w=w)
    return out


def check_individual(sol: Dict[str, Any], cells: List[int], regime: Dict[int, str], fwd: ForwardResult,
                     inst: Instance, pre: Preprocessed, cfg: Config, tol: float = 1e-6) -> Dict[str, Any]:
    """Bireysel güçlü SP kısıtlarının ÇÖZÜCÜSÜZ değerlendirmesi (bağımsız doğrulama). Kısıt ailesi başına
    en büyük ihlal; 'ok' = hepsi ≤ tol; 'obj' = Σ p."""
    midpoint = (cfg.omega_mode == "midpoint")
    K = list(inst.K)
    x = sol.get("x") or {}; s = sol.get("s") or {}; w = sol.get("w") or {}
    tc, p, om, omx = sol["tc"], sol["p"], sol["omega"], sol["omega_max"]
    omn = sol.get("omega_min") or {}
    viol: Dict[str, float] = {}
    def bump(name, v):
        viol[name] = max(viol.get(name, 0.0), v)
    assigned = {i: [k for k in K if x.get((i, k), 0.0) > 0.5] for i in cells}
    for i in cells:
        lo, hi = _tc_window(fwd, i, regime[i])
        bump("tc_lo", lo - tc[i]); bump("tc_hi", tc[i] - hi)
        bump("p_reward", p[i] - (inst.pi[i] - inst.beta[i] * (tc[i] - fwd.ts[i]))); bump("p_cap", p[i] - inst.pi[i])
        bump("p_nonneg", -p[i])
        bump("must_serve", 1 - len(assigned[i]))
        water = 0.0
        for k in K:
            xik = x.get((i, k), 0.0); sik = s.get((i, k), 0.0)
            if xik > 0.5:
                vm = _v_min(fwd, inst, i, k)
                bump("omax", inst.a[i] * (vm - fwd.ts[i]) - omx[i])
                bump("s_win", sik - (tc[i] - vm))
            else:
                bump("s_off", sik)
            bump("s_nonneg", -sik)
            water += inst.mu[k] * sik
        bump("water", om[i] - water)
        bump("omega_cap", om[i] - pre.M_s[i])
        if midpoint:
            wsum = sum(w.get((i, k), 0.0) for k in K)
            bump("omin_sel", abs(wsum - 1.0))
            omin_val = sum(inst.a[i] * (_v_min(fwd, inst, i, k) - fwd.ts[i]) * w.get((i, k), 0.0) for k in K)
            bump("omin", abs(omn[i] - omin_val))
            for k in K:
                bump("omin_le_x", w.get((i, k), 0.0) - x.get((i, k), 0.0))
            bump("omega_mid", abs(2 * om[i] - omx[i] - omn[i] - 2 * inst.delta_wat))
        else:
            bump("omega_wc", abs(om[i] - omx[i] - inst.delta_wat))
    for k in K:
        bump("cap", sum(x.get((i, k), 0.0) for i in cells) - 1)
    mx = max(viol.values()) if viol else 0.0
    return dict(ok=(mx <= tol), max_violation=mx, violations={k: v for k, v in viol.items() if v > tol},
                obj=sum(p[i] for i in cells))


def _set_start(h, sol_agg: Dict[str, Any], integer_only: bool = False) -> int:
    """Grup çözümünü MIP start olarak ver (sabitleme DEĞİL). Döndürür: Start verilen değişken sayısı."""
    pd = h["pd"]; cnt = 0
    for (i, gid) in h["pairs"]:
        nv = int(round(sol_agg.get("n", {}).get((i, gid), 0)))
        h["n"][i, gid].Start = nv; h["u"][i, gid].Start = 1.0 if nv >= 1 else 0.0; cnt += 2
        for b, bb in enumerate(h["bits"][i, gid]):
            bb.Start = float((nv >> b) & 1); cnt += 1
        if integer_only:
            continue
        tci = sol_agg["tc"][i]
        winv = max(0.0, tci - pd["vmin"][i, gid]) if nv >= 1 else 0.0
        h["win"][i, gid].Start = winv
        h["W"][i, gid].Start = sol_agg.get("W", {}).get((i, gid), 0.0)
        for b, gg in enumerate(h["gam"][i, gid]):
            gg.Start = winv * float((nv >> b) & 1)
        if h["w"] is not None:
            h["w"][i, gid].Start = sol_agg.get("w", {}).get((i, gid), 0.0)
        cnt += 3 + len(h["gam"][i, gid])
    if not integer_only:
        for i in h["cells"]:
            for name in ("tc", "p", "omega", "omega_max"):
                h[name][i].Start = sol_agg[name][i]; cnt += 1
            if h["omega_min"] is not None and "omega_min" in sol_agg:
                h["omega_min"][i].Start = sol_agg["omega_min"][i]; cnt += 1
    return cnt


def extract_solution(h, inst: Instance, pre: Preprocessed) -> Dict[str, Any]:
    cells = h["cells"]
    sol = {"tc": {i: h["tc"][i].X for i in cells}, "p": {i: h["p"][i].X for i in cells},
           "omega": {i: h["omega"][i].X for i in cells}, "omega_max": {i: h["omega_max"][i].X for i in cells},
           "n": {}, "W": {}, "w": {}}
    if h["omega_min"] is not None:
        sol["omega_min"] = {i: h["omega_min"][i].X for i in cells}
    for (i, gid) in h["pairs"]:
        nv = int(round(h["n"][i, gid].X))
        if nv >= 1:
            sol["n"][i, gid] = nv
            sol["W"][i, gid] = h["W"][i, gid].X
            if h["w"] is not None:
                sol["w"][i, gid] = h["w"][i, gid].X
    return sol


# ----------------------------------------------------------------------------- çözücü
def solve_groupagg_subproblem(fwd: ForwardResult, C: Iterable[int], regime: Dict[int, str],
                              inst: Instance, pre: Preprocessed, cfg: Config,
                              time_budget: Optional[float] = None,
                              warm_x: Optional[Dict[Tuple[int, int], float]] = None,
                              symmetry_break: Optional[bool] = None,
                              mip_gap: Optional[float] = None,
                              warm_full: Optional[Dict[str, Any]] = None,
                              timing: Optional[dict] = None) -> SPResult:
    """Boru hattı imzası güçlü SP ile aynı. warm_x (bireysel x) → n/u/β Start'a dönüştürülür;
    warm_full (tam bireysel çözüm) → tüm değişkenlere Start. symmetry_break anlamsız (simetri yapısal olarak yok).
    Dönen solution: n, W, w, tc, p, omega + yeniden üretilmiş bireysel x, s (disaggregate) + recon_check."""
    cells = sorted(C)
    if not cells:
        return SPResult(SPStatus.OPTIMAL, 0.0, 0.0, 0.0, 0.0, {})
    tb = time.time()
    m, h = build_groupagg(cells, regime, fwd, inst, pre, cfg, integer=True, mip_gap=mip_gap)
    build_s = time.time() - tb
    tt = time.time()
    if warm_full is not None:
        _set_start(h, aggregate_solution(warm_full, pre, cells), integer_only=False)
    elif warm_x is not None:
        _set_start(h, aggregate_solution({"x": warm_x}, pre, cells), integer_only=True)
    transform_s = time.time() - tt
    t0 = time.time()
    _optimize_capped(m, time_budget)
    if m.Status == GRB.INF_OR_UNBD:
        m.setParam("DualReductions", 0)
        rem = None if time_budget is None else max(1.0, time_budget - (time.time() - t0))
        _optimize_capped(m, rem)
    runtime = time.time() - t0
    status = _map_status(m)
    obj = m.ObjVal if m.SolCount > 0 else None
    obj_bound = m.ObjBound if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL) else None
    try:
        gap = m.MIPGap if m.SolCount > 0 else None
    except Exception:
        gap = None
    solution: Dict[str, Any] = {}
    recon_s = 0.0
    if m.SolCount > 0:
        tr = time.time()
        sol = extract_solution(h, inst, pre)
        ind = disaggregate_solution(sol, pre, cells, inst.K)
        chk = check_individual(ind, cells, regime, fwd, inst, pre, cfg)
        recon_s = time.time() - tr
        solution = dict(sol)
        solution["x"] = ind["x"]; solution["s"] = ind["s"]
        solution["recon_check"] = {"ok": chk["ok"], "max_violation": chk["max_violation"], "obj": chk["obj"]}
        if not chk["ok"]:
            # yeniden üretilen bireysel çözüm özgün kısıtları ihlal ediyorsa DOĞRULANMIŞ değildir
            status = SPStatus.NUMERICAL_FAILURE
    if timing is not None:
        timing.update(build_s=build_s, transform_s=transform_s, solve_s=runtime, recon_s=recon_s,
                      n_vars=m.NumVars, n_bin=m.NumBinVars, n_int=m.NumIntVars, n_constrs=m.NumConstrs)
    return SPResult(status, obj, obj_bound, gap, runtime, solution)
