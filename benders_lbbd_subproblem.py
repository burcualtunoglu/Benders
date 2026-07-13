"""
benders_lbbd_subproblem.py — LBBD alt problemi: ATAMA + ÇİZELGELEME (rapor §4.6).

Klasik varyantta $x_{ik}$ master'daydı ve alt problem saf LP idi. LBBD'de $x_{ik}$
alt problemE devredilir; master yalnız ağaç+rejime karar verir. Master'ın verdiği
yayılım (ȳ,z̄,q̄) ve rejim (ū,ūpre,ūpost) ile ileri geçiş zamanları (ts,tm,te,ts_min)
SABİT olduğunda, alt problem küçük bir MILP'tir:

  * karar: her KONTROL EDİLEN hücre için hangi araçların atanacağı (x_ik ikili) +
           çizelgeleme (t,v,s,δ,ω,tc,p).
  * kısıt: (2)-(7),(16w),(18),(21)-(25),(29),(33)-(38); araç tekliği (33) hücreleri
           GLOBAL olarak bağlar -> tek (kök-ayrık değil) MILP.
  * amaç : max Σ_{i∈C} p_i.

omega tanımı cfg.model.omega_mode ile seçilir:
  * "worstcase" : ω_i = ω_max_i + Δwat·u_i  (16w) — h_max/h_min/ω_min DÜŞER, model küçük.
  * "midpoint"  : 2ω_i = ω_max_i + ω_min_i + 2Δwat·u_i  (makale (8)-(16)) — h_max/h_min/ω_min
                  ikilileri eklenir (model_monolithic.py ile BİREBİR); alt problem büyür.
Zamanlar (ts,tm,te,ts_min) master'dan sabit olduğundan iki modda da model küçüktür.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import gurobipy as gp
from gurobipy import GRB

from data_loader import Instance
from config import Config
from benders_forward import ForwardResult


@dataclass
class LBBDSubResult:
    status: str                              # "optimal" | "infeasible"
    obj: float = 0.0
    p_cell: Dict[int, float] = field(default_factory=dict)   # per-hücre p_i*
    x_assign: Dict[tuple, float] = field(default_factory=dict)
    conflict: List[int] = field(default_factory=list)        # minimal çakışan hücreler (IIS)


def _cells_in_iis(sp, C):
    """IIS'teki kısıt adlarından çakışan hücreleri (C içindekiler) çıkar."""
    import re
    Cset = set(C)
    conflict = set()
    for c in sp.getConstrs():
        if c.IISConstr:
            for tok in re.findall(r"-?\d+", c.ConstrName):
                v = int(tok)
                if v in Cset:
                    conflict.add(v)
    return sorted(conflict)


def cell_feasible(inst: Instance, cfg: Config, i: int, fp: ForwardResult,
                  pre_i: float, post_i: float) -> bool:
    """
    Tek hücrenin BİREYSEL kontrol edilebilirliği (tüm araçlar serbest, teklik yok).
    x sürekli [0,1] gevşetmesiyle ucuz bir LP; uygunsuz ise hücre GERÇEKTEN kontrol
    edilemez (necessary koşul) -> singleton uygunluk kesmesi. Araç-çekişmesini test
    ETMEZ (onu global MILP çözer).
    """
    mc = cfg.model
    K = inst.K
    Md, Mi, Dwat, Dbuf = inst.Md, inst.Mi, mc.delta_wat, mc.delta_buf
    Ms = inst.Ms_cell[i] if mc.ms_mode == "percell" else inst.Ms
    ts, tm, te, ts_min = fp.ts[i], fp.tm[i], fp.te[i], fp.ts_min[i]

    sp = gp.Model(f"feas_{i}")
    sp.setParam("OutputFlag", 0)
    x = sp.addVars(K, lb=0.0, ub=1.0, name="x")
    t = sp.addVars(K, lb=0.0); v = sp.addVars(K, lb=0.0)
    s = sp.addVars(K, lb=0.0); delta = sp.addVars(K, lb=0.0)
    omax = sp.addVar(lb=0.0); omega = sp.addVar(lb=0.0); tc = sp.addVar(lb=0.0)
    # worstcase: ω = ω_max + Δwat (kesin).  midpoint: ω = Δwat kullanılır — midpoint
    # ω ≥ Δwat olduğundan bu GEÇERLİ bir GEVŞETMEdir → gereklilik testi sağlam kalır
    # (yalnız su ≥ Δwat bile sağlanamıyorsa hücre gerçekten kontrol edilemez).
    if mc.omega_mode == "midpoint":
        sp.addConstr(omega == Dwat)
    else:
        sp.addConstr(omega == omax + Dwat)
    sp.addConstr(gp.quicksum(inst.mu[k] * s[k] for k in K) >= omega)
    sp.addConstr(tc <= te); sp.addConstr(tc >= ts)
    if pre_i > 0.5:
        sp.addConstr(tc <= tm)
    if post_i > 0.5:
        sp.addConstr(tc >= tm)
    sp.addConstr(gp.quicksum(x[k] for k in K) >= 1)
    for k in K:
        sp.addConstr(delta[k] >= inst.a[i] * (v[k] - ts) - Mi[i] * (1 - x[k]))
        sp.addConstr(delta[k] <= Mi[i] * x[k])
        sp.addConstr(omax >= delta[k] - Ms * (1 - x[k]))
        sp.addConstr(v[k] >= t[k] + inst.d[i, k] - Md * (1 - x[k]))
        sp.addConstr(s[k] <= Md * x[k]); sp.addConstr(v[k] <= Md * x[k])
        sp.addConstr(v[k] + s[k] <= tc + Md * (1 - x[k]))
        sp.addConstr(v[k] >= ts - Md * (1 - x[k]))
        sp.addConstr(t[k] >= ts_min + Dbuf - Md * (1 - x[k]))
    sp.setObjective(0, GRB.MAXIMIZE)
    sp.optimize()
    return sp.Status == GRB.OPTIMAL


def cell_solo_reward(inst: Instance, cfg: Config, i: int, fp: ForwardResult,
                     pre_i: float, post_i: float) -> float:
    """
    i hücresinin ÇEKİŞMESİZ (tüm araçlar serbest) en yüksek ödülü — ρ_i için GEÇERLİ
    üst sınır (çekişme ödülü yalnız DÜŞÜRÜR). x sürekli gevşetmesiyle ucuz LP; verilen
    ts_i (yol) ve rejime bağlıdır -> imza aynıyken tekrar kullanılabilir. Uygunsuzsa 0.
    """
    mc = cfg.model
    K = inst.K
    Md, Mi, Dwat, Dbuf = inst.Md, inst.Mi, mc.delta_wat, mc.delta_buf
    Ms = inst.Ms_cell[i] if mc.ms_mode == "percell" else inst.Ms
    ts, tm, te, ts_min = fp.ts[i], fp.tm[i], fp.te[i], fp.ts_min[i]

    sp = gp.Model(f"solo_{i}")
    sp.setParam("OutputFlag", 0)
    x = sp.addVars(K, lb=0.0, ub=1.0); t = sp.addVars(K, lb=0.0); v = sp.addVars(K, lb=0.0)
    s = sp.addVars(K, lb=0.0); delta = sp.addVars(K, lb=0.0)
    omax = sp.addVar(lb=0.0); omega = sp.addVar(lb=0.0); tc = sp.addVar(lb=0.0); p = sp.addVar(lb=0.0)
    sp.addConstr(p <= inst.pi[i] - inst.beta[i] * (tc - ts))
    sp.addConstr(p <= inst.pi[i])
    # worstcase: ω = ω_max + Δwat.  midpoint: ω = Δwat — midpoint ω ≥ Δwat olduğundan
    # daha KÜÇÜK ω → daha erken tc → daha YÜKSEK ödül ⇒ ρ_i için GEÇERLİ üst sınır.
    if mc.omega_mode == "midpoint":
        sp.addConstr(omega == Dwat)
    else:
        sp.addConstr(omega == omax + Dwat)
    sp.addConstr(gp.quicksum(inst.mu[k] * s[k] for k in K) >= omega)
    sp.addConstr(tc <= te); sp.addConstr(tc >= ts)
    if pre_i > 0.5:
        sp.addConstr(tc <= tm)
    if post_i > 0.5:
        sp.addConstr(tc >= tm)
    sp.addConstr(gp.quicksum(x[k] for k in K) >= 1)
    for k in K:
        sp.addConstr(delta[k] >= inst.a[i] * (v[k] - ts) - Mi[i] * (1 - x[k]))
        sp.addConstr(delta[k] <= Mi[i] * x[k])
        sp.addConstr(omax >= delta[k] - Ms * (1 - x[k]))
        sp.addConstr(v[k] >= t[k] + inst.d[i, k] - Md * (1 - x[k]))
        sp.addConstr(s[k] <= Md * x[k]); sp.addConstr(v[k] <= Md * x[k])
        sp.addConstr(v[k] + s[k] <= tc + Md * (1 - x[k]))
        sp.addConstr(v[k] >= ts - Md * (1 - x[k]))
        sp.addConstr(t[k] >= ts_min + Dbuf - Md * (1 - x[k]))
    sp.setObjective(p, GRB.MAXIMIZE)
    sp.optimize()
    return sp.ObjVal if sp.Status == GRB.OPTIMAL else 0.0


def solve_assignment_subproblem(inst: Instance, cfg: Config, controlled: List[int],
                                fp: ForwardResult, prebar: Dict[int, float],
                                postbar: Dict[int, float],
                                compute_conflict: bool = False,
                                tl: float = None) -> LBBDSubResult:
    """Kontrol edilen hücreler için atama+çizelgeleme MILP'i (zamanlar sabit)."""
    if not controlled:
        return LBBDSubResult(status="optimal", obj=0.0)

    mc = cfg.model
    K = inst.K
    Md, Mi, Dwat, Dbuf = inst.Md, inst.Mi, mc.delta_wat, mc.delta_buf

    def Ms(i):
        return inst.Ms_cell[i] if mc.ms_mode == "percell" else inst.Ms

    ts, tm, te, ts_min = fp.ts, fp.tm, fp.te, fp.ts_min
    C = controlled
    P = [i for i in C if prebar[i] > 0.5]        # PRE: zorunlu (ağacı etkiler)
    Q = [i for i in C if postbar[i] > 0.5]       # POST: opsiyonel (yalnız ödül)
    pairs = [(i, k) for i in C for k in K]

    sp = gp.Model("lbbd_sub")
    sp.setParam("OutputFlag", 0)
    sp.setParam("DualReductions", 0)      # INF_OR_UNBD yerine kesin INFEASIBLE (model sınırlı)
    if tl and tl > 0:
        sp.setParam("TimeLimit", max(1.0, tl))

    x = sp.addVars(pairs, vtype=GRB.BINARY, name="x")
    t = sp.addVars(pairs, lb=0.0, name="t")
    v = sp.addVars(pairs, lb=0.0, name="v")
    s = sp.addVars(pairs, lb=0.0, name="s")
    delta = sp.addVars(pairs, lb=0.0, name="delta")
    omax = sp.addVars(C, lb=0.0, name="omax")
    omega = sp.addVars(C, lb=0.0, name="omega")
    tc = sp.addVars(C, lb=0.0, name="tc")
    p = sp.addVars(C, lb=0.0, name="p")
    # POST hücreleri için "servis edildi mi" ikilisi: servis edilmezse p=0 (kontrolsüz),
    # UYGUNSUZLUK yaratmaz (post yayılımı zaten durdurmaz -> ağaç değişmez).
    served = {i: sp.addVar(vtype=GRB.BINARY, name=f"served[{i}]") for i in Q}

    # midpoint (makale (8)-(16)) için ek değişkenler; worstcase'de hiç oluşturulmaz.
    midpoint = (mc.omega_mode == "midpoint")
    if midpoint:
        omin = sp.addVars(C, lb=0.0, name="omin")                       # ω_min_i (54)
        hmax = sp.addVars(pairs, vtype=GRB.BINARY, name="hmax")          # (53) hangi araç max
        hmin = sp.addVars(pairs, vtype=GRB.BINARY, name="hmin")          # (53) hangi araç min

    for i in C:
        is_post = postbar[i] > 0.5
        act = served[i] if is_post else 1.0     # aktif-kontrol göstergesi u_i
        sp.addConstr(p[i] <= inst.pi[i] - inst.beta[i] * (tc[i] - ts[i]), f"c2[{i}]")        # (2)
        sp.addConstr(p[i] <= inst.pi[i] * act, f"c3[{i}]")                                    # (3): p=0 servissiz
        if midpoint:
            # (16): 2ω = ω_max + ω_min + 2Δwat·u ;  (14)-(15): tam bir araç max, bir araç min
            sp.addConstr(2 * omega[i] == omax[i] + omin[i] + 2 * Dwat * act, f"c16[{i}]")     # (16)
            sp.addConstr(gp.quicksum(hmax[i, k] for k in K) == act, f"c14[{i}]")              # (14)
            sp.addConstr(gp.quicksum(hmin[i, k] for k in K) == act, f"c15[{i}]")              # (15)
        else:
            sp.addConstr(omega[i] == omax[i] + Dwat * act, f"c16w[{i}]")                      # (16w) worstcase
        sp.addConstr(gp.quicksum(inst.mu[k] * s[i, k] for k in K) >= omega[i], f"c18[{i}]")   # (18)
        sp.addConstr(tc[i] <= te[i], f"c34[{i}]")                                             # (34)
        if is_post:
            sp.addConstr(tc[i] >= tm[i] - Md * (1 - served[i]), f"c38[{i}]")                  # (38) servisliyken
            sp.addConstr(gp.quicksum(x[i, k] for k in K) >= served[i], f"ctrl_needs_veh[{i}]")
            for k in K:
                sp.addConstr(x[i, k] <= served[i], f"x_le_served[{i},{k}]")                   # servissiz -> araç yok
        else:                                    # PRE: zorunlu, tc ≤ tm
            sp.addConstr(tc[i] >= ts[i], f"c35[{i}]")                                         # (35)
            sp.addConstr(tc[i] <= tm[i], f"c37[{i}]")                                         # (37)
            sp.addConstr(gp.quicksum(x[i, k] for k in K) >= 1, f"ctrl_needs_veh[{i}]")        # zorunlu ≥1 araç
        for k in K:
            sp.addConstr(delta[i, k] >= inst.a[i] * (v[i, k] - ts[i]) - Mi[i] * (1 - x[i, k]), f"c4[{i},{k}]")   # (4)
            sp.addConstr(delta[i, k] <= inst.a[i] * (v[i, k] - ts[i]) + Mi[i] * (1 - x[i, k]), f"c5[{i},{k}]")   # (5)
            sp.addConstr(delta[i, k] <= Mi[i] * x[i, k], f"c6[{i},{k}]")                                          # (6)
            sp.addConstr(omax[i] >= delta[i, k] - Ms(i) * (1 - x[i, k]), f"c7[{i},{k}]")                          # (7)
            if midpoint:
                # (8)-(12): ω_max/ω_min'i atanan araçların gerçek max/min δ'sına kilitle
                sp.addConstr(omax[i] <= delta[i, k] + Ms(i) * (1 - hmax[i, k]), f"c8[{i},{k}]")                   # (8)
                sp.addConstr(hmax[i, k] <= x[i, k], f"c9[{i},{k}]")                                               # (9)
                sp.addConstr(omin[i] <= delta[i, k] + Ms(i) * (1 - x[i, k]), f"c10[{i},{k}]")                     # (10)
                sp.addConstr(omin[i] >= delta[i, k] - Ms(i) * (1 - hmin[i, k]), f"c11[{i},{k}]")                  # (11)
                sp.addConstr(hmin[i, k] <= x[i, k], f"c12[{i},{k}]")                                              # (12)
            sp.addConstr(v[i, k] >= t[i, k] + inst.d[i, k] - Md * (1 - x[i, k]), f"c21[{i},{k}]")                 # (21)
            sp.addConstr(v[i, k] <= t[i, k] + inst.d[i, k] + Md * (1 - x[i, k]), f"c22[{i},{k}]")                 # (22)
            sp.addConstr(t[i, k] <= Md * x[i, k], f"c23t[{i},{k}]")                                               # (23)
            sp.addConstr(s[i, k] <= Md * x[i, k], f"c23s[{i},{k}]")                                               # (23)
            sp.addConstr(v[i, k] <= Md * x[i, k], f"c23v[{i},{k}]")                                               # (23)
            sp.addConstr(v[i, k] + s[i, k] <= tc[i] + Md * (1 - x[i, k]), f"c24[{i},{k}]")                        # (24)
            sp.addConstr(v[i, k] >= ts[i] - Md * (1 - x[i, k]), f"c25[{i},{k}]")                                  # (25)
            sp.addConstr(t[i, k] >= ts_min[i] + Dbuf - Md * (1 - x[i, k]), f"c29[{i},{k}]")                       # (29)

    # (33) araç tekliği — hücreleri global bağlar
    for k in K:
        sp.addConstr(gp.quicksum(x[i, k] for i in C) <= 1, f"c33[{k}]")

    sp.setObjective(gp.quicksum(p[i] for i in C), GRB.MAXIMIZE)
    sp.optimize()

    if sp.Status == GRB.OPTIMAL:
        return LBBDSubResult(status="optimal", obj=sp.ObjVal,
                             p_cell={i: p[i].X for i in C},
                             x_assign={(i, k): x[i, k].X for (i, k) in pairs if x[i, k].X > 0.5})
    if sp.Status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        # zaman-limiti vb: optimal/infeasible KANITLANAMADI -> "unknown" (kesme üretme)
        return LBBDSubResult(status="unknown")
    # INFEASIBLE / INF_OR_UNBD (model sınırlı -> gerçek uygunsuzluk)
    # uygunsuz (araç çekişmesi). İstenirse IIS ile minimal çakışma; aksi halde tam-C.
    conflict = list(C)
    if compute_conflict:
        try:
            sp.computeIIS()
            c = _cells_in_iis(sp, C)
            if c:
                conflict = c
        except Exception:
            pass
    return LBBDSubResult(status="infeasible", conflict=conflict)
