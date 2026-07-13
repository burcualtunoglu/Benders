"""
lbbd_subproblem_typeagg.py — Theme 3 HİBRİT: LBBD alt problemini TİP-AGREGASYONU ile
küçült (lit Theme 3 + P6).

LBBD'nin darboğazı: atama+çizelgeleme alt problemi TEK büyük MILP'tir (araç tekliği
(33) tüm hücreleri global bağlar); 75/150 araçta çözülemez. Çözüm: bireysel araç
`x_ik` yerine araç TİPİ sayacı `n_it` (aynı üs+tip → aynı d, μ). Alt problem
|C|×|K| ikiliden |C|×|tip| tam-sayıya iner → FİLO-BAĞIMSIZ (~15 tip).

Bu, `pricing_typeagg.py`'nin (ana proje, B&P) SABİT-ZAMANLI, hücre-başına biçimidir —
LBBD alt probleminde zamanlar (ts,tm,te,ts_min) master'dan sabit olduğundan tip t'nin
i'ye varışı v_it = ts_min_i+Δbuf+d_t[i] bir SABİTtir ve gecikme a_i(v_it−ts_i) sabittir
→ ω_max makinesi çok basitleşir. Su bilineeri μ_t·n_it·(tc_i−v_it), n'in ikili açılımı
× McCormick ile TAM lineer (gevşetme değil).

YALNIZ worstcase ω. Aynı Φ'yi (per-araç alt problemle birebir) verir — doğrulanır.
Arayüz `benders_lbbd_subproblem.LBBDSubResult` ile uyumludur.
"""

from __future__ import annotations

import math
from typing import Dict, List

import gurobipy as gp
from gurobipy import GRB

from data_loader import Instance
from config import Config
from benders_forward import ForwardResult
from benders_lbbd_subproblem import LBBDSubResult


def vehicle_types(inst: Instance):
    """(base_id,type) grupları → [(N_t, mu_t, {i:d_t[i]})]; grup içi d,μ özdeş."""
    groups: Dict[tuple, List[int]] = {}
    for v in inst.vehicle_info:
        groups.setdefault((v["base_id"], v["type"]), []).append(v["id"])
    types = []
    for key, ids in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        rep = ids[0]
        types.append((len(ids), inst.mu[rep], {i: inst.d[i, rep] for i in inst.Nf}))
    return types


def solve_assignment_subproblem_typeagg(inst: Instance, cfg: Config, controlled: List[int],
                                        fp: ForwardResult, prebar: Dict[int, float],
                                        postbar: Dict[int, float],
                                        compute_conflict: bool = False,
                                        tl: float = None) -> LBBDSubResult:
    """Tip-agregasyonlu atama+çizelgeleme alt problemi (worstcase, sabit zamanlar)."""
    if cfg.model.omega_mode != "worstcase":
        raise ValueError("tip-agregasyonlu LBBD alt problemi yalnız worstcase içindir")
    if not controlled:
        return LBBDSubResult(status="optimal", obj=0.0)

    mc = cfg.model
    Dwat, Dbuf = mc.delta_wat, mc.delta_buf
    ts, tm, te, ts_min = fp.ts, fp.tm, fp.te, fp.ts_min
    C = controlled

    def Ms(i):
        return inst.Ms_cell[i] if mc.ms_mode == "percell" else inst.Ms

    types = vehicle_types(inst)
    T = list(range(len(types)))
    Ncap = {t: types[t][0] for t in T}
    mu_t = {t: types[t][1] for t in T}
    d_t = {t: types[t][2] for t in T}
    bits = {t: max(1, int(math.ceil(math.log2(Ncap[t] + 1)))) for t in T}

    sp = gp.Model("lbbd_sub_typeagg")
    sp.setParam("OutputFlag", 0)
    sp.setParam("DualReductions", 0)
    if tl and tl > 0:
        sp.setParam("TimeLimit", max(1.0, tl))

    # deadline_i (rejime göre) ve tip t'nin i'ye SABİT varışı v_it, gecikme delay_it.
    # (i,t) yalnız pencere pozitifse yaratılır (win_ub = deadline_i − v_it > 0).
    deadline = {}
    for i in C:
        deadline[i] = tm[i] if prebar[i] > 0.5 else te[i]
    v_it, delay_it, win_ub = {}, {}, {}
    it_pairs = []
    for i in C:
        for t in T:
            # SABİT en erken varış: (21)+(29) v ≥ ts_min+Δbuf+d  VE  (25) v ≥ ts_i.
            v = max(ts_min[i] + Dbuf + d_t[t][i], ts[i])
            w = deadline[i] - v                            # pencere üst sınırı (tc ≤ deadline)
            if w > 1e-9:
                v_it[i, t] = v
                delay_it[i, t] = inst.a[i] * (v - ts[i])   # SABİT gecikme a_i(v−ts) ≥ 0
                win_ub[i, t] = w
                it_pairs.append((i, t))
    Ti = {i: [t for t in T if (i, t) in v_it] for i in C}
    It = {t: [i for i in C if (i, t) in v_it] for t in T}

    # değişkenler
    tc = sp.addVars(C, lb=0.0, name="tc")
    p = sp.addVars(C, lb=0.0, name="p")
    omax = sp.addVars(C, lb=0.0, name="omax")
    omega = sp.addVars(C, lb=0.0, name="omega")
    n = sp.addVars(it_pairs, vtype=GRB.INTEGER, lb=0.0, name="n")
    used = sp.addVars(it_pairs, vtype=GRB.BINARY, name="used")
    win = sp.addVars(it_pairs, lb=0.0, name="win")
    WS = sp.addVars(it_pairs, lb=0.0, name="WS")
    bit = {(i, t, b): sp.addVar(vtype=GRB.BINARY, name=f"bit[{i},{t},{b}]")
           for (i, t) in it_pairs for b in range(bits[t])}
    g = {(i, t, b): sp.addVar(lb=0.0, name=f"g[{i},{t},{b}]")
         for (i, t) in it_pairs for b in range(bits[t])}
    for (i, t) in it_pairs:
        win[i, t].UB = win_ub[i, t]

    Q = [i for i in C if postbar[i] > 0.5]
    served = {i: sp.addVar(vtype=GRB.BINARY, name=f"served[{i}]") for i in Q}

    for i in C:
        is_post = postbar[i] > 0.5
        act = served[i] if is_post else 1.0
        # ödül (2),(3)
        sp.addConstr(p[i] <= inst.pi[i] - inst.beta[i] * (tc[i] - ts[i]), f"c2[{i}]")
        sp.addConstr(p[i] <= inst.pi[i] * act, f"c3[{i}]")
        # ω_max (7 tipe): delay_it SABİT → ω_max ≥ delay_it·used_it ;  (16w) ω = ω_max + Δwat·act
        for t in Ti[i]:
            sp.addConstr(omax[i] >= delay_it[i, t] * used[i, t], f"c7[{i},{t}]")
        sp.addConstr(omega[i] == omax[i] + Dwat * act, f"c16w[{i}]")
        sp.addConstr(omega[i] <= Ms(i) * act, f"c17[{i}]")
        # su (18): Σ_t μ_t·WS_it ≥ ω_i
        sp.addConstr(gp.quicksum(mu_t[t] * WS[i, t] for t in Ti[i]) >= omega[i], f"c18[{i}]")
        # rejim: pre → ts ≤ tc ≤ tm ; post → tm ≤ tc ≤ te (servisliyken)
        sp.addConstr(tc[i] <= te[i], f"c34[{i}]")
        if is_post:
            sp.addConstr(tc[i] >= tm[i] - inst.Md * (1 - served[i]), f"c38[{i}]")
        else:
            sp.addConstr(tc[i] >= ts[i], f"c35[{i}]")
            sp.addConstr(tc[i] <= tm[i], f"c37[{i}]")
        # su-zamanı WS = n·win ; win ≤ tc − v ; ikili açılım × McCormick (TAM)
        for t in Ti[i]:
            WUB = win_ub[i, t]
            sp.addConstr(n[i, t] == gp.quicksum((2 ** b) * bit[i, t, b] for b in range(bits[t])),
                         f"nbits[{i},{t}]")
            sp.addConstr(gp.quicksum((2 ** b) * bit[i, t, b] for b in range(bits[t])) <= Ncap[t],
                         f"ncap[{i},{t}]")
            for b in range(bits[t]):
                gb = g[i, t, b]
                sp.addConstr(gb <= win[i, t], f"gle[{i},{t},{b}]")
                sp.addConstr(gb <= WUB * bit[i, t, b], f"gleM[{i},{t},{b}]")
                sp.addConstr(gb >= win[i, t] - WUB * (1 - bit[i, t, b]), f"gge[{i},{t},{b}]")
            sp.addConstr(WS[i, t] == gp.quicksum((2 ** b) * g[i, t, b] for b in range(bits[t])),
                         f"WSdef[{i},{t}]")
            sp.addConstr(win[i, t] <= tc[i] - v_it[i, t], f"winle[{i},{t}]")
            # used = (n ≥ 1)
            for b in range(bits[t]):
                sp.addConstr(used[i, t] >= bit[i, t, b], f"usedge[{i},{t},{b}]")
            sp.addConstr(used[i, t] <= gp.quicksum(bit[i, t, b] for b in range(bits[t])), f"usedle[{i},{t}]")
            sp.addConstr(used[i, t] <= act, f"usedact[{i},{t}]")          # servissiz → araç yok
        # kontrol için ≥1 araç: pre zorunlu, post servisliyken
        sp.addConstr(gp.quicksum(n[i, t] for t in Ti[i]) >= act, f"needveh[{i}]")

    # (33) araç tekliği → tip kapasitesi: Σ_i n_it ≤ N_t
    for t in T:
        if It[t]:
            sp.addConstr(gp.quicksum(n[i, t] for i in It[t]) <= Ncap[t], f"c33[{t}]")

    sp.setObjective(gp.quicksum(p[i] for i in C), GRB.MAXIMIZE)
    sp.optimize()

    if sp.Status == GRB.OPTIMAL:
        # x_assign yerine tip sayaçları döndür (kesme mantığı yalnız p_cell/obj kullanır)
        x_assign = {(i, t): round(n[i, t].X) for (i, t) in it_pairs if n[i, t].X > 0.5}
        return LBBDSubResult(status="optimal", obj=sp.ObjVal,
                             p_cell={i: p[i].X for i in C}, x_assign=x_assign)
    if sp.Status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return LBBDSubResult(status="unknown")
    conflict = list(C)
    if compute_conflict:
        try:
            sp.computeIIS()
            import re
            Cset = set(C); cf = set()
            for c in sp.getConstrs():
                if c.IISConstr:
                    for tok in re.findall(r"-?\d+", c.ConstrName):
                        if int(tok) in Cset:
                            cf.add(int(tok))
            if cf:
                conflict = sorted(cf)
        except Exception:
            pass
    return LBBDSubResult(status="infeasible", conflict=conflict)
