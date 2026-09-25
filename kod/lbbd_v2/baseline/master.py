"""master.py — DÜZELTİLMİŞ LBBD master (geç-ateşleme `mccap` tavanı KALDIRILMIŞ).

lbbd_v2.lbbd_master ile aynı geçerli-gevşetme yapısıdır; TEK fark: master zaman-tutarlılığı
katmanındaki (F-MTC) `mccap` kısıtı — `t^s_i ≤ M_i/a_i + M_d(1−y_i)` — BURADA YOKTUR. O tavan,
düzeltilmemiş modelin geç-ateşleme infizibilitesini kodlar; düzeltilmiş modelde (deaktivasyon
a_i·M_d) geç ateşleme fizibildir, dolayısıyla `mccap` GEÇERSİZ ve optimum içerebilen yapıyı keser
(docs/asama0_rapor.md §4-C1). F-MTC katmanı yalnız (46) z-yay tutarlılığını dayatır (geçerli,
tembel yayılım kesmesinin master karşılığı).

Bayrakların tümü BaselineConfig'ten okunur (aynı öznitelik adları); minimal baseline'da hepsi kapalı.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from lbbd_v2.config import Config
from lbbd_v2.data_loader import Instance
from lbbd_v2.preprocessing import Preprocessed
from lbbd_v2.lbbd_master import MasterSolution   # aynı sonuç yapısı; yeniden kullan


class BaselineMaster:
    def __init__(self, inst: Instance, pre: Preprocessed, cfg: Config):
        self.inst, self.pre, self.cfg = inst, pre, cfg
        self.m = gp.Model("lbbd_master_baseline")
        self.m.setParam("OutputFlag", 1 if cfg.gurobi_output else 0)
        self.m.setParam("Seed", cfg.gurobi_seed)
        self.m.setParam("Threads", cfg.gurobi_threads)
        self.cut_records: List[dict] = []
        self._build()

    def _build(self) -> None:
        inst, pre, cfg = self.inst, self.pre, self.cfg
        m = self.m
        Nf, Na, Nplus = inst.Nf, inst.Na, inst.Nplus
        nb, arcs = inst.neighbors, inst.arcs
        pi, beta, e = inst.pi, inst.beta, inst.e

        y = m.addVars(Nf, vtype=GRB.BINARY, name="y")
        z = m.addVars(arcs, vtype=GRB.BINARY, name="z")
        q = m.addVars(arcs, vtype=GRB.BINARY, name="q")
        u_pre = m.addVars(Nf, vtype=GRB.BINARY, name="u_pre")
        u_post = m.addVars(Nf, vtype=GRB.BINARY, name="u_post")
        rho = m.addVars(Nf, lb=0.0, name="rho")

        def u(i):
            return u_pre[i] + u_post[i]

        # amaç: max Σ π_i(1−y_i) + Σ ρ_i  (geçerli gevşetme => geçerli UB)
        m.setObjective(gp.quicksum(pi[i] * (1 - y[i]) for i in Nf)
                       + gp.quicksum(rho[i] for i in Nf), GRB.MAXIMIZE)

        # --- ÇEKİRDEK (her zaman) ---
        m.addConstr(gp.quicksum(y[i] for i in Na) == len(Na), "c47")        # kökler yanar
        m.addConstrs((u(i) <= y[i] for i in Nf), "u_le_y")
        m.addConstrs((u(i) <= 1 for i in Nf), "c39")
        m.addConstrs((rho[i] <= pi[i] * u(i) for i in Nf), "rho_proxy")     # ρ_i ≤ π_i u_i
        m.addConstrs((gp.quicksum(z[i, j] for j in nb[i]) == len(nb[i]) * (y[i] - u_pre[i])
                      for i in Nf), "c40")
        m.addConstrs((len(nb[j]) * y[j] >= gp.quicksum(z[i, j] for i in nb[j]) for j in Nf), "c41")
        m.addConstrs((q[i, j] <= z[i, j] for (i, j) in arcs), "c42")
        m.addConstrs((gp.quicksum(q[i, j] for i in nb[j]) == y[j] - e[j] for j in Nf), "c43")

        # --- PERFORMANS-ONLY sıkılaştırmalar (bayrakla) ---
        if cfg.enable_C0:
            for i in Nf:
                if not pre.controllable[i]:
                    m.addConstr(u(i) == 0, f"C0[{i}]")
        if cfg.enable_C0prime:
            for i in Nf:
                if not pre.pre_possible[i]:
                    m.addConstr(u_pre[i] == 0, f"C0p[{i}]")
        if cfg.enable_C3:
            for i in Nf:
                m.addConstr(rho[i] <= (pi[i] - beta[i] * pre.delta_min[i]) * u(i), f"C3[{i}]")
        if cfg.enable_C2:
            m.addConstr(gp.quicksum(u(i) for i in Nf) <= len(inst.K), "C2")

        f = None
        if cfg.enable_flow_connectivity:
            f = m.addVars(arcs, lb=0.0, name="flow")
            cap = float(len(Nf))
            m.addConstrs((f[i, j] <= cap * q[i, j] for (i, j) in arcs), "flow_cap")
            for i in Nf:
                if i in Na:
                    continue
                inflow = gp.quicksum(f[j, i] for j in nb[i])
                outflow = gp.quicksum(f[i, j] for j in nb[i])
                m.addConstr(inflow - outflow == y[i], f"flow_cons[{i}]")

        ts = None
        if getattr(cfg, "enable_master_time_consistency", False):
            Md = pre.M_d
            alpha, lam = inst.alpha, inst.lam
            ts = m.addVars(Nf, lb=0.0, name="ts")
            m.addConstr(gp.quicksum(ts[r] for r in Na) == 0, "mc48")
            m.addConstrs((ts[i] <= Md * y[i] for i in Nf), "mc23ts")
            for (i, j) in arcs:
                tmi = ts[i] + alpha / lam[i]
                m.addConstr(ts[j] >= tmi - Md * (1 - q[i, j]), f"mc44[{i},{j}]")
                m.addConstr(ts[j] <= tmi + Md * (1 - q[i, j]), f"mc45[{i},{j}]")
                m.addConstr(ts[j] <= tmi + Md * (1 - z[i, j]), f"mc46[{i},{j}]")
            # NOT: `mccap` (t^s ≤ M_i/a_i tavanı) BİLEREK EKLENMEDİ — düzeltilmiş modelde geçersiz.

        self.vars = dict(y=y, z=z, q=q, u_pre=u_pre, u_post=u_post, rho=rho, flow=f, ts=ts)

    def add_cut(self, lhs: gp.LinExpr, sense: str, rhs: float,
                name: str = "cut", record: Optional[dict] = None) -> None:
        if sense == "<=":
            self.m.addConstr(lhs <= rhs, name=name)
        elif sense == ">=":
            self.m.addConstr(lhs >= rhs, name=name)
        elif sense == "==":
            self.m.addConstr(lhs == rhs, name=name)
        else:
            raise ValueError(f"bad sense {sense!r}")
        if record is not None:
            self.cut_records.append(record)

    def ensure_binary(self, vn: str, key, name: str):
        """Kesme yardımcı ikilisi (2026-09-24 SDC-t: komşu 'erken yandı' göstergesi ε). Aynı anahtar için bir kez
        oluşturulur; self.vars[vn] sözlüğüne kaydedilir (Cut.to_linexpr erişir). Döndürür: (var, yeni_mi)."""
        d = self.vars.setdefault(vn, {})
        if key in d:
            return d[key], False
        v = self.m.addVar(vtype=GRB.BINARY, name=name)
        d[key] = v
        return v, True

    def fix_zero(self, vn: str, key, name: str, record: Optional[dict] = None) -> None:
        """Tek bir ikili değişkeni 0'a sabitle (APRI: a-priori kanıtlanmış u^r_i = 0). Bir 'kesme'
        gibi kaydedilir (cut_records) ki geçerlilik izi JSON'da kalsın."""
        v = self.vars[vn][key]
        self.m.addConstr(v <= 0, name=name)
        if record is not None:
            self.cut_records.append(record)

    def cb_solution(self, model) -> MasterSolution:
        """MIPSOL callback içinde: o anki tamsayı inkümbentini MasterSolution olarak döndür (ikililer
        0.5 eşiğiyle yuvarlanır; ileri geçiş/kesme üreticileri normal çözümle AYNI nesneyi görür)."""
        out = MasterSolution(status="CALLBACK", obj_val=None, obj_bound=None, runtime=0.0,
                             solution_count=1)
        try:
            out.obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        except Exception:
            out.obj_val = None
        for name in ("y", "z", "q", "u_pre", "u_post", "rho"):
            tv = self.vars[name]
            keys = list(tv.keys())
            vals = model.cbGetSolution([tv[k] for k in keys])
            if name == "rho":
                setattr(out, name, {k: float(v) for k, v in zip(keys, vals)})
            else:
                setattr(out, name, {k: (1.0 if v > 0.5 else 0.0) for k, v in zip(keys, vals)})
        return out

    def discard_solution(self) -> None:
        """F-BCH güvenli durdurma sonrası: çözüm bilgisini (iç inkümbent / MIP start) AT. Model ve kısıtlar
        korunur; bir sonraki optimize() doğrulanmamış iç inkümbenti başlangıç çözümü olarak TAŞIMAZ."""
        self.m.reset(0)

    @staticmethod
    def cb_bound(model) -> Optional[float]:
        """MIPSOL callback anındaki KÜRESEL en iyi sınır (MIPSOL_OBJBND): o anki dal-sınır ağacının tüm açık
        düğümleri ve inkümbenti üzerinden geçerli dual sınır; +inf ise None."""
        try:
            b = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            return float(b) if b < GRB.INFINITY else None
        except Exception:
            return None

    def solve(self, budget: Optional[float] = None, mip_gap: float = 0.0,
              callback=None) -> MasterSolution:
        """callback verilirse (F-BCH) LazyConstraints=1 ile `optimize(callback)`; callback her
        MIPSOL'da cbLazy ile GEÇERLİ kesme ekleyebilir. Model/kısıt kümesi aksi halde aynıdır."""
        m = self.m
        m.setParam("MIPGap", mip_gap)
        if budget is not None:
            m.setParam("TimeLimit", budget)
        t0 = time.time()
        if callback is not None:
            m.setParam("LazyConstraints", 1)
            m.optimize(callback)
        else:
            m.optimize()
        runtime = time.time() - t0
        st = m.Status
        status = {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
                  GRB.INF_OR_UNBD: "INFEASIBLE", GRB.TIME_LIMIT: "TIME_LIMIT",
                  GRB.UNBOUNDED: "UNBOUNDED",
                  GRB.INTERRUPTED: "INTERRUPTED"}.get(st, f"STATUS_{st}")   # F-BCH: callback terminate()
        sol = MasterSolution(status=status, obj_val=None, obj_bound=None, runtime=runtime,
                             solution_count=int(m.SolCount))
        if m.SolCount > 0:
            vv = self.vars
            sol.obj_val = m.ObjVal
            sol.y = {i: vv["y"][i].X for i in vv["y"].keys()}
            sol.z = {a: vv["z"][a].X for a in vv["z"].keys()}
            sol.q = {a: vv["q"][a].X for a in vv["q"].keys()}
            sol.u_pre = {i: vv["u_pre"][i].X for i in vv["u_pre"].keys()}
            sol.u_post = {i: vv["u_post"][i].X for i in vv["u_post"].keys()}
            sol.rho = {i: vv["rho"][i].X for i in vv["rho"].keys()}
        if st in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED):
            try:
                if m.ObjBound < GRB.INFINITY:
                    sol.obj_bound = m.ObjBound          # kesintide de geçerli dual sınır
            except Exception:
                sol.obj_bound = None
        return sol

    def lp_relaxation_bound(self, budget: Optional[float] = None) -> dict:
        """Ana problemin KÖK LP-gevşetme optimumu (reviewer §4). Tamsayısızlaştırılmış master'ı bir kez
        çözer; sonuç, master MIP amacının — dolayısıyla OPT'un — GEÇERLİ bir ÜST sınırıdır. Bu ölçüt,
        MIP çözücüsünün kök-kesme/dallanma SONRASI dual sınırından AYRIDIR (burada saf LP; hiçbir kesme
        veya dallanma yok). LBBD kesmeleri master'a EKLENMEDEN (ana döngüden ÖNCE) çağrılmalıdır; aksi
        halde 'kök' LP-gevşetmesi olmaz. Ayrı bir teşhis çözümüdür (LBBD zaman bütçesine dahil değil;
        süresi `runtime` ile raporlanır)."""
        self.m.update()                            # relax() bekleyen değişiklikleri görmeli (yoksa boş/bayat model)
        r = self.m.relax()
        r.setParam("OutputFlag", 0)
        r.setParam("Threads", self.cfg.gurobi_threads)
        r.setParam("Seed", self.cfg.gurobi_seed)
        if budget is not None:
            r.setParam("TimeLimit", max(1.0, budget))
        t0 = time.time()
        r.optimize()
        runtime = time.time() - t0
        st = r.Status
        optimal = (st == GRB.OPTIMAL)
        # initial_lp_ub YALNIZ LP optimalliği KANITLANINCA yazılır (reviewer §4/2). Zaman aşımında
        # fizibil LP amaç değeri (r.ObjVal) bir ALT değerdir (maksimizasyonda ≤ LP-opt), ÜST SINIR
        # DEĞİLDİR -> initial_lp_ub'e KOYMA. Geçerli üst sınır varsa dual sınırdır (r.ObjBound ≥ LP-opt
        # ≥ OPT); ayrı alanda + durumla raporlanır.
        lp_ub = r.ObjVal if optimal else None
        dual_bound = None
        if not optimal and st in (GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED):
            try:
                if r.ObjBound < GRB.INFINITY:
                    dual_bound = r.ObjBound        # geçerli dual UB (LP kanıtlanmadı)
            except Exception:
                dual_bound = None
        return {"lp_ub": lp_ub, "dual_bound": dual_bound, "status": int(st),
                "optimal": optimal, "runtime": runtime}


def build_baseline_master(inst: Instance, pre: Preprocessed, cfg: Config) -> BaselineMaster:
    return BaselineMaster(inst, pre, cfg)
