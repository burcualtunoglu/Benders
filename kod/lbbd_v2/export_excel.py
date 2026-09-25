"""export_excel.py — Çözülmüş monolitik modelden (orijinal VEYA düzeltilmiş) örnek
sonucunu, verilen referans Excel biçimiyle uyumlu şekilde .xlsx'e döker; ayrıca iki
modelin karar değişkenlerini yan yana (ORİJİNAL vs DÜZELTİLMİŞ + Δ) karşılaştıran
bir çalışma kitabı üretir.

Değişkenler Gurobi VarName'lerinden okunur (p[i], x[i,k], z[i,j], ...), böylece
orijinal ve düzeltilmiş model için AYNI çıkarıcı çalışır. Orijinal dosyalar değişmez.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from gurobipy import GRB

_STAT = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED",
         9: "TIME_LIMIT", 11: "INTERRUPTED", 13: "SUBOPTIMAL"}

_VNAME = re.compile(r"^([A-Za-z_]+)\[([^\]]+)\]$")

# gap eşiklerinin ilk kez altına inilen çözüm süresini (runtime, s) yakalar
_GAP_THRESHOLDS = (0.10, 0.05, 0.03, 0.01)


def make_gap_time_callback(thresholds=_GAP_THRESHOLDS):
    """Gurobi callback + hit sözlüğü döndürür. hit[t] = gap ilk kez <= t olduğu runtime (s)."""
    hit = {t: None for t in thresholds}

    def cb(model, where):
        if where == GRB.Callback.MIP:
            best = model.cbGet(GRB.Callback.MIP_OBJBST)
            bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            if best < GRB.INFINITY and abs(best) > 1e-9:
                gap = abs(bnd - best) / abs(best)
                rt = model.cbGet(GRB.Callback.RUNTIME)
                for t in thresholds:
                    if hit[t] is None and gap <= t:
                        hit[t] = rt
    return cb, hit


def hit_to_meta(hit):
    """hit sözlüğünü meta satırlarına çevirir: 'Sure_gap<=3%_sec' vb."""
    out = {}
    for t in sorted(hit.keys(), reverse=True):
        out[f"Sure_gap<={int(round(t*100))}%_sec"] = (None if hit[t] is None else round(hit[t], 1))
    return out


def extract_solution(m) -> Dict[str, Dict]:
    """m.getVars() -> {family: {index_tuple: value}}. index_tuple: int veya (int,int)."""
    fam: Dict[str, Dict] = {}
    if m.SolCount == 0:
        return fam
    for var in m.getVars():
        mt = _VNAME.match(var.VarName)
        if not mt:
            continue
        name, idx = mt.group(1), mt.group(2)
        parts = tuple(int(float(p)) for p in idx.split(","))
        key = parts[0] if len(parts) == 1 else parts
        fam.setdefault(name, {})[key] = var.X
    return fam


def _meta_rows(m, input_path: str, extra=None):
    obj = m.ObjVal if m.SolCount > 0 else None
    gap = m.MIPGap if m.SolCount > 0 else None
    df = pd.DataFrame(
        {"metric": ["InputFile", "Status", "Status_text", "Objective",
                    "Runtime_sec", "MIPGap", "MIPGap_%", "SolCount", "BestBound"],
         "value": [input_path, m.Status, _STAT.get(m.Status, str(m.Status)),
                   obj, round(m.Runtime, 2), gap,
                   (gap * 100 if gap is not None else None), m.SolCount,
                   (m.ObjBound if m.Status in (2, 9, 13) else None)]})
    if extra:
        df = pd.concat([df, pd.DataFrame({"metric": list(extra.keys()),
                                          "value": list(extra.values())})],
                       ignore_index=True)
    return df


def _vehicle_sheets(inst, fam):
    vinfo = inst.vehicle_info
    vml = pd.DataFrame([{"Vehicle_ID": v["id"], "Base": v["base_id"], "Type": v["type"],
                         "Speed": v["speed"], "Capacity": v["capacity"]} for v in vinfo])
    x = fam.get("x", {})
    rows = []
    for v in vinfo:
        k = v["id"]
        target = next((i for (i, kk), val in x.items() if kk == k and val > 0.5), None)
        tt = inst.d.get((target, k)) if target is not None else None
        rows.append({"Vehicle_ID": k, "Type": v["type"], "Base": v["base_id"],
                     "Speed_km_h": v["speed"], "Capacity_L": v["capacity"],
                     "Target_Node": target, "Travel_Time_min": tt})
    return vml, pd.DataFrame(rows)


# (family, sheet_name, value_col)  — hücre-bazlı
_SCALAR = [("p", "p", "p_i"), ("y", "y", "y_i"), ("ts", "ts", "t_i_s"),
           ("tm", "tm", "t_i_m"), ("te", "te", "t_i_e"), ("tc", "tc", "t_i_c"),
           ("u_pre", "u_pre", "u_i_pre"), ("u_post", "u_post", "u_i_post"),
           ("omega", "omega", "omega_i"), ("omega_max", "omega_max", "omega_i_max"),
           ("omega_min", "omega_min", "omega_i_min"), ("ts_min", "ts_min", "t_i_s_min")]
# atama-bazlı (i,k)
_IK = [("x", "x", "x_ik"), ("t", "t", "t_ik"), ("s", "s", "s_ik"),
       ("v", "v", "v_ik"), ("delta", "delta", "delta_ik")]
# yay-bazlı (i,j)
_IJ = [("z", "z", "z_ij"), ("q", "q", "q_ij"), ("bmin", "bmin", "b_ij_min")]


def _is_bin(name):
    return name in ("y", "u_pre", "u_post", "x", "z", "q", "bmin")


def write_solution_xlsx(m, inst, input_path: str, out_path: str, extra_meta=None):
    fam = extract_solution(m)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        _meta_rows(m, input_path, extra_meta).to_excel(w, sheet_name="meta", index=False)
        if fam:
            vml, dep = _vehicle_sheets(inst, fam)
            vml.to_excel(w, sheet_name="Vehicle_Master_List", index=False)
            dep.to_excel(w, sheet_name="Deployment_Details", index=False)
        for name, sheet, col in _SCALAR:
            d = fam.get(name)
            if d is None:
                continue
            df = pd.DataFrame({"i": list(d.keys()), col: list(d.values())}).sort_values("i")
            if _is_bin(name):
                df[col] = df[col].round().astype(int)
            df.to_excel(w, sheet_name=sheet, index=False)
        for name, sheet, col in _IK:
            d = fam.get(name)
            if d is None:
                continue
            if name == "x":
                items = [(i, k, val) for (i, k), val in d.items() if val > 0.5]
            elif name == "delta":
                items = [(i, k, val) for (i, k), val in d.items() if abs(val) > 1e-9]
            else:  # t,s,v: yalnız atanmış (x=1) çiftler
                xd = fam.get("x", {})
                items = [(i, k, val) for (i, k), val in d.items() if xd.get((i, k), 0) > 0.5]
            df = pd.DataFrame(items, columns=["i", "k", col]).sort_values(["i", "k"])
            if _is_bin(name):
                df[col] = df[col].round().astype(int)
            df.to_excel(w, sheet_name=sheet, index=False)
        for name, sheet, col in _IJ:
            d = fam.get(name)
            if d is None:
                continue
            items = [(i, j, val) for (i, j), val in d.items() if val > 0.5]
            df = pd.DataFrame(items, columns=["i", "j", col]).sort_values(["i", "j"])
            df[col] = df[col].round().astype(int)
            df.to_excel(w, sheet_name=sheet, index=False)
    return out_path


def _cmp_scalar(inst, f0, f1, name, col):
    d0, d1 = f0.get(name, {}), f1.get(name, {})
    keys = sorted(set(d0) | set(d1))
    rows = []
    for i in keys:
        a = d0.get(i); b = d1.get(i)
        dd = (b - a) if (a is not None and b is not None) else None
        rows.append({"i": i, f"{col}_ORIJINAL": a, f"{col}_DUZELTILMIS": b,
                     "delta": dd, "FARK": (dd is not None and abs(dd) > 1e-6)})
    return pd.DataFrame(rows)


def _cmp_ik(inst, f0, f1, name, col):
    d0, d1 = f0.get(name, {}), f1.get(name, {})
    keys = sorted(set(d0) | set(d1))
    rows = []
    for (i, k) in keys:
        a = d0.get((i, k)); b = d1.get((i, k))
        dd = (b - a) if (a is not None and b is not None) else None
        if (a in (None, 0) or abs(a or 0) < 1e-9) and (b in (None, 0) or abs(b or 0) < 1e-9):
            continue  # ikisinde de sıfır/atanmamış -> atla
        rows.append({"i": i, "k": k, f"{col}_ORIJINAL": a, f"{col}_DUZELTILMIS": b,
                     "delta": dd, "FARK": (dd is not None and abs(dd) > 1e-6) or (a is None) != (b is None)})
    return pd.DataFrame(rows)


def write_comparison_xlsx(inst, m0, m1, input_path, out_path,
                          extra0=None, extra1=None):
    f0, f1 = extract_solution(m0), extract_solution(m1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        # özet
        def g(m):
            return {"status": _STAT.get(m.Status, str(m.Status)),
                    "obj": (m.ObjVal if m.SolCount > 0 else None),
                    "bound": (m.ObjBound if m.Status in (2, 9, 13) else None),
                    "gap_%": (m.MIPGap * 100 if m.SolCount > 0 else None),
                    "runtime_s": round(m.Runtime, 2), "solcount": m.SolCount}
        s0, s1 = g(m0), g(m1)
        do = (s1["obj"] - s0["obj"]) if (s0["obj"] is not None and s1["obj"] is not None) else None
        metrics = ["InputFile", "status", "objective", "bound", "gap_%", "runtime_s", "solcount"]
        orij = [input_path, s0["status"], s0["obj"], s0["bound"], s0["gap_%"],
                s0["runtime_s"], s0["solcount"]]
        duz = [input_path, s1["status"], s1["obj"], s1["bound"], s1["gap_%"],
               s1["runtime_s"], s1["solcount"]]
        # gap-eşiği süreleri (ör. Sure_gap<=3%_sec)
        for key in (list((extra0 or {}).keys()) or list((extra1 or {}).keys())):
            metrics.append(key)
            orij.append((extra0 or {}).get(key))
            duz.append((extra1 or {}).get(key))
        summ = pd.DataFrame({"metric": metrics, "ORIJINAL": orij, "DUZELTILMIS": duz})
        summ.to_excel(w, sheet_name="OZET", index=False)
        pd.DataFrame({"metric": ["obj_farki(DUZ-ORIJ)"], "value": [do]}).to_excel(
            w, sheet_name="OZET", index=False, startrow=len(summ) + 2)
        for name, _s, col in _SCALAR:
            df = _cmp_scalar(inst, f0, f1, name, col)
            if len(df):
                df.to_excel(w, sheet_name=f"cmp_{name}", index=False)
        for name, _s, col in _IK:
            df = _cmp_ik(inst, f0, f1, name, col)
            if len(df):
                df.to_excel(w, sheet_name=f"cmp_{name}", index=False)
    return out_path
