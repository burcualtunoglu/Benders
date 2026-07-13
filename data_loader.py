"""
data_loader.py — Excel instance dosyalarını okuyup model-hazır bir `Instance`
nesnesine dönüştürür.

Veri formatı (bkz. inputs/*.xlsx, 3 sayfa):

  * inputs_df   : hücre başına bir satır
        node_id, x_coordinate, y_coordinate   -> merkez koordinat (METRE)
        value_at_start        -> pi   (tam ödül)
        fire_degradation_rate -> lambda_i (yayılım hızı, (49))
        fire_amelioration_rate-> sigma_i  (sönme hızı,  (50))
        beta                  -> beta_i (değer azalması, (2))
        su                    -> a_i    (gecikme başına ek su, (4)-(5))
        state                 -> 0 yanıcı / 1 aktif yangın / 2 su / 3 yol
        neighborhood_list     -> N(i)   (road_pairs + su/yol kenarları ZATEN
                                         çıkarılmış; doğrulandı)
        time, Mi              -> türetilmiş referans alanları (modelde kullanılmaz;
                                 big-M'ler preprocessing.py'de yeniden hesaplanır)

  * bases       : üs başına bir satır
        Base, x_coordinate, y_coordinate      -> üs (HÜCRE-KÖŞE indeksi 0..N)
        Helicopter, Fire Engine, FRV          -> o üsteki araç adetleri

  * parameters  : parametre/değer/not
        cell_side_length, water_required_per_cell (Ms), Mtime (Md ref.),
        road_pairs (bilgi), *_speed, *_capacity (mu_k), ...

Mesafe metriği: Helicopter -> Öklid, Fire Engine/FRV -> Manhattan.
d_ik = mesafe / speed_k.

Bu modül `load_inputs_p4.py` (wildfire_benders projesi) ile birebir aynı okuma
mantığını izler; kendi kendine yeterli olması için buraya taşınmıştır.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


# --------------------------------------------------------------------------- #
#  Instance veri yapısı — modelin gördüğü her şey burada
# --------------------------------------------------------------------------- #
@dataclass
class Instance:
    name: str

    # kümeler
    N: List[int]                         # tüm düğümler
    Nf: List[int]                        # yanıcı hücreler  (state in {0,1})
    Na: List[int]                        # aktif yangın     (state == 1, e_i=1)
    Nw: List[int]                        # su kaynağı       (state == 2)
    Np: List[int]                        # yol              (state == 3)
    K: List[int]                         # araçlar

    # düğüm parametreleri (yalnız Nf anlamlı; diğerleri 0)
    pi: Dict[int, float]                 # pi_i   (value_at_start)
    lam: Dict[int, float]                # lambda_i (fire_degradation_rate)
    sig: Dict[int, float]                # sigma_i  (fire_amelioration_rate)
    beta: Dict[int, float]               # beta_i
    a: Dict[int, float]                  # a_i    (su)
    alpha: Dict[int, float]              # alpha_i = cell_side_length/2 (yarı-kenar)
    state: Dict[int, int]
    e: Dict[int, int]                    # e_i = 1 <=> i in Na
    neighbors: Dict[int, Tuple[int, ...]]  # N(i)
    coords: Dict[int, Tuple[float, float]]

    # araç parametreleri
    mu: Dict[int, float]                 # mu_k (capacity = su akış hızı)
    d: Dict[Tuple[int, int], float]      # d_ik (seyahat süresi)
    vehicle_info: List[dict]

    # ham parametre sözlüğü ve seçili sabitler
    params: Dict[str, Any]
    cell_side_length: float
    water_required_per_cell: float       # Ms (constant) kaynağı
    mtime: float                         # Md referansı (parameters!Mtime)

    # --- preprocessing.py tarafından doldurulur --------------------------- #
    ts_ub: Dict[int, float] = field(default_factory=dict)   # ts üst sınırı
    tm_ub: Dict[int, float] = field(default_factory=dict)   # tm üst sınırı (pre son tarihi)
    te_ub: Dict[int, float] = field(default_factory=dict)   # te üst sınırı
    Md: float = 0.0                                          # zaman big-M
    Mi: Dict[int, float] = field(default_factory=dict)      # delta big-M (hücre)
    Ms: float = 0.0                                          # su big-M (skaler)
    Ms_cell: Dict[int, float] = field(default_factory=dict) # su big-M (percell)

    # --- LBBD çekişme gevşetmesi için ulaşılabilirlik (compute_bigM doldurur) --- #
    # A/kappa PRE (tm son tarihi) tabanlıdır çünkü eşleme VI pre-hücrelere uygulanır.
    A: Dict[int, list] = field(default_factory=dict)         # A_i: i'yi tm_UB'den önce basabilen araçlar
    kappa: Dict[int, int] = field(default_factory=dict)      # kappa_i: Δwat için min araç (pre penceresi)
    controllable: Dict[int, bool] = field(default_factory=dict)  # i hiç kontrol edilebilir mi (te tabanlı)

    # kolaylık
    def Nplus(self, i: int) -> Tuple[int, ...]:
        """Genişletilmiş komşuluk N+(i) = {i} ∪ N(i)."""
        return (i,) + tuple(self.neighbors[i])


# --------------------------------------------------------------------------- #
#  Yardımcılar
# --------------------------------------------------------------------------- #
def _parse_neighbors(obj) -> Tuple[int, ...]:
    if isinstance(obj, (list, tuple)):
        vals = list(obj)
    elif pd.isna(obj):
        vals = []
    else:
        try:
            parsed = ast.literal_eval(str(obj))
            vals = list(parsed) if isinstance(parsed, (list, tuple, set)) else [int(parsed)]
        except Exception:
            vals = [int(x) for x in str(obj).replace('[', '').replace(']', '').split(',')
                    if str(x).strip()]
    return tuple(sorted({int(v) for v in vals}))


def _num(v, default=0.0) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    if isinstance(v, str):
        v = v.strip().replace(",", ".")
    return float(v)


# --------------------------------------------------------------------------- #
#  Ana yükleyici
# --------------------------------------------------------------------------- #
def load_instance(excel_path: str) -> Instance:
    excel_path = str(excel_path)
    xls = pd.ExcelFile(excel_path)
    df_nodes = pd.read_excel(xls, sheet_name="inputs_df")
    df_bases = pd.read_excel(xls, sheet_name="bases")
    df_params = pd.read_excel(xls, sheet_name="parameters")

    # --- parametreler ----------------------------------------------------- #
    params = {str(r["parameter"]).strip(): r["value"] for _, r in df_params.iterrows()}
    cell_side = _num(params.get("cell_side_length", 500.0), 500.0)
    alpha_val = cell_side / 2.0
    water_per_cell = _num(params.get("water_required_per_cell", 9080.5), 9080.5)
    mtime = _num(params.get("Mtime", 0.0), 0.0)

    # --- araçlar (bases + parameters) ------------------------------------- #
    vehicle_types_map = {"Helicopter": "helicopter",
                         "Fire Engine": "fire_engine",
                         "FRV": "FRV"}
    vehicle_records: List[dict] = []
    current_k = 1
    for _, row in df_bases.iterrows():
        b_id = str(row["Base"])
        # bases koordinatları hücre-köşe indeksi -> metre  (P4 fix)
        bx = _num(row["x_coordinate"]) * cell_side
        by = _num(row["y_coordinate"]) * cell_side
        for col_name, p_prefix in vehicle_types_map.items():
            if col_name in df_bases.columns and pd.notna(row[col_name]) and row[col_name] > 0:
                count = int(row[col_name])
                v_speed = _num(params[f"{p_prefix}_speed"])
                v_cap = _num(params[f"{p_prefix}_capacity"])
                for _ in range(count):
                    vehicle_records.append({
                        "id": current_k, "base_id": b_id, "type": col_name,
                        "speed": v_speed, "capacity": v_cap, "bx": bx, "by": by,
                    })
                    current_k += 1
    if not vehicle_records:
        raise ValueError("Hiçbir araç oluşturulamadı — 'bases'/'parameters' sayfalarını kontrol edin.")

    # --- düğümler --------------------------------------------------------- #
    N = sorted(pd.to_numeric(df_nodes["node_id"]).astype(int).unique().tolist())
    coords = {int(r["node_id"]): (_num(r["x_coordinate"]), _num(r["y_coordinate"]))
              for _, r in df_nodes.iterrows()}
    state = {int(r["node_id"]): int(r["state"]) for _, r in df_nodes.iterrows()}
    pi = {int(r["node_id"]): _num(r["value_at_start"]) for _, r in df_nodes.iterrows()}
    lam = {int(r["node_id"]): _num(r["fire_degradation_rate"]) for _, r in df_nodes.iterrows()}
    sig = {int(r["node_id"]): _num(r["fire_amelioration_rate"]) for _, r in df_nodes.iterrows()}
    beta = {int(r["node_id"]): _num(r.get("beta", 0.0)) for _, r in df_nodes.iterrows()}
    a = {int(r["node_id"]): _num(r.get("su", 0.0)) for _, r in df_nodes.iterrows()}
    neighbors = {int(r["node_id"]): _parse_neighbors(r["neighborhood_list"])
                 for _, r in df_nodes.iterrows()}

    K = [v["id"] for v in vehicle_records]
    mu = {v["id"]: float(v["capacity"]) for v in vehicle_records}

    # --- d_ik ------------------------------------------------------------- #
    d: Dict[Tuple[int, int], float] = {}
    for v in vehicle_records:
        k, spd = v["id"], float(v["speed"])
        bx, by = float(v["bx"]), float(v["by"])
        for i in N:
            ix, iy = coords[i]
            if v["type"] == "Helicopter":          # hava -> Öklid
                dist = math.hypot(ix - bx, iy - by)
            else:                                  # kara -> Manhattan
                dist = abs(ix - bx) + abs(iy - by)
            d[(i, k)] = dist / spd

    # --- kümeler ---------------------------------------------------------- #
    Na = [i for i in N if state[i] == 1]
    Nw = [i for i in N if state[i] == 2]
    Np = [i for i in N if state[i] == 3]
    Nf = [i for i in N if state[i] in (0, 1)]
    e = {i: (1 if i in Na else 0) for i in N}
    alpha = {i: alpha_val for i in N}

    inst = Instance(
        name=Path(excel_path).stem,
        N=N, Nf=Nf, Na=Na, Nw=Nw, Np=Np, K=K,
        pi=pi, lam=lam, sig=sig, beta=beta, a=a, alpha=alpha,
        state=state, e=e, neighbors=neighbors, coords=coords,
        mu=mu, d=d, vehicle_info=vehicle_records,
        params=params, cell_side_length=cell_side,
        water_required_per_cell=water_per_cell, mtime=mtime,
    )
    _sanity_check(inst)
    return inst


def _sanity_check(inst: Instance) -> None:
    """Yüklemede erken hata yakalama: komşuluk simetrisi, küme tutarlılığı."""
    nf = set(inst.Nf)
    for i in inst.Nf:
        for j in inst.neighbors[i]:
            if j not in nf:
                raise ValueError(f"N({i}) yanıcı olmayan {j} düğümünü içeriyor "
                                 f"(su/yol kenarı çıkarılmamış?).")
            if i not in inst.neighbors[j]:
                raise ValueError(f"Komşuluk simetrik değil: {i}->{j} var ama {j}->{i} yok.")
    if not inst.Na:
        raise ValueError("Aktif yangın (Na) yok — en az bir state==1 hücre gerekli.")
    for i in inst.Nf:
        if inst.lam[i] <= 0 or inst.sig[i] <= 0:
            raise ValueError(f"lambda/sigma pozitif olmalı (düğüm {i}).")


if __name__ == "__main__":
    import sys
    inst = load_instance(sys.argv[1] if len(sys.argv) > 1 else "inputs/inputs_4x4.xlsx")
    print(f"{inst.name}: |N|={len(inst.N)} |Nf|={len(inst.Nf)} "
          f"|Na|={len(inst.Na)} |K|={len(inst.K)}")
    print("Na =", inst.Na, " Nw =", inst.Nw, " Np =", inst.Np)
