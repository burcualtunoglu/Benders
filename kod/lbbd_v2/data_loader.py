"""data_loader.py — Excel -> Instance with strict validation.

Clean-room port of the corrected loader (the untouched legacy load_inputs_p4.py stays as
historical reference). Reuses the P4 coordinate-unit fix and travel-time metric
(helicopter Euclidean, ground Manhattan) verified against the notes §2.

Canonical set definitions (03 §1.1):
    N_f = {i : state_i ∈ {0,1}}   burnable cells
    N_a = {i : state_i = 1}       active ignition roots
    N(i)  = burnable neighbours of i   (raw neighbourhood ∩ N_f)   <-- MANDATORY
    N^+(i) = {i} ∪ N(i)

Water (state 2) and road (state 3) cells never enter the model. The mandatory
assertion N(i) ⊆ N_f is enforced by construction + recorded.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from lbbd_v2.config import DELTA_WAT, DELTA_BUF


class DataError(Exception):
    """Raised when input data is structurally invalid."""


@dataclass
class Instance:
    # sets
    N: List[int]
    Nf: List[int]
    Na: List[int]
    K: List[int]
    # graph (already restricted to burnable neighbours)
    neighbors: Dict[int, Tuple[int, ...]]    # N(i) ⊆ N_f
    Nplus: Dict[int, Tuple[int, ...]]        # N^+(i)
    arcs: List[Tuple[int, int]]              # A = {(i,j): i∈N_f, j∈N(i)}
    # cell parameters
    pi: Dict[int, float]     # π_i  value_at_start
    beta: Dict[int, float]   # β_i
    lam: Dict[int, float]    # λ_i  fire_degradation_rate
    sig: Dict[int, float]    # σ_i  fire_amelioration_rate
    a: Dict[int, float]      # a_i  per-delay extra water (su)
    e: Dict[int, int]        # 1 if root
    # resources
    mu: Dict[int, float]     # µ_k
    d: Dict[Tuple[int, int], float]  # d_ik travel time
    vehicle_info: List[dict]
    # scalars
    alpha: float
    delta_wat: float
    delta_buf: float
    # provenance
    source_path: str = ""
    dropped_nonburnable_neighbors: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    raw_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return len(self.N)

    @property
    def n_vehicles(self) -> int:
        return len(self.K)


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


def _get_float_param(P: Dict[str, Any], key: str) -> float:
    if key not in P or P[key] is None:
        raise DataError(f"Missing parameter '{key}'. Available: {sorted(P.keys())}")
    val = P[key]
    try:
        if isinstance(val, str):
            val = val.strip().replace(",", ".")
        return float(val)
    except Exception as ex:
        raise DataError(f"Parameter '{key}'={P[key]} not numeric") from ex


def _finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def load_instance(path: str | Path) -> Instance:
    path = Path(path)
    if not path.exists():
        raise DataError(f"Input file not found: {path}")

    xls = pd.ExcelFile(path)
    for sheet in ("inputs_df", "bases", "parameters"):
        if sheet not in xls.sheet_names:
            raise DataError(f"Sheet '{sheet}' missing in {path}")

    df_nodes = pd.read_excel(xls, sheet_name="inputs_df")
    df_bases = pd.read_excel(xls, sheet_name="bases")
    df_params = pd.read_excel(xls, sheet_name="parameters")

    P = {str(r["parameter"]).strip(): r["value"] for _, r in df_params.iterrows()}
    cell_side = _get_float_param(P, "cell_side_length")
    alpha = cell_side / 2.0

    # ---- nodes -----------------------------------------------------------
    if "node_id" not in df_nodes.columns:
        raise DataError("inputs_df missing 'node_id'")
    if df_nodes["node_id"].isna().any():
        raise DataError("NaN node_id present")

    N = sorted(pd.to_numeric(df_nodes["node_id"]).astype(int).unique().tolist())
    coords: Dict[int, Tuple[float, float]] = {}
    state: Dict[int, int] = {}
    pi: Dict[int, float] = {}
    lam: Dict[int, float] = {}
    sig: Dict[int, float] = {}
    raw_neighbors: Dict[int, Tuple[int, ...]] = {}

    for _, row in df_nodes.iterrows():
        i = int(row["node_id"])
        coords[i] = (float(row["x_coordinate"]), float(row["y_coordinate"]))
        state[i] = int(row["state"])
        pi[i] = float(row["value_at_start"])
        lam[i] = float(row["fire_degradation_rate"])
        sig[i] = float(row["fire_amelioration_rate"])
        raw_neighbors[i] = _parse_neighbors(row["neighborhood_list"])

    def _num_col(col: str, default: float = 0.0) -> Dict[int, float]:
        if col in df_nodes.columns:
            s = pd.to_numeric(df_nodes[col], errors="coerce").fillna(default)
        else:
            s = pd.Series(default, index=df_nodes.index)
        return {int(df_nodes.loc[idx, "node_id"]): float(s.loc[idx]) for idx in df_nodes.index}

    a = _num_col("su", 0.0)
    beta = _num_col("beta", 0.0)

    bad_state = sorted({state[i] for i in N} - {0, 1, 2, 3})
    if bad_state:
        raise DataError(f"Undefined state value(s): {bad_state}; allowed {{0,1,2,3}}")

    Nf = [i for i in N if state[i] in (0, 1)]
    Na = [i for i in N if state[i] == 1]
    Nf_set = set(Nf)

    if not Na:
        raise DataError("No active ignition roots (state==1)")

    # a_i ('su') and beta_i are read with a silent 0.0 default by _num_col; a column
    # that is missing or entirely blank would disable the delay-dependent workload
    # (delta_ik = a_i (v_ik - t^s_i) == 0) or the reward decay WITHOUT any error.
    # Refuse that silently-degenerate model.
    if all(abs(a[i]) < 1e-12 for i in Nf):
        raise DataError(
            "Column 'su' missing/blank: a_i == 0 for every cell, so delta_ik == 0 and "
            "the delay-dependent workload omega_i degenerates to Delta_wat.")
    if all(abs(beta[i]) < 1e-12 for i in Nf):
        raise DataError(
            "Column 'beta' missing/blank: beta_i == 0 for every cell, so the reward "
            "never decays with control delay.")

    # ---- domain checks on burnable cells ---------------------------------
    for i in Nf:
        for name, val in (("pi", pi[i]), ("beta", beta[i]), ("lam", lam[i]),
                          ("sig", sig[i]), ("a", a[i])):
            if not _finite(val):
                raise DataError(f"Non-finite {name} at cell {i}: {val}")
        if lam[i] <= 0:
            raise DataError(f"lambda_{i} <= 0 (={lam[i]}); (49) undefined")
        if sig[i] <= 0:
            raise DataError(f"sigma_{i} <= 0 (={sig[i]}); (50) undefined")

    # ---- neighbours restricted to burnable + symmetry --------------------
    neighbors: Dict[int, Tuple[int, ...]] = {}
    dropped: Dict[int, Tuple[int, ...]] = {}
    for i in Nf:
        burnable = tuple(j for j in raw_neighbors[i] if j in Nf_set)
        drop = tuple(j for j in raw_neighbors[i] if j not in Nf_set)
        neighbors[i] = burnable
        if drop:
            dropped[i] = drop
    for i in Nf:
        for j in neighbors[i]:
            if i not in neighbors.get(j, ()):
                raise DataError(f"Asymmetric adjacency: {j} in N({i}) but {i} not in N({j})")
        # mandatory assertion: N(i) ⊆ N_f  (DataError, not assert: `python -O` strips asserts)
        outside = [j for j in neighbors[i] if j not in Nf_set]
        if outside:
            raise DataError(f"N({i}) not subset of N_f: {outside}")

    Nplus = {i: tuple(sorted({i, *neighbors[i]})) for i in Nf}
    arcs = [(i, j) for i in Nf for j in neighbors[i]]
    e = {i: (1 if state[i] == 1 else 0) for i in N}

    # ---- vehicles --------------------------------------------------------
    # (column name) -> (parameter prefix, travel metric).  The metric is part of the type
    # definition so a new aerial type cannot silently inherit the ground metric.
    vehicle_types_map = {"Helicopter": ("helicopter", "euclid"),
                         "Fire Engine": ("fire_engine", "manhattan"),
                         "FRV": ("FRV", "manhattan")}
    vehicle_records: List[dict] = []
    current_k = 1
    for _, row in df_bases.iterrows():
        b_id = str(row["Base"])
        bx = float(row["x_coordinate"]) * cell_side  # P4 corner-index -> metres
        by = float(row["y_coordinate"]) * cell_side
        for col_name, (prefix, metric) in vehicle_types_map.items():
            if col_name in df_bases.columns and pd.notna(row[col_name]) and row[col_name] > 0:
                count = int(row[col_name])
                v_speed = _get_float_param(P, f"{prefix}_speed")
                v_cap = _get_float_param(P, f"{prefix}_capacity")
                if v_speed <= 0:
                    raise DataError(f"Non-positive speed for {prefix}")
                for _ in range(count):
                    vehicle_records.append({
                        "id": current_k, "base_id": b_id, "type": col_name,
                        "metric": metric,
                        "speed": v_speed, "capacity": v_cap, "bx": bx, "by": by,
                    })
                    current_k += 1
    if not vehicle_records:
        raise DataError("No vehicles constructed from 'bases'/'parameters'")

    K = [v["id"] for v in vehicle_records]
    mu = {v["id"]: float(v["capacity"]) for v in vehicle_records}

    d: Dict[Tuple[int, int], float] = {}
    for v in vehicle_records:
        k = v["id"]
        vk = float(v["speed"])
        bx, by = float(v["bx"]), float(v["by"])
        for i in Nf:  # d only needed for burnable cells
            ix, iy = coords[i]
            if v["metric"] == "euclid":
                dist = math.hypot(ix - bx, iy - by)
            else:
                dist = abs(ix - bx) + abs(iy - by)
            dik = dist / vk
            if not _finite(dik):
                raise DataError(f"Non-finite d[{i},{k}]")
            d[(i, k)] = dik

    return Instance(
        N=N, Nf=Nf, Na=Na, K=K,
        neighbors=neighbors, Nplus=Nplus, arcs=arcs,
        pi=pi, beta=beta, lam=lam, sig=sig, a=a, e=e,
        mu=mu, d=d, vehicle_info=vehicle_records,
        alpha=alpha, delta_wat=DELTA_WAT, delta_buf=DELTA_BUF,
        source_path=str(path), dropped_nonburnable_neighbors=dropped,
        raw_params=P,
    )
