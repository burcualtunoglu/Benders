"""
benders_forward.py — Benders alt problemi öncesi İLERİ GEÇİŞ (rapor Gözlem 1).

Master'ın (ȳ, z̄, q̄) yayılım yapısı verildiğinde, ts/tm/te/ts_min sabitleri LP
çözmeden deterministik olarak hesaplanır (Önerme 1 iyi tanımlı olduğunu garanti eder):

    ts_r = 0            (r ∈ Na)
    q̄_ij = 1  =>  ts_j = tm_i
    tm_i = ts_i + alpha_i/lambda_i        (49)
    te_i = tm_i + alpha_i/sigma_i         (50)
    ts_min_i = min{ ts_j : j ∈ N+(i), ȳ_j = 1 }   (düzeltilmiş (26'))

Ayrıca (46) tutarlılığı kontrol edilir: bir z̄_ij = 1 yayı için ts_j > tm_i ise
kombinatoryal uygunluk kesmesi (rapor (18)) döndürülür:

    sum_{(a,b) in P(j)} q_ab + z_ij <= |P(j)|

P(j) kökten j'ye giden q-yoludur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from data_loader import Instance

TOL = 1e-6


@dataclass
class ForwardResult:
    ts: Dict[int, float]
    tm: Dict[int, float]
    te: Dict[int, float]
    ts_min: Dict[int, float]
    parent: Dict[int, int]                 # q-ebeveyni (kök hariç yanan hücreler)
    root_of: Dict[int, int]                # her yanan hücrenin kökü
    feas_cuts: List[List[Tuple[int, int]]]  # (46) ihlalleri: [(P(j) yayları)+, (i,j)]


def _q_parent(inst: Instance, y, q) -> Dict[int, int]:
    """q̄'dan ebeveyn eşlemesi: her yanan (e=0) hücrenin tek kaynağı."""
    parent = {}
    for j in inst.Nf:
        if y[j] > 0.5 and inst.e[j] == 0:
            for i in inst.neighbors[j]:
                if q.get((i, j), 0.0) > 0.5:
                    parent[j] = i
                    break
    return parent


def forward_pass(inst: Instance, y, z, q) -> ForwardResult:
    Nf = inst.Nf
    parent = _q_parent(inst, y, q)

    # --- ts: köklerden ebeveyn zinciriyle çöz ---------------------------- #
    ts = {i: (0.0 if inst.e[i] == 1 and y[i] > 0.5 else None) for i in Nf}
    # topolojik gevşetme (arborescence olduğundan |Nf| tur yeterli)
    for _ in range(len(Nf) + 1):
        changed = False
        for j in Nf:
            if ts[j] is None and j in parent:
                pi = parent[j]
                if ts.get(pi) is not None:
                    ts[j] = ts[pi] + inst.alpha[pi] / inst.lam[pi]   # ts_j = tm_parent
                    changed = True
        if not changed:
            break
    ts = {i: (v if v is not None else 0.0) for i, v in ts.items()}

    tm, te = {}, {}
    for i in Nf:
        if y[i] > 0.5:
            tm[i] = ts[i] + inst.alpha[i] / inst.lam[i]
            te[i] = tm[i] + inst.alpha[i] / inst.sig[i]
        else:
            ts[i] = tm[i] = te[i] = 0.0

    # --- ts_min: yalnız yanan komşular (düzeltilmiş (26')) --------------- #
    ts_min = {}
    for i in Nf:
        burning = [ts[j] for j in inst.Nplus(i) if y[j] > 0.5]
        ts_min[i] = min(burning) if burning else 0.0

    # --- kök etiketi (ebeveyn zincirinin tepesi) ------------------------- #
    root_of = {}
    for i in Nf:
        if y[i] > 0.5:
            cur, steps = i, 0
            while cur in parent and steps <= len(Nf):
                cur = parent[cur]; steps += 1
            root_of[i] = cur

    # --- (46) tutarlılık kontrolü --------------------------------------- #
    feas_cuts: List[List[Tuple[int, int]]] = []
    for i in Nf:
        if y[i] < 0.5:
            continue
        for j in inst.neighbors[i]:
            if z.get((i, j), 0.0) > 0.5 and ts.get(j, 0.0) > tm[i] + 1e-4:
                # kökten j'ye q-yolu P(j)
                path, cur, steps = [], j, 0
                while cur in parent and steps <= len(Nf):
                    path.append((parent[cur], cur)); cur = parent[cur]; steps += 1
                feas_cuts.append(path + [(i, j)])

    return ForwardResult(ts, tm, te, ts_min, parent, root_of, feas_cuts)
