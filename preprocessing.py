"""
preprocessing.py — hücre-bazlı big-M sıkılaştırması (rapor Böl. 2.3).

Rapordaki öneri:
  * Md için doğal üst sınır  T = max_i { ts_UB_i + alpha_i/lambda_i + alpha_i/sigma_i }
    (= max_i te_UB_i).
  * ts_UB_i, yayılım grafında köklerden i'ye giden yollardaki sum(alpha/lambda)
    toplamlarının en büyüğüdür.
  * Mi = a_i * te_UB_i  (delta_ik = a_i(v_ik - ts_i), v_ik - ts_i <= te_UB_i).

ts_UB hesabı:
  Yangın bir q-yayı boyunca ilerlerken ts_j = tm_i = ts_i + alpha_i/lambda_i.
  Dolayısıyla i'nin ateşleme zamanı, kökten i'ye giden zincirdeki ÖNCEKI hücrelerin
  alpha/lambda toplamıdır. Grafik simetrik/döngülü olduğundan basit-yol en-uzunu
  NP-zordur; yerine en fazla |Nf|-1 kenarlı en-uzun-YÜRÜYÜŞü Bellman-Ford tarzı
  gevşetme ile buluruz — gerçek ateşleme zamanının GEÇERLİ bir üst sınırı.
"""

from __future__ import annotations

from typing import Dict

from data_loader import Instance
from config import ModelConfig


def _ts_upper_bounds(inst: Instance) -> Dict[int, float]:
    """En fazla |Nf|-1 kenarlı en-uzun-yürüyüş ile ts üst sınırı (köklerden)."""
    Nf = inst.Nf
    w = {i: inst.alpha[i] / inst.lam[i] for i in Nf}   # i ateşlenince eklenen süre
    NEG = float("-inf")

    dist = {i: (0.0 if inst.e[i] == 1 else NEG) for i in Nf}   # kökler ts=0
    edges = [(i, j) for i in Nf for j in inst.neighbors[i]]     # i -> j, ağırlık w[i]

    for _ in range(max(1, len(Nf) - 1)):
        updated = False
        for i, j in edges:
            if dist[i] > NEG and dist[i] + w[i] > dist[j] + 1e-12:
                dist[j] = dist[i] + w[i]
                updated = True
        if not updated:
            break

    reachable_max = max([v for v in dist.values() if v > NEG], default=0.0)
    walk = {i: (dist[i] if dist[i] > NEG else reachable_max) for i in Nf}

    # --- SIKILAŞTIRMA: ata-kümesi toplamı ---------------------------------- #
    # ts_i, i'ye giden ateşleme yolundaki ATALARıN alpha/lambda toplamıdır; atalar
    # ayrık ve "i'ye ulaşabilen hücreler" (anc(i)) kümesinin alt kümesidir. Dolayısıyla
    # ts_i <= sum_{a in anc(i)} w_a geçerli bir üst sınırdır — walk sınırının döngüde
    # şişmesini keser (walk ile min alınır). Kökler: ts=0 (48).
    anc = {i: set() for i in Nf}
    for a in Nf:                                  # a'dan ileri erişilebilenlerin atası a'dır
        seen, stack = {a}, [a]
        while stack:
            xx = stack.pop()
            for yy in inst.neighbors[xx]:
                if yy not in seen:
                    seen.add(yy); stack.append(yy)
        for j in seen:
            if j != a:
                anc[j].add(a)
    out = {}
    for i in Nf:
        if inst.e[i] == 1:
            out[i] = 0.0                           # kök: ts=0
        else:
            anc_sum = sum(w[a] for a in anc[i])
            out[i] = min(walk[i], anc_sum)
    return out


def compute_bigM(inst: Instance, cfg: ModelConfig) -> None:
    """inst nesnesine ts_ub, te_ub, Md, Mi, Ms, Ms_cell alanlarını doldurur."""
    Nf = inst.Nf
    # burn_i = te_i - ts_i = alpha/lambda + alpha/sigma  (hücrenin KENDİ yanma süresi;
    # ts_i'den bağımsız sabittir — Excel'deki 'time' sütunuyla aynı)
    burn = {i: inst.alpha[i] / inst.lam[i] + inst.alpha[i] / inst.sig[i] for i in Nf}

    ts_ub = _ts_upper_bounds(inst)                          # ateşleme zamanı üst sınırı (sıkı)
    tm_ub = {i: ts_ub[i] + inst.alpha[i] / inst.lam[i] for i in Nf}   # pre son tarihi üst sınırı
    te_ub = {i: ts_ub[i] + burn[i] for i in Nf}             # doğal sönme üst sınırı
    inst.ts_ub, inst.tm_ub, inst.te_ub = ts_ub, tm_ub, te_ub

    # -------- Md: tüm zaman değişkenlerini kapsayan tek skaler big-M -------- #
    # ts,tm,te,tc <= max te_UB;  t_ik/v_ik ek olarak seyahat+buffer kadar açılabilir.
    T = max(te_ub.values())
    max_d = max(inst.d.values())
    md_tight = (T + cfg.delta_buf + max_d) * cfg.bigm_safety
    inst.Md = (inst.mtime if (cfg.bigm_mode == "data" and inst.mtime > 0) else md_tight)

    # -------- Mi: hücre-bazlı delta big-M --------------------------------- #
    # delta_ik = a_i (v_ik - ts_i); (24)-(25),(34) gereği 0 <= v_ik - ts_i <= burn_i.
    # Dolayısıyla YEREL yanma penceresi kullanılır (yayılımla biriken te_UB DEĞİL).
    inst.Mi = {i: inst.a[i] * burn[i] * cfg.bigm_safety for i in Nf}

    # -------- Ms: su big-M ------------------------------------------------- #
    # omega_max_i <= max_k delta_ik <= a_i*burn_i;  omega_i <= a_i*burn_i + Dwat.
    inst.Ms_cell = {i: inst.a[i] * burn[i] * cfg.bigm_safety + cfg.delta_wat for i in Nf}
    inst.Ms = (max(inst.Ms_cell.values()) if cfg.ms_mode == "percell"
               else inst.water_required_per_cell)

    _compute_reachability(inst, cfg)


def _compute_reachability(inst: Instance, cfg: ModelConfig) -> None:
    """
    LBBD master çekişme gevşetmesi için ön hesap (rapor yol haritası, V2/V4).

    Eşleme VI PRE-hücrelere uygulandığından son tarih olarak (te değil) tm_UB kullanılır
    --- kökte tm_r = alpha_r/lambda_r küçüktür, dolayısıyla A_i^pre çok daralır ve Hall
    kısıtı gerçekten bağlayıcı olur.

    A_i  : i'yi tm_UB_i'den önce basabilen araçlar (window^pre_ik = tm_UB_i - (Δbuf+d_ik)).
    kappa_i : Δwat'ı bu pre penceresinde karşılamak için gereken MİN araç (μ·window büyükten).
    controllable_i : hücre HİÇ kontrol edilebilir mi (post için gevşek te_UB penceresi).
    """
    Dbuf, Dwat = cfg.delta_buf, cfg.delta_wat
    inst.A, inst.kappa, inst.controllable = {}, {}, {}
    for i in inst.Nf:
        caps = []                                        # pre penceresi (tm_UB)
        for k in inst.K:
            window = inst.tm_ub[i] - (Dbuf + inst.d[i, k])
            if window > 1e-9:
                caps.append((k, inst.mu[k] * window))
        caps.sort(key=lambda t: -t[1])
        inst.A[i] = [k for k, _ in caps]
        cum, kap = 0.0, 0
        for _, c in caps:
            if cum >= Dwat - 1e-9:
                break
            cum += c
            kap += 1
        inst.kappa[i] = max(1, kap)                       # pre penceresinde min araç (>=1)
        # controllable: post (gevşek te_UB) penceresinde Δwat hiç karşılanabiliyor mu?
        cap_te = sum(inst.mu[k] * max(0.0, inst.te_ub[i] - (Dbuf + inst.d[i, k])) for k in inst.K)
        inst.controllable[i] = (cap_te >= Dwat - 1e-9)


def bigM_report(inst: Instance) -> str:
    return (f"  Md = {inst.Md:.2f}\n"
            f"  Ms = {inst.Ms:.2f}\n"
            f"  Mi in [{min(inst.Mi.values()):.1f}, {max(inst.Mi.values()):.1f}]\n"
            f"  te_UB max = {max(inst.te_ub.values()):.3f}")
