"""preprocessing.py — baseline ön işleme sarmalayıcısı (F-MD faktörü).

Değişmeyen ``lbbd_v2.preprocessing.preprocess_instance`` yeniden kullanılır. Tek fark: F-MD KAPALI
iken (baseline) zaman ufku M_d, GEVŞEK küresel-toplam ile kurulur; F-MD AÇIK iken mevcut
sıkılaştırılmış (en-uzun-ateşleme-yolu) M_d korunur.

Her iki M_d de GEÇERLİ üst sınırdır (küresel toplam ≥ en-uzun-yol); yalnız gevşeklik farkı vardır
(rapor: sıkılaştırma monolitiği ~4× hızlandırır). M_i, M_s ts_ub'ye bağlı olmadığından (yalnız burn)
değişmez; sadece M_d yeniden hesaplanır — dolayısıyla kesinlik her iki durumda da korunur.
"""
from __future__ import annotations

from lbbd_v2.config import Config
from lbbd_v2.data_loader import Instance
from lbbd_v2.preprocessing import Preprocessed, preprocess_instance, _burn


def _loose_md(inst: Instance, margin: float) -> float:
    """GEVŞEK zaman ufku: ts_ub = Σ_i α/λ_i (her zaman geçerli, en-uzun-yol ≤ bu)."""
    burn = _burn(inst)
    ts_ub_loose = sum(inst.alpha / inst.lam[i] for i in inst.Nf)
    te_ub = ts_ub_loose + max(burn.values())
    max_d = max(inst.d[(i, k)] for i in inst.Nf for k in inst.K)
    return (te_ub + inst.delta_buf + max_d) * margin


def prepare(inst: Instance, cfg: Config) -> Preprocessed:
    """cfg.tighten_md=False iken M_d'yi gevşek küresel-toplama düşür; True iken sıkı bırak."""
    pre = preprocess_instance(inst, cfg)
    if not getattr(cfg, "tighten_md", False):
        loose = _loose_md(inst, cfg.big_m_margin)
        # Güvenlik: gevşek ufuk sıkı olandan küçük olamaz; olursa (imkânsız) sıkıyı koru.
        if loose >= pre.M_d - 1e-9:
            pre.M_d = loose
            pre.ts_ub_exact = False
    return pre
