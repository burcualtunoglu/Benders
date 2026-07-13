"""
config.py — Wildfire Suppression and Spread Control (WSC) modeli için merkezi
yapılandırma.

Buradaki tek bir `Config` nesnesi hem monolitik MILP'i (bkz. model_monolithic.py)
hem de Benders ayrışımını (Bölüm B) besler. Amaç: makaledeki modelin
davranışını ve rapordaki düzeltmeleri (benders_analiz_raporu) *veri değiştirmeden*
açıp kapatabilmek.

Kısıt numaraları makaledeki (1)–(55) numaralandırmasına atıf yapar.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
#  Model varyant seçenekleri
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    # --- omega (iş yükü) tanımı ------------------------------------------- #
    # "midpoint"  : orijinal makale (16): 2*omega = omega_max + omega_min
    #               + 2*Dwat*u_i  (h_max, h_min, omega_min ikilileriyle).
    # "worstcase" : rapor Gözlem 2(a): omega_i = omega_max_i + Dwat*u_i.
    #               h_max/h_min/omega_min tamamen düşer; alt problem saf LP olur.
    omega_mode: str = "midpoint"

    # --- rapordaki düzeltmeler (hepsi varsayılan olarak AÇIK) ------------- #
    # (26)-(28) düzeltmesi: (26') ts_min_i <= ts_j + Md*(1-y_j) ve
    #                       (28') b_min_ij <= y_j + (1-y_i).
    apply_correction_26: bool = True
    # (18)-(19) tek kısıtta birleştir: sum_k mu_k*s_ik >= omega_i - Ms*(1-u_i).
    merge_18_19: bool = True
    # (20)'yi kaldır ((23)+(31) tarafından domine).
    drop_20: bool = True

    # --- Big-M sıkılaştırma ---------------------------------------------- #
    # "tight" : ts_UB yayılım grafındaki en uzun sum(alpha/lambda) yolundan
    #           önişlemede hesaplanır; Md, Mi bundan türetilir (preprocessing.py).
    # "data"  : Excel'deki hazır Mi ve parameters!Mtime kullanılır (baseline).
    bigm_mode: str = "tight"
    bigm_safety: float = 1.02       # tight modda üst sınırlara küçük emniyet payı

    # Ms (su kısıtlarının big-M'i) kaynağı:
    # "constant" : parameters!water_required_per_cell  (baseline = 9080.5 ~ 9081)
    # "percell"  : Ms_i = a_i * te_UB_i + Dwat   (hücre-bazlı sıkı)
    ms_mode: str = "constant"

    # --- veride bulunmayan, baseline'da sabit-kodlu model sabitleri ------- #
    # (kullanıcı onayı: Dwat=500, Dbuf=5)
    delta_wat: float = 500.0        # Δwat  — kontrol edilen her yangın için taban su
    delta_buf: float = 5.0          # Δbuf  — operasyonel hazırlık tamponu (kısıt 29)


# --------------------------------------------------------------------------- #
#  Çözücü (Gurobi) seçenekleri
# --------------------------------------------------------------------------- #
@dataclass
class SolverConfig:
    time_limit: Optional[float] = None   # saniye (None = sınırsız)
    mip_gap: Optional[float] = None      # örn. 0.0 = optimal
    threads: Optional[int] = None
    output_flag: int = 1
    seed: int = 0
    log_dir: str = "output"


# --------------------------------------------------------------------------- #
#  Üst düzey config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    excel_path: str
    model: ModelConfig = field(default_factory=ModelConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    result_dir: str = "result"

    # kolay erişim kısayolları -------------------------------------------- #
    @property
    def omega_mode(self) -> str:
        return self.model.omega_mode

    def summary(self) -> str:
        m = self.model
        return (
            f"instance={self.excel_path}\n"
            f"  omega_mode        = {m.omega_mode}\n"
            f"  correction_26     = {m.apply_correction_26}\n"
            f"  merge_18_19       = {m.merge_18_19}\n"
            f"  drop_20           = {m.drop_20}\n"
            f"  bigm_mode         = {m.bigm_mode} (safety={m.bigm_safety})\n"
            f"  ms_mode           = {m.ms_mode}\n"
            f"  delta_wat/buf     = {m.delta_wat} / {m.delta_buf}\n"
        )
