"""lbbd_v2.baseline — düzeltilmiş (bigmfix) model üzerinde SADE, kesin LBBD baseline'ı + ablasyon çatısı.

AŞAMA 1 (docs/asama0_rapor.md onaylı). Bu alt-paket mevcut ``lbbd_v2`` dosyalarını DEĞİŞTİRMEZ;
değişmeyen bileşenleri (data_loader, preprocessing, benders_forward, lbbd_cuts, lbbd_conflicts,
lbbd_heuristics, güçlü alt problem, model_monolithic_bigmfix) İÇE AKTARIR ve yalnızca düzeltilmiş
modelle tutarlı olması için değişmesi gereken parçaları yeniden yazar:

  * config.py         — BaselineConfig (tüm performans-only bayraklar KAPALI) + FAKTÖR registry
  * preprocessing.py  — ince sarmalayıcı: F-MD kapalıyken M_d'yi gevşek (küresel-toplam) yapar
  * subproblem_weak.py— zayıf (Big-M'li) alt problem; düzeltilmiş modelin kısıtlaması
  * subproblem.py     — motor seçici: "strong_resource" (mevcut) | "weak_resource" (yeni)
  * master.py         — düzeltilmiş master: geç-ateşleme `mccap` tavanı KALDIRILMIŞ
  * solver.py         — düzeltilmiş döngü: `_late_ignition_cells` geçidi ve `cut_ignition_time` YOK
  * runner.py         — tek koşu -> tidy CSV/JSON metrik satırı (SLURM giriş noktası)

Kesinlik sözleşmesi (docs/asama0_rapor.md §2) korunur: UB yalnız master dual sınırından; LB yalnız
doğrulanmış inkümbentten; kesme yalnız kanıtlanmış SP durumlarından; sertifikasız iken "optimal"
denmez. Referans = model_monolithic_bigmfix (düzeltilmemiş model DEĞİL).
"""
