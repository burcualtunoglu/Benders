# LBBD – orman yangını müdahale planlaması

Bu klasörde kodu çalıştırmak ve sonuçları incelemek için gereken dosyalar var.
Yapılanların açıklaması: `LBBD_Calismasi_Yapilanlarin_Ozeti.docx`.

## Klasörler

| Klasör | İçerik |
|---|---|
| `kod/lbbd_v2/` | LBBD çözücüsü ve düzeltilmiş monolitik MILP modeli |
| `kod/inputs/` | Excel örnekleri (`inputs_4x4.xlsx` … `inputs_12x12.xlsx`) |
| `sonuclar/1_lbbd_ana_sonuclar/` | LBBD sonuçları, FS yapılandırması, 26 örnek × 3 seed (sunucu işleri 1933–1935) |
| `sonuclar/2_milp_referans/` | Düzeltilmiş monolitik MILP sonuçları: `tek_is_parcacigi/` (iş 1932), `sekiz_is_parcacigi/` (iş 2285) |
| `sonuclar/3_pilotlar/` | GAGG (5x5_2) ve SDC-t (10x10_v_full) yerel pilotları; `analiz.txt` iki kolun karşılaştırması |

## Kurulum

Python 3.9–3.12 gerekir.

```bash
bash kurulum.sh
```

Betik `.venv` sanal ortamını kurar ve `requirements.txt`'i yükler.
`pip install gurobipy` boyut sınırlı bir deneme lisansıyla gelir: 4x4 LBBD bununla çalışır, diğer örnekler ve MILP tam (akademik) Gurobi lisansı ister.

## LBBD çalıştırma

`kod/` klasöründen:

```bash
cd kod
../.venv/bin/python -m lbbd_v2.baseline.runner --instance inputs_4x4 \
  --factors C0,C0p,C3,FLOW,MTC,S11,WARM,PRE,K5,K7,FILT,MD,SEED,SEEDLS \
  --omega midpoint --seed 0 --threads 1 \
  --total 1800 --master 600 --sp 300 --seed-budget 120 \
  --outdir ../yeni_sonuclar --excel
```

* `--instance`: `kod/inputs/` içindeki dosyanın uzantısız adı (ör. `inputs_5x5_2`).
* GAGG için faktör listesine `,GAGG`, SDC-t için `,SDC` eklenir.
* 4x4 bu komutla yaklaşık 12 dakikada OPTIMAL, 532,62 verir (5 yineleme).

## Monolitik MILP çalıştırma

```bash
cd kod
../.venv/bin/python -m lbbd_v2.run_bigmfix_excel inputs/inputs_4x4.xlsx 1800 1 ../yeni_sonuclar/milp fix 0 deneme
```

Argümanlar: girdi dosyası, süre (s), iş parçacığı sayısı, çıktı klasörü, `fix` (düzeltilmiş model), seed, etiket.

## Sonuç dosyalarını okuma

LBBD JSON dosya adı: `<faktörler>__<örnek>__midpoint__seed<n>__g0__t0__c1__<tarih_saat>__<kimlik>.json`

| JSON alanı | Anlamı |
|---|---|
| `metrics.status` | OPTIMAL, TIME_LIMIT veya INCONCLUSIVE |
| `metrics.LB`, `metrics.UB`, `metrics.gap` | alt sınır, üst sınır, (UB−LB)/max(1,\|UB\|); problem en büyüklemedir |
| `iteration_log` | her yinelemenin LB/UB'si, kontrol kümesi, alt problem durumu |
| `incumbent_solution` | en iyi çözüm (yapı kararları ve araç atamaları) |
| `metrics.witness_check` | yeni koşularda çözümün özgün modelde denetimi (DOGRULANDI vb.) |

`ablation_tidy.csv` klasördeki koşuların bir satırlık özetidir. `--excel` verilirse aynı adla bir Excel dosyası da yazılır.
MILP sonuçları: `milp_<örnek>_seed<n>_<tarih>.json` (özet) ve `result_DUZELTILMIS_<örnek>_seed<n>_<tarih>.xlsx` (karar değişkenleri).
