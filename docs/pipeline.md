# Model karşılaştırma pipeline'ı

Karşılaştırılan her model (Strong Baseline, Dist-Tracker, ileride TIR-JDE) **aynı veriyle, aynı
doğrulama grubuyla ve aynı değerlendirme koduyla** eğitilir ve test edilir. Her çalıştırmanın
veri seti, model tipi, parametreleri, eğitim eğrileri, test metrikleri ve çıktıları MLflow'a
(DagsHub) kaydedilir.

```
            configs/experiment/<deney>.yaml  (+ seed=…, model.detector.epochs=… gibi değişiklikler)
                                   │
 ┌─────────────┐   ┌─────────────────────────────┐   ┌──────────────────┐   ┌─────────────┐
 │ preprocess  │ → │ train (adaptör)              │ → │ evaluate         │ → │ visualize   │
 │ tir_jde/    │   │ tir_jde/modeling/adapters/   │   │ tir_jde/eval/    │   │ tir_jde/    │
 │ dataset.py  │   │ yazarın kodu, kendi ortamında│   │ TrackEval (ortak)│   │ plots.py    │
 └─────────────┘   └─────────────────────────────┘   └──────────────────┘   └─────────────┘
        └──────────────────────── hepsi aynı MLflow run'ına yazar ────────────────────────┘
```

## Komutlar

```bash
python -m tir_jde.pipeline make-val   strong_baseline_bsb --fraction 0.15  # bir kez: data/splits/val.txt
python -m tir_jde.pipeline preprocess strong_baseline_bsb
python -m tir_jde.pipeline run        dist_tracker_bsb seed=0               # train + evaluate + visualize
python -m tir_jde.pipeline run        strong_baseline_bsb seed=0 tag=smoke model.detector.epochs=1
python -m tir_jde.pipeline compare    strong_baseline_bsb dist_tracker_bsb --seeds 0 1 2
```

Colab'da baştan sona: [`notebooks/colab_pipeline.ipynb`](../notebooks/colab_pipeline.ipynb).

## Adımlar

**Preprocess** (`tir_jde/dataset.py`) — ham BSB'den (video veya kare + MOTChallenge etiketi):

| Çıktı | Kim kullanır |
|---|---|
| `yolo/{train,val}/{images,labels}` + `dataset.yaml` | iki dedektör; dosya adları sekansı korur (`C_01_00001.jpg`) |
| `reid/MOT20/bounding_box_{train,test}` | Strong Baseline ReID; train = eğitim sekansları, test = doğrulama sekansları |
| `track/val/Test_imgs` + `TestLabels_FirstFrameOnly` | takip girdisi (Track 3 düzeni) |
| `gt/val/<seq>.txt` | ortak değerlendirme |

Çıktılar **split hash**'iyle adlandırılır (`bsb-<hash>`): sekans listeleri veya dönüştürme ayarları
değişirse yeni bir klasör oluşur, eskisi yanlışlıkla kullanılmaz. Hash MLflow'a `data.split_hash`
etiketi olarak yazılır.

**Train** (`tir_jde/modeling/adapters/`) — her model kendi deposunda, kendi Python ortamında,
**kodu değiştirilmeden** çalışır. Adaptör yalnızca veri yolunu, seed'i ve config'teki parametreleri verir.

| Model | Ortam | Not |
|---|---|---|
| Strong Baseline | Python 3.11, torch 2.2.2 + flash-attn | yazarın `train.py`'sindeki `YOLO(...).train(...)` çağrısı + seed; ReID için yazarın `train_net.py`'si |
| Dist-Tracker | Python 3.8, torch 2.4.1 | yazarın Ultralytics fork'u; pip'teki `ultralytics` yüklenirse çalışma durdurulur (aksi halde NWD/SSQ kaybı sessizce devre dışı kalır) |

Depo, config'teki commit'e sabitlenir; depoda yerel değişiklik varsa çalıştırma reddedilir.

**Evaluate** (`tir_jde/eval/trackeval_runner.py`) — her modelin MOTChallenge çıktısı aynı TrackEval
koduyla puanlanır: HOTA, DetA, AssA, LocA, IDF1, IDP, IDR, MOTA, MOTP, IDSW, FP, FN, Frag ve uçtan uca FPS
(model yükleme dahil duvar saati). Sekans bazında sonuçlar `per_sequence.csv` olarak kaydedilir.

**Visualize** (`tir_jde/plots.py`) — sekans başına HOTA grafiği ve seçilen sekanslar için takip videosu
(renkli kutular = tahmin + ID, gri = GT).

**Compare** — her deneyin seed'leri üzerinden ortalama ± std tablo (`comparison.csv`) ve grafik.

## Değerlendirme protokolü (BSB)

Üç yöntem (Strong Baseline, Dist-Tracker, TIR-JDE) aynı protokolle çalıştırılır ve bu protokol BSB'ninkiyle aynıdır.

| Kural | İçerik |
|---|---|
| Girdi | Sekansın bütün kareleri + `TestLabels_FirstFrameOnly` (1. karenin verilen kutuları) |
| 1. kare | Takipçi verilen kutularla başlatılır (güven 1.0); 1. karede dedektör kullanılmaz |
| 2. kare ve sonrası | Dedektör + takipçi |
| Boş 1. kare | 1. karede iz olmaz (örn. MultiUAV-068) |
| Çıktı | `frame,id,x,y,w,h,score,-1,-1,-1`; kare 1'den başlar, koordinatlar orijinal 640×512 |
| Değerlendirme | TrackEval, tüm kareler (1. kare dahil); ana ölçüt HOTA, yanında IDF1, MOTA, IDSW |

Yöntemlerin uyumu:

- **Strong Baseline:** yazarın `tools/predict_track3.py`'si 1. karede dedektör çıktısı yerine verilen kutuları kullanır; değişiklik yok.
- **Dist-Tracker:** makale problemi "ilk karedeki bilinen tespitlerle başlatılan takip" diye tanımlar, ancak
  FLIT'in algoritmasında ilk kareye özel bir adım yoktur; repodaki `track.py` `model.track` çağırır, ilk kare GT'si
  almaz ve FLIT yerine Ultralytics BoT-SORT (`botsort.yaml`) çalıştırır. **İlk kare başlatması makalenin ya da
  yazar kodunun parçası değildir; BSB protokolüne uymak için eklenen bir protokol uyarlamasıdır.** TIR-JDE'nin
  runner'ı (`_ultralytics_runner.py`), yazarın koduna dokunmadan, 1. karenin tespitlerini verilen kutularla değiştiren
  bir `on_predict_postprocess_end` callback'i kaydeder. Callback takipçininkinden önce kaydedilir ve 1. kareyi dosya
  adından tanır; sonraki kareler `model.track` ile birebir aynıdır. Config: `model.tracker.first_frame_gt`
  (gerçek Ultralytics ile `protocol.first_frame_match = 1.0` doğrulanana kadar varsayılan `false`).
- **TIR-JDE:** takipçi bu protokolle yazılır.

**Bilinen davranış (boş 1. kare):** Ultralytics takipçisi tespit olmayan karede çağrılmaz (`if len(det) == 0: continue`),
bu yüzden iç kare sayacı ilerlemez ve ilk tespit gelen kare takipçi için "1. kare" olur; izler orada hemen aktif
edilir. Bu yazar kodundaki mevcut davranıştır (SB'de de benzeri vardır) ve yalnızca 1. karesi boş birkaç sekansı etkiler.

**Otomatik denetim:** her değerlendirmede çıktının 1. karesi verilen kutularla karşılaştırılır (IoU ≥ 0.99) ve MLflow'a
`protocol.first_frame_match` (1.0 olmalı) ve `protocol.first_frame_extra` (0 olmalı) olarak yazılır. Uyumsuzlukta
ekrana uyarı basılır.

## MLflow'da ne kaydedilir

| | |
|---|---|
| Parametreler | birleşik config'in tamamı (`model.detector.epochs`, `data.reid.first_fraction`, `seed`, …), veri sayıları |
| Etiketler | `model`, `dataset`, `seed`, `data.split_hash`, `baseline.commit`, `tir_jde.commit`, `config.hash` |
| Metrikler | `train.*` (epoch başına kayıp ve mAP), `val.*` (HOTA, IDF1, MOTA, …), `val.FPS_end_to_end` |
| Dosyalar | `config/config.yaml`, ortamların `pip freeze` çıktıları, eğitim logları, `best.pt`, tahminler, grafikler, videolar |

Kayıt yeri: `MLFLOW_TRACKING_URI` → yoksa `DAGSHUB_TOKEN` ile
`https://dagshub.com/slaaylc/TIR-JDE.mlflow` → yoksa `<work_dir>/mlflow.db` (yerel SQLite;
`mlflow ui --backend-store-uri sqlite:///<work_dir>/mlflow.db` ile açılır).

Bir deney + seed (+ `tag`) **tek bir MLflow run'ıdır**; train, evaluate ve visualize aynı run'a yazar.
Aynı run adıyla farklı ayarlar (ör. başka epoch sayısı) çalıştırılırsa pipeline durur; yeni ayar için
yeni `seed` veya `tag=...` verin.

## Yeni model eklemek

1. `configs/model/<model>.yaml` ve `configs/experiment/<model>_bsb.yaml`
2. `tir_jde/modeling/adapters/<model>.py`: `Adapter`'dan türeyen sınıf, `train()` ve `track()`
3. `ADAPTERS` sözlüğüne kayıt

`track()` yalnızca sekans başına bir MOTChallenge dosyası üretmelidir; değerlendirme ve loglama ortaktır.
