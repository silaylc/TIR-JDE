# Baseline'lar

Baseline kodları bu depoya kopyalanmaz. Her biri kendi deposunda, sabit bir commit'te ve kendi Python
ortamında çalışır; TIR-JDE pipeline'ı onları `tir_jde/modeling/adapters/` altındaki adaptörlerle çağırır.

| Model | Kullanılan depo (eğitilmemiş, yazar kodu) | Commit | Config |
|---|---|---|---|
| Strong Baseline | [silaylc/Yolov12_BoT-SORT_Re-ID](https://github.com/silaylc/Yolov12_BoT-SORT_Re-ID) | `c7c9ab0` | `configs/model/strong_baseline.yaml` |
| Dist-Tracker | [silaylc/YOLOv12-Dist-Tracker](https://github.com/silaylc/YOLOv12-Dist-Tracker) | `52b1729` | `configs/model/dist_tracker.yaml` |

Kurallar:
- Yazarların yayımladığı ağırlıklar kullanılmaz; iki baseline da BSB eğitim sekanslarında sıfırdan eğitilir.
  (Strong Baseline'ın yazar ReID ağırlığı istenirse `model.reid.weights=<yol>` ile verilebilir.)
- Kod değiştirilmez; adaptör yalnızca veri yolunu, seed'i ve config parametrelerini verir.
- Depo yerel değişiklik içeriyorsa çalıştırma reddedilir.
- Çıktılar MOTChallenge formatında yazılır ve `tir_jde/eval/` altındaki aynı TrackEval koduyla değerlendirilir.

Ayrıntılar: [`docs/pipeline.md`](../docs/pipeline.md).
