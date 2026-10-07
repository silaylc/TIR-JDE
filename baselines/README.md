# Baseline'lar

Baseline kodları bu depoya kopyalanmaz; kendi depolarında ve ayrı conda ortamlarında çalıştırılır.

- Strong Baseline (ana baseline): github.com/wish44165/YOLOv12-BoT-SORT-ReID
- Dist-Tracker (ikinci baseline): github.com/earth-insights/Dist-Tracker

Kurallar:
- Yazarların yayımladığı ağırlıklar kullanılmaz; iki baseline da BSB eğitim sekanslarında yeniden eğitilir.
- Çıktılar MOTChallenge formatına çevrilip eval/ altındaki aynı TrackEval betiğiyle değerlendirilir.

Bu klasörde yalnızca çalıştırma notları ve çıktıları dönüştüren betikler tutulur.
