# Dist-Tracker

Track 3 (4. Anti-UAV Challenge) birincisi; yalnızca hareket bilgisi kullanır, görünüm embedding'i yoktur.
Kod: [silaylc/YOLOv12-Dist-Tracker](https://github.com/silaylc/YOLOv12-Dist-Tracker) (ağırlıksız kopya),
adaptör: `tir_jde/modeling/adapters/dist_tracker.py`.

- Ortam: Python 3.8, torch 2.4.1, yazarın Ultralytics fork'u. pip'ten `ultralytics` yüklenmemelidir:
  yüklenirse fork yerine o kullanılır ve NWD/SSQ kaybı sessizce devre dışı kalır. Adaptör bunu tespit
  edip çalıştırmayı durdurur.
- Eğitim: `yolo12-UAV.yaml`, 50 epoch, 640 px. Takip: `botsort.yaml`, 1280 px, conf 0.
- Yazarlar yalnızca MOTA raporlar; burada IDF1, IDSW, HOTA ve FPS de raporlanır.

```bash
python -m tir_jde.pipeline run dist_tracker_bsb seed=0
```
