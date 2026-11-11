# Strong Baseline tracker (patched)

Strong Baseline'ın BoT-SORT'u ([wish44165/YOLOv12-BoT-SORT-ReID](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID), commit `a97bba3`), kaydedilmiş tespitler üzerinde çalışır. Kaynak koda dokunulmaz; değişiklikler `runner.py` içinde çalışma anında uygulanır ve her koşu, hangi yamanın gerçekten çalıştığını gösteren sayaçlar (`PATCH_STATS`) basar.

| Dosya | Ne yapar |
|---|---|
| `track.py` | Sekans başına yeni tracker; 1. karede GT (Track 3 protokolü), sonra skor > 0,09 tespitler |
| `runner.py` | Yamaları kurar, sonra `track.py`'yi çalıştırır |
| `costs.py` | NWD maliyeti (numpy; test edilir) |
| `api.py` | Python'dan tek çağrı: `run_tracker(...)`; istenen yama çalışmadıysa hata verir |

## Yamalar ve deney sonuçları

| Yama | Nasıl açılır | Durum |
|---|---|---|
| NWD maliyeti | `cost_mode="nwd"` | **Açık** (E1: HOTA +0,067) |
| Coasting | `coast_k=K` | **Açık, K = 10** (E1c) |
| Kamera hareketi (CMC) | `cmc="sparseOptFlow"` | **Açık** (E2) |
| Kalman Q/R ölçeği | `--kf_q`, `--kf_r` | Kapalı (Deney 4b: kazanç yok) |
| Mahalanobis gate, güven sönümü | `patches="gate"` / `"gate,conf"` | Kapalı (E5a: NWD ile birlikte yalnızca daraltıyor) |
| Görünüş, ağırlıklı birleşim | `emb_dir=..., fuse_w=w` | Kapalı (E7a: head-only embedding ile HOTA artmadı) |

Dondurulmuş konfigürasyon: `api.FROZEN` ve [`configs/tracker/frozen.yaml`](../../../configs/tracker/frozen.yaml).

## Kullanım

```python
from tir_jde.trackers.sb import FROZEN, run_tracker

stats = run_tracker(
    dets_dir="dets/", gt_dir="gt/", images="frames/", seqs=["BB1_00", "C_04"], out_dir="out/",
    sb_botsort="YOLOv12-BoT-SORT-ReID/BoT-SORT", **FROZEN,
)
```

Tespit dosyası: `kare, x1, y1, x2, y2, skor`. Çıktı: `kare, id, x, y, w, h, skor, -1, -1, -1` (MOTChallenge).
