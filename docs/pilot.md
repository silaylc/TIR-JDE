# Ortak dedektör

Katkımız tracker'da olduğu için başlık tablosundaki üç yöntem (Strong Baseline, Dist-Tracker, TIR-JDE) **aynı
dedektörün** tespitlerini kullanır; yöntemler yalnızca tracker'da ayrışır.

- Başlangıç: SB'nin SOT ağırlığı (YOLOv12-L, 640).
- Eğitim ayarları ve kutu kaybı: **Dist-Tracker'ınki** (Dist'in `train.py`'si + fork'unun `default.yaml`'ı,
  NWD + SSQ kaybı).
- Tek koşu, 10 epoch, bütün eğitim kareleri: `python -m tir_jde.pipeline train-detector common_detector_bsb seed=0`.
- Ayarlar `configs/setting/common_detector.yaml`. Daha kısa bir deneme gerekirse `common_detector_bsb_pilot`
  (aynı ayarlar, her 2. kare).

## Kod sürümü

Dedektör, SB yazarının YOLOv12 kodunda (Ultralytics 8.3.63 fork'u) **a97bba3** commit'inde (2 Temmuz 2025) çalışır
(`configs/model/common_detector.yaml`). Yazar 3 Temmuz 2025'te (255f987) kodu YOLOv12-turbo'ya geçirdi: dikkat
katmanı `qkv` + 7×7 konum katmanı yerine `qk` + `v` + 5×5, L/X için MLP oranı 1.2 yerine 1.5. Yayınlanan
`SOT_yolov12l.pt` ve `MOT_yolov12n.pt` eski sürümle kaydedildi; yeni kodda değerlendirme çöker, eğitimde ise
Ultralytics yalnızca uyan tensörleri aktardığı için dikkat katmanları **sessizce rastgele** başlar. Runner bu yüzden
her `.pt` başlangıcında ağırlığı kodla yeniden kurulan modelle karşılaştırır; tek bir tensör uymazsa eğitim
başlamaz. SB'nin tracker'ı ve ReID'i kendi sabit kopyasında (`configs/model/strong_baseline.yaml`) kalır.

## Başlangıç ağırlığı

**SB'nin SOT ağırlığı** (`SOT_yolov12l.pt`): YOLOv12-L, 640 px, Track 1/2 SOT verisi, SOT val'inde AP50-95 0.672.
Yazar reposundan (commit a97bba3) indirilir, SHA-256 denetlenir (`model.detector.pretrained.sot`).

- Ağırlığın eğitim kaydındaki başlangıç `yolov12l.pt` (COCO); makale ise 1. aşamayı "sıfırdan" diye anlatıyor.
  Raporda "SB'nin SOT ağırlığı" yazılır, COCO/sıfırdan ayrımı yapılmaz.
- `MOT_yolov12n.pt` ve ReID `model_0016.pth` kullanılmaz: Track 3 eğitim verisinin tamamıyla, kare bazlı bölmeyle
  eğitilmişler; BSB'nin 16 val sekansının hepsinin karelerini görmüşler, test sekanslarının ikizlerini de.
- Açık kontrol: SOT verisi ile BSB val/test arasında sahne çakışması (SOT karelerine parmak izi).

## Eğitim ayarları (Dist-Tracker ile karşılaştırmalı)

| Ayar | Dist-Tracker | Ortak dedektör | Not |
|---|---|---|---|
| Model, başlangıç | YOLO12-s, sıfırdan | YOLOv12-L, SOT ağırlığı | ortak dedektör kararı |
| Kutu kaybı | NWD + SSQ | NWD + SSQ (3 düzeltme, aşağıda) | |
| imgsz | 640 | 640 | |
| epochs | 50 | **10** | tek, kısa ince ayar |
| batch | 16 | **32** | 64 daha hızlıydı ama `multi_scale` (960'a kadar) ile 80 GiB'a sığmaz; `nbs` 64 ile etkin batch aynı |
| optimizer | auto | **SGD** | Dist'in uzun koşusunda auto SGD seçer; 10 epoch'ta AdamW'ye kayardı |
| lr0, lrf, momentum, weight_decay | 0.01, 0.01, 0.937, 0.0005 | aynı | SB'nin kendi MOT ince ayarı da 0.01 |
| warmup_epochs | 3 (50 epoch'ta) | **0.6** | aynı oran (3/50) |
| close_mosaic | 10 (50 epoch'ta) | **2** | aynı oran; 10 olsa mosaic 10 epoch'ta hiç açılmazdı |
| warmup_bias_lr, copy_paste, multi_scale | 0.1, 0.0, True | aynı | SB fork varsayılanları farklı (0.0, 0.1, False); açıkça yazılır |
| mosaic, scale, translate, flip, hsv, mixup | Ultralytics varsayılanları | aynı | |
| box, cls, dfl, nbs, amp | 7.5, 0.5, 1.5, 64, True | aynı | |
| workers | 8 | 8 | A100 makinesinde 12 çekirdek: 8 veri işçisi, kalan çekirdekler ana süreç, val ve izleme için |
| Doğrulama | Ultralytics varsayılanı | epoch sonlarında val'in her 10. karesi, sonda bütün val, `conf` 0.01 | eğitimi etkilemez |

## Kutu kaybı: Dist-Tracker SSQ, üç düzeltmeyle

`tir_jde/modeling/adapters/_dist_ssq_loss.py`, Dist'in `BboxLoss`'unun (Shape-NWD + dinamik WIoU, DFL) uyarlaması;
çalışma anında SB fork'unun `BboxLoss`'unun yerine konur (fork dosyaları değişmez). Yayınlanan Dist kodundaki üç
hata makalenin anlattığı davranış için düzeltilir:

1. Wasserstein terimi normalize edilmemiş (ham kare mesafe); karma hassasiyette taşar, smoke testteki NaN box loss
   bundandı. Düzeltme: `1 − exp(−√W₂ / C)`, `C = 5 × ortalama hedef boyutu` (Dist reposundaki kullanılmayan
   `NWDLoss`'un normalizasyonu).
2. Eğitim ilerlemesi kayba hiç ulaşmıyor (`batch['epoch']` Ultralytics'te yok); NWD → WIoU geçişi hiç olmuyor.
   Düzeltme: `on_train_epoch_start` geri çağrısı `epoch / epochs` oranını kayba iletir.
3. WIoU için IoU, `xyxy` kutulara varsayılan `xywh=True` ile hesaplanıyor. Düzeltme: `xywh=False`.

Tezde: "Dist-Tracker'ın kaybı makaledeki tanıma göre, yayınlanan koddaki üç hata düzeltilerek kullanıldı."

## Sınırlar (tezde yazılacak)

- Tek koşu, tek tohum: ayarların etkisi ayrıca ölçülmedi; ayarlar Dist-Tracker'dan alındı.
- Ortak dedektör SB'nin YOLOv12 kodunda eğitilir; Dist-Tracker'ın kendi dedektörü (YOLO12-s) ana tabloda yoktur.
- SOT ağırlığının başlangıcı (COCO mu sıfırdan mı) yazar kaynaklarında çelişkili.
