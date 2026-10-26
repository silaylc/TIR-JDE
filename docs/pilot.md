# Ortak dedektör ve pilotu

Katkımız tracker'da olduğu için başlık tablosundaki üç yöntem (Strong Baseline, Dist-Tracker, TIR-JDE) **aynı
dedektörün** tespitlerini kullanır; yöntemler yalnızca tracker'da ayrışır. Dist-Tracker'ın SSQ kaybı ve SB'nin
kendi dedektör tarifi ana tabloya girmez, ek ablasyondadır.

Ayarlar `configs/setting/common_detector.yaml`, deneyler `common_detector_bsb` (ana koşu) ve
`common_detector_bsb_pilot`.

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
Yazar reposundan (commit 071fe76) indirilir, SHA-256 denetlenir (`model.detector.pretrained.sot`).

- Ağırlığın eğitim kaydındaki başlangıç `yolov12l.pt` (COCO); makale ise 1. aşamayı "sıfırdan" diye anlatıyor.
  Raporda "SB'nin SOT ağırlığı" yazılır, COCO/sıfırdan ayrımı yapılmaz.
- `MOT_yolov12n.pt` ve ReID `model_0016.pth` kullanılmaz: Track 3 eğitim verisinin tamamıyla, kare bazlı bölmeyle
  eğitilmişler; BSB'nin 16 val sekansının hepsinin karelerini görmüşler, test sekanslarının ikizlerini de.
- Açık kontrol: SOT verisi ile BSB val/test arasında sahne çakışması (SOT karelerine parmak izi).

## Ortak ayarlar

| Ayar | Değer |
|---|---|
| Model | YOLOv12-L, tek sınıf, SOT ağırlığından ince ayar |
| Çözünürlük | 640 (SOT ağırlığının çözünürlüğü); 1280 yalnızca tek tohumlu duyarlılık deneyi |
| Veri | BSB train: 102 − T_13, BB2P_01 − 16 val = 84 sekans; sekans bazlı bölme |
| Optimizasyon | SGD (Ultralytics'te Nesterov), `lr0` 0.003, `lrf` 0.01, `batch` 16, `nbs` 64, AMP |
| Isınma | `warmup_epochs` 1, `warmup_bias_lr` 0.0 (hazır tek sınıflı başlığı 0.1'lik bias lr'siyle bozmamak için) |
| Augmentasyon | mosaic 1.0, `scale` 0.5, translate 0.1, fliplr 0.5; flipud, döndürme, eğme, perspektif, mixup, copy_paste 0; `hsv_h`/`hsv_s` 0 (termal gri), `hsv_v` 0.4 |
| `close_mosaic` | ana koşu 10, pilot 2 |
| Doğrulama | dondurulmuş `val.txt` (16 sekans), `conf` 0.01, `iou` 0.7, `max_det` 300 |

Fork varsayılanlarına güvenilmez; bütün argümanlar açıkça verilir (SB fork'unda `copy_paste` 0.1,
`warmup_bias_lr` 0.0, `optimizer` auto).

## Sıra

1. **Sıfır-atış değerlendirme:** `python -m tir_jde.pipeline eval-detector common_detector_bsb tag=zeroshot`.
   SOT ağırlığı eğitimsiz, val'in bütün karelerinde (`det_val_zero_shot.*`). İnce ayarın başlangıç noktası;
   beklenmedik yüksek bir skor sızıntı ipucudur.
2. **Zamanlama koşusu (P0):** `fraction=0.05`, 1 epoch. `train.seconds_per_iteration`, epoch süresi,
   GPU belleği ve kullanımı. `fraction` listenin ilk %5'ini alır (rastgele değil); yalnızca süre için.
3. **Pilot:** her 2. kare, 10 epoch, epoch sonlarında val'in her 10. karesi, sonunda tam val.

   | Koşu | Değişiklik | Tohum |
   |---|---|---|
   | R0a | — | 0 |
   | R0b | — | 1 |
   | R2 | `lr0` 0.005 | 0 |
   | R3 | `lr0` 0.002 | 0 |
   | R4 | `scale` 0.25 | 0 |

4. **Ana koşu:** epoch sayısı pilot eğrisinden ve bütçeden; 3 tohum (`common_detector_bsb seed=0|1|2`).
   Test seti hiçbir seçimde kullanılmaz.

## Karar kuralı

Birincil ölçüt tam val mAP50, recall yanında. Bir ayar R0a'dan |R0a − R0b| farkının iki katından fazla iyiyse
benimsenir, değilse varsayılan kalır. `python -m tir_jde.pipeline pilot-report common_detector_bsb_pilot`
tabloyu ve kararı `<work_dir>/runs/common_detector_bsb_pilot-pilot_report.md` dosyasına yazar.

## MLflow'a kaydedilenler

Koşu adı, tohum, tam config, `split_hash`, kare listesi hash'leri, epoch başına kayıp ve doğrulama metrikleri
(`train.*`), epoch süresi ve iterasyon başına saniye, tam val (`det_val_full.*`), sıfır-atış
(`det_val_zero_shot.*`), GPU belleği ve kullanımı (`gpu.*`).

## Sınırlar (tezde yazılacak)

- Gürültü yalnızca iki tohumla tahmin edildi; "iki katı" eşiği kabadır.
- 10 epoch kısa bir ufuktur; `lr0` ve `scale` etkileri uzun eğitimde farklı çıkabilir.
- Ortak dedektör SB fork'unda seçilip eğitilir; Dist-Tracker'ın kendi dedektörü (YOLO12 + SSQ) ana tabloda yoktur.
- SOT ağırlığının başlangıcı (COCO mu sıfırdan mı) yazar kaynaklarında çelişkili.
