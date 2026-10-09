# TIR-JDE

**Degradation-Adaptive Small Target Detection and Joint Identity Embedding for Multi-Object Tracking in Thermal Infrared UAV Imagery**

Termal kızılötesi görüntüde çoklu İHA takibi. Tek bir hafif ağ, kutuyla
birlikte kimlik gömmesi üretiyor; aynı ağdan çıkan bir görüntü bozulma
tahmini hem tespit özelliklerini koşullandırıyor hem de takip aşamasında
görünüm ile hareketin ağırlığını **tespit başına** belirliyor.

> **Durum: iskelet.** Henüz çalışan kod yok. Bu depo şu an yapıyı,
> yöntemsel kaynakları ve değerlendirme protokolünü sabitliyor.

---

## Problem

Track 3 verisinde ortalama hedef 10.56 × 9.06 piksel; alt uçta 2 pikselin
altına inen kutular var. Bu boyutta hedefin dokusu yok. Görünürlüğü ise
sensör gürültüsü, otomatik kazanç kontrolü, termal sürüklenme, bulanıklık
ve atmosferik zayıflamayla kareden kareye değişiyor. Sürüdeki İHA'lar
birbirinin neredeyse aynısı.

Mevcut Track 3 çözümlerinin tepkisi iki uçta:

- **Görünümü tamamen bırakmak.** [Dist-Tracker](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/html/Wang_Dist-Tracker_A_Small_Object-aware_Detector_and_Tracker_for_UAV_Tracking_CVPRW_2025_paper.html)
  yalnızca hareket kullanıyor (IoU + L2 füzyonu), görünüm gömmesi yok.
- **Görünümü sabit ağırlıkla kullanmak.** [Strong Baseline](https://arxiv.org/abs/2503.17237)
  ve [PPTracker](https://mlanthology.org/cvprw/2025/qin2025cvprw-pptracker/),
  BoT-SORT'un sabit eşiklerinden geçirdiği ayrı eğitilmiş bir ReID kullanıyor.

[Context-Aware Identity Prediction](https://doi.org/10.3390/rs18132084) bu
ikisinin arasına giriyor: görünüm gömmesini uçtan uca öğreniyor ve
güvenilirliğe göre ağırlıklıyor — ama **kare başına tek bir ağırlıkla**,
sahne genelinden hesaplanan bir betimleyiciden.

TIR-JDE'nin iddiası: görünüm güvenilirliği karenin değil **tespitin**
özelliği. Aynı karede, aynı sensörde, aynı anda bir İHA net görünürken
diğeri arka plana karışabiliyor — dolayısıyla tek bir kare-düzeyi skor
hiçbirini doğru temsil etmiyor.

## Yaklaşım

1. **Hafif dedektör.** Ultralytics YOLO (Nano/Small), minik hedefler için
   P2 (stride-4) başlığı ve 1280 px girdi.
2. **Bozulma koşullandırma.** Bozulma tahmininden gelen bir kod,
   [FiLM](https://arxiv.org/abs/1709.07871) (`γ(z) ⊙ F + β(z)`) ile
   dedektör özelliklerini modüle ediyor.
3. **Ortak kimlik başlığı.** Dedektörle birlikte eğitilen kısa (64–128
   boyutlu) gömme — ayrı bir ReID ağı yok.
4. **Bozulmaya duyarlı eşleştirme.** Aynı bozulma tahmininden tespit başına
   bir güvenilirlik ağırlığı `r` türüyor:
   `maliyet = r × görünüm farkı + (1 − r) × konum farkı`.

Dördüncüsü asıl katkı. İncelediğimiz bozulmaya duyarlı yöntemlerin hepsi
([FiLM](https://arxiv.org/abs/1709.07871),
[DASR](https://arxiv.org/abs/2104.00416),
[AirNet](https://openaccess.thecvf.com/content/CVPR2022/html/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.html),
[DTRDNet](https://doi.org/10.3390/s24196330),
[RDMNet](https://github.com/xfwang23/RDMNet),
[DAISOD](https://arxiv.org/abs/2608.09311)) iyileştirme veya tespitte
duruyor; hiçbiri bozulma tahminini eşleştirmeye taşımıyor.

---

## Yöntemsel kaynaklar

| Bileşen | Kaynak | Bizim değişikliğimiz | Kod |
|---|---|---|---|
| Dedektör omurgası | [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics); [YOLO11-JDE](https://arxiv.org/abs/2501.13710) (mimari şablon) | P2 başlığı, 1280 px, FiLM modülü eklenmesi | `detector/` |
| Görünüm başlığı tasarımı | [JDE](https://arxiv.org/abs/1909.12605) (ortak başlık fikri), [FairMOT](https://arxiv.org/abs/2004.01888) (anchor'sız, yüksek çözünürlük, kısa vektör), [YOLO11-JDE](https://arxiv.org/abs/2501.13710) (2×3×3 + 1×1 konv.) | Başlık P2 seviyesinde; üç görevli otomatik kayıp dengesi (tespit + kimlik + bozulma) | `detector/heads/` |
| Kimlik denetimi | [YOLO11-JDE](https://arxiv.org/abs/2501.13710) (Mosaic ile etiketsiz triplet), [FairMOT](https://arxiv.org/abs/2004.01888) (etiketsiz ön eğitim) | Gerçek ID'lerle farklı karelerden pozitif çiftler; Track 1/2 SOT verisinde sekans = kimlik ön eğitimi | `detector/losses/` |
| Çoklu nokta okuma | [Motion-Guided Multi-Offset ReID Readout](https://doi.org/10.3390/rs18183238) | 6 piksellik kutuda tek merkez noktası kırılgan; birden fazla noktadan örnekleme | `detector/heads/` |
| Bozulma koşullandırma | [FiLM](https://arxiv.org/abs/1709.07871) (mekanizma), [DASR](https://arxiv.org/abs/2104.00416) / [AirNet](https://openaccess.thecvf.com/content/CVPR2022/html/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.html) (karşıtsal bozulma kodlayıcı), [DAISOD](https://arxiv.org/abs/2608.09311) (IR'de açık bozulma kestirimi) | TIR'e özgü sentetik bozulma (sütun gürültüsü, AGC kontrast sıkıştırma, bulanıklık, zayıflama); sentetik parametreler bedava etiket | `degradation/` |
| Takip çatısı | [ByteTrack](https://arxiv.org/abs/2110.06864) (iki turlu eşleştirme), [BoT-SORT](https://arxiv.org/abs/2206.14651) (w/h Kalman, iki kapılı füzyon) | İHA'ya göre yeniden ayarlanmış eşikler; [Edge-Aware](https://arxiv.org/abs/2607.12544)'den TFPS ve 1 karelik öngörülü devam | `trackers/` |
| Eşleştirme maliyeti | [BoT-SORT](https://arxiv.org/abs/2206.14651)'un sabit kapıları, [ByteTrack](https://arxiv.org/abs/2110.06864)'in skor kuralı, [Dist-Tracker](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/html/Wang_Dist-Tracker_A_Small_Object-aware_Detector_and_Tracker_for_UAV_Tracking_CVPRW_2025_paper.html)'ın L2 + IoU füzyonu | Sabit kapılar yerine sürekli, tespit başına `r`; minik kutularda IoU yerine merkez uzaklığı / [NWD](https://arxiv.org/abs/2110.13389) denemesi | `trackers/*/matching.py` |
| Görünüm hafızası | [JDE](https://arxiv.org/abs/1909.12605) (EMA güncellemesi) | Güncelleme hızı sabit değil, `r`'ye bağlı — bozuk karedeki görünüm hafızayı daha az değiştiriyor | `trackers/*/track.py` |
| Kamera hareketi (GMC) | [BoT-SORT](https://arxiv.org/abs/2206.14651) (piramidal Lucas-Kanade + RANSAC) | Çekirdek katkının dışında; açık/kapalı ablasyonla ölçülüp karara bağlanacak (bazı sekanslarda kamera tamamen sabit) | `trackers/*/gmc.py` |
| Ana baseline | [Strong Baseline](https://arxiv.org/abs/2503.17237) ([kod](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID)) | BSB eğitim bölümünde yeniden eğitim; ayrı ReID'e karşı ortak gömme karşılaştırması | `baselines/strong_baseline/` |
| İkinci baseline | [Dist-Tracker](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/html/Wang_Dist-Tracker_A_Small_Object-aware_Detector_and_Tracker_for_UAV_Tracking_CVPRW_2025_paper.html) (Track 3 şampiyonu) | Yalnızca MOTA bildirmiş; IDF1/IDSW/FPS'ini ilk kez biz raporlayacağız | `baselines/dist_tracker/` |
| Karşılaştırma hedefi | [Context-Aware Identity Prediction](https://doi.org/10.3390/rs18132084) (kare-düzeyi güvenilirlik kapısı) | Kare-düzeyi kapıyı kendi çatımızda yeniden kurup tespit-düzeyi `r`'ye karşı ablasyon | `eval/ablations/` |
| Metrikler | [HOTA](https://doi.org/10.1007/s11263-020-01375-2), IDF1, CLEAR MOT, [TrackEval](https://github.com/JonathonLuiten/TrackEval) | Ortalamanın yanında bozulma şiddetine göre katmanlı IDSW / AssA raporu | `eval/` |

### Esinlendiğimiz ama almadığımız

- [MOTIP](https://arxiv.org/abs/2403.16848) / DETR tabanlı uçtan uca kimlik
  tahmini (Context-Aware'in omurgası) — kenar hesaplama bütçesine sığmıyor;
  gerçek zamanlı olma kısıtıyla çelişiyor.
- **AKKF** ([Edge-Aware](https://arxiv.org/abs/2607.12544)'in uyarlanır
  Kalman filtresi) — bildirilen kazanç üçüncü ondalıkta
  (HOTA 0.82612 → 0.82661), FPS 132 → 95. Fikir (dinamik ölçüm gürültüsü)
  değerli, uygulaması maliyetli.
- [Çevrimdışı iz yeniden bağlama](https://arxiv.org/abs/2606.01694) —
  çevrimiçi takip kısıtını bozuyor.
- SOT şablon eşleştirme ([SiamSTA](https://openaccess.thecvf.com/content/ICCV2021W/AntiUAV/html/Huang_SiamSTA_Spatio-Temporal_Attention_Based_Siamese_Tracker_for_Tracking_UAVs_ICCVW_2021_paper.html),
  SiamDT) — tek hedefte çalışıyor, 30 hedefe ölçeklenmiyor.

---

## Veri

[Track 3](https://anti-uav.github.io/), 4. Anti-UAV Challenge (CVPR 2025).
640×512 termal, 200 eğitim + 100 test sekansı. Resmi test etiketleri
[yarışma sunucusunda](https://codalab.lisn.upsaclay.fr/competitions/21806).

Etiket formatı (MOTChallenge, 9 kolon, kareler 1'den başlar):

```
frame, id, x_sol, y_ust, w, h, conf, class, visibility
```

`x,y` sol-üst köşe (ampirik olarak doğrulandı). 7–9. kolonlar bu veri
setinde sabit (`1,1,1.0`), taşıyıcı bilgi değil.

Videolar MPEG-4 ile sıkıştırılmış. Ölçtüğümüz bozulmanın bir kısmı sensör
değil codec kaynaklı; tez metninde "görüntü bozulması" (sensör gürültüsü,
termal sürüklenme, bulanıklık ve sıkıştırma artifaktlarının bileşimi)
ifadesi kullanılıyor, "sensör bozulması" değil.

**Bölümleme: BSB (Beyond Strong Baseline).** Strong Baseline kare düzeyinde
bölme yapmış — aynı videonun kareleri hem eğitime hem doğrulamaya düşmüş.
Yazar bunun aşırı öğrenmeye yol açtığını kendisi kabul ediyor. Biz
[Edge-Aware](https://arxiv.org/abs/2607.12544)'in sekans bazlı 102/98
yeniden bölümlemesini kullanıyoruz.

BSB test etiketleri gizli (yalnızca ilk kare kutuları verilmiş):

- Yerel ablasyonlar → 102 eğitim sekansından ayrılan sabit bir doğrulama grubu
- Final sayılar → BSB liderlik tablosu
- Bölünme listeleri `data/splits/` altında, veri dosyaları Git'te değil

**Bilinen etiket hataları** (Strong Baseline'ın bildirdiği):
MultiUAV-230 (yanlış), MultiUAV-256 (fazla), MultiUAV-294 (eksik),
MultiUAV-068 (test, bozuk kare).

## Değerlendirme

Birincil metrik [HOTA](https://doi.org/10.1007/s11263-020-01375-2)
([TrackEval](https://github.com/JonathonLuiten/TrackEval) ile), yanında IDF1 ve IDSW.

MOTA birincil değil: Strong Baseline'ın ablasyonunda ReID modülünün MOTA'ya etkisi ~0.01 çıkmış, oysa katkısı kimlik korumada. Detektör hatalarının baskın olduğu bir metrik, katkısı eşleştirme tarafında olan bir tezi ölçemez.

Her sonuç ayrıca:

- **Tek, ilan edilmiş bölümlemede.** Yayınlanmış Track 3 sayıları resmi
  test, yazarların kendi doğrulama bölümleri ve BSB'yi karıştırıyor;
  birbirine karşı sıralanamaz.
- **Kenar verimliliğiyle birlikte.** Parametre, GFLOPs, model boyutu ve
  uçtan uca FPS (dedektör + gömme + eşleştirme), doğrulukla aynı
  çözünürlükte, belirtilen donanımda.
- **Bozulma şiddetine göre katmanlı.** "Ortalama HOTA +2" yerine "bozuk
  sekanslarda +5, temizlerde +0.3".

**Önceden ilan edilen olasılık:** "Görünüm yalnızca bozulmanın düşük olduğu
yerde yardım ediyor" sonucu da geçerli bir bulgudur. Hipotez çürütülürse
tez çürümez.

## Kurulum

```bash
conda env create -f environment.yml
conda activate tir-jde
```

PyTorch ve Ultralytics dedektör aşamasında eklenecek.

## Repo yapısı

```
baselines/      yeniden çalıştırılan karşılaştırma yöntemleri
configs/        deney yapılandırmaları
data/splits/    BSB sekans listeleri (veri dosyaları Git'te değil)
detector/       dedektör, görünüm başlığı, kayıplar
degradation/    bozulma tahmini ve FiLM modülü
eval/           TrackEval sarmalayıcısı, ablasyonlar, katmanlı rapor
notebooks/      EDA ve analiz
tools/          veri hazırlama, format dönüştürme, görselleştirme
trackers/       takip çatısı ve eşleştirme
```




