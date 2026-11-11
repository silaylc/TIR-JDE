# Deney kaydı

Tüm sonuçlar **geliştirme setinde** (BSB train'den ayrılan 16 val sekansı, 11.868 kare), **tek seed** ile alınmıştır. Final sayılar yalnızca dondurulmuş konfigürasyonla, test setinde bir kez üretilecektir.

## Yöntem

- **Ön kayıt.** Her deneyin karar kuralı (birincil metrik, eşik, koruma metriği) sonuçlar görülmeden önce bir JSON dosyasına yazılır ve özetiyle kilitlenir. Kural sonradan değiştirilirse notebook çalışmayı durdurur. Bu sayede "işe yarayan ayarı sonradan seçme" riski önlenir.
- **İstatistik.** Karşılaştırmalar aynı sekanslar üzerinde eşleştirilmiştir: fark, sekans başına farkların ortalamasıdır; %95 güven aralığı sekans bootstrap'ı ile (B = 10.000) hesaplanır; birden fazla karşılaştırma Holm ile düzeltilir. Kod: [`tir_jde/eval/stats.py`](../tir_jde/eval/stats.py).
- **Metrikler.** HOTA (takip kalitesi), MOTA_seq (Track 3 resmî skoru: sekans ortalaması MOTA), DetA, AssA, IDSW. Hepsi aynı TrackEval koduyla ([`tir_jde/eval/trackeval_runner.py`](../tir_jde/eval/trackeval_runner.py)).
- **Kayıt.** Her deney MLflow'a (parametreler, metrikler, görseller) ve Drive'a (ön kayıt, sonuç JSON'u, video) yazılır.

## Özet

| Deney | Soru | Karar |
|---|---|---|
| E0 | Protokol: resmî MOTA tanımı, baseline sayıları, SOT sızıntısı, kayıp fonksiyonu, OSD | MOTA_seq; sızıntı yok; maskeleme gereksiz |
| E1 | NWD maliyeti ve coasting tracker'a katkı veriyor mu? | İkisi de **eklendi** (HOTA 0,511 → 0,615) |
| E1b | Kimlik hataları nereden geliyor? | Yakın çiftler 2,66 kat fazla temsil ediliyor |
| E1c | Coasting kaç kare? | **K = 10** |
| E2 | CMC × NWD × coasting | **Donduruldu:** sparseOptFlow + NWD + K10 (HOTA 0,618) |
| 4b | Kalman Q/R ölçeği | Kazanç yok, varsayılan kaldı |
| D0 | SAHI (girdiyi dilimleme) | Recall +0,13 ama DetA kanıtlanamadı → **elendi** |
| D1 | NWD loss ile fine-tune | Kontrole göre fark yok → **elendi** |
| **D2** | **P2 (stride 4) çıkışı** | **DetA +0,083 → kabul** (HOTA 0,618 → 0,660) |
| D2b | Yinelenen kutular için merkez NMS | IDSW −%12 ama HOTA değişmedi → **eklenmedi** |
| E5a | Uzun kayıplar için Mahalanobis kapısı | Uzun boşluk IDSW'nin %16'sı; kapı NWD ile yalnızca daraltıyor → **kapatıldı** |
| E3 | Küçük termal hedeflerde görünüş kimlik taşıyor mu? | Evet (%78, şans %5), ama tek başına konumdan zayıf (%86) |
| E7a | Görünüşü sabit ağırlıkla eklemek | HOTA artmadı, IDSW arttı → **eklenmedi** |

![İlerleme](figures/progress_light.png)

## Tracker (E1, E1c, E2)

Strong Baseline'ın BoT-SORT'u kaydedilmiş tespitler üzerinde, kaynak koda dokunmadan yamalarla çalıştırıldı ([`tir_jde/trackers/sb`](../tir_jde/trackers/sb)).

| Konfigürasyon | MOTA_seq | HOTA | IDSW |
|---|---|---|---|
| IoU, coasting yok (Strong Baseline) | 0,629 | 0,511 | 2.379 |
| NWD | 0,642 | 0,571 | 495 |
| NWD + coasting K = 5 | 0,693 | 0,615 | 469 |
| **sparseOptFlow + NWD + K = 10 (dondurulmuş)** | **0,688** | **0,618** | **461** |
| CMC kapalı, NWD + K = 10 | 0,631 | 0,519 | 1.874 |

- **NWD:** 4–8 piksellik kutularda IoU, 1–2 piksellik kaymada sıfıra düşer; NWD kademeli bir benzerlik verir.
- **Coasting:** Kaybolan iz, Kalman tahminiyle en fazla K kare daha yazılır. FN'yi azaltır, FP'yi biraz artırır.
- **CMC:** Kamera hareketli sekanslarda tahmin kaymasını ve "hayalet" izleri önler; CMC olmadan coasting zarar verir.

## Detector (D0, D1, D2, D2b)

Ortak detector (detector-v1): Strong Baseline'ın SOT ağırlığından BSB üzerinde ince ayarlanmış YOLOv12-L, 640 px.

**D2, P2 çıkışı.** Ultralytics'in atayıcısı yalnızca merkezi GT kutusunun içine düşen anchor'ları aday alır. Stride 8'de (P3) anchor merkezleri 8 px aralıklı olduğu için **8 pikselden küçük hedeflerin %38'i hiç aday anchor alamıyordu**; stride 4'te bu oran %0,8. Mimari: [`configs/arch/yolov12l-p2.yaml`](../configs/arch/yolov12l-p2.yaml), kurulum: [`tir_jde/detector/p2.py`](../tir_jde/detector/p2.py).

![D2 recall](figures/d2_recall_by_size.png)

| | DetA (D1 karar protokolü) | HOTA (dondurulmuş tracker) | <8 px recall | Gecikme |
|---|---|---|---|---|
| Kontrol (aynı reçeteyle fine-tune) | 0,594 | 0,621 | 0,779 | 7,4 ms |
| **detector-v1-P2** | **0,693** | **0,660** | **0,928** | 8,6 ms |

Birincil karar: DetA farkı +0,083, CI95 (0,035; 0,132).

**D2b, merkez NMS.** P2 aynı hedefe birbirine çok yakın birkaç kutu çiziyor (yinelenen çift oranı ‰151, kontrolde ‰51). Merkezleri 3 px'ten yakın kutuları silen bir NMS bunları tamamen temizledi ve IDSW'yi %12 azalttı, ama HOTA değişmedi (+0,002, CI sıfırı kapsıyor) ve küçük hedef recall'ı 0,009 düştü. Sonuç: yinelenen kutular takip hatasının yalnızca ~%1–2'si; **darboğaz FP ve FN**. Kod kaldı ama kapalı: [`tir_jde/detector/postprocess.py`](../tir_jde/detector/postprocess.py).

## Kimlik (E1b, E5a, E3, E7a)

**Kimlik hataları nereden geliyor?** P2 + dondurulmuş tracker'da IDSW'nin %40'ı yakın çiftlerde (iki hedef ≤ 12,7 px), %44'ü kısa süreli karışmalarda, yalnızca %16'sı 10 kareden uzun boşluklardan sonra.

**E5a, uzun kayıp kapısı.** Planlanan Mahalanobis kapısı NWD eşiğiyle birlikte uygulandığında yalnızca daraltabiliyor; küçük hedeflerde kapı yarıçapı ~30 kareye kadar NWD'ninkinden küçük (10 karede 6,5 px'e karşı 24 px). Etkileyebileceği aralık tüm IDSW'nin ~%6'sı olduğu için koşulmadan kapatıldı.

**E3, görünüş sondajı.** Detector donuk; P2 ve P3 neck feature'ları üstüne küçük bir kimlik başlığı eğitildi (CE, 128 boyut; triplet zarar verdi). Yakın çiftte, 10 kare sonra doğru kimliği seçme oranı:

![E3](figures/e3_identity_probe.png)

Görünüş, 4–8 piksellik termal hedeflerde bile şansın ~16 katı doğru, ama tek başına konumdan zayıf. Konumun yanıldığı sorguların %53'ünde ise görünüş doğru: iki kaynak birbirini tamamlıyor.

**E7a, sabit ağırlıklı birleşim.** Aynı başlık tracker'a eklendi: BoT-SORT'un `min(konum, görünüş)` kuralı ve `(1 − w) · konum + w · görünüş`.

![E7a](figures/e7a_appearance_fusion.png)

Hiçbiri HOTA'yı artırmadı; ağırlık arttıkça IDSW yükseldi. E3'teki konum referansında hareket modeli yoktu; gerçek tracker'da Kalman ve CMC konumu çok daha güçlü yapıyor ve head-only embedding'ler (adayların kosinüs benzerliği 0,7–0,9) bu modele bilgi değil gürültü ekliyor.

## Sonraki adımlar

1. **B2:** detector ve kimlik başlığının ortak eğitimi (JDE). Head-only embedding tracker için yeterince ayırt edici değil.
2. **E7b:** tespit başına öğrenilmiş güven ağırlığı `r`. E3'te görünüş yanıldığında en iyi iki adayın benzerliği birbirine çok yakın çıkıyor; bu, `r`'nin kullanabileceği bir sinyal.
3. **E6:** zor hedefler için FiLM ile bozulmaya uyarlanan detector.
4. Dondurmadan önce: tracker ayarlarının (coasting K) P2 ile yeniden kontrolü, BB2_03 GT hizası, en az 3 seed.
5. **Final test:** dondurulmuş konfigürasyonla bir kez; Strong Baseline ve Dist-Tracker ile aynı protokolde.
