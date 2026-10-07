# Sıfırdan takipçi

BoT-SORT'u adım adım yeniden yazarak bir takipçinin nasıl çalıştığını öğrenmek için.

Aşamalar:

1. SORT: IoU maliyeti, Macar eşleştirmesi, iz yaşam döngüsü
2. ByteTrack: iki turlu eşleştirme
3. BoT-SORT Kalman: genişlik/yükseklikli durum
4. GMC: kamera hareket düzeltmesi
5. Görünüm füzyonu: iki kapı, görünüm hafızası
6. TIR-JDE: bozulma koşullu güvenilirlik ağırlığı (r)

Her aşama sahte tespitlerle (tools/) test edilir ve bir öncekiyle karşılaştırılır.
