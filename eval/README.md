# Değerlendirme

Tüm takip metrikleri (HOTA, DetA, AssA, IDF1, IDSW, MOTA) aynı TrackEval sürümüyle hesaplanır.

Çıktı formatı (MOTChallenge): `kare, id, x, y, w, h, skor, -1, -1, -1`
- Kareler 1'den başlar.
- (x, y) kutunun sol üst köşesidir.

Yerel değerlendirme yalnızca etiketleri elimizde olan doğrulama grubunda yapılır. BSB test sonuçları liderlik tablosundan alınır.
