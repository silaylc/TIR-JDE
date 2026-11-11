# Deney notebook'ları

Colab'da (RTX PRO 6000) çalıştırılan deney notebook'ları, çıktıları temizlenmiş halde. Sonuçlar ve kararlar: [`docs/experiments.md`](../../docs/experiments.md).

| Notebook | İçerik |
|---|---|
| `D2_p2_head_and_followups.ipynb` | D2 (P2 çıkışı: kurulum, ağırlık aktarımı ve doğrulama, eğitim, değerlendirme), D2b (merkez NMS), E5a.0 (IDSW teşhisi), E3 (kimlik başlığı sondajı), E7a (görünüş birleştirme) |

Her deney aynı düzende ilerler: markdown tanım hücresi, karar kuralını kilitleyen ön kayıt hücresi, uygulama, değerlendirme (karar, görsel, MLflow) ve rapor.

Notebook'lardaki tekrar kullanılan kod paket içine taşınmıştır: `tir_jde/detector/p2.py`, `tir_jde/trackers/sb/`, `tir_jde/eval/`, `tir_jde/reid/`.
