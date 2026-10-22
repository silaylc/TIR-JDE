# Sekans bölümleri

Bu klasördeki listeler Git'e eklenir ve bir kez oluşturulduktan sonra değiştirilmez. Her satırda bir sekans adı.

- `bsb_train.txt`: BSB eğitim sekansları (102)
- `bsb_test.txt`: BSB test sekansları (98, etiketler gizli)
- `val.txt`: BSB eğitim sekanslarından ayrılan doğrulama grubu. Bir kez oluşturulur:
  ```bash
  python -m tir_jde.pipeline make-val strong_baseline_bsb --fraction 0.15 --seed 0
  ```
  Her ailenin (`BB2P_02` → `BB2P`) %15'i, en az bir sekans, sabit seed'le rastgele seçilir; her ailede en az
  bir sekans eğitimde kalır. Böylece doğrulamadaki aile karışımı eğitim verisindekiyle aynı oranda olur
  (16 sekans: T 5, C 3, diğer ailelerden 1'er; Takeoff tek sekans olduğu için eğitimde kalır).
  **Sahne kümeleri bölünmez:** neredeyse aynı çekimler (görüntü parmak izi benzerliği ≥ 0.999) ya bütünüyle
  doğrulamaya ya bütünüyle eğitime gider; aksi hâlde doğrulama skoru sızıntıyla şişer.
  Dosya zaten varsa komut `--force` olmadan çalışmaz.
- `bsb_track3_map.csv`: her BSB eğitim sekansının Track 3 (`MultiUAV-xxx`) karşılığı ve sahne kümesi.
  Etiketler karşılaştırılarak eşleştirildi (102/102 tekil); BSB eğitim etiketleri Track 3 ile birebir aynı.

Hiçbir bölümde kullanılmayan sekanslar `configs/data/bsb.yaml` → `exclude`: Strong Baseline'ın bildirdiği
etiket hataları BSB'de düzeltilmemiş.

| BSB | Track 3 | Sorun |
|---|---|---|
| T_13 | MultiUAV-230 | yanlış etiket |
| BB2P_01 | MultiUAV-294 | eksik etiket |

Eğitim kümesi = `bsb_train.txt` − `val.txt` − `exclude` (84 sekans).
- `dev.txt`: Takipçi geliştirme için seçilen birkaç sekans

Listeler değişirse preprocess çıktısı yeni bir split hash'iyle ayrı klasöre yazılır.
