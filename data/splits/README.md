# Sekans bölümleri

Bu klasördeki listeler Git'e eklenir ve bir kez oluşturulduktan sonra değiştirilmez. Her satırda bir sekans adı.

- `bsb_train.txt`: BSB eğitim sekansları (102)
- `bsb_test.txt`: BSB test sekansları (98, etiketler gizli)
- `val.txt`: BSB eğitim sekanslarından ayrılan doğrulama grubu. Bir kez oluşturulur:
  ```bash
  python -m tir_jde.pipeline make-val strong_baseline_bsb --per-family 1 --seed 0
  ```
  Her aileden (`BB2P_02` → `BB2P`) bir sekans seçilir; her ailede en az bir sekans eğitimde kalır.
  Dosya zaten varsa komut `--force` olmadan çalışmaz.
- `dev.txt`: Takipçi geliştirme için seçilen birkaç sekans

Eğitim kümesi = `bsb_train.txt` − `val.txt`. Listeler değişirse preprocess çıktısı yeni bir split hash'iyle
ayrı klasöre yazılır.
