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
  (102 sekanstan 16'sı: T 5, C 3, diğer ailelerden 1'er; Takeoff tek sekans olduğu için eğitimde kalır).
  Dosya zaten varsa komut `--force` olmadan çalışmaz.
- `dev.txt`: Takipçi geliştirme için seçilen birkaç sekans

Eğitim kümesi = `bsb_train.txt` − `val.txt`. Listeler değişirse preprocess çıktısı yeni bir split hash'iyle
ayrı klasöre yazılır.
