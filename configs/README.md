# Ayar dosyaları

Her deney tek bir YAML dosyasıdır ve üç parçadan birleşir:

```
configs/
├── base.yaml            ortak ayarlar: seed, klasörler, MLflow/DagsHub
├── data/bsb.yaml        veri seti: ham veri yolu, split listeleri, ReID kırpıntı ayarları
├── model/*.yaml         model tipi, baseline deposu + commit, eğitim/takip parametreleri
└── experiment/*.yaml    defaults: [base, data/..., model/...] + deneye özel değişiklikler
```

Komut satırından herhangi bir değer ezilebilir:

```bash
python -m tir_jde.pipeline train configs/experiment/strong_baseline_bsb.yaml seed=1 model.detector.epochs=2
```

Değerlerde `${DEGISKEN:-varsayilan}` yazımı ortam değişkenlerinden okunur (ör. `BSB_ROOT`, `TIRJDE_WORK`).
Çalıştırmada kullanılan birleşik ayar, MLflow'a hem parametre hem `config.yaml` dosyası olarak kaydedilir.
