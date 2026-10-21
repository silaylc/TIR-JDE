# Testler

GPU ve baseline ortamı gerektirmez; sentetik bir veri seti (`conftest.py`) kullanılır.

| Dosya | Kapsam |
|---|---|
| `test_experiment.py` | config birleştirme, değişkenler, override, run adı, config hash |
| `test_dataset.py` | split, preprocess çıktıları (YOLO, ReID, takip girdisi, GT) |
| `test_eval.py` | TrackEval sarmalayıcısı (mükemmel / ID değişimi / eksik tahmin) |
| `test_adapters.py` | adaptör komutları, sahte `ultralytics` ile runner, pip paketi koruması |
| `test_pipeline.py` | uçtan uca: preprocess → train → evaluate → visualize → compare, MLflow kaydı |

```bash
pytest
```
