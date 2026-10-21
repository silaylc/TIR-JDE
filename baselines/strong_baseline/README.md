# Strong Baseline

YOLOv12 + BoT-SORT + FastReID (SBS-S50). Kod: [silaylc/Yolov12_BoT-SORT_Re-ID](https://github.com/silaylc/Yolov12_BoT-SORT_Re-ID)
(yazarın deposunun ağırlıksız kopyası), adaptör: `tir_jde/modeling/adapters/strong_baseline.py`.

- Ortam: Python 3.11, torch 2.2.2 (CUDA 11.8), flash-attn 2.7.3 → Ampere GPU (A100) gerekir.
- Dedektör: `BoT-SORT/yolov12`, yazarın `train.py`'sindeki `YOLO(...).train(...)` çağrısı + seed.
- ReID: yazarın `fast_reid/tools/train_net.py`'si, preprocess'in ürettiği `reid/MOT20` kırpıntılarıyla.
- Takip: `tools/predict_track3.py`, 1600 px, `--with-reid`.

```bash
python -m tir_jde.pipeline run strong_baseline_bsb seed=0
```
