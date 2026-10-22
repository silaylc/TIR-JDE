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

## Yazarın kodundan farklar (kopyadaki commit'ler)

| Commit | Değişiklik | Neden |
|---|---|---|
| `f74b91d` | `assets/existing_methods_overview.png` geri eklendi | Yazarın AMP kontrolü (`ultralytics/utils/checks.py`) bu resmi kullanıyor; kopyada eksikti |
| `4ecdd24` | `fast_reid/fastreid/engine/train_loop.py`: `zero_grad()` → `zero_grad(set_to_none=False)` | FastReID'in `ContiguousParams` optimizasyonu gradyan tamponunun silinmemesini gerektiriyor; PyTorch ≥ 2.0'da varsayılan `set_to_none=True` olduğu için ReID eğitimi ilk adımda duruyordu. PyTorch 1.x davranışının aynısıdır; eğitim değişmez. |

ReID, yazarın config'indeki gibi ImageNet ön-eğitimli backbone'dan başlar (`MODEL.WEIGHTS: ''`); yazarın
`train_net.py`'sinin indirdiği eğitilmiş `model_0016.pth` eğitimde kullanılmaz.

