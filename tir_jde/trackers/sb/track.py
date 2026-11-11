"""Strong Baseline's BoT-SORT (``tracker.mc_bot_sort``, as in its ``predict_track3``) on dumped detections.

- A new tracker per sequence; frame 1 uses the GT boxes ``[x1, y1, x2, y2, 1, 0]`` (the Track 3 protocol).
- Later frames use detections with ``score > conf_thres`` (Strong Baseline: 0.09); frames without
  detections are skipped, as in Strong Baseline.
- ``--emb_dir`` turns on ``with_reid``: instead of FastReID, our own per-detection vectors are used
  (``<emb_dir>/<seq>.npz`` with ``det_emb`` aligned to the rows of the detection file, ``gt1_emb`` to the
  frame-1 GT rows).
- Output: ``<out>/<seq>.txt`` (``frame,id,x,y,w,h,score,-1,-1,-1``) and ``<out>/run.json`` (settings, FPS).

Usually launched through ``runner.py`` (which adds the cost/coasting/gate/fusion patches); see README.md.
"""
import argparse, glob, json, os, sys, time, types
import numpy as np


class Bank:
    boxes = np.zeros((0, 4)); embs = np.zeros((0, 128), np.float32)


def install_stubs():
    """fast_reid'i (fastreid bagimliliklari) ve gerekirse cython_bbox'i yuklemeden tracker'i ice aktarmak icin."""
    class FastReIDInterface:
        def __init__(self, *a, **k):
            pass

        def inference(self, img, dets):
            dets = np.asarray(dets, dtype=np.float64).reshape(-1, 4)
            if len(dets) == 0:
                return np.zeros((0, Bank.embs.shape[1]), np.float32)
            d = np.abs(dets[:, None, :] - Bank.boxes[None, :, :4]).max(2)
            j = d.argmin(1)
            if not (d[np.arange(len(j)), j] < 1e-4).all():
                raise RuntimeError("tespit kutusu vektor bankasinda yok")
            return Bank.embs[j]

    pkg = types.ModuleType("fast_reid"); mod = types.ModuleType("fast_reid.fast_reid_interfece")
    mod.FastReIDInterface = FastReIDInterface; pkg.fast_reid_interfece = mod
    sys.modules["fast_reid"] = pkg; sys.modules["fast_reid.fast_reid_interfece"] = mod
    try:
        import cython_bbox  # noqa: F401
    except ImportError:
        def bbox_overlaps(a, b):
            a = np.asarray(a, float); b = np.asarray(b, float)
            iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1, 0, None)
            ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1, 0, None)
            inter = iw * ih
            aa = ((a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1))[:, None]
            bb = ((b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1))[None, :]
            return inter / (aa + bb - inter)
        m = types.ModuleType("cython_bbox"); m.bbox_overlaps = bbox_overlaps; sys.modules["cython_bbox"] = m


def load(p, cols):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return np.zeros((0, cols))
    return np.loadtxt(p, delimiter=",", ndmin=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot_root", required=True, help=".../BoT-SORT (SB kodu)")
    ap.add_argument("--data_dir", required=True, help="detector-v1_val: dets/, gt/, val.txt")
    ap.add_argument("--images", default="/content/data/BSB/TrainImages")
    ap.add_argument("--emb_dir", default="", help="bos: 5a (ReID kapali); dolu: 7a vektorleri")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmc", default="sparseOptFlow", help="sparseOptFlow | none | orb | ecc")
    ap.add_argument("--conf_thres", type=float, default=0.09)
    ap.add_argument("--track_high_thresh", type=float, default=0.3)
    ap.add_argument("--track_low_thresh", type=float, default=0.05)
    ap.add_argument("--new_track_thresh", type=float, default=0.4)
    ap.add_argument("--track_buffer", type=int, default=60, help="SB README'deki Track 3 komutu: 60")
    ap.add_argument("--match_thresh", type=float, default=0.7)
    ap.add_argument("--proximity_thresh", type=float, default=0.5)
    ap.add_argument("--appearance_thresh", type=float, default=0.25)
    ap.add_argument("--min_box_area", type=float, default=10)
    ap.add_argument("--mot20", action="store_true", help="SB --fuse-score (varsayilan kapali)")
    ap.add_argument("--seqs", default="", help="virgulle; bos = val.txt")
    ap.add_argument("--cost_mode", default="iou", help="iou | nwd")
    ap.add_argument("--nwd_C", type=float, default=12.0, help="NWD sabiti (px)")
    ap.add_argument("--coast_k", type=int, default=0, help="kayip iz tahmini kare sayisi")
    ap.add_argument("--confirm_only", action="store_true")
    ap.add_argument("--kf_q", type=float, default=1.0)
    ap.add_argument("--kf_r", type=float, default=1.0)
    ap.add_argument("--gmc_downscale", type=int, default=2, help="GMC downscale: 1 veya 2")
    a = ap.parse_args()

    install_stubs()
    sys.path.insert(0, a.bot_root)
    import cv2
    from tracker.mc_bot_sort import BoTSORT
    from tracker import matching as _m; _m.COST["mode"] = a.cost_mode; _m.COST["C"] = a.nwd_C

    targs = types.SimpleNamespace(
        track_high_thresh=a.track_high_thresh, track_low_thresh=a.track_low_thresh, new_track_thresh=a.new_track_thresh,
        track_buffer=a.track_buffer, match_thresh=a.match_thresh, proximity_thresh=a.proximity_thresh,
        appearance_thresh=a.appearance_thresh, mot20=a.mot20, cmc_method=a.cmc, with_reid=bool(a.emb_dir),
        fast_reid_config="", fast_reid_weights="", device="cpu", name="", ablation=False)

    seqs = a.seqs.split(",") if a.seqs else [l.strip() for l in open(os.path.join(a.data_dir, "val.txt"))
                                              if l.strip() and not l.startswith("#")]
    os.makedirs(a.out, exist_ok=True)
    t_upd, n_frames, per_seq = 0.0, 0, {}
    for s in seqs:
        imgs = sorted(glob.glob(os.path.join(a.images, s, "*.jpg")))
        dets = load(os.path.join(a.data_dir, "dets", s + ".txt"), 6)
        gt = load(os.path.join(a.data_dir, "gt", s + ".txt"), 6)
        g1 = gt[gt[:, 0] == 1]
        prior = np.stack([g1[:, 2], g1[:, 3], g1[:, 2] + g1[:, 4], g1[:, 3] + g1[:, 5],
                          np.ones(len(g1)), np.zeros(len(g1))], 1) if len(g1) else np.zeros((0, 6))
        if a.emb_dir:
            z = np.load(os.path.join(a.emb_dir, s + ".npz"))
            det_emb, gt1_emb = z["det_emb"].astype(np.float32), z["gt1_emb"].astype(np.float32)
            assert len(det_emb) == len(dets) and len(gt1_emb) == len(prior), f"{s}: vektor sayisi uymuyor"
        by_frame = {}
        for i, f in enumerate(dets[:, 0].astype(int)):
            by_frame.setdefault(f, []).append(i)

        tracker = BoTSORT(targs, frame_rate=30.0); from tracker.gmc import GMC; tracker.gmc = GMC(method=a.cmc, downscale=a.gmc_downscale)
        tracker.coast_k = a.coast_k
        tracker.confirm_only = a.confirm_only
        from tracker import kalman_filter as _kf; _kf.KF_SCALE["Q"] = a.kf_q; _kf.KF_SCALE["R"] = a.kf_r
        lines, ts = [], time.time()
        for f in range(1, len(imgs) + 1):
            if f == 1:
                out = prior.copy(); emb = gt1_emb if a.emb_dir else None
            else:
                rows = np.array(by_frame.get(f, []), int)
                rows = rows[dets[rows, 5] > a.conf_thres] if len(rows) else rows
                out = np.concatenate([dets[rows, 1:6], np.zeros((len(rows), 1))], 1) if len(rows) else np.zeros((0, 6))
                emb = det_emb[rows] if a.emb_dir else None
            if len(out) == 0:
                continue  # SB: pred bossa tracker.update cagrilmaz
            img = cv2.imread(imgs[f - 1]) if a.cmc != "none" else np.zeros((512, 640, 3), np.uint8)
            if a.emb_dir:
                Bank.boxes, Bank.embs = out[:, :4], emb
            t0 = time.time()
            tracks, _ = tracker.update(out, img)
            t_upd += time.time() - t0; n_frames += 1
            for t in tracks:
                x, y, w, h = t.tlwh
                if w * h > a.min_box_area:
                    lines.append(f"{f},{t.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{t.score:.2f},-1,-1,-1\n")
        open(os.path.join(a.out, s + ".txt"), "w").writelines(lines)
        per_seq[s] = {"kare": len(imgs), "satir": len(lines), "sn": round(time.time() - ts, 1)}
        print(f"{s}: {len(imgs)} kare, {len(lines)} satir, {time.time() - ts:.1f} sn", flush=True)

    rec = {"ayarlar": vars(a), "tracker_fps": n_frames / max(t_upd, 1e-9), "sekanslar": per_seq}
    json.dump(rec, open(os.path.join(a.out, "run.json"), "w"), indent=1)
    print(f"[bitti] tracker FPS (yalniz update, GMC dahil): {rec['tracker_fps']:.1f}", flush=True)


if __name__ == "__main__":
    main()
