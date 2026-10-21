from pathlib import Path

import pytest

pytest.importorskip("trackeval")

from tir_jde.eval.trackeval_runner import evaluate  # noqa: E402


def _rows(n_frames, ids, swap_at=None, drop_every=None):
    out = []
    for f in range(1, n_frames + 1):
        for k, i in enumerate(ids):
            if drop_every and k == 1 and f % drop_every == 0:
                continue
            tid = ids[(k + 1) % len(ids)] if swap_at and f >= swap_at else i
            out.append(f"{f},{tid},{50 + 100 * k + f},{60 + 40 * k},12,10,1,1,1.0")
    return "\n".join(out) + "\n"


@pytest.fixture
def gt_dir(tmp_path: Path) -> Path:
    d = tmp_path / "gt"
    d.mkdir()
    for seq, n in (("C_01", 30), ("T_05", 20)):
        (d / f"{seq}.txt").write_text(_rows(n, [1, 2, 3]))
        (d / f"{seq}.len").write_text(str(n))
    return d


def _pred(tmp_path, name, **kw):
    d = tmp_path / name
    d.mkdir()
    for seq, n in (("C_01", 30), ("T_05", 20)):
        (d / f"{seq}.txt").write_text(_rows(n, [1, 2, 3], **kw))
    return d


def test_perfect_tracker_scores_one(gt_dir, tmp_path):
    r = evaluate(gt_dir, _pred(tmp_path, "perfect"))["overall"]
    assert r["HOTA"] == pytest.approx(1.0)
    assert r["IDF1"] == pytest.approx(1.0)
    assert r["IDSW"] == 0


def test_identity_swaps_hurt_association_not_detection(gt_dir, tmp_path):
    r = evaluate(gt_dir, _pred(tmp_path, "swapped", swap_at=10))["overall"]
    assert r["DetA"] == pytest.approx(1.0)
    assert r["AssA"] < 0.6
    assert r["IDSW"] > 0


def test_missed_detections_are_counted(gt_dir, tmp_path):
    r = evaluate(gt_dir, _pred(tmp_path, "partial", drop_every=2))["overall"]
    assert r["FN"] == 15 + 10
    assert r["MOTA"] < 1.0


def test_missing_prediction_file_counts_as_no_detections(gt_dir, tmp_path):
    pred = _pred(tmp_path, "one_missing")
    (pred / "T_05.txt").unlink()
    result = evaluate(gt_dir, pred)
    assert result["per_sequence"]["C_01"]["HOTA"] == pytest.approx(1.0)
    assert result["per_sequence"]["T_05"]["FN"] == 60
