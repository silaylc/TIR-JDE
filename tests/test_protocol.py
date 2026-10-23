"""BSB protocol: frame 1 of a tracker's output must reproduce the given first-frame boxes."""

from tir_jde.eval.protocol import first_frame_check

GT = "1,1,10,20,8,6,1,1,1\n1,2,40,30,6,5,1,1,1\n2,1,11,20,8,6,1,1,1\n"


def _write(d, name, text):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.txt").write_text(text)


def test_reproduced_first_frame_passes(tmp_path):
    _write(tmp_path / "gt", "A", GT)
    _write(tmp_path / "pred", "A", "1,7,40,30,6,5,1,-1,-1,-1\n1,3,10,20,8,6,1,-1,-1,-1\n2,3,30,30,8,6,0.4,-1,-1,-1\n")
    r = first_frame_check(tmp_path / "gt", tmp_path / "pred", ["A"])
    assert (r["match"], r["extra"], r["failed_sequences"]) == (1.0, 0, [])


def test_detector_output_on_frame_one_is_flagged(tmp_path):
    _write(tmp_path / "gt", "A", GT)
    _write(tmp_path / "pred", "A", "1,1,12,21,8,6,0.7,-1,-1,-1\n1,2,90,90,5,5,0.3,-1,-1,-1\n")
    r = first_frame_check(tmp_path / "gt", tmp_path / "pred", ["A"])
    assert r["match"] == 0.0 and r["extra"] == 2 and r["failed_sequences"] == ["A"]


def test_empty_first_frame_needs_an_empty_first_frame(tmp_path):
    _write(tmp_path / "gt", "E", "2,1,10,20,8,6,1,1,1\n")  # no boxes in frame 1 (like MultiUAV-068)
    _write(tmp_path / "pred", "E", "2,1,10,20,8,6,0.9,-1,-1,-1\n")
    assert first_frame_check(tmp_path / "gt", tmp_path / "pred", ["E"])["match"] == 1.0
    _write(tmp_path / "pred", "E", "1,1,10,20,8,6,0.9,-1,-1,-1\n")
    r = first_frame_check(tmp_path / "gt", tmp_path / "pred", ["E"])
    assert r["extra"] == 1 and r["failed_sequences"] == ["E"]
