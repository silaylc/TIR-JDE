from tir_jde.experiment import config_hash, flatten, load_config, run_name


def test_experiment_composes_base_data_and_model():
    cfg = load_config("strong_baseline_bsb")
    assert cfg["name"] == "strong_baseline_bsb"
    assert cfg["data"]["name"] == "bsb"
    assert cfg["model"]["adapter"] == "strong_baseline"
    assert cfg["tracking"]["experiment"]
    assert "defaults" not in cfg


def test_overrides_are_typed():
    cfg = load_config("dist_tracker_bsb", ["seed=3", "model.detector.epochs=2", "tag=smoke", "data.reid.min_size=4"])
    assert cfg["seed"] == 3 and isinstance(cfg["seed"], int)
    assert cfg["model"]["detector"]["epochs"] == 2
    assert run_name(cfg) == "dist_tracker_bsb-s3-smoke"


def test_environment_variables_fill_placeholders(monkeypatch):
    monkeypatch.setenv("BSB_ROOT", "/data/bsb")
    monkeypatch.setenv("SB_PYTHON", "/envs/sb/bin/python")
    cfg = load_config("strong_baseline_bsb")
    assert cfg["data"]["raw_root"] == "/data/bsb"
    assert cfg["model"]["python"] == "/envs/sb/bin/python"


def test_config_hash_ignores_machine_paths_but_not_settings():
    a = load_config("strong_baseline_bsb", ["paths.work_dir=/a", "model.python=/x/python"])
    b = load_config("strong_baseline_bsb", ["paths.work_dir=/b", "model.python=/y/python"])
    c = load_config("strong_baseline_bsb", ["paths.work_dir=/a", "model.detector.epochs=7"])
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)


def test_flatten_produces_dotted_keys():
    flat = flatten({"a": {"b": 1, "c": [1, 2]}, "d": "x"})
    assert flat == {"a.b": 1, "a.c": "[1, 2]", "d": "x"}
