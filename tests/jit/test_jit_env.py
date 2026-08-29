from pathlib import Path

from flashinfer.jit import env


def test_aot_dir_environment_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FLASHINFER_AOT_DIR", str(tmp_path))

    assert env._get_aot_dir() == tmp_path
