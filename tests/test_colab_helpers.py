import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import colab_helpers as ch
import run_seeds


def _write(path, text="x", mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mtime:
        os.utime(path, (mtime, mtime))


def test_sync_backs_up_reports_but_skips_heavy_weights(tmp_path):
    local, drive = tmp_path / "reports", tmp_path / "drive" / "reports"
    _write(local / "a.json")
    _write(local / "sub" / "b.csv")
    _write(local / "model.safetensors")
    assert ch.sync_dirs([(local, drive)], "to_drive") == 2
    assert (drive / "sub" / "b.csv").exists() and not (drive / "model.safetensors").exists()
    assert ch.sync_dirs([(local, drive)], "to_drive") == 0                  # lần hai: không có gì mới


def test_sync_only_copies_newer_files_and_restores_from_drive(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    _write(drive / "r.json", "new", mtime=2_000_000_000)
    _write(local / "r.json", "old", mtime=1_000_000_000)
    assert ch.sync_dirs([(local, drive)], "from_drive") == 1
    assert (local / "r.json").read_text(encoding="utf-8") == "new"
    assert ch.sync_dirs([(tmp_path / "nope", drive)], "to_drive") == 0       # nguồn chưa có: không lỗi
    with pytest.raises(ValueError):
        ch.sync_dirs([(local, drive)], "sideways")


def test_extract_tunnel_url():
    log = "INF |  https://quiet-river-1234.trycloudflare.com  |"
    assert ch.extract_tunnel_url(log) == "https://quiet-river-1234.trycloudflare.com"
    assert ch.extract_tunnel_url("Requesting new quick Tunnel") is None and ch.extract_tunnel_url(None) is None


def test_secret_reads_env_and_never_requires_colab(monkeypatch):
    monkeypatch.setenv("MY_TEST_SECRET", "abc")
    assert ch.secret("MY_TEST_SECRET") == "abc" and ch.secret("NOT_SET_ANYWHERE", "d") == "d"


def test_run_seeds_skip_done(tmp_path, monkeypatch):
    monkeypatch.setattr(run_seeds.config, "REPORT_DIR", tmp_path)
    assert not run_seeds.is_done("focal", 42)
    (tmp_path / "focal_s42_results.json").write_text("{}", encoding="utf-8")
    assert run_seeds.is_done("focal", 42) and not run_seeds.is_done("focal", 43)
