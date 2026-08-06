import os

from tradeforge.data.cleanup import clear_external_files


def test_clear_external_files_deletes_only_matching_pattern(tmp_path):
    (tmp_path / "keep_1.csv").write_text("a")
    (tmp_path / "trial_5.csv").write_text("b")
    (tmp_path / "trial_6.csv").write_text("c")

    clear_external_files(str(tmp_path), "trial_*.csv")

    assert (tmp_path / "keep_1.csv").exists()
    assert not (tmp_path / "trial_5.csv").exists()
    assert not (tmp_path / "trial_6.csv").exists()


def test_clear_external_files_default_pattern_deletes_everything(tmp_path):
    (tmp_path / "a.csv").write_text("a")
    (tmp_path / "b.txt").write_text("b")

    clear_external_files(str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_clear_external_files_no_matches_is_a_noop(tmp_path):
    (tmp_path / "keep.csv").write_text("a")

    clear_external_files(str(tmp_path), "nomatch_*.csv")

    assert (tmp_path / "keep.csv").exists()


def test_clear_external_files_skips_directories(tmp_path):
    (tmp_path / "subdir").mkdir()
    (tmp_path / "file.csv").write_text("a")

    clear_external_files(str(tmp_path), "*")

    assert (tmp_path / "subdir").is_dir()
    assert not (tmp_path / "file.csv").exists()


def test_clear_external_files_permission_error_is_caught_and_reported(tmp_path, monkeypatch, capsys):
    target = tmp_path / "locked.csv"
    target.write_text("a")

    def fake_remove(path):
        raise PermissionError("locked by another process")

    monkeypatch.setattr(os, "remove", fake_remove)

    clear_external_files(str(tmp_path), "*")  # should not raise

    out = capsys.readouterr().out
    assert "locked.csv" in out
    assert "locked by another trial" in out


def test_clear_external_files_file_not_found_is_silently_skipped(tmp_path, monkeypatch):
    target = tmp_path / "vanishing.csv"
    target.write_text("a")

    def fake_remove(path):
        raise FileNotFoundError()

    monkeypatch.setattr(os, "remove", fake_remove)

    clear_external_files(str(tmp_path), "*")  # should not raise
