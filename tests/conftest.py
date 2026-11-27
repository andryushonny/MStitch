import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest


@pytest.fixture(autouse=True)
def _timing_file_in_tmp(tmp_path, monkeypatch):
    """Sessions learn their timing into the user's file: keep tests out of it."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
