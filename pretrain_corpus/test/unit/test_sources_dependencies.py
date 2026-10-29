from __future__ import annotations

import pytest

from src.errors import SourceConfigError
from src.sources import check_git_dependencies


def test_passes_when_git_is_installed():
    check_git_dependencies()  # git is a real dependency of this test env


def test_missing_git_binary_raises_a_clear_config_error(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(SourceConfigError, match="git"):
        check_git_dependencies()
