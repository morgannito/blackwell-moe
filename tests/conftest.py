"""Pytest fixtures: CUDA-only tests auto-skip when no GPU available."""

from __future__ import annotations

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: test requires CUDA device")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
