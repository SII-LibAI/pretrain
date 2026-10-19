"""Object gathering without GPUs, model construction or training."""
from types import SimpleNamespace

import pytest
import torch

from lerobot.utils.utils import gather_object


@pytest.mark.parametrize("obj", [31023, 105, {"robot": {"mean": [0.1, 0.2]}}])
def test_single_process_gather_does_not_require_a_process_group(monkeypatch, obj):
    def forbidden(*args):
        pytest.fail("Single-process gathering must not call a distributed collective")

    monkeypatch.setattr(torch.distributed, "all_gather_object", forbidden)
    gathered = gather_object(obj, SimpleNamespace(num_processes=1))
    assert len(gathered) == 1 and gathered[0] is obj
    if isinstance(obj, int):
        assert sum(gathered) == obj


def test_multi_process_gather_preserves_all_rank_values(monkeypatch):
    values = [31023, 20000, 12000]
    calls = []

    def collective(output, obj):
        assert output == [None, None, None]
        calls.append(obj)
        output[:] = values

    monkeypatch.setattr(torch.distributed, "all_gather_object", collective)
    gathered = gather_object(values[1], SimpleNamespace(num_processes=3))
    assert gathered == values and sum(gathered) == 63023
    assert calls == [values[1]]


def test_multi_process_collective_errors_are_not_hidden(monkeypatch):
    def collective(*args):
        raise ValueError("Default process group has not been initialized")

    monkeypatch.setattr(torch.distributed, "all_gather_object", collective)
    with pytest.raises(ValueError, match="Default process group"):
        gather_object(31023, SimpleNamespace(num_processes=2))
