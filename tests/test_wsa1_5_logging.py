"""Concise startup reports and teacher lifecycle with tiny random test modules."""
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.wsa1_5.initialization import load_report
from lerobot.policies.wsa1_5.teacher import OmegaTeacher
from test_wsa1_5_observation_encoding import make_policy


def test_loading_does_not_log_layer_or_parameter_counts(caplog):
    module = nn.Module()
    module.blocks = nn.ModuleList([nn.Linear(2, 3), nn.Linear(3, 2)])
    with caplog.at_level(logging.INFO):
        load_report(module, module.state_dict(), name="test backbone")
    assert len(caplog.records) == 1
    assert "[WSA load] test backbone missing=[] unexpected=[]" in caplog.text
    assert "[WSA parameters]" not in caplog.text
    assert "[WSA layer parameters]" not in caplog.text
    with pytest.raises(RuntimeError, match="Incomplete test backbone checkpoint"):
        load_report(module, {}, name="test backbone")


@pytest.mark.parametrize("queries,weight", [(True, 1.0), (True, 0.0), (False, 1.0)])
def test_training_startup_reports_only_main_modules_in_billions(monkeypatch, caplog, queries, weight):
    policy, _ = make_policy(monkeypatch, enable_3d_queries=queries, lambda_3d=weight)
    calls = []

    def load_teacher(self, device):
        calls.append(device)
        self.model = nn.Linear(2, 3).eval().requires_grad_(False)

    monkeypatch.setattr(OmegaTeacher, "_load", load_teacher)
    assert policy.teacher is None or policy.teacher.model is None
    with caplog.at_level(logging.INFO):
        policy.prepare_for_training(torch.device("cpu"))
    assert len(calls) == int(queries and weight > 0)
    modules = {"RGB/Observation": policy.flow.rgb, "Motion": policy.flow.motion,
               "Future3D": policy.future_3d, "Wan VAE": policy.vae, "UMT5": policy.text_encoder,
               "VGGT-Omega": policy.teacher.model if policy.teacher else None}
    records = [record for record in caplog.records if record.msg.startswith("[WSA parameters]")]
    assert len(records) == len(modules)
    for record in records:
        name = record.args[0]
        module = modules[name]
        if module is None:
            assert record.getMessage() == f"[WSA parameters] {name} disabled"
        else:
            assert record.args[1] == sum(p.numel() for p in module.parameters()) / 1e9
            assert record.args[2] == sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e9
            assert "B trainable=" in record.getMessage() and record.getMessage().endswith("B")
    assert "teacher" not in policy._modules
    if policy.teacher is not None:
        teacher_ids = {id(p) for p in policy.teacher.model.parameters()}
        optimizer_ids = {id(p) for group in policy.get_optim_params() for p in group["params"]}
        assert not teacher_ids & optimizer_ids
        policy.train()
        assert not policy.teacher.model.training
        assert not any(p.requires_grad for p in policy.teacher.model.parameters())


def test_nonmain_rank_loads_teacher_without_parameter_report(monkeypatch, caplog):
    policy, _ = make_policy(monkeypatch, enable_3d_queries=True)
    calls = []
    monkeypatch.setattr(policy.teacher, "_load", lambda device: calls.append(device))
    with caplog.at_level(logging.INFO):
        policy.prepare_for_training(torch.device("cpu"), log_parameters=False)
    assert calls == [torch.device("cpu")]
    assert "[WSA parameters]" not in caplog.text


@pytest.mark.parametrize("valid_checkpoint", [True, False])
def test_teacher_load_is_visible_frozen_strict_and_preserves_rng(monkeypatch, caplog, valid_checkpoint):
    class TinyOmega(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.projection = nn.Linear(2, 3)
            self.aggregator = SimpleNamespace(patch_token_start=17, cached_layer_indices=(4, 11, 17, 23))

    package = ModuleType("vggt_omega")
    package.__path__ = []
    models = ModuleType("vggt_omega.models")
    models.VGGTOmega = TinyOmega
    monkeypatch.setitem(sys.modules, "vggt_omega", package)
    monkeypatch.setitem(sys.modules, "vggt_omega.models", models)
    state = TinyOmega().state_dict() if valid_checkpoint else {}
    monkeypatch.setattr("lerobot.policies.wsa1_5.teacher.read_checkpoint", lambda path: state)
    teacher = OmegaTeacher("test-only-weights-not-read.pt", dtype=torch.float32)
    rng = torch.random.get_rng_state().clone()
    with caplog.at_level(logging.INFO):
        if valid_checkpoint:
            teacher._load(torch.device("cpu"))
            teacher._load(torch.device("cpu"))
        else:
            with pytest.raises(RuntimeError, match="Incomplete VGGT-Omega checkpoint"):
                teacher._load(torch.device("cpu"))
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert caplog.text.count("loading VGGT-Omega from test-only-weights-not-read.pt") == 1
    assert caplog.text.count("VGGT-Omega ready") == int(valid_checkpoint)
    assert "[WSA layer parameters]" not in caplog.text
    if valid_checkpoint:
        assert not teacher.model.training
        assert not any(p.requires_grad for p in teacher.model.parameters())
    else:
        assert teacher.model is None
