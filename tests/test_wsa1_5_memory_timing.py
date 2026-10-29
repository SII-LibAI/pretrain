"""Small random modules and clocks only; no optimizer update or training run."""
from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA1_5_Policy
from lerobot.policies.wsa1_5.utils.timing import StageTimer
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from test_wsa1_5_policy_contract import small_config, small_policy


def test_bf16_construction_omits_umt5_and_tokenizer(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline training must never construct UMT5 or tokenizer")
    monkeypatch.setitem(sys.modules, "lerobot.policies.wsa1_5.vendor.wan_video_text_encoder",
                        SimpleNamespace(WanTextEncoder=forbidden, HuggingfaceTokenizer=forbidden))
    monkeypatch.setattr("lerobot.policies.wsa1_5.vendor.wan_video_vae.WanVideoVAE38", lambda: nn.Linear(2, 2))
    config = replace(small_config(), dtype="bfloat16")
    assert not config.trainable_weights_fp32 and not config.load_text_encoder
    policy = WSA1_5_Policy(config, initialize_pretrained=False)
    assert {parameter.dtype for parameter in policy.parameters()} == {torch.bfloat16}
    assert policy.text_encoder is None and policy.tokenizer is None


@pytest.mark.parametrize("wsa", [True, False])
def test_adamw_foreach_is_disabled_for_wsa_including_resume(wsa):
    policy = small_policy()
    config = SimpleNamespace(policy=policy.config if wsa else SimpleNamespace(type="other"),
                             use_policy_training_preset=True, optimizer=policy.config.get_optimizer_preset(),
                             scheduler=None, steps=100)
    optimizer, _ = make_optimizer_and_scheduler(config, policy)
    assert optimizer.defaults["foreach"] is (False if wsa else None)
    state = optimizer.state_dict()
    for group in state["param_groups"]:
        group["foreach"] = True
    optimizer.load_state_dict(state)
    assert all(group["foreach"] is (False if wsa else True) for group in optimizer.param_groups)
    assert not optimizer.state  # Never call optimizer.step in local checks.


def test_cpu_stage_times_and_wsa_progress_fields(monkeypatch):
    stamps = iter([1.0, 1.2, 1.5, 1.6])
    monkeypatch.setattr("lerobot.policies.wsa1_5.utils.timing.time", SimpleNamespace(perf_counter=lambda: next(stamps)))
    timer = StageTimer("cpu")
    for stage in ["input_s", "mot_s", "pred_s"]:
        timer.mark(stage)
    assert timer.values() == pytest.approx({"input_s": .2, "mot_s": .3, "pred_s": .1})
    tracker = MetricsTracker(2, 40, 1, {"loss": AverageMeter("loss", ":.3f")}, log_episode_progress=False)
    tracker.step()
    assert "episode:" not in str(tracker) and "epoch:" not in str(tracker)
    assert "episodes" not in tracker.to_dict() and "epochs" not in tracker.to_dict()
    assert tracker.to_dict()["steps"] == 1 and tracker.to_dict()["samples"] == 2
    regular = MetricsTracker(2, 40, 1, {})
    assert "episode:" in str(regular) and "epochs" in regular.to_dict()


def test_cuda_stage_timer_syncs_once_at_end(monkeypatch):
    events, streams = [], []
    class Event:
        def __init__(self, **kwargs):
            self.index = len(events)
            self.syncs = 0
            events.append(self)
        def record(self, stream):
            streams.append(stream)
        def elapsed_time(self, other):
            return (other.index - self.index) * 100
        def synchronize(self):
            self.syncs += 1
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: str(device))
    timer = StageTimer("cuda:1")
    timer.mark("input_s")
    timer.mark("mot_s")
    assert not any(event.syncs for event in events)
    assert timer.values() == {"input_s": .1, "mot_s": .1}
    assert [event.syncs for event in events] == [0, 0, 1]
    assert streams == ["cuda:1"] * 3
