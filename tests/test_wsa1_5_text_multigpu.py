"""Simulate torchrun ranks with tiny CPU encoders; no GPU or pretrained weights."""
from collections import Counter
import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.wsa1_5.text_embeddings import TextEmbeddingCache, text_embedding_path
from util_scripts import precompute_wsa1_5_text_embeddings as script
from test_wsa1_5_launch import add_dataset
from test_wsa1_5_text_embeddings import TinyEncoder, tokenize


@pytest.mark.parametrize("dataset_count,world_size", [(1, 4), (3, 4), (1, 8)])
@pytest.mark.parametrize("use_subtask", [False, True])
def test_multigpu_partitions_all_prompts_without_duplicates_or_missing_files(
    tmp_path, monkeypatch, dataset_count, world_size, use_subtask,
):
    root = tmp_path / "dataset collection"
    repos = ["."] if dataset_count == 1 else [".", "a/b", "z"]
    for repo in repos:
        path = root / repo
        add_dataset(path)
        sidecar = [{"episode_index": 0, "task_instruction": repo, "subtasks": [
            {"start_frame": index, "end_frame": index, "subtask_id": index, "subtask_text": f"move {index}"}
            for index in range(5)
        ]}]
        (path / "meta/sidecar.json").write_text(json.dumps(sidecar), encoding="utf-8")
    encoder_path = tmp_path / "fake-encoder.pt"
    encoder_path.touch()
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    all_tasks = [(path, text) for path in sorted((root / repo).resolve() for repo in repos)
                 for text in script.dataset_prompts(path, use_subtask=use_subtask)]
    model_ranks, bindings, assignments = [], [], []

    class Encoder(TinyEncoder):
        def __init__(self):
            super().__init__()
            model_ranks.append(int(os.environ["RANK"]))

        def to(self, *, device, dtype):
            assert device == f"cuda:{os.environ['LOCAL_RANK']}"
            return self  # No CUDA calls: simulate device binding with CPU tensors.

        def forward(self, ids, valid):
            return ids[..., None].expand(-1, -1, 4096).to(torch.bfloat16)

    monkeypatch.setitem(sys.modules, "lerobot.policies.wsa1_5.vendor.wan_video_text_encoder", SimpleNamespace(
        WanTextEncoder=Encoder, HuggingfaceTokenizer=lambda *args, **kwargs: tokenize))
    monkeypatch.setattr("lerobot.policies.wsa1_5.initialization.read_checkpoint", lambda path: {})
    monkeypatch.setattr("lerobot.policies.wsa1_5.initialization.load_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(script.torch.cuda, "set_device", lambda device: bindings.append(device))
    precompute = script.precompute_dataset

    def run_precompute(path, encoder, tokenizer, device, **kwargs):
        rank = int(os.environ["RANK"])
        assignments.extend((rank, path, text) for text in kwargs["prompts"])
        return precompute(path, encoder, tokenizer, "cpu", **kwargs)

    monkeypatch.setattr(script, "precompute_dataset", run_precompute)
    monkeypatch.setattr(sys, "argv", ["precompute", "--data-root", str(root),
                                     "--text-encoder-path", str(encoder_path),
                                     "--tokenizer-path", str(tokenizer_path),
                                     "--batch-size", "2", "--max-text-length", "4",
                                     "--use-subtask" if use_subtask else "--no-use-subtask"])
    def run_rank(rank, workers):
        monkeypatch.setenv("WORLD_SIZE", str(workers))
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        # Reproduce shuffled discovery and duplicate aliases independently on each rank.
        shuffled = repos[rank % len(repos):] + repos[:rank % len(repos)] + [".", "./."]
        monkeypatch.setattr(script.subprocess, "check_output", lambda *args, **kwargs: " ".join(shuffled))
        script.main()

    for rank in range(world_size):
        run_rank(rank, world_size)
        assert [(path, text) for task_rank, path, text in assignments if task_rank == rank] == all_tasks[rank::world_size]
    assigned_tasks = [(path, text) for _, path, text in assignments]
    assert Counter(assigned_tasks) == Counter(all_tasks)
    assert len(set(assigned_tasks)) == len(all_tasks)
    active_ranks = [rank for rank in range(world_size) if all_tasks[rank::world_size]]
    assert model_ranks == bindings == active_ranks
    for path, text in all_tasks:
        context, valid = TextEmbeddingCache(path / "meta/wsa1_5_text_embeddings", 4, 4096).get(text)
        assert context.shape == (4, 4096) and valid.tolist() == [True, True, False, False]
    assert not list(root.rglob("*.tmp"))

    # Resume with a different GPU count and partially completed output files.
    files = {task: text_embedding_path(task[0] / "meta/wsa1_5_text_embeddings", task[1]) for task in all_tasks}
    snapshots = {task: (path.read_bytes(), path.stat().st_mtime_ns) for task, path in files.items()}
    missing = set(all_tasks[::3])
    for task in missing:
        files[task].unlink()
    model_ranks.clear()
    bindings.clear()
    assignments.clear()
    resumed_world_size = world_size // 2
    for rank in range(resumed_world_size):
        run_rank(rank, resumed_world_size)
        expected = [task for index, task in enumerate(all_tasks) if index % resumed_world_size == rank and task in missing]
        assert [(path, text) for task_rank, path, text in assignments if task_rank == rank] == expected
    assert Counter((path, text) for _, path, text in assignments) == Counter(missing)
    for task, path in files.items():
        if task not in missing:
            assert path.read_bytes() == snapshots[task][0]
            assert path.stat().st_mtime_ns == snapshots[task][1]
        context, valid = TextEmbeddingCache(task[0] / "meta/wsa1_5_text_embeddings", 4, 4096).get(task[1])
        assert (context[0] == 1).all() and (context[1] == 2).all() and not context[2:].any()
        assert valid.tolist() == [True, True, False, False]

    # All-complete reruns must not instantiate UMT5 or bind any CUDA device.
    model_ranks.clear()
    bindings.clear()
    assignments.clear()
    for rank in range(resumed_world_size):
        run_rank(rank, resumed_world_size)
    assert not model_ranks and not bindings and not assignments


@pytest.mark.parametrize("environment,device", [
    ({"WORLD_SIZE": "0"}, "cuda"),
    ({"WORLD_SIZE": "2", "RANK": "2", "LOCAL_RANK": "0"}, "cuda"),
    ({"WORLD_SIZE": "2", "RANK": "0"}, "cuda"),
    ({"WORLD_SIZE": "2", "LOCAL_RANK": "0"}, "cuda"),
    ({"WORLD_SIZE": "2", "RANK": "bad", "LOCAL_RANK": "0"}, "cuda"),
    ({"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "-1"}, "cuda"),
    ({"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "bad"}, "cuda"),
    ({"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}, "cuda:0"),
    ({"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}, "cpu"),
])
def test_multigpu_invalid_rank_or_device_fails_before_loading(monkeypatch, environment, device):
    for name in ["WORLD_SIZE", "RANK", "LOCAL_RANK"]:
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(script.subprocess, "check_output", lambda *args, **kwargs: pytest.fail("Invalid launch reached discovery"))
    monkeypatch.setattr(sys, "argv", ["precompute", "--data-root", "unused", "--text-encoder-path", "unused",
                                     "--tokenizer-path", "unused", "--device", device])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
