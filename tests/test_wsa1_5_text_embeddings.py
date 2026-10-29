"""Offline cache tests use a tiny fake encoder; no pretrained assets are loaded."""
import json
import sys
from types import SimpleNamespace

import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from lerobot.policies.wsa1_5.dataset_wsa1_5 import WSA1_5Dataset, condition_text, wsa1_5_collate
from lerobot.policies.wsa1_5.text_embeddings import TextEmbeddingCache, text_embedding_path
from util_scripts.precompute_wsa1_5_text_embeddings import dataset_prompts, precompute_dataset
from util_scripts import precompute_wsa1_5_text_embeddings as precompute_script
from test_wsa1_5_dataset import make_data
from test_wsa1_5_launch import add_dataset, bash
from test_wsa1_5_observation_encoding import make_policy


class TinyEncoder(nn.Module):
    def forward(self, ids, valid):
        return ids[..., None].expand(-1, -1, 8).to(torch.bfloat16)


def tokenize(texts, **kwargs):
    ids = torch.zeros(len(texts), 4, dtype=torch.long)
    ids[:, :2] = torch.tensor([1, 2])
    return ids, ids != 0


def precompute(root):
    if not (root / "meta/episodes").exists():
        add_dataset(root)
    return precompute_dataset(root, TinyEncoder(), tokenize, "cpu", max_length=4, text_dim=8)


def test_offline_cache_covers_actual_prompts_collates_and_never_loads_encoder(tmp_path, monkeypatch):
    dataset, _ = make_data(tmp_path, monkeypatch)
    assert set(dataset_prompts(tmp_path)) == {condition_text({"task_instruction": "knit"}),
                                             *(segment.text for segment in dataset.segments)}
    with pytest.raises(FileNotFoundError, match="offline text"):
        WSA1_5Dataset(dataset.base, dataset.config, text_embedding_shape=(4, 8))
    assert precompute(tmp_path) == 3
    dataset = WSA1_5Dataset(dataset.base, dataset.config, text_embedding_shape=(4, 8))
    batch = wsa1_5_collate([dataset[0], dataset[12]])
    assert batch["text_embedding"].shape == (2, 4, 8)
    assert batch["text_embedding"].dtype == torch.bfloat16
    assert batch["text_valid"].tolist() == [[True, True, False, False]] * 2
    assert not batch["text_embedding"][:, 2:].any()
    policy, _ = make_policy(monkeypatch)
    policy.config.max_text_length = 4
    monkeypatch.setattr(policy, "_encode_text", lambda texts: pytest.fail("Must use offline embeddings"))
    # Use a small pre-encoded observation to avoid a real VAE.
    result = policy.encode_observation(torch.zeros(2, 3, 96, 32), batch["condition_text"],
                                      latent=torch.zeros(2, 4, 1, 6, 2),
                                      text_embedding=batch["text_embedding"], text_valid=batch["text_valid"])
    torch.testing.assert_close(result[1], batch["text_embedding"].float())
    assert policy.text_encoder is None and policy.tokenizer is None


def test_original_task_cache_for_missing_annotations(tmp_path, monkeypatch):
    dataset, _ = make_data(tmp_path, monkeypatch, annotations=False)
    prompt = condition_text({"task_instruction": "knit"})
    assert dataset_prompts(tmp_path) == [prompt]
    assert precompute(tmp_path) == 1
    dataset = WSA1_5Dataset(dataset.base, dataset.config, text_embedding_shape=(4, 8))
    assert dataset[0]["condition_text"] == prompt
    assert dataset[0]["text_valid"].any()


@pytest.mark.parametrize("subtasks", [[], None])
def test_episode_without_subtasks_uses_sidecar_task_in_training_and_precompute(tmp_path, monkeypatch, subtasks):
    dataset, _ = make_data(tmp_path, monkeypatch)
    sidecar = [{"episode_index": 0, "task_instruction": "original task", "subtasks": subtasks}]
    (tmp_path / "meta/sidecar.json").write_text(json.dumps(sidecar), encoding="utf-8")
    prompt = condition_text({"task_instruction": "original task"})
    assert dataset_prompts(tmp_path) == [prompt]
    assert precompute(tmp_path) == 1
    rebuilt = WSA1_5Dataset(dataset.base, dataset.config, text_embedding_shape=(4, 8))
    assert rebuilt[0]["condition_text"] == prompt and not rebuilt[0]["metadata_valid"]
    assert not rebuilt[0]["targets"]["goal_view_valid"].any()


def test_multiple_original_tasks_follow_actual_frame_task_indices(tmp_path, monkeypatch):
    dataset, _ = make_data(tmp_path, monkeypatch, annotations=False)
    dataset.base.meta.episodes[0]["tasks"] = ["knit", "pull"]
    dataset.base.meta.tasks = pd.DataFrame({"task_index": [0, 1]}, index=["knit", "pull"])
    dataset.base.hf_dataset["task_index"] = [0] * 20 + [1] * 20
    pq.write_table(pa.table({"episode_index": [0], "tasks": [["knit", "pull"]],
                             "dataset_from_index": [0], "dataset_to_index": [40]}),
                   tmp_path / "meta/episodes/chunk-000/file-000.parquet")
    prompts = {condition_text({"task_instruction": task}) for task in ("knit", "pull")}
    assert set(dataset_prompts(tmp_path)) == prompts and "" not in prompts
    assert precompute(tmp_path) == 2
    rebuilt = WSA1_5Dataset(dataset.base, dataset.config, text_embedding_shape=(4, 8))
    assert len(rebuilt) == 40
    for index in (0, 19, 20, 39):
        sample = rebuilt[index]
        task = "knit" if index < 20 else "pull"
        assert sample["condition_text"] == condition_text({"task_instruction": task})
        assert sample["text_valid"].any() and not sample["metadata_valid"]
        assert not sample["targets"]["goal_view_valid"].any()


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_precompute_batches_texts_and_saves_each_prompt_in_order(tmp_path, monkeypatch, batch_size):
    dataset, _ = make_data(tmp_path, monkeypatch)
    prompts = dataset_prompts(tmp_path)
    calls = []

    def tokenizer(texts, **kwargs):
        calls.append(list(texts))
        ids = torch.zeros(len(texts), 4, dtype=torch.long)
        for row, text in enumerate(texts):
            ids[row, :prompts.index(text) + 1] = prompts.index(text) + 1
        return ids, ids != 0

    assert precompute_dataset(tmp_path, TinyEncoder(), tokenizer, "cpu", max_length=4, text_dim=8,
                              batch_size=batch_size) == len(prompts)
    assert calls == [prompts[offset:offset + batch_size] for offset in range(0, len(prompts), batch_size)]
    cache = TextEmbeddingCache(tmp_path / "meta/wsa1_5_text_embeddings", 4, 8)
    for index, text in enumerate(prompts):
        context, valid = cache.get(text)
        assert (context[:index + 1] == index + 1).all()
        assert valid.sum() == index + 1 and not context[index + 1:].any()


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_precompute_rejects_invalid_batch_size(tmp_path, batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        precompute_dataset(tmp_path, TinyEncoder(), tokenize, "cpu", batch_size=batch_size)


@pytest.mark.parametrize("corruption", ["nan", "mask", "metadata", "dimension", "dtype"])
def test_bad_text_cache_is_rejected_without_recomputing(tmp_path, corruption):
    precompute(tmp_path)
    cache = TextEmbeddingCache(tmp_path / "meta/wsa1_5_text_embeddings", 4, 8)
    prompt = condition_text({"task_instruction": "knit"})
    path = text_embedding_path(cache.directory, prompt)
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
        tensors = {key: handle.get_tensor(key).clone() for key in handle.keys()}
    if corruption == "nan":
        tensors["context"][0, 0] = float("nan")
    elif corruption == "mask":
        tensors["valid"] = tensors["valid"].float()
    elif corruption == "metadata":
        metadata["text"] = "another prompt"
    elif corruption == "dimension":
        tensors["context"] = tensors["context"][:, :7].contiguous()
    else:
        tensors["context"] = tensors["context"].float()
    save_file(tensors, path, metadata=metadata)
    with pytest.raises(ValueError):
        cache.get(prompt)


def test_cache_read_checks_expected_model_shape_and_has_no_online_fallback(tmp_path, monkeypatch):
    precompute(tmp_path)
    with pytest.raises(ValueError, match="metadata"):
        TextEmbeddingCache(tmp_path / "meta/wsa1_5_text_embeddings", 512, 4096).get(
            condition_text({"task_instruction": "knit"}))
    policy, _ = make_policy(monkeypatch)
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="no online fallback"):
        policy._encode_text(["missing"])


def test_sidecar_schema_errors_are_not_silenced(tmp_path):
    add_dataset(tmp_path)
    sidecar = tmp_path / "meta/sidecar.json"
    for payload in [{}, [{"episode_index": 0, "subtasks": {}}]]:
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            dataset_prompts(tmp_path)


def test_precompute_main_uses_the_shared_find_script(tmp_path, monkeypatch, bash):
    root = tmp_path / "dataset collection"
    add_dataset(root)
    add_dataset(root / "a/b/c")
    encoder_path = tmp_path / "fake-encoder.pt"
    encoder_path.touch()
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    check_output = precompute_script.subprocess.check_output

    def run_discovery(command, **kwargs):
        assert command[0] == "bash"
        assert command[1] == str(precompute_script.PROJECT_ROOT / "util_scripts/discover_wsa1_5_datasets.sh")
        return check_output([bash, *command[1:]], **kwargs)

    monkeypatch.setattr(precompute_script.subprocess, "check_output", run_discovery)
    monkeypatch.setitem(sys.modules, "lerobot.policies.wsa1_5.vendor.wan_video_text_encoder",
                        SimpleNamespace(WanTextEncoder=TinyEncoder, HuggingfaceTokenizer=lambda *args, **kwargs: object()))
    monkeypatch.setattr("lerobot.policies.wsa1_5.initialization.read_checkpoint", lambda path: {})
    monkeypatch.setattr("lerobot.policies.wsa1_5.initialization.load_report", lambda *args, **kwargs: None)
    roots = []
    monkeypatch.setattr(precompute_script, "precompute_dataset", lambda path, *args, **kwargs: roots.append(path.resolve()))
    monkeypatch.setattr(sys, "argv", ["precompute", "--data-root", str(root),
                                     "--text-encoder-path", str(encoder_path),
                                     "--tokenizer-path", str(tokenizer_path), "--device", "cpu"])
    precompute_script.main()
    assert set(roots) == {root.resolve(), (root / "a/b/c").resolve()}
