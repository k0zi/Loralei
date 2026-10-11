"""The job document is the whole interface to the C# side; this pins how it is read."""

import json
from pathlib import Path

import pytest

from loralei.cache.store import CacheLayout, caption_hash
from loralei.job import Components, Config, Prepared


def make_job(tmp_path: Path) -> dict:
    return {
        "job": "train",
        "config": {
            "name": "run",
            "model": "krea-2-raw",
            "dataset": str(tmp_path / "ds"),
            "output_root": "runs",
            "model_path": None,
            "resolution": 512,
            "rank": 16,
            "alpha": None,
            "sample_prompts": ["a cat"],
        },
        "spec": {
            "key": "krea-2-raw",
            "pipeline_class": "Krea2Pipeline",
            "transformer_class": "Krea2Transformer2DModel",
            "blocks_attr": "transformer_blocks",
            "max_sequence_length": 512,
            "family": "krea2",
        },
        "components": {"transformer": str(tmp_path / "m" / "transformer")},
        "cache": {"latents": str(tmp_path / "c" / "latents"), "text": str(tmp_path / "c" / "text")},
        "fingerprint": "abc",
        "items": [
            {"image": str(tmp_path / "ds" / "a.png"), "caption": "one", "key": "k1", "bucket": [512, 512]},
            {"image": str(tmp_path / "ds" / "b.png"), "caption": "two", "key": "k2", "bucket": [640, 384]},
            {"image": str(tmp_path / "ds" / "c.png"), "caption": "three", "key": "k3", "bucket": [512, 512]},
        ],
    }


def test_a_job_becomes_what_the_training_loop_reads(tmp_path: Path):
    prepared = Prepared.of(make_job(tmp_path))

    assert prepared.cfg.run_dir == Path("runs") / "run"
    assert prepared.spec.family == "krea2"
    # A sampler child is handed the same facts, unchanged.
    assert prepared.spec.as_dict()["family"] == "krea2"
    assert prepared.fingerprint == "abc"
    assert [item.image.name for item in prepared.items] == ["a.png", "b.png", "c.png"]
    # Buckets are assigned upstream and only grouped here.
    assert prepared.plan.bucket_for(prepared.items[1]) == (640, 384)
    assert prepared.plan.summary() == "512x512:2, 640x384:1"


def test_alpha_defaults_to_rank_and_the_dump_is_what_arrived(tmp_path: Path):
    job = make_job(tmp_path)
    cfg = Config(job["config"])
    assert cfg.effective_alpha == 16.0
    assert Config({**job["config"], "alpha": 8}).effective_alpha == 8.0
    # Stamped into the safetensors metadata, so it must survive json.dumps unchanged.
    assert json.loads(json.dumps(cfg.dump(), sort_keys=True)) == job["config"]


def test_an_unresolved_component_says_how_to_get_it(tmp_path: Path):
    components = Components("krea-2-raw", make_job(tmp_path)["components"])
    assert components.path("transformer").name == "transformer"
    with pytest.raises(FileNotFoundError, match="loralei models pull krea-2-raw"):
        components.path("vae")


def test_cache_file_names_match_the_other_side(tmp_path: Path):
    layout = CacheLayout(latents=tmp_path / "l", text=tmp_path / "t")
    assert layout.latent_path("abc", 640, 384).name == "abc_640x384.safetensors"
    # sha256("")[:24] - CacheLayout.CaptionHash in Loralei.Core asserts the same constant.
    assert caption_hash("") == "e3b0c44298fc1c149afbf4c8"
    assert layout.text_path("").name == "e3b0c44298fc1c149afbf4c8.safetensors"
