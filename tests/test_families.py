"""Each model family against its own diffusers pipeline.

The families in `loralei.families` re-implement, for training, what a pipeline does for
inference: how a prompt becomes embeddings, how latents are normalised, what the
transformer is called with. A mistake there does not crash - it trains a useless LoRA.
So these tests do not check the families against expectations written down here; they
run the real pipeline on the same inputs and require the family to have done the same.

They need tiny random-weight models of each architecture (a few MB each, structurally
identical to the real ones). `python tests/fetch_tiny_models.py` downloads them; without
them the tests skip.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from loralei.families import family_of
from loralei.job import Components, Spec

TINY = Path(os.environ.get("LORALEI_TINY_MODELS", Path(__file__).parent / "tiny-models"))

#: family, folder under TINY, what the registry says about a model of that family.
MODELS = {
    "qwen-image": dict(
        folder="qwen", pipeline_class="QwenImagePipeline", transformer_class="QwenImageTransformer2DModel",
        blocks_attr="transformer_blocks", max_sequence_length=1024,
    ),
    "krea2": dict(
        folder="krea", pipeline_class="Krea2Pipeline", transformer_class="Krea2Transformer2DModel",
        blocks_attr="transformer_blocks", max_sequence_length=512,
    ),
    "ltx2": dict(
        folder="ltx", pipeline_class="LTX2Pipeline", transformer_class="LTX2VideoTransformer3DModel",
        blocks_attr="transformer_blocks", max_sequence_length=1024,
    ),
}

PROMPT = "a red circle on a wooden table"
SIZE = 128


def load(name: str):
    facts = dict(MODELS[name])
    folder = TINY / facts.pop("folder")
    if not folder.is_dir():
        pytest.skip(f"no tiny {name} model in {TINY}; run tests/fetch_tiny_models.py")
    spec = Spec.of({"key": name, "family": name, **facts})
    components = Components(name, {p.name: str(p) for p in folder.iterdir() if p.is_dir()})
    family = family_of(spec)
    torch.manual_seed(0)
    pipe = family.build_pipeline(components, torch.float32, torch.float32)
    return family, components, pipe


class Recorder:
    """Stands in for `transformer.forward` and keeps what it was called with."""

    def __init__(self, transformer) -> None:
        self.transformer = transformer
        self.forward = transformer.forward
        self.calls: list[dict] = []

    def __enter__(self):
        def recording(*args, **kwargs):
            assert not args, "every caller here passes keyword arguments"
            self.calls.append(kwargs)
            return self.forward(**kwargs)

        self.transformer.forward = recording
        return self

    def __exit__(self, *exc) -> None:
        self.transformer.forward = self.forward


def same(left, right, what: str) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), what
        assert left.shape == right.shape, f"{what}: {tuple(left.shape)} != {tuple(right.shape)}"
        assert torch.allclose(left.float(), right.float(), atol=1e-5), what
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), what
        for index, (a, b) in enumerate(zip(left, right)):
            same(a, b, f"{what}[{index}]")
    else:
        assert left == right, f"{what}: {left!r} != {right!r}"


def text_from_pipeline(name: str, family, pipe):
    """The prompt, encoded by the pipeline's own method, in the family's cache layout."""
    if name == "qwen-image":
        embeds, mask = pipe.encode_prompt(prompt=[PROMPT], device="cpu")
        return {"embedding": embeds[0] if mask is None else embeds[0][mask[0].bool()]}
    if name == "krea2":
        embeds, mask = pipe.encode_prompt(prompt=[PROMPT], device="cpu", max_sequence_length=512)
        return {"embedding": embeds[0][mask[0]]}
    embeds, mask, _, _ = pipe.encode_prompt(
        prompt=[PROMPT], do_classifier_free_guidance=False, max_sequence_length=1024, device="cpu"
    )
    video, audio, connector_mask = pipe.connectors(embeds, mask, padding_side="left")
    return {"video": video[0], "audio": audio[0], "mask": connector_mask[0]}


@pytest.mark.parametrize("name", list(MODELS))
def test_a_prompt_is_encoded_exactly_as_the_pipeline_encodes_it(name: str):
    family, components, pipe = load(name)
    expected = text_from_pipeline(name, family, pipe)

    ctx = family.load_text(components, torch.float32, "cpu")
    actual = family.encode_text(ctx, [PROMPT])[0]

    assert actual.keys() == expected.keys()
    for key in expected:
        same(actual[key], expected[key], f"{name} text[{key}]")


@pytest.mark.parametrize("name", list(MODELS))
def test_latents_are_normalised_the_way_the_pipeline_undoes_before_decoding(name: str):
    family, _, pipe = load(name)
    vae = pipe.vae
    pixels = torch.rand(1, 3, SIZE, SIZE) * 2 - 1

    mean, std = family.encode(vae, pixels)
    posterior = vae.encode(pixels.unsqueeze(2)).latent_dist
    assert mean.ndim == 4 and std.shape == mean.shape
    assert tuple(mean.unsqueeze(2).shape) == family.latent_shape(vae, SIZE, SIZE)

    # The exact lines every pipeline runs on its final latents before `vae.decode`.
    latents = mean.unsqueeze(2)
    if name == "ltx2":
        decoded_from = pipe._denormalize_latents(
            latents, vae.latents_mean, vae.latents_std, vae.config.scaling_factor
        )
    else:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
        decoded_from = latents / latents_std + latents_mean
    same(decoded_from, posterior.mean, f"{name} latents")


@pytest.mark.parametrize("name", list(MODELS))
def test_the_transformer_is_called_exactly_as_the_pipeline_calls_it(name: str):
    family, _, pipe = load(name)
    text = text_from_pipeline(name, family, pipe)
    shape = family.latent_shape(pipe.vae, SIZE, SIZE)
    noisy = torch.randn(shape[0], shape[1], shape[3], shape[4])
    # Two steps, of which the first is compared: some schedules are undefined for one.
    common = dict(height=SIZE, width=SIZE, num_inference_steps=2, output_type="latent")

    with Recorder(pipe.transformer) as inference:
        if name == "qwen-image":
            from loralei.families import pack_2x2

            embeds, mask = pipe.encode_prompt(prompt=[PROMPT], device="cpu")
            pipe(prompt_embeds=embeds, prompt_embeds_mask=mask, true_cfg_scale=1.0,
                 latents=pack_2x2(noisy), **common)
        elif name == "krea2":
            from loralei.families import pack_2x2

            # The pipeline pads the prompt to 512 tokens under a mask; the cache keeps only
            # the real ones. Handing it the cached form makes the two calls comparable, and
            # the test below shows the padding changes nothing.
            pipe(prompt_embeds=text["embedding"][None],
                 prompt_embeds_mask=torch.ones(1, text["embedding"].shape[0], dtype=torch.bool),
                 guidance_scale=0.0, latents=pack_2x2(noisy), **common)
        else:
            packed = noisy.unsqueeze(2).permute(0, 2, 3, 4, 1).reshape(1, -1, noisy.shape[1])
            # Every guidance off: one plain conditional call, as a training step makes.
            pipe(prompt=PROMPT, num_frames=1, frame_rate=family.FPS, latents=packed,
                 guidance_scale=1.0, audio_guidance_scale=1.0, stg_scale=0.0, audio_stg_scale=0.0,
                 modality_scale=1.0, audio_modality_scale=1.0, **common)
    assert len(inference.calls) == 2
    theirs = inference.calls[0]

    # The families assume 1000 training timesteps, as every real model here has.
    assert pipe.scheduler.config.num_train_timesteps == 1000

    # The noise level the pipeline was actually at, as a sigma in [0, 1]. Feeding the
    # family that sigma must reproduce the pipeline's `timestep` - in its own scale.
    sigma = pipe.scheduler.sigmas[0].reshape(1).float()
    with Recorder(pipe.transformer) as training:
        prediction = family.predict(pipe.transformer, noisy, [text], sigma)
    ours = training.calls[0]
    assert prediction.shape == noisy.shape

    #: Arguments that only steer inference-time features, or the audio stream that
    #: training deliberately fills differently (see `Ltx2`).
    ignored = {"attention_kwargs", "return_dict", "audio_hidden_states", "audio_timestep", "audio_sigma"}
    for key in sorted((theirs.keys() | ours.keys()) - ignored):
        want, got = theirs.get(key), ours.get(key)
        if want is None and got is None:
            continue
        # A default left unset on one side is the same call as passing that default.
        if (want is None or got is None) and key == "guidance":
            continue
        if want is False and got is None or got is False and want is None:
            continue
        assert want is not None and got is not None, f"{name}: only one side passes `{key}`"
        if key.endswith("mask") and isinstance(want, torch.Tensor):
            want, got = want.bool(), got.bool()
        same(got, want, f"{name} transformer({key})")


@pytest.mark.parametrize("name", list(MODELS))
def test_the_loss_is_finite_and_reaches_the_weights(name: str):
    family, _, pipe = load(name)
    text = text_from_pipeline(name, family, pipe)
    shape = family.latent_shape(pipe.vae, SIZE, SIZE)
    latents = torch.randn(2, shape[1], shape[3], shape[4])

    transformer = pipe.transformer
    transformer.requires_grad_(True)
    loss = family.loss(transformer, latents, [text, text], torch.tensor([0.3, 0.8]), None)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in transformer.parameters())


def test_dropping_padded_prompt_tokens_does_not_change_what_krea_predicts():
    family, _, pipe = load("krea2")
    from loralei.families import pack_2x2, unpack_2x2

    embeds, mask = pipe.encode_prompt(prompt=[PROMPT], device="cpu", max_sequence_length=512)
    assert mask.sum() < mask.shape[1], "the prompt should leave padding to drop"
    shape = family.latent_shape(pipe.vae, SIZE, SIZE)
    noisy = torch.randn(1, shape[1], shape[3], shape[4])
    sigma = torch.tensor([0.6])

    with torch.no_grad():
        trimmed = family.predict(pipe.transformer, noisy, [{"embedding": embeds[0][mask[0]]}], sigma)
        padded = pipe.transformer(
            hidden_states=pack_2x2(noisy),
            encoder_hidden_states=embeds,
            timestep=sigma,
            position_ids=pipe.prepare_position_ids(embeds.shape[1], noisy.shape[2] // 2, noisy.shape[3] // 2, "cpu"),
            encoder_attention_mask=mask,
            return_dict=False,
        )[0]
    same(trimmed, unpack_2x2(padded, noisy.shape[2], noisy.shape[3]), "krea2 prediction")
