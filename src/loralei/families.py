"""What differs between base models, in one place.

Every supported model is a rectified-flow transformer trained the same way - interpolate
between latents and noise, predict the velocity `noise - latents` - but each one wants
its inputs differently: which VAE and how its latents are normalised, which text encoder
and what comes out of it, how latents are packed, what the transformer's `forward` is
called with. Getting any of that wrong is silent: the run trains, the loss falls, and the
LoRA is useless. So each family below mirrors its diffusers pipeline line for line, and
says which lines.

A dataset here is images. LTX 2.3, a video model, is trained on them as one-frame clips,
which the architecture accepts natively.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers.training_utils import compute_loss_weighting_for_sd3

NUM_TRAIN_TIMESTEPS = 1000

Text = dict[str, torch.Tensor]
"""Everything one caption turns into; what the text cache stores per caption."""


def class_named(path: Path, *modules):
    """The class a component folder says it is, looked up in the given modules."""
    name = json.loads((path / "config.json").read_text())["_class_name"]
    for module in modules:
        found = getattr(module, name, None)
        if found is not None:
            return found
    raise RuntimeError(f"{path} is a {name}, which this diffusers install does not provide")


def load_scheduler(path: Path):
    import diffusers

    name = json.loads((path / "scheduler_config.json").read_text())["_class_name"]
    return getattr(diffusers, name).from_pretrained(path)


def load_text_model(path: Path, dtype: torch.dtype, device: str):
    """The text encoder, as the class its own config names."""
    import transformers
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(path)
    model_cls = None
    for name in getattr(config, "architectures", None) or []:
        model_cls = getattr(transformers, name, None)
        if model_cls is not None:
            break
    model = (model_cls or transformers.AutoModel).from_pretrained(path, dtype=dtype)
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def pad(tensors: list[torch.Tensor], length: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length `(n, ...)` tensors, zero-padded, with a boolean mask."""
    length = length or max(t.shape[0] for t in tensors)
    stacked = torch.stack(
        [torch.cat([t, t.new_zeros(length - t.shape[0], *t.shape[1:])]) for t in tensors]
    )
    mask = torch.zeros(len(tensors), length, dtype=torch.bool, device=stacked.device)
    for i, t in enumerate(tensors):
        mask[i, : t.shape[0]] = True
    return stacked, mask


def pack_2x2(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H/2 * W/2, C * 4), the patch layout Qwen-Image and Krea 2 share."""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_2x2(packed: torch.Tensor, height: int, width: int) -> torch.Tensor:
    b, _, channels = packed.shape
    c = channels // 4
    latents = packed.view(b, height // 2, width // 2, c, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(b, c, height, width)


@dataclass
class TextContext:
    """A loaded text encoder and whatever goes with it, for the duration of a caching pass."""

    tokenizer: Any
    model: Any
    device: str
    extra: Any = None


class Family:
    """One architecture. Subclasses fill in the four places models differ."""

    vae_class = "AutoencoderKLQwenImage"
    text_batch_size = 4
    #: Where the text encoder runs when it cannot share the training card. None means "the GPU".
    text_device: str | None = None

    def __init__(self, spec) -> None:
        self.spec = spec

    # ------------------------------------------------------------------ latents

    def load_vae(self, path: Path, dtype: torch.dtype, device: str):
        import diffusers

        vae = getattr(diffusers, self.vae_class).from_pretrained(path, dtype=dtype)
        vae.requires_grad_(False)
        vae.eval()
        return vae.to(device)

    def _latent_stats(self, vae, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel mean and std, shaped to broadcast over (B, C, F, H, W)."""
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        return mean.to(like.device, like.dtype), std.to(like.device, like.dtype)

    def encode(self, vae, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Images in [-1, 1], (B, 3, H, W) -> the posterior's mean and std in the
        transformer's latent space, (B, C, h, w).

        The inverse of what every pipeline does before decoding:
        `latents / (1 / std) + mean`, i.e. `latents * std + mean`.
        """
        posterior = vae.encode(pixels.unsqueeze(2)).latent_dist  # one frame
        mean, std = self._latent_stats(vae, posterior.mean)
        return ((posterior.mean - mean) / std).squeeze(2), (posterior.std / std).squeeze(2)

    def latent_shape(self, vae, height: int, width: int) -> tuple[int, ...]:
        """The latent tensor a decode at this size takes, for the VAE warm-up."""
        factor = 2 ** len(vae.temperal_downsample)
        return (1, vae.config.z_dim, 1, height // factor, width // factor)

    # --------------------------------------------------------------------- text

    def load_text(self, components, dtype: torch.dtype, device: str) -> TextContext:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(components.path("tokenizer"))
        model = load_text_model(components.path("text_encoder"), dtype, device)
        return TextContext(tokenizer, model, device)

    def encode_text(self, ctx: TextContext, prompts: list[str]) -> list[Text]:
        raise NotImplementedError

    # ---------------------------------------------------------------- objective

    def noised(self, latents: torch.Tensor, sigmas: torch.Tensor, generator):
        """Rectified-flow interpolation and its exact derivative."""
        noise = torch.randn(
            latents.shape, device=latents.device, dtype=latents.dtype, generator=generator
        )
        sigma = sigmas.view(-1, *([1] * (latents.ndim - 1))).to(latents.dtype)
        return (1.0 - sigma) * latents + sigma * noise, noise - latents

    def predict(self, transformer, noisy: torch.Tensor, text: list[Text], sigmas: torch.Tensor):
        """The transformer's velocity prediction, back in the shape of `noisy`."""
        raise NotImplementedError

    def loss(
        self,
        transformer,
        latents: torch.Tensor,
        text: list[Text],
        sigmas: torch.Tensor,
        generator: torch.Generator | None,
        weighting_scheme: str = "none",
    ) -> torch.Tensor:
        noisy, target = self.noised(latents, sigmas, generator)
        prediction = self.predict(transformer, noisy, text, sigmas)
        weighting = compute_loss_weighting_for_sd3(weighting_scheme, sigmas).view(
            -1, *([1] * (latents.ndim - 1))
        )
        return (weighting.float() * (prediction.float() - target.float()) ** 2).mean()

    # ----------------------------------------------------------------- sampling

    def pipeline_extras(self, components, dtype: torch.dtype) -> dict:
        return {}

    def build_pipeline(self, components, dtype: torch.dtype, vae_dtype: torch.dtype):
        import diffusers
        from transformers import AutoTokenizer

        spec = self.spec
        transformer_cls = getattr(diffusers, spec.transformer_class)
        return getattr(diffusers, spec.pipeline_class)(
            transformer=transformer_cls.from_pretrained(components.path("transformer"), dtype=dtype),
            vae=getattr(diffusers, self.vae_class).from_pretrained(components.path("vae"), dtype=vae_dtype),
            text_encoder=load_text_model(components.path("text_encoder"), dtype, "cpu"),
            tokenizer=AutoTokenizer.from_pretrained(components.path("tokenizer")),
            scheduler=load_scheduler(components.path("scheduler")),
            **self.pipeline_extras(components, dtype),
        )

    def render(self, pipe, prompt: str, negative: str, height: int, width: int, steps: int,
               cfg: float, generator):
        """One image, as PIL."""
        raise NotImplementedError


class QwenImage(Family):
    """Qwen-Image. Mirrors `QwenImagePipeline`: `_get_qwen_prompt_embeds` for the text,
    `_pack_latents` and the denoising call for the objective."""

    TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    DROP = 34
    """Tokens of the template before the prompt; the pipeline drops them from the embedding."""

    @torch.no_grad()
    def encode_text(self, ctx, prompts):
        inputs = ctx.tokenizer(
            [self.TEMPLATE.format(p) for p in prompts],
            max_length=self.spec.max_sequence_length + self.DROP,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(ctx.device)
        hidden = ctx.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1]
        mask = inputs.attention_mask.bool()
        return [{"embedding": hidden[i][mask[i]][self.DROP :]} for i in range(len(prompts))]

    def predict(self, transformer, noisy, text, sigmas):
        _, _, height, width = noisy.shape
        embeds, mask = pad([t["embedding"] for t in text])
        prediction = transformer(
            hidden_states=pack_2x2(noisy),
            # The pipeline passes `timestep / 1000`: this transformer wants sigma itself.
            timestep=sigmas.to(noisy.dtype),
            guidance=None,
            encoder_hidden_states=embeds,
            # As `encode_prompt` does: a mask with nothing to hide is not passed at all.
            encoder_hidden_states_mask=None if mask.all() else mask.long(),
            img_shapes=[[(1, height // 2, width // 2)]] * noisy.shape[0],
            return_dict=False,
        )[0]
        return unpack_2x2(prediction, height, width)

    def render(self, pipe, prompt, negative, height, width, steps, cfg, generator):
        # Qwen-Image has no guidance embedding; its guidance is true CFG against a negative.
        return pipe(
            prompt=prompt,
            negative_prompt=negative or " ",
            true_cfg_scale=cfg,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]


class Krea2(Family):
    """Krea 2. Mirrors `Krea2Pipeline`: `get_text_hidden_states` for the text - twelve
    hidden layers of Qwen3-VL stacked per token - and `prepare_position_ids` plus the
    denoising call for the objective. It shares Qwen-Image's VAE and patch layout."""

    PREFIX = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
        "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
    START = 34
    SUFFIX_TOKENS = 5
    #: `text_encoder_select_layers`, the pipeline's default.
    LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

    @torch.no_grad()
    def encode_text(self, ctx, prompts):
        tokens = ctx.tokenizer(
            [self.PREFIX + p for p in prompts],
            truncation=True,
            padding="max_length",
            max_length=self.spec.max_sequence_length + self.START - self.SUFFIX_TOKENS,
            return_tensors="pt",
        ).to(ctx.device)
        suffix = ctx.tokenizer([self.SUFFIX] * len(prompts), return_tensors="pt").to(ctx.device)

        input_ids = torch.cat([tokens.input_ids, suffix.input_ids], dim=1)
        mask = torch.cat([tokens.attention_mask, suffix.attention_mask], dim=1).bool()
        position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = ctx.model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden = torch.stack([outputs.hidden_states[i] for i in self.LAYERS], dim=2)
        hidden, mask = hidden[:, self.START :], mask[:, self.START :]
        # Padding carries no position (text ids are all zero) and is masked out of attention,
        # so only the real tokens are kept; a batch re-pads them under a mask.
        return [{"embedding": hidden[i][mask[i]]} for i in range(len(prompts))]

    def predict(self, transformer, noisy, text, sigmas):
        _, _, height, width = noisy.shape
        embeds, mask = pad([t["embedding"] for t in text])

        grid_h, grid_w = height // 2, width // 2
        ids = torch.zeros(grid_h, grid_w, 3, device=noisy.device)
        ids[..., 1] = torch.arange(grid_h, device=noisy.device)[:, None]
        ids[..., 2] = torch.arange(grid_w, device=noisy.device)[None, :]
        position_ids = torch.cat(
            [torch.zeros(embeds.shape[1], 3, device=noisy.device), ids.reshape(grid_h * grid_w, 3)]
        )

        prediction = transformer(
            hidden_states=pack_2x2(noisy),
            encoder_hidden_states=embeds,
            timestep=sigmas.to(noisy.dtype),  # `t / num_train_timesteps` in the pipeline
            position_ids=position_ids,
            encoder_attention_mask=mask,
            return_dict=False,
        )[0]
        return unpack_2x2(prediction, height, width)

    def render(self, pipe, prompt, negative, height, width, steps, cfg, generator):
        return pipe(
            prompt=prompt,
            negative_prompt=negative or None,
            guidance_scale=cfg,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]


class Ltx2(Family):
    """LTX 2.3. Mirrors `LTX2Pipeline`: `_get_gemma_prompt_embeds` followed by the text
    connectors for the text, `_normalize_latents` / `_pack_latents` and the denoising
    call for the objective.

    The model denoises video and audio jointly. A dataset of images has no audio, so the
    audio stream is given pure noise at the final timestep - which is what it holds at
    the start of every generation - and only the video prediction is trained on.
    """

    vae_class = "AutoencoderKLLTX2Video"
    text_batch_size = 1
    #: Gemma 3 12B is 24 GiB in bf16; it does not fit beside anything on a 16 GiB card.
    text_device = "cpu"
    FPS = 24.0

    def encode(self, vae, pixels):
        posterior = vae.encode(pixels.unsqueeze(2)).latent_dist
        mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(posterior.mean)
        std = vae.latents_std.view(1, -1, 1, 1, 1).to(posterior.mean)
        scale = vae.config.scaling_factor
        return ((posterior.mean - mean) * scale / std).squeeze(2), (posterior.std * scale / std).squeeze(2)

    def latent_shape(self, vae, height, width):
        factor = vae.spatial_compression_ratio
        return (1, vae.config.latent_channels, 1, height // factor, width // factor)

    def _connectors(self, components, dtype):
        import diffusers
        from diffusers.pipelines import ltx2

        path = components.path("connectors")
        return class_named(path, diffusers, ltx2).from_pretrained(path, dtype=dtype)

    def load_text(self, components, dtype, device):
        ctx = super().load_text(components, dtype, device)
        ctx.extra = self._connectors(components, dtype).to(device).eval()
        return ctx

    @torch.no_grad()
    def encode_text(self, ctx, prompts):
        tokenizer = ctx.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(
            [p.strip() for p in prompts],
            padding="max_length",
            max_length=self.spec.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        mask = inputs.attention_mask.to(ctx.device)
        states = ctx.model(
            input_ids=inputs.input_ids.to(ctx.device), attention_mask=mask, output_hidden_states=True
        ).hidden_states
        # Every layer of Gemma, side by side per token; the connectors reduce that to the
        # two sequences the transformer cross-attends to. Those are what is worth caching.
        embeds = torch.stack(states, dim=-1).flatten(2, 3)
        video, audio, connector_mask = ctx.extra(embeds, mask, padding_side="left")
        return [
            {"video": video[i], "audio": audio[i], "mask": connector_mask[i]}
            for i in range(len(prompts))
        ]

    def predict(self, transformer, noisy, text, sigmas):
        batch, channels, height, width = noisy.shape
        frames = audio_frames = 1
        # Patch size 1 in space and time: `_pack_latents` reduces to moving channels last.
        packed = noisy.unsqueeze(2).permute(0, 2, 3, 4, 1).reshape(batch, frames * height * width, channels)

        audio = torch.randn(
            batch, audio_frames, transformer.config.audio_in_channels,
            device=noisy.device, dtype=noisy.dtype,
        )
        timestep = sigmas * NUM_TRAIN_TIMESTEPS
        silent = torch.full_like(timestep, NUM_TRAIN_TIMESTEPS)
        mask = torch.stack([t["mask"] for t in text])

        video_prediction, _ = transformer(
            hidden_states=packed,
            audio_hidden_states=audio,
            encoder_hidden_states=torch.stack([t["video"] for t in text]),
            audio_encoder_hidden_states=torch.stack([t["audio"] for t in text]),
            timestep=timestep,
            audio_timestep=silent,
            sigma=timestep,  # the pipeline passes the timestep here too; LTX 2.3 reads it
            audio_sigma=silent,
            encoder_attention_mask=mask,
            audio_encoder_attention_mask=mask,
            num_frames=frames,
            height=height,
            width=width,
            fps=self.FPS,
            audio_num_frames=audio_frames,
            video_coords=transformer.rope.prepare_video_coords(
                batch, frames, height, width, noisy.device, fps=self.FPS
            ),
            audio_coords=transformer.audio_rope.prepare_audio_coords(batch, audio_frames, noisy.device),
            use_cross_timestep=True,  # the pipeline's default, and how LTX 2.3 was trained
            return_dict=False,
        )
        return video_prediction.reshape(batch, frames, height, width, channels).permute(0, 4, 1, 2, 3).squeeze(2)

    def pipeline_extras(self, components, dtype):
        import diffusers
        from diffusers.pipelines import ltx2

        def load(name):
            path = components.path(name)
            return class_named(path, diffusers, ltx2).from_pretrained(path, dtype=dtype)

        return {"audio_vae": load("audio_vae"), "vocoder": load("vocoder"), "connectors": load("connectors")}

    def render(self, pipe, prompt, negative, height, width, steps, cfg, generator):
        return pipe(
            prompt=prompt,
            negative_prompt=negative or None,
            height=height,
            width=width,
            num_frames=1,
            frame_rate=self.FPS,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            output_type="pil",
        ).frames[0][0]


FAMILIES: dict[str, type[Family]] = {
    "qwen-image": QwenImage,
    "krea2": Krea2,
    "ltx2": Ltx2,
}


def family_of(spec) -> Family:
    try:
        return FAMILIES[spec.family](spec)
    except KeyError:
        raise RuntimeError(
            f"{spec.key} is of family {spec.family!r}, which this worker has no training code for"
        ) from None
