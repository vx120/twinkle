# Copyright (c) ModelScope Contributors. All rights reserved.
import logging
from typing import Optional

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
)

from twinkle.patch import Patch

logger = logging.getLogger(__name__)


def fast_pos_embed_interpolate(self, grid_thw):
    grid_thw_list = grid_thw.tolist()
    grid_ts = [row[0] for row in grid_thw_list]
    grid_hs = [row[1] for row in grid_thw_list]
    grid_ws = [row[2] for row in grid_thw_list]
    # Keep interpolation tensors on the same device as the incoming grid.
    # This avoids FSDP2 cpu_offload mismatches when the embedding weight sits on CPU.
    device = grid_thw.device

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for t, h, w in grid_thw_list:
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
    pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws, strict=False)])

    patch_pos_embeds_permute = []
    merge_size = self.config.spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=False):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
    return patch_pos_embeds


def _maybe_warmup_visual_once(
    model: "Qwen3_5CausalLMOutputWithPast",
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if getattr(model, "_twinkle_visual_warmed_up", False):
        return

    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    visual = getattr(model, "visual", None)
    if vision_config is None or visual is None:
        return

    patch_dim = vision_config.in_channels * vision_config.temporal_patch_size * vision_config.patch_size**2
    dummy_pixel_values = torch.zeros((16, patch_dim), dtype=dtype, device=device)
    dummy_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=device)

    with torch.no_grad():
        visual(dummy_pixel_values.type(visual.dtype), grid_thw=dummy_grid_thw)

    model._twinkle_visual_warmed_up = True
    logger.info("Ran one-time Qwen3.5 visual warmup on %s", device)


def _get_input_embeds(
    model: "Qwen3_5CausalLMOutputWithPast",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask.to(inputs_embeds.device), image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw).pooler_output
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        video_mask = (input_ids == model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask.to(inputs_embeds.device), video_embeds)

    # Keep real multimodal requests on the vision path, but reduce pure-text
    # overhead by doing at most one lazy visual warmup per model instance.
    if pixel_values is None and pixel_values_videos is None:
        _maybe_warmup_visual_once(model, device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}


def qwen3_5_base_forward(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    **kwargs,
):
    kwargs.pop("input_ids", None)
    kwargs.pop("pixel_values", None)
    kwargs.pop("pixel_values_videos", None)
    kwargs.pop("image_grid_thw", None)
    kwargs.pop("video_grid_thw", None)

    if inputs_embeds is None:
        if input_ids is None:
            raise ValueError("Qwen3.5 forward requires input_ids when inputs_embeds is not provided.")
        input_kwargs = _get_input_embeds(
            self,
            input_ids,
            attention_mask,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
        )
        kwargs.update(input_kwargs)
    else:
        kwargs["inputs_embeds"] = inputs_embeds
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(inputs_embeds.device)

    return self.language_model(input_ids=None, **kwargs)


def forward_with_normal_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3_5CausalLMOutputWithPast":
    del labels, temperature
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    return Qwen3_5CausalLMOutputWithPast(
        logits=logits,
        past_key_values=getattr(outputs, "past_key_values", None),
        hidden_states=outputs.hidden_states,
        attentions=getattr(outputs, "attentions", None),
    )


def patch_qwen3_5_model(model) -> bool:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type not in {"qwen3_5", "qwen3_5_moe"}:
        return False

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Model,
        Qwen3_5VisionModel,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5MoeModel,
        Qwen3_5MoeVisionModel,
    )

    Qwen3_5Model.forward = qwen3_5_base_forward
    Qwen3_5MoeModel.forward = qwen3_5_base_forward
    Qwen3_5ForConditionalGeneration.forward = forward_with_normal_backend
    Qwen3_5MoeForConditionalGeneration.forward = forward_with_normal_backend
    Qwen3_5VisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate
    Qwen3_5MoeVisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate
    logger.info("Patched Qwen3.5 runtime hooks for model_type=%s", model_type)
    return True


class Qwen3_5Patch(Patch):

    def __call__(self, module, *args, **kwargs):
        del args, kwargs
        patch_qwen3_5_model(module)
