import torch
import math
import numpy as np
import torch.nn.functional as F

from einops import rearrange
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from omnigen2.utils.teacache_util import TeaCacheParams
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import retrieve_timesteps
from dataclasses import dataclass, field

MAG_RATIOS={
    "t2i_cond": np.array([1.0] + [1.03906, 1.00781, 1.02344, 1.01562, 1.01562, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.0, 1.00781, 1.00781, 1.0, 1.00781, 1.0, 1.00781, 1.00781, 1.0, 1.0, 1.00781, 1.0, 1.0, 1.00781, 1.0, 1.0, 1.00781, 1.0, 1.00781, 1.0, 1.00781, 1.0, 1.0, 1.0, 1.0, 0.99609, 1.0, 0.99609, 0.99609, 0.99219, 0.98828, 0.98047, 0.96875, 0.94531, 0.92969]),
    "t2i_uncond": np.array([1.0] + [1.03906, 1.00781, 1.01562, 1.01562, 1.01562, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.0, 1.0, 1.00781, 1.0, 1.00781, 1.0, 1.00781, 1.00781, 1.0, 1.0, 1.0, 1.0, 1.0, 1.00781, 1.0, 1.0, 1.00781, 1.0, 1.00781, 1.0, 1.00781, 1.0, 1.0, 1.0, 1.0, 0.99609, 1.0, 0.99609, 0.99609, 0.99219, 0.98828, 0.98047, 0.96875, 0.94922, 0.93359]),
    "edit_cond": np.array([1.0] + [1.07031, 1.03906, 1.02344, 1.01562, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99609, 1.0, 0.99609, 0.99609, 0.99609, 0.99609, 0.99219, 0.99219, 0.98828, 0.98828, 0.98047, 0.96875, 0.95312, 0.94531]),
    "edit_uncond": np.array([1.0] + [1.03906, 1.03125, 1.02344, 1.01562, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.00781, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99609, 1.0, 1.0, 0.99609, 1.0, 0.99609, 0.99609, 0.99609, 0.99609, 0.99219, 0.99219, 0.98828, 0.98438, 0.97656, 0.96484, 0.94531, 0.93359]),
    "edit_ref": np.array([1.0] + [1.02344, 1.04688, 1.03906, 1.02344, 1.00781, 1.00781, 1.00781, 1.00781, 1.0, 1.0, 1.0, 1.0, 0.99609, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99609, 1.0, 0.99609, 0.99609, 0.99609, 0.99609, 0.99219, 0.99219, 0.98828, 0.98438, 0.98047, 0.96875, 0.95312, 0.94531]),
}

@dataclass
class MagCacheParams:
    """
    MagCache parameters for `OmniGen2Transformer2DModel`
    See https://github.com/zehong-ma/MagCache/ for a more comprehensive understanding

    Args:
        mag_ratios:
            The mag_ratios that calibrated by magcache_calibration function.
        previous_residual (Optional[torch.Tensor]):
            The tensor difference between the output and the input of the transformer layers from the previous timestep.
        accumulated_ratio:
            The accumulated mag_ratio between current step and the cached step.
        accumulated_err:
            The accumulated relative L1 distance.
        accumulated_steps (bool):
            The accumulated skipped steps.
    """
    mag_ratios: Optional[torch.Tensor] = None
    previous_residual: Optional[torch.Tensor] = None
    accumulated_ratio: float = 1.0
    accumulated_err: float = 0.0
    accumulated_steps: int = 3
    cnt: int = 0
    mode_type: Optional[str] = field(default=None, repr=False)
    num_inference_step: int = 50
    
    # calibratio params
    norm_ratio, norm_std, cos_dis = [], [], []

    def __post_init__(self):
        if self.mode_type is not None:
            self._initialize_from_mode(self.mode_type)

    def _initialize_from_mode(self, mode_type: str):
        self.mag_ratios = MAG_RATIOS[mode_type]
        ### Nearest interpolation when the num_inferece_steps is different from the length of mag_ratios
        if len(self.mag_ratios) != self.num_inference_step:
            interpolated_mag_ratios = nearest_interp(self.mag_ratios, self.num_inference_step)
            self.mag_ratios = interpolated_mag_ratios
        

def set_magcache_params(pipeline, args):
    """ MagCache Setting
    'magcache_calibration' is used to calibrate the above MAG_RATIOS list.
    'magcache_forward' is the forward function of MagCache.
    Please modify the magcache_thresh to achieve a good trade-off between quality and speed.
    We recommend to set magcache_thresh=0.05 to achieve a 2x speedup.
    """
    pipeline.__class__.processing = magcache_processing
    pipeline.transformer.__class__.cnt = 0
    pipeline.transformer.__class__.enable_magcache = True
    pipeline.transformer.__class__.num_steps = args.num_inference_step # num_inference_steps
    ## MagCache Calibration
    pipeline.transformer.__class__.forward = magcache_forward # magcache_calibration
    
    ## MagCache forward
    pipeline.transformer.__class__.magcache_params = None
    # hyprameters: please modify the hyper-parameters to get a trade-off between latency and quality in your own task.
    pipeline.transformer.__class__.magcache_thresh = args.magcache_thresh
    pipeline.transformer.__class__.K = 3
    pipeline.transformer.__class__.retention_ratio = 0.2
        
def nearest_interp(src_array, target_length):
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])

    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]

def magcache_calibration(
    self,
    hidden_states: Union[torch.Tensor, List[torch.Tensor]],
    timestep: torch.Tensor,
    text_hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
    text_attention_mask: torch.Tensor,
    ref_image_hidden_states: Optional[List[List[torch.Tensor]]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = False,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    Forward this function twice in t2i mode and three times in edit mode:
    1. Conditional 2. Unconditional
    1. Conditional 2. Reference 3. Unconditional
    """
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    # 1. Condition, positional & patch embedding
    batch_size = len(hidden_states)
    is_hidden_states_tensor = isinstance(hidden_states, torch.Tensor)

    if is_hidden_states_tensor:
        assert hidden_states.ndim == 4
        hidden_states = [_hidden_states for _hidden_states in hidden_states]

    device = hidden_states[0].device

    temb, text_hidden_states = self.time_caption_embed(timestep, text_hidden_states, hidden_states[0].dtype)

    (
        hidden_states,
        ref_image_hidden_states,
        img_mask,
        ref_img_mask,
        l_effective_ref_img_len,
        l_effective_img_len,
        ref_img_sizes,
        img_sizes,
    ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)
    
    (
        context_rotary_emb,
        ref_img_rotary_emb,
        noise_rotary_emb,
        rotary_emb,
        encoder_seq_lengths,
        seq_lengths,
    ) = self.rope_embedder(
        freqs_cis,
        text_attention_mask,
        l_effective_ref_img_len,
        l_effective_img_len,
        ref_img_sizes,
        img_sizes,
        device,
    )

    # 2. Context refinement
    for layer in self.context_refiner:
        text_hidden_states = layer(text_hidden_states, text_attention_mask, context_rotary_emb)
    
    combined_img_hidden_states = self.img_patch_embed_and_refine(
        hidden_states,
        ref_image_hidden_states,
        img_mask,
        ref_img_mask,
        noise_rotary_emb,
        ref_img_rotary_emb,
        l_effective_ref_img_len,
        l_effective_img_len,
        temb,
    )

    # 3. Joint Transformer blocks
    max_seq_len = max(seq_lengths)

    attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
    joint_hidden_states = hidden_states.new_zeros(batch_size, max_seq_len, self.config.hidden_size)
    for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
        attention_mask[i, :seq_len] = True
        joint_hidden_states[i, :encoder_seq_len] = text_hidden_states[i, :encoder_seq_len]
        joint_hidden_states[i, encoder_seq_len:seq_len] = combined_img_hidden_states[i, :seq_len - encoder_seq_len]

    hidden_states = joint_hidden_states

    ori_hidden_states = hidden_states.clone()
    for layer_idx, layer in enumerate(self.layers):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                layer, hidden_states, attention_mask, rotary_emb, temb
            )
        else:
            hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)

    # calibrate the mag_ratios for the MagCache
    cur_residual = hidden_states -  ori_hidden_states
    if self.magcache_params.cnt>=1:
        norm_ratio = ((cur_residual.norm(dim=-1)/self.magcache_params.previous_residual.norm(dim=-1)).mean()).item()
        norm_std = (cur_residual.norm(dim=-1)/self.magcache_params.previous_residual.norm(dim=-1)).std().item()
        cos_dis = (1-F.cosine_similarity(cur_residual, self.magcache_params.previous_residual, dim=-1, eps=1e-8)).mean().item()
        self.magcache_params.norm_ratio.append(round(norm_ratio, 5))
        self.magcache_params.norm_std.append(round(norm_std, 5))
        self.magcache_params.cos_dis.append(round(cos_dis, 5))
        print(f"time: {self.magcache_params.cnt}, norm_ratio: {norm_ratio}, norm_std: {norm_std}, cos_dis: {cos_dis}")
    self.magcache_params.previous_residual = cur_residual

    # 4. Output norm & projection
    hidden_states = self.norm_out(hidden_states, temb)

    p = self.config.patch_size
    output = []
    for i, (img_size, img_len, seq_len) in enumerate(zip(img_sizes, l_effective_img_len, seq_lengths)):
        height, width = img_size
        output.append(rearrange(hidden_states[i][seq_len - img_len:seq_len], '(h w) (p1 p2 c) -> c (h p1) (w p2)', h=height // p, w=width // p, p1=p, p2=p))
    if is_hidden_states_tensor:
        output = torch.stack(output, dim=0)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)
        
    self.magcache_params.cnt += 1
    if self.magcache_params.cnt>=self.num_steps:
        print("norm ratio")
        print(self.magcache_params.norm_ratio)
        print("norm std")
        print(self.magcache_params.norm_std)
        print("cos_dis")
        print(self.magcache_params.cos_dis)
        self.magcache_params.cnt = 0
        self.magcache_params.norm_ratio = []
        self.magcache_params.norm_std = []
        self.magcache_params.cos_dis = []
        self.magcache_params.previous_residual = None

    if not return_dict:
        return output
    return Transformer2DModelOutput(sample=output)

def magcache_forward(
    self,
    hidden_states: Union[torch.Tensor, List[torch.Tensor]],
    timestep: torch.Tensor,
    text_hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
    text_attention_mask: torch.Tensor,
    ref_image_hidden_states: Optional[List[List[torch.Tensor]]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = False,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    # 1. Condition, positional & patch embedding
    batch_size = len(hidden_states)
    is_hidden_states_tensor = isinstance(hidden_states, torch.Tensor)

    if is_hidden_states_tensor:
        assert hidden_states.ndim == 4
        hidden_states = [_hidden_states for _hidden_states in hidden_states]

    device = hidden_states[0].device

    temb, text_hidden_states = self.time_caption_embed(timestep, text_hidden_states, hidden_states[0].dtype)

    (
        hidden_states,
        ref_image_hidden_states,
        img_mask,
        ref_img_mask,
        l_effective_ref_img_len,
        l_effective_img_len,
        ref_img_sizes,
        img_sizes,
    ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)
    
    (
        context_rotary_emb,
        ref_img_rotary_emb,
        noise_rotary_emb,
        rotary_emb,
        encoder_seq_lengths,
        seq_lengths,
    ) = self.rope_embedder(
        freqs_cis,
        text_attention_mask,
        l_effective_ref_img_len,
        l_effective_img_len,
        ref_img_sizes,
        img_sizes,
        device,
    )

    # 2. Context refinement
    for layer in self.context_refiner:
        text_hidden_states = layer(text_hidden_states, text_attention_mask, context_rotary_emb)
    
    combined_img_hidden_states = self.img_patch_embed_and_refine(
        hidden_states,
        ref_image_hidden_states,
        img_mask,
        ref_img_mask,
        noise_rotary_emb,
        ref_img_rotary_emb,
        l_effective_ref_img_len,
        l_effective_img_len,
        temb,
    )

    # 3. Joint Transformer blocks
    max_seq_len = max(seq_lengths)

    attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
    joint_hidden_states = hidden_states.new_zeros(batch_size, max_seq_len, self.config.hidden_size)
    for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
        attention_mask[i, :seq_len] = True
        joint_hidden_states[i, :encoder_seq_len] = text_hidden_states[i, :encoder_seq_len]
        joint_hidden_states[i, encoder_seq_len:seq_len] = combined_img_hidden_states[i, :seq_len - encoder_seq_len]

    hidden_states = joint_hidden_states

    skip_forward = False
    if self.magcache_params.cnt>=math.ceil(self.retention_ratio*self.num_steps):
        cur_scale = self.magcache_params.mag_ratios[self.magcache_params.cnt]
        self.magcache_params.accumulated_ratio = self.magcache_params.accumulated_ratio*cur_scale
        self.magcache_params.accumulated_steps += 1
        self.magcache_params.accumulated_err += np.abs(1-self.magcache_params.accumulated_ratio)
        if self.magcache_params.accumulated_err<=self.magcache_thresh and self.magcache_params.accumulated_steps<=self.K:
            cur_residual = self.magcache_params.previous_residual
            skip_forward = True
        else:
            self.magcache_params.accumulated_ratio = 1.0
            self.magcache_params.accumulated_steps = 0
            self.magcache_params.accumulated_err = 0
    if skip_forward:
        hidden_states = hidden_states + cur_residual
    else: # original forward
        ori_hidden_states = hidden_states
        for layer_idx, layer in enumerate(self.layers):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    layer, hidden_states, attention_mask, rotary_emb, temb
                )
            else:
                hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)
        cur_residual = hidden_states -  ori_hidden_states
    self.magcache_params.previous_residual = cur_residual
    self.magcache_params.cnt += 1
    
    # init the hyper-parameters for the next generation
    if self.magcache_params.cnt>=self.num_steps:
        self.magcache_params.cnt = 0
        self.magcache_params.accumulated_ratio = 1.0
        self.magcache_params.accumulated_steps = 0
        self.magcache_params.accumulated_err = 0 
        
    # 4. Output norm & projection
    hidden_states = self.norm_out(hidden_states, temb)

    p = self.config.patch_size
    output = []
    for i, (img_size, img_len, seq_len) in enumerate(zip(img_sizes, l_effective_img_len, seq_lengths)):
        height, width = img_size
        output.append(rearrange(hidden_states[i][seq_len - img_len:seq_len], '(h w) (p1 p2 c) -> c (h p1) (w p2)', h=height // p, w=width // p, p1=p, p2=p))
    if is_hidden_states_tensor:
        output = torch.stack(output, dim=0)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return output
    return Transformer2DModelOutput(sample=output)

def magcache_processing(
    self,
    latents,
    ref_latents,
    prompt_embeds,
    freqs_cis,
    negative_prompt_embeds,
    prompt_attention_mask,
    negative_prompt_attention_mask,
    num_inference_steps,
    timesteps,
    device,
    dtype,
    verbose,
    step_func=None
):
    batch_size = latents.shape[0]

    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        num_tokens=latents.shape[-2] * latents.shape[-1]
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    enable_taylorseer = getattr(self, "enable_taylorseer", False)
    if enable_taylorseer:
        model_pred_cache_dic, model_pred_current = cache_init(self, num_inference_steps)
        model_pred_ref_cache_dic, model_pred_ref_current = cache_init(self, num_inference_steps)
        model_pred_uncond_cache_dic, model_pred_uncond_current = cache_init(self, num_inference_steps)
        self.transformer.enable_taylorseer = True
    elif self.transformer.enable_teacache:
        # Use different TeaCacheParams for different conditions
        teacache_params = TeaCacheParams()
        teacache_params_uncond = TeaCacheParams()
        teacache_params_ref = TeaCacheParams()
    elif self.transformer.enable_magcache:
        if self.image_guidance_scale<=1.0: # t2i
            magcache_params_cond = MagCacheParams(mode_type='t2i_cond', num_inference_step=num_inference_steps)
            magcache_params_uncond = MagCacheParams(mode_type='t2i_uncond', num_inference_step=num_inference_steps)
        else:
            magcache_params_cond = MagCacheParams(mode_type='edit_cond', num_inference_step=num_inference_steps)
            magcache_params_uncond = MagCacheParams(mode_type='edit_uncond', num_inference_step=num_inference_steps)
            magcache_params_ref = MagCacheParams(mode_type='edit_ref', num_inference_step=num_inference_steps)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if enable_taylorseer:
                self.transformer.cache_dic = model_pred_cache_dic
                self.transformer.current = model_pred_current
            elif self.transformer.enable_teacache:
                teacache_params.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                self.transformer.teacache_params = teacache_params
            elif self.transformer.enable_magcache:
                self.transformer.magcache_params = magcache_params_cond 

            model_pred = self.predict(
                t=t,
                latents=latents,
                prompt_embeds=prompt_embeds,
                freqs_cis=freqs_cis,
                prompt_attention_mask=prompt_attention_mask,
                ref_image_hidden_states=ref_latents,
            )
            text_guidance_scale = self.text_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
            image_guidance_scale = self.image_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
            
            if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
                if enable_taylorseer:
                    self.transformer.cache_dic = model_pred_ref_cache_dic
                    self.transformer.current = model_pred_ref_current
                elif self.transformer.enable_teacache:
                    teacache_params_ref.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                    self.transformer.teacache_params = teacache_params_ref
                elif self.transformer.enable_magcache:
                    self.transformer.magcache_params = magcache_params_ref

                model_pred_ref = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=negative_prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=negative_prompt_attention_mask,
                    ref_image_hidden_states=ref_latents,
                )

                if enable_taylorseer:
                    self.transformer.cache_dic = model_pred_uncond_cache_dic
                    self.transformer.current = model_pred_uncond_current
                elif self.transformer.enable_teacache:
                    teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                    self.transformer.teacache_params = teacache_params_uncond
                elif self.transformer.enable_magcache:
                    self.transformer.magcache_params = magcache_params_uncond 

                model_pred_uncond = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=negative_prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=negative_prompt_attention_mask,
                    ref_image_hidden_states=None,
                )

                model_pred = model_pred_uncond + image_guidance_scale * (model_pred_ref - model_pred_uncond) + \
                    text_guidance_scale * (model_pred - model_pred_ref)
            elif text_guidance_scale > 1.0:
                if enable_taylorseer:
                    self.transformer.cache_dic = model_pred_uncond_cache_dic
                    self.transformer.current = model_pred_uncond_current
                elif self.transformer.enable_teacache:
                    teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                    self.transformer.teacache_params = teacache_params_uncond
                elif self.transformer.enable_magcache:
                    self.transformer.magcache_params = magcache_params_uncond 

                model_pred_uncond = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=negative_prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=negative_prompt_attention_mask,
                    ref_image_hidden_states=None,
                )
                model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)

            latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]

            latents = latents.to(dtype=dtype)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
            
            if step_func is not None:
                step_func(i, self._num_timesteps)

    if enable_taylorseer:
        del model_pred_cache_dic, model_pred_ref_cache_dic, model_pred_uncond_cache_dic
        del model_pred_current, model_pred_ref_current, model_pred_uncond_current

    latents = latents.to(dtype=dtype)
    if self.vae.config.scaling_factor is not None:
        latents = latents / self.vae.config.scaling_factor
    if self.vae.config.shift_factor is not None:
        latents = latents + self.vae.config.shift_factor
    image = self.vae.decode(latents, return_dict=False)[0]
    
    return image