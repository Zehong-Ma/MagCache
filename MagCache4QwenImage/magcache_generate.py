from diffusers import DiffusionPipeline
import torch
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F  # Added import for F

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models import QwenImageTransformer2DModel
import numpy as np
from copy import deepcopy

# Added nearest neighbor interpolation function
def nearest_interp(original: np.ndarray, target_length: int) -> np.ndarray:
    """Nearest neighbor interpolation for 1D array to target length."""
    original_length = len(original)
    if original_length == target_length:
        return original.copy()
    # Compute indices for original array corresponding to target positions
    indices = np.round(np.linspace(0, original_length - 1, target_length)).astype(int)
    return original[indices]

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an image from a text prompt using Qwen Image with MagCache"
    )
    parser.add_argument(
        "--sample_steps", type=int, default=50, help="The sampling steps.")
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--magcache_thresh",
        type=float,
        default=0.06,
        help="Upper bound of accumulated error for MagCache")
    parser.add_argument(
        "--retention_ratio",
        type=float,
        default=0.2,
        help="Retention ratio of unchanged steps for MagCache")
    parser.add_argument(
        "--magcache_K",
        type=int,
        default=2,
        help="Max skip steps for MagCache")
    parser.add_argument(
        "--use_magcache",
        action="store_true",
        default=True,
        help="Use MagCache for inference after calibration")
    parser.add_argument(
        "--magcache_calibration",
        action="store_true",
        default=False,
        help="Calibrate magnitude ratios for MagCache")
    
    args = parser.parse_args()
    return args

def init_magcache(model, mag_ratios, args, split_steps=None, mode="t2v"):
    model.__class__.forward = magcache_forward
    model.__class__.cnt = torch.tensor(0)
    model.__class__.num_steps = args.sample_steps * 2
    model.__class__.split_step = split_steps * 2 if split_steps else None
    model.__class__.mode = mode
    model.__class__.magcache_thresh = args.magcache_thresh
    model.__class__.K = args.magcache_K
    model.__class__.accumulated_err = [0.0, 0.0]
    model.__class__.accumulated_steps = [0, 0]
    model.__class__.accumulated_ratio = [1.0, 1.0]
    model.__class__.retention_ratio = args.retention_ratio
    model.__class__.residual_cache = [None, None]
    model.__class__.mag_ratios = np.array([1.0] * 2 + mag_ratios)  # [pad, pad, ratios...]
    
    # Interpolate if length mismatch
    if len(model.__class__.mag_ratios) != args.sample_steps * 2:
        mag_ratio_con = nearest_interp(model.__class__.mag_ratios[0::2], args.sample_steps)
        mag_ratio_ucon = nearest_interp(model.__class__.mag_ratios[1::2], args.sample_steps)
        interpolated = np.concatenate([mag_ratio_con.reshape(-1, 1), mag_ratio_ucon.reshape(-1, 1)], axis=1).flatten()
        model.__class__.mag_ratios = interpolated

def init_magcache_calibration(model, args):
    model.__class__.forward = magcache_calibration
    model.__class__.cnt = torch.tensor(0)
    model.__class__.num_steps = args.sample_steps * 2
    model.__class__.norm_ratio = []  # Mean of magnitude ratios
    model.__class__.norm_std = []   # Std of magnitude ratios
    model.__class__.cos_dis = []    # Cosine distance of residuals
    model.__class__.residual_cache = [None, None]

def magcache_calibration(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: Optional[List[Tuple[int, int, int]]] = None,
    txt_seq_lens: Optional[List[int]] = None,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    hidden_states = self.img_in(hidden_states)
    timestep = timestep.to(hidden_states.dtype)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = self.time_text_embed(timestep, hidden_states) if guidance is None else self.time_text_embed(timestep, guidance, hidden_states)
    image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
                
    ori_hidden_states = hidden_states
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, image_rotary_emb
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )

    residual_x = hidden_states - ori_hidden_states
    if self.cnt >= 2:
        # Calculate metrics
        prev_residual = self.residual_cache[self.cnt % 2]
        norm_ratio = (residual_x.norm(dim=-1) / prev_residual.norm(dim=-1)).mean().item()
        norm_std = (residual_x.norm(dim=-1) / prev_residual.norm(dim=-1)).std().item()
        cos_dis = (1 - F.cosine_similarity(residual_x, prev_residual, dim=-1, eps=1e-8)).mean().item()
        
        self.norm_ratio.append(round(norm_ratio, 5))
        self.norm_std.append(round(norm_std, 5))
        self.cos_dis.append(round(cos_dis, 5))
        print(f"Step {self.cnt}: norm_ratio={norm_ratio:.5f}, norm_std={norm_std:.5f}, cos_dis={cos_dis:.5f}")
    
    self.residual_cache[self.cnt % 2] = residual_x
    self.cnt += 1

    if self.cnt >= self.num_steps:
        self.cnt = 0
        print("\nCalibration Results:")
        print("norm_ratio:", self.norm_ratio)
        print("norm_std:", self.norm_std)
        print("cos_dis:", self.cos_dis)

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    return Transformer2DModelOutput(sample=output) if return_dict else (output,)

def magcache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: Optional[List[Tuple[int, int, int]]] = None,
    txt_seq_lens: Optional[List[int]] = None,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    hidden_states = self.img_in(hidden_states)
    timestep = timestep.to(hidden_states.dtype)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = self.time_text_embed(timestep, hidden_states) if guidance is None else self.time_text_embed(timestep, guidance, hidden_states)
    image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
    
    skip_forward = False
    if self.cnt >= int(self.num_steps * self.retention_ratio):
        cur_mag_ratio = self.mag_ratios[self.cnt]
        self.accumulated_ratio[self.cnt % 2] *= cur_mag_ratio
        self.accumulated_steps[self.cnt % 2] += 1
        cur_skip_err = np.abs(1 - self.accumulated_ratio[self.cnt % 2])
        self.accumulated_err[self.cnt % 2] += cur_skip_err
        
        if self.accumulated_err[self.cnt % 2] < self.magcache_thresh and self.accumulated_steps[self.cnt % 2] <= self.K:
            skip_forward = True
            residual_x = self.residual_cache[self.cnt % 2]
        else:
            self.accumulated_err[self.cnt % 2] = 0.0
            self.accumulated_steps[self.cnt % 2] = 0
            self.accumulated_ratio[self.cnt % 2] = 1.0

    if skip_forward:
        hidden_states += residual_x
    else:
        ori_hidden_states = hidden_states
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, image_rotary_emb
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )
        residual_x = hidden_states - ori_hidden_states

    self.residual_cache[self.cnt % 2] = residual_x
    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    return Transformer2DModelOutput(sample=output) if return_dict else (output,)

# Inference Code
args = _parse_args()
model_name = "Qwen/Qwen-Image"  # Or "Qwen/Qwen-Image"

# Load pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

# Initialize MagCache or Calibration
if args.magcache_calibration:
    init_magcache_calibration(pipe.transformer, args)
elif args.use_magcache:
    # Replace with precomputed mag_ratios from calibration
    # Example: mag_ratios = [0.98, 0.97, ...] (length should match sample_steps - 2)
    mag_ratios = [1.64062, 1.64062, 1.45312, 1.45312, 1.1875, 1.1875, 1.17188, 1.17188, 1.05469, 1.05469, 1.21094, 1.21094, 1.10938, 1.10938, 1.11719, 1.11719, 1.13281, 1.13281, 1.11719, 1.11719, 1.07812, 1.07812, 1.07031, 1.07031, 1.08594, 1.08594, 1.08594, 1.08594, 1.07812, 1.07812, 1.03906, 1.03906, 1.04688, 1.04688, 1.07812, 1.07812, 1.07031, 1.07031, 1.03125, 1.03125, 1.07812, 1.07812, 1.04688, 1.04688, 1.04688, 1.04688, 1.04688, 1.04688, 1.03906, 1.03906, 1.01562, 1.01562, 1.03125, 1.03125, 1.02344, 1.02344, 1.02344, 1.02344, 1.02344, 1.02344, 1.03906, 1.03906, 1.0, 1.0, 1.01562, 1.01562, 1.0, 1.0, 0.99219, 0.99219, 1.00781, 1.00781, 0.98047, 0.98047, 0.95703, 0.95703, 0.96875, 0.96875, 0.99219, 0.99219, 0.92578, 0.92578, 0.92578, 0.92578, 0.90625, 0.90625, 0.85938, 0.85938, 0.80469, 0.80469, 0.87891, 0.87891, 0.75, 0.75, 0.60938, 0.60938, 0.55078, 0.55078]  # Insert your calibrated ratios here
    init_magcache(pipe.transformer, mag_ratios, args)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图."
}

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''
negative_prompt = " "

aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}
width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=args.sample_steps,
    true_cfg_scale=args.true_cfg_scale,
    generator=torch.Generator(device=device).manual_seed(42)
).images[0]

image.save("example.png")