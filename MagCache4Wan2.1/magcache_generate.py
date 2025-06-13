# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

import gc
from contextlib import contextmanager
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp
import numpy as np
import math
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from tqdm import tqdm
import torch.nn.functional as F
import json

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}

def save_json(filename, obj_list):
    with open(filename+".json", "w") as f:
        json.dump(obj_list, f)

def t2v_generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None



def i2v_generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, 
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F-1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
                # 'cond_flag': True,
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
                # 'cond_flag': False,
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

def magcache_calibration(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    r"""
    Forward pass through the diffusion model
    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x
    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
    ])

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
    
    skip_forward = False
    ori_x = x
    
  
    for ind, block in enumerate(self.blocks):
        x = block(x, **kwargs)
    residual_x = x - ori_x
    if self.cnt>=2:
        norm_ratio = ((residual_x.norm(dim=-1)/self.residual_cache[self.cnt%2].norm(dim=-1)).mean()).item()
        norm_std = (residual_x.norm(dim=-1)/self.residual_cache[self.cnt%2].norm(dim=-1)).std().item()
        cos_dis = (1-F.cosine_similarity(residual_x, self.residual_cache[self.cnt%2], dim=-1, eps=1e-8)).mean().item()
        self.norm_ratio.append(round(norm_ratio, 5))
        self.norm_std.append(round(norm_std, 5))
        self.cos_dis.append(round(cos_dis, 5))
        print(f"time: {self.cnt}, norm_ratio: {norm_ratio}, norm_std: {norm_std}, cos_dis: {cos_dis}")
    
    self.residual_cache[self.cnt%2] = residual_x
    
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    self.cnt += 1
    if self.cnt >= self.num_steps: # clear the history of current video and prepare for generating the next video.
        self.cnt = 0
        self.accumulated_ratio = [1.0, 1.0]
        self.accumulated_err = [0.0, 0.0]
        self.accumulated_steps = [0, 0]
        print("norm ratio")
        print(self.norm_ratio)
        print("norm std")
        print(self.norm_std)
        print("cos_dis")
        print(self.cos_dis)
        save_json("wan2_1_mag_ratio", self.norm_ratio)
        save_json("wan2_1_mag_std", self.norm_std)
        save_json("wan2_1_cos_dis", self.cos_dis)
    return [u.float() for u in x]

def magcache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    r"""
    Forward pass through the diffusion model
    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x
    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
    ])

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
    
    skip_forward = False
    ori_x = x
    if self.cnt>=int(self.num_steps*self.retention_ratio):
        cur_mag_ratio = self.mag_ratios[self.cnt] # conditional and unconditional in one list
        self.accumulated_ratio[self.cnt%2] = self.accumulated_ratio[self.cnt%2]*cur_mag_ratio # magnitude ratio between current step and the cached step
        self.accumulated_steps[self.cnt%2] += 1 # skip steps plus 1
        cur_skip_err = np.abs(1-self.accumulated_ratio[self.cnt%2]) # skip error of current steps
        self.accumulated_err[self.cnt%2] += cur_skip_err # accumulated error of multiple steps
        
        if self.accumulated_err[self.cnt%2]<self.magcache_thresh and self.accumulated_steps[self.cnt%2]<=self.K:
            skip_forward = True
            residual_x = self.residual_cache[self.cnt%2]
        else:
            self.accumulated_err[self.cnt%2] = 0
            self.accumulated_steps[self.cnt%2] = 0
            self.accumulated_ratio[self.cnt%2] = 1.0

    if skip_forward: # skip this step with cached residual
        x =  x + residual_x 
    else:
        for ind, block in enumerate(self.blocks):
            x = block(x, **kwargs)
        residual_x = x - ori_x
        
    self.residual_cache[self.cnt%2] = residual_x 
    
    
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    self.cnt += 1
    if self.cnt >= self.num_steps: # clear the history of current video and prepare for generating the next video.
        self.cnt = 0
        self.accumulated_ratio = [1.0, 1.0]
        self.accumulated_err = [0.0, 0.0]
        self.accumulated_steps = [0, 0]
    return [u.float() for u in x]

def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="ch",
        choices=["ch", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--magcache_thresh",
        type=float,
        default=0.12,
        help="the upper bound of accumulated err")
    parser.add_argument(
        "--retention_ratio",
        type=float,
        default=0.2,
        help="Retention ratio of unchanged steps")
    parser.add_argument(
        "--magcache_K",
        type=int,
        default=2,
        help="max skip steps")
    parser.add_argument(
        "--use_magcache",
        action="store_true",
        default=False,
        help="Use MagCache for inference")
    parser.add_argument(
        "--magcache_calibration",
        action="store_true",
        default=False,
        help="Calibrate for MagCache")
        

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        # MagCache
        if args.use_magcache:
            wan_t2v.__class__.generate = t2v_generate
            wan_t2v.model.__class__.forward = magcache_forward
            wan_t2v.model.__class__.cnt = 0
            wan_t2v.model.__class__.num_steps = args.sample_steps*2
            wan_t2v.model.__class__.magcache_thresh = args.magcache_thresh
            wan_t2v.model.__class__.K = args.magcache_K
            wan_t2v.model.__class__.accumulated_err = [0.0, 0.0]
            wan_t2v.model.__class__.accumulated_steps = [0, 0]
            wan_t2v.model.__class__.accumulated_ratio = [1.0, 1.0]
            wan_t2v.model.__class__.retention_ratio = args.retention_ratio
            wan_t2v.model.__class__.residual_cache = [None, None]
            # note: we utilze the sdpa to forward the calibration, so the mag_ratios may be different with those using flash attention2 in the last two decimal digits
            # the [1.0]*1 is the padding value of first magnitude ratio. 
            if 'T2V-14B' in args.ckpt_dir:
                wan_t2v.model.__class__.mag_ratios = np.array([1.0]*2+[1.02504, 1.03017, 1.00025, 1.00251, 0.9985, 0.99962, 0.99779, 0.99771, 0.9966, 0.99658, 0.99482, 0.99476, 0.99467, 0.99451, 0.99664, 0.99656, 0.99434, 0.99431, 0.99533, 0.99545, 0.99468, 0.99465, 0.99438, 0.99434, 0.99516, 0.99517, 0.99384, 0.9938, 0.99404, 0.99401, 0.99517, 0.99516, 0.99409, 0.99408, 0.99428, 0.99426, 0.99347, 0.99343, 0.99418, 0.99416, 0.99271, 0.99269, 0.99313, 0.99311, 0.99215, 0.99215, 0.99218, 0.99215, 0.99216, 0.99217, 0.99163, 0.99161, 0.99138, 0.99135, 0.98982, 0.9898, 0.98996, 0.98995, 0.9887, 0.98866, 0.98772, 0.9877, 0.98767, 0.98765, 0.98573, 0.9857, 0.98501, 0.98498, 0.9838, 0.98376, 0.98177, 0.98173, 0.98037, 0.98035, 0.97678, 0.97677, 0.97546, 0.97543, 0.97184, 0.97183, 0.96711, 0.96708, 0.96349, 0.96345, 0.95629, 0.95625, 0.94926, 0.94929, 0.93964, 0.93961, 0.92511, 0.92504, 0.90693, 0.90678, 0.8796, 0.87945, 0.86111, 0.86189])
            if 'T2V-1.3B' in args.ckpt_dir:
                wan_t2v.model.__class__.mag_ratios = np.array([1.0]*2+[1.0124, 1.02213, 1.00166, 1.0041, 0.99791, 1.00061, 0.99682, 0.99762, 0.99634, 0.99685, 0.99567, 0.99586, 0.99416, 0.99422, 0.99578, 0.99575, 0.9957, 0.99563, 0.99511, 0.99506, 0.99535, 0.99531, 0.99552, 0.99549, 0.99541, 0.99539, 0.9954, 0.99536, 0.99489, 0.99485, 0.99518, 0.99514, 0.99484, 0.99478, 0.99481, 0.99479, 0.99415, 0.99413, 0.99419, 0.99416, 0.99396, 0.99393, 0.99388, 0.99386, 0.99349, 0.99349, 0.99309, 0.99304, 0.9927, 0.9927, 0.99228, 0.99226, 0.99171, 0.9917, 0.99137, 0.99135, 0.99068, 0.99063, 0.99005, 0.99003, 0.98944, 0.98942, 0.98849, 0.98849, 0.98758, 0.98757, 0.98644, 0.98643, 0.98504, 0.98503, 0.9836, 0.98359, 0.98202, 0.98201, 0.97977, 0.97978, 0.97717, 0.97718, 0.9741, 0.97411, 0.97003, 0.97002, 0.96538, 0.96541, 0.9593, 0.95933, 0.95086, 0.95089, 0.94013, 0.94019, 0.92402, 0.92414, 0.90241, 0.9026, 0.86821, 0.86868, 0.81838, 0.81939])#**(0.5)# In our papaer, we utilize the sqrt to smooth the ratio, which has little impact on the performance and can be deleted.
        if args.magcache_calibration:
            wan_t2v.__class__.generate = t2v_generate
            wan_t2v.model.__class__.forward = magcache_calibration
            wan_t2v.model.__class__.cnt = 0
            wan_t2v.model.__class__.num_steps = args.sample_steps*2
            wan_t2v.model.__class__.norm_ratio = [] # mean of magnitude ratio
            wan_t2v.model.__class__.norm_std = [] # std of magnitude ratio
            wan_t2v.model.__class__.cos_dis = [] # cosine distance of residual features
            wan_t2v.model.__class__.residual_cache = [None, None]

        logging.info(
            f"Generating {'image' if 't2i' in args.task else 'video'} ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input image: {args.image}")

        img = Image.open(args.image).convert("RGB")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
        # MagCache
        if args.use_magcache:
            wan_i2v.__class__.generate = i2v_generate
            wan_i2v.model.__class__.forward = magcache_forward
            wan_i2v.model.__class__.cnt = 0
            wan_i2v.model.__class__.num_steps = args.sample_steps*2
            wan_i2v.model.__class__.magcache_thresh = args.magcache_thresh
            wan_i2v.model.__class__.K = args.magcache_K
            wan_i2v.model.__class__.retention_ratio = args.retention_ratio
            wan_i2v.model.__class__.accumulated_err = [0.0, 0.0]
            wan_i2v.model.__class__.accumulated_steps = [0, 0]
            wan_i2v.model.__class__.accumulated_ratio = [1.0, 1.0]
            wan_i2v.model.__class__.residual_cache = [None, None]
            
            if '480P' in args.ckpt_dir:
                wan_i2v.model.__class__.mag_ratios = np.array([1.0]*2+[0.98783, 0.98993, 0.97559, 0.97593, 0.98311, 0.98319, 0.98202, 0.98225, 0.9888, 0.98878, 0.98762, 0.98759, 0.98957, 0.98971, 0.99052, 0.99043, 0.99383, 0.99384, 0.98857, 0.9886, 0.99065, 0.99068, 0.98845, 0.98847, 0.99057, 0.99057, 0.98957, 0.98961, 0.98601, 0.9861, 0.98823, 0.98823, 0.98756, 0.98759, 0.98808, 0.98814, 0.98721, 0.98724, 0.98571, 0.98572, 0.98543, 0.98544, 0.98157, 0.98165, 0.98411, 0.98413, 0.97952, 0.97953, 0.98149, 0.9815, 0.9774, 0.97742, 0.97825, 0.97826, 0.97355, 0.97361, 0.97085, 0.97087, 0.97056, 0.97055, 0.96588, 0.96587, 0.96113, 0.96124, 0.9567, 0.95681, 0.94961, 0.94969, 0.93973, 0.93988, 0.93217, 0.93224, 0.91878, 0.91896, 0.90955, 0.90954, 0.92617, 0.92616])
            if '720P' in args.ckpt_dir:
                wan_i2v.model.__class__.mag_ratios = np.array([1.0]*2+[0.99428, 0.99498, 0.98588, 0.98621, 0.98273, 0.98281, 0.99018, 0.99023, 0.98911, 0.98917, 0.98646, 0.98652, 0.99454, 0.99456, 0.9891, 0.98909, 0.99124, 0.99127, 0.99102, 0.99103, 0.99215, 0.99212, 0.99515, 0.99515, 0.99576, 0.99572, 0.99068, 0.99072, 0.99097, 0.99097, 0.99166, 0.99169, 0.99041, 0.99042, 0.99201, 0.99198, 0.99101, 0.99101, 0.98599, 0.98603, 0.98845, 0.98844, 0.98848, 0.98851, 0.98862, 0.98857, 0.98718, 0.98719, 0.98497, 0.98497, 0.98264, 0.98263, 0.98389, 0.98393, 0.97938, 0.9794, 0.97535, 0.97536, 0.97498, 0.97499, 0.973, 0.97301, 0.96827, 0.96828, 0.96261, 0.96263, 0.95335, 0.9534, 0.94649, 0.94655, 0.93397, 0.93414, 0.91636, 0.9165, 0.89088, 0.89109, 0.8679, 0.86768])
        if args.magcache_calibration:
            wan_i2v.__class__.generate = i2v_generate
            wan_i2v.model.__class__.forward = magcache_calibration
            wan_i2v.model.__class__.cnt = 0
            wan_i2v.model.__class__.num_steps = args.sample_steps*2
            wan_i2v.model.__class__.norm_ratio = [] # mean of magnitude ratio
            wan_i2v.model.__class__.norm_std = [] # std of magnitude ratio
            wan_i2v.model.__class__.cos_dis = [] # cosine distance of residual features
            wan_i2v.model.__class__.residual_cache = [None, None]

        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            if args.use_magcache:
                magcache_params = f"_E{args.magcache_thresh}_K{args.magcache_K}_R{args.retention_ratio}".replace(".", "")
            else:
                magcache_params = ""
            args.save_file = f"magcache_{args.task}{magcache_params}_{args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}".replace(" ", "").replace(",", "") + suffix

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    logging.info("Finished.")    
    
    

if __name__ == "__main__":
    args = _parse_args()
    generate(args)
