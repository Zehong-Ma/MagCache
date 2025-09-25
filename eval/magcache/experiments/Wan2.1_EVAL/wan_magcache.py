from utils import generate_func, read_prompt_list
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import torch
import numpy as np
import random
import torch.nn.functional as F
from copy import deepcopy

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 确定性卷积算法
    torch.backends.cudnn.benchmark = False     # 关闭自动优化

set_seed(0)  # 设置种子为0

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
from copy import deepcopy
import time

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

def push_tensor_roll(queue, new_tensor):
    """
    queue: (b, s, d, k)
    new_tensor: (b, s, d)

    Uses torch.roll to shift the queue and insert the new tensor.
    """
    # Roll the queue by -1 along the last dimension (shifts elements towards index 0)
    # queue[1] goes to queue[0], queue[2] to queue[1], ..., queue[k-1] to queue[k-2]
    # The last element (queue[k-1]) is moved to queue[k-2] in the rolled tensor,
    # effectively making the original queue[0] disappear.
    if new_tensor is None:
        return queue
    rolled_queue = torch.roll(queue, shifts=-1, dims=-1)

    # Place the new tensor in the last position
    rolled_queue[..., -1] = new_tensor

    return rolled_queue

def fit_affine_lstsq(I_s: torch.Tensor,
                      O_s: torch.Tensor,
                      eps: float=1e-6):  # 估计更准，但是first frame的一致性差
    """
    用 torch.linalg.lstsq 加速拟合 O ≈ sum_k (I[...,d,k]-mean_I_sub) * a[b,d,k] + c_sub，
    lstsq 内置了对奇异矩阵的处理，保证数值稳定性。
    此版本避免了手动 SVD 和 B*D 维度合并，可能更高效且内存占用更低。

    输入：
      I_s: Tensor, shape = (B, Q, D, K), dtype=float32/float64
         (批次大小, 观测/样本数, 输出维度/通道数, 输入特征数)
      O_s: Tensor, shape = (B, Q, D), dtype 同 I
         (批次大小, 观测/样本数, 输出维度/通道数)
      sub_Q: 若指定，则随机抽样 sub_Q 条样本进行拟合，以加速计算。
      eps: 用于 lstsq 的截断阈值 (rcond)，处理奇异性。
           rcond 是相对于最大奇异值的阈值。设为 None 可使用 Pytorch 默认值。
           使用与原 SVD 版本相同的 eps 可以保持类似行为。

    返回：
      a: Tensor, shape = (B, D, K)  (线性变换系数)
      c: Tensor, shape = (B, 1, D)  (偏置项/截距)
    """
    B, Q, D, K = I_s.shape
    device, dtype = I_s.device, I_s.dtype

    # --- 2) 计算子样本中心化 ---
    # 计算均值时务必使用 float64 以提高精度，特别是对于大的 Q 或 sub_Q
    mean_I = I_s.to(torch.float32).mean(dim=1, keepdim=True)   # (B, 1, D, K)
    mean_O = O_s.to(torch.float32).mean(dim=1, keepdim=True)   # (B, 1, D)

    # 应用中心化，转换回原始或计算精度(float32/float64)
    I_c = I_s - mean_I.to(dtype)                       # (B, s, D, K)
    O_c = O_s - mean_O.to(dtype)                       # (B, s, D)

    # --- 3) 准备 lstsq 输入 ---
    # 我们想为每个 B 和 D 求解 O_c[b, :, d] ≈ I_c[b, :, d, :] @ a[b, d, :]
    # lstsq(A, B) 求解 AX = B，其中 A 是系数矩阵，B 是目标向量/矩阵
    # A 需要是 (..., s, K) -> I_c.permute(0, 2, 1, 3) -> (B, D, s, K)
    # B 需要是 (..., s, 1) -> O_c.permute(0, 2, 1).unsqueeze(-1) -> (B, D, s, 1)
    # X (解 a) 将是 (..., K, 1)

    A = I_c.permute(0, 2, 1, 3)  # Shape: (B, D, s, K)
    B_target = O_c.permute(0, 2, 1).unsqueeze(-1) # Shape: (B, D, s, 1)

    # --- 4) 调用 lstsq 求解 ---
    # 使用 float32 或 float64 进行计算以保证数值稳定性
    try:
        result = torch.linalg.lstsq(A.float(), B_target.float(), rcond=eps) # driver='gelsd' 或其他可能更优
        # result.solution 的 shape 是 (B, D, K, 1)
        A_sol = result.solution.squeeze(-1) # Shape: (B, D, K)
    except torch.linalg.LinAlgError as e:
         print(f"torch.linalg.lstsq failed: {e}")
         # 根据需要进行错误处理，例如返回 None 或 零
         # 这里我们先返回 None
         return None, None
    except RuntimeError as e:
         # 捕获可能的 CUDA 错误等
         print(f"Runtime error during lstsq: {e}")
         return None, None


    # a 的形状已经是 (B, D, K)，符合要求
    a = A_sol.to(dtype)

    # --- 5) 计算偏置 c 基于子样本均值 ---
    # c[b, 1, d] = mean_O[b, 0, d] - sum_k (mean_I[b, 0, d, k] * a[b, d, k])
    # mean_I: (B, 1, D, K) -> squeeze(1) -> (B, D, K)
    # a: (B, D, K)
    # mean_O: (B, 1, D)
    mean_I_squeezed = mean_I.squeeze(1) # (B, D, K)
    # 确保计算 c 时使用与均值相同的高精度
    correction = torch.sum(mean_I_squeezed.to(torch.float64) * a.to(torch.float64), dim=-1) # (B, D)
    c_val = mean_O.squeeze(1).to(torch.float64) - correction # (B, D)

    # 恢复 c 的形状 (B, 1, D)
    c = c_val.unsqueeze(1).to(dtype) # (B, 1, D)

    # --- 6) 返回结果，转换回原始数据类型 ---
    pred_sample = torch.einsum('bqdk,bdk->bqd', I_s, a) + c
    # sample_err = (pred_sample - O_s).abs().mean() / (O_s.abs().mean() + 1e-12)
    return a, c, pred_sample


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


def magcache_intro(
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
    use_full_attn = True
    cache_time = 0
    skip_time = 1
    ori_x = x
    
    for ind, block in enumerate(self.blocks):
        x = block(x, **kwargs, t=self.t, use_full_attn=use_full_attn)
    residual_x = x - ori_x

    with torch.autocast(device_type="cuda:0", dtype=torch.float32):
        
        if self.t>=skip_time:
            # print("residual_norm:", residual_x.norm(dim=-1, p=2).mean().item())
            # scale_cos = F.cosine_similarity(scale_residual, self.scale_cache[self.t%2][..., -1], dim=-1, eps=1e-8).mean().item()
            # print((scale_residual-self.scale_cache[self.t%2][..., -1]).mean(), (scale_residual-self.scale_cache[self.t%2][..., -1]).std())
            # print(F.cosine_similarity(scale_residual-self.scale_cache[self.t%2][..., -1], self.scale_cache[self.t%2][..., -1]-self.scale_cache[self.t%2][..., -2], dim=-1, eps=1e-8).mean())
            
            residual_cos = F.cosine_similarity(residual_x, self.residual_cache[self.t%2][..., -1], dim=-1, eps=1e-8).mean().item()
            residual_norm_ratio = ((residual_x.norm(dim=-1, p=2))/(self.residual_cache[self.t%2][..., -1].norm(dim=-1, p=2)+1e-8)).mean().item()
            residual_norm_std = ((residual_x.norm(dim=-1, p=2))/(self.residual_cache[self.t%2][..., -1].norm(dim=-1, p=2)+1e-8)).std().item()
            
            self.residual_cos.append(round(residual_cos, 5))
            self.residual_ratio.append(round(residual_norm_ratio, 5))
            self.residual_std.append(round(residual_norm_std, 5))
            print(f"time: {self.t}, cos1: {residual_cos:.4f}, ratio_mean: {residual_norm_ratio:.4f}, ratio_std: {residual_norm_std:.4f}")
    if self.t>=cache_time:
        if self.residual_cache is None:
            self.residual_cache = torch.zeros((2,  *x.shape, 1), device=x.device, dtype=x.dtype)
            # self.scale_cache = torch.zeros((2,  *x.shape[:2], 1), device=x.device, dtype=x.dtype)
        self.residual_cache[self.t%2] = push_tensor_roll(self.residual_cache[self.t%2], residual_x)

    x = self.head(x, e)
    # print("avg_sparsity_before: ", self.total_sparsity/self.total_t)
    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    
    self.t += 1
    if self.t>=100:
        print("residual cos")
        print(self.residual_cos)
        print("residual norm ratio")
        print(self.residual_ratio)
        print("residual norm std")
        print(self.residual_std)
        self.t = 0
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
    # if self.t%2==1 and self.t//2>=30 and self.t//2<=48:
    #     self.skip_steps+=1
    #     self.t+=1
    #     print(f"skip, total_steps: {self.skip_steps}")
    #     return self.pre_con
    
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
    cache_time = 10
    skip_time = int(self.num_steps*0.2) # keep the first 20% steps unchanged
    ori_x = x
    if self.t>=skip_time:
        cur_scale = self.ratio[self.t-10] # minus 10 because the ratios are cached after 10 steps
        self.accumulated_sim[self.t%2] = self.accumulated_sim[self.t%2]*cur_scale
        self.accumulated_steps[self.t%2] += 1
        self.accumulated_err[self.t%2] += np.abs(1-self.accumulated_sim[self.t%2])
        if self.accumulated_err[self.t%2]<=self.magcache_thresh and self.accumulated_steps[self.t%2]<=self.magcache_K:
            # residual_x = cur_scale*self.residual_cache[self.t%2][..., -1]
            residual_x = self.residual_cache[self.t%2][..., -1]
            skip_forward = True
        else:
            self.accumulated_sim[self.t%2] = 1.0
            self.accumulated_steps[self.t%2] = 0
            self.accumulated_err[self.t%2] = 0
    if skip_forward:
        x = x + residual_x
        self.skip_steps += 1
        print(f"skip time {self.t}, cur_scale: {cur_scale}, acc_sim: {self.accumulated_sim[self.t%2]}, total_steps: {self.skip_steps}")
    else:
        for ind, block in enumerate(self.blocks):
            x = block(x, **kwargs)
        residual_x = x - ori_x
                        
    if self.t>=cache_time:
        if self.residual_cache is None:
            self.residual_cache = torch.zeros((2,  *x.shape, 1), device=x.device, dtype=x.dtype)
        self.residual_cache[self.t%2] = push_tensor_roll(self.residual_cache[self.t%2], residual_x)

    # head
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    if self.t%2==0:
        self.pre_con = [u.float() for u in x]

    self.t += 1
    if self.t >= self.num_steps:
        self.t = 0
        self.accumulated_err[0] = 0
        self.accumulated_err[1] = 0
        self.skip_steps = 0
        self.accumulated_sim = [1.0, 1.0]
        self.accumulated_steps = [0,0]
        self.accumulated_err = [0,0]

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
        default=0.015,
        help="Higher speedup will cause to worse quality -- 0.1 for 2.0x speedup -- 0.2 for 3.0x speedup")
    parser.add_argument(
        "--magcache_K",
        type=int,
        default=-1,
        help="maximum number of skip steps")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./generated_samples",
        help="directory that stores the generated videos.")
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        default=False,
        help="Using Retention Steps will result in faster generation speed and better generation quality.")
    parser.add_argument(
        "--enable_magcache",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="start index of prompt list")
    parser.add_argument(
        "--end_index",
        type=int,
        default=10,
        help="start index of prompt list")
    
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


def generate(args, prompt_list=None, save_dir=None):
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

        # magcache
        if args.enable_magcache:
            wan_t2v.__class__.generate = t2v_generate
            wan_t2v.model.__class__.forward = magcache_forward
            wan_t2v.model.__class__.magcache_thresh = args.magcache_thresh
            wan_t2v.model.__class__.magcache_K = args.magcache_K
            wan_t2v.model.__class__.t = 0
            wan_t2v.model.__class__.accumulated_err = [0,0]
            wan_t2v.model.__class__.skip_steps = 0
            wan_t2v.model.__class__.skip_time = 1
            wan_t2v.model.__class__.cache_time = 0
            wan_t2v.model.__class__.pre_con = None

            wan_t2v.model.__class__.num_steps = args.sample_steps*2
            import numpy as np
            wan_t2v.model.__class__.ratio = np.array([0.99635, 0.99662, 0.99546, 0.99564, 0.99438, 0.9944, 0.99564, 0.99562, 0.9958, 0.99575, 0.9952, 0.99512, 0.99539, 0.99531, 0.99554, 0.9955, 0.99542, 0.99539, 0.99545, 0.99544, 0.99495, 0.9949, 0.99515, 0.99512, 0.99491, 0.99486, 0.99484, 0.99482, 0.99421, 0.99418, 0.99425, 0.99421, 0.99405, 0.99403, 0.99395, 0.99393, 0.99359, 0.99356, 0.99315, 0.99312, 0.9928, 0.99277, 0.99237, 0.99237, 0.99183, 0.9918, 0.99145, 0.99143, 0.99077, 0.99074, 0.99014, 0.99012, 0.98952, 0.98951, 0.98859, 0.98857, 0.98765, 0.98765, 0.98647, 0.98648, 0.98511, 0.9851, 0.98367, 0.98365, 0.98206, 0.98207, 0.9798, 0.9798, 0.97721, 0.97723, 0.97412, 0.97412, 0.9701, 0.97009, 0.96541, 0.96541, 0.95939, 0.95941, 0.95088, 0.95089, 0.94016, 0.94019, 0.92404, 0.92412, 0.90251, 0.90264, 0.86845, 0.86891, 0.81871, 0.81975])**(0.5)
            # wan_t2v.model.__class__.ratio = [0.99661, 0.99658, 0.99482, 0.99476, 0.99467, 0.99452, 0.99662, 0.99657, 0.99434, 0.99435, 0.99535, 0.99543, 0.99459, 0.99454, 0.99442, 0.99439, 0.99519, 0.99517, 0.99387, 0.99383, 0.99396, 0.99396, 0.99511, 0.99512, 0.99407, 0.99407, 0.9943, 0.99423, 0.99346, 0.99343, 0.99416, 0.99414, 0.99273, 0.99271, 0.99315, 0.99313, 0.99215, 0.99214, 0.99218, 0.99217, 0.99218, 0.99216, 0.99164, 0.9916, 0.99138, 0.99137, 0.98984, 0.9898, 0.99001, 0.98998, 0.98871, 0.98869, 0.98772, 0.9877, 0.98772, 0.98769, 0.98576, 0.98572, 0.98498, 0.98495, 0.98378, 0.98376, 0.98174, 0.98173, 0.98034, 0.9803, 0.97679, 0.97675, 0.97546, 0.97546, 0.97188, 0.97187, 0.96713, 0.96711, 0.9635, 0.96348, 0.95627, 0.95625, 0.9493, 0.94928, 0.93964, 0.93963, 0.92513, 0.92505, 0.90709, 0.90691, 0.87981, 0.8797, 0.86091, 0.86164]
            wan_t2v.model.__class__.residual_cache = None
            wan_t2v.model.__class__.residual_cos = []
            wan_t2v.model.__class__.residual_std = []
            wan_t2v.model.__class__.residual_ratio = []
            wan_t2v.model.__class__.accumulated_sim = [1,1]
            wan_t2v.model.__class__.accumulated_steps = [0,0]
    
        logging.info(
            f"Generating {'image' if 't2i' in args.task else 'video'} ...")
        
        if prompt_list is not None:
            start_time = time.time()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for ind, prompt_str in enumerate(tqdm(prompt_list)):
                video = wan_t2v.generate(
                    prompt_str,
                    size=SIZE_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model)
                
                save_name = f"{save_dir}/{prompt_str}-0.mp4"
                cache_video(
                    tensor=video[None],
                    save_file=save_name,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
            end_time = time.time()
            print("generate_total_Time:", end_time-start_time)
        else:
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

    if rank == 0 and prompt_list is None:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

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
    prompt_list = read_prompt_list("../../vbench/VBench_full_info.json")
    st_index = args.start_index
    end_index = args.end_index
    prompt_list = prompt_list[st_index:end_index]
    generate(args, prompt_list, args.save_dir)
  
def push_tensor_roll(queue, new_tensor):
    """
    queue: (b, s, d, k)
    new_tensor: (b, s, d)

    Uses torch.roll to shift the queue and insert the new tensor.
    """
    # Roll the queue by -1 along the last dimension (shifts elements towards index 0)
    # queue[1] goes to queue[0], queue[2] to queue[1], ..., queue[k-1] to queue[k-2]
    # The last element (queue[k-1]) is moved to queue[k-2] in the rolled tensor,
    # effectively making the original queue[0] disappear.
    if new_tensor is None:
        return queue
    rolled_queue = torch.roll(queue, shifts=-1, dims=-1)

    # Place the new tensor in the last position
    rolled_queue[..., -1] = new_tensor

    return rolled_queue