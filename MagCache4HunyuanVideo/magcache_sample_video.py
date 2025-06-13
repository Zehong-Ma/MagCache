import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

from hyvideo.modules.modulate_layers import modulate
from hyvideo.modules.attenion import attention, parallel_attention, get_cu_seqlens
from typing import Any, List, Tuple, Optional, Union, Dict
import torch
import json
import numpy as np
import torch.nn.functional as F
import argparse

def magcache_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        skip_forward = False
        
        if self.cnt>=int(self.retention_ratio*self.num_steps):
            cur_mag_ratio = self.mag_ratios[self.cnt]
            self.accumulated_ratio = self.accumulated_ratio*cur_mag_ratio
            cur_skip_err = np.abs(1-self.accumulated_ratio)
            self.accumulated_err += cur_skip_err
            self.accumulated_steps += 1
            if self.accumulated_err<=self.magcache_thresh and self.accumulated_steps<=self.K:
                cur_residual = self.residual_cache
                skip_forward = True
            else:
                self.accumulated_ratio = 1.0
                self.accumulated_steps = 0
                self.accumulated_err = 0
        if skip_forward:
            img = img + cur_residual
        else:
            ori_img = img
            # --------------------- Pass through DiT blocks ------------------------
            for _, block in enumerate(self.double_blocks):
                double_block_args = [
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                ]

                img, txt = block(*double_block_args)

            # Merge txt and img to pass through single stream blocks.
            x = torch.cat((img, txt), 1)
            if len(self.single_blocks) > 0:
                for _, block in enumerate(self.single_blocks):
                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                    ]

                    x = block(*single_block_args)

            img = x[:, :img_seq_len, ...]
            cur_residual = img - ori_img
        self.residual_cache = cur_residual
       
        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        
        self.cnt+=1
        if self.cnt >= self.num_steps:
            self.cnt = 0
            self.accumulated_ratio = 1.0
            self.accumulated_steps = 0
            self.accumulated_err = 0
        
        if return_dict:
            out["x"] = img
            return out
        return img

def magcache_calibration(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None 
        ori_img = img
        # --------------------- Pass through DiT blocks ------------------------
        for _, block in enumerate(self.double_blocks):
            double_block_args = [
                img,
                txt,
                vec,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis,
            ]

            img, txt = block(*double_block_args)

        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)
        if len(self.single_blocks) > 0:
            for _, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    (freqs_cos, freqs_sin),
                ]

                x = block(*single_block_args)

        img = x[:, :img_seq_len, ...]
        cur_residual = img - ori_img
        if self.cnt >= 1:
            norm_ratio = ((cur_residual.norm(dim=-1)/self.residual_cache.norm(dim=-1)).mean()).item()
            norm_std = (cur_residual.norm(dim=-1)/self.residual_cache.norm(dim=-1)).std().item()
            cos_dis = (1-F.cosine_similarity(cur_residual, self.residual_cache, dim=-1, eps=1e-8)).mean().item()
            self.norm_ratio.append(round(norm_ratio, 5))
            self.norm_std.append(round(norm_std, 5))
            self.cos_dis.append(round(cos_dis, 5))
            print(f"time: {self.cnt}, norm_ratio: {norm_ratio}, norm_std: {norm_std}, cos_dis: {cos_dis}")
        
        self.residual_cache = cur_residual
        if self.cnt >= 49:
            print("norm ratio")
            print(self.norm_ratio)
            print("norm std")
            print(self.norm_std)
            print("cos_dis")
            print(self.cos_dis)
        
        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        
        self.cnt+=1
        
        if return_dict:
            out["x"] = img
            return out
        return img

def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args
    
    # add magcache config here
    args.magcache_thresh = 0.24 # 0.12
    args.magcache_K = 6 # 4
    args.retention_ratio = 0.2
    
    hunyuan_video_sampler.pipeline.transformer.__class__.enable_teacache = True
    hunyuan_video_sampler.pipeline.transformer.__class__.cnt = 0
    hunyuan_video_sampler.pipeline.transformer.__class__.num_steps = args.infer_steps
    hunyuan_video_sampler.pipeline.transformer.__class__.magcache_thresh = args.magcache_thresh
    hunyuan_video_sampler.pipeline.transformer.__class__.K = args.magcache_K
    hunyuan_video_sampler.pipeline.transformer.__class__.norm_ratio = []
    hunyuan_video_sampler.pipeline.transformer.__class__.norm_std = []
    hunyuan_video_sampler.pipeline.transformer.__class__.cos_dis = []
    hunyuan_video_sampler.pipeline.transformer.__class__.residual_cache = None
    if args.video_size[0]==720:
        hunyuan_video_sampler.pipeline.transformer.__class__.mag_ratios = np.array([1.0]+[1.0754, 1.27807, 1.11596, 1.09504, 1.05188, 1.00844, 1.05779, 1.00657, 1.04142, 1.03101, 1.00679, 1.02556, 1.00908, 1.06949, 1.05438, 1.02214, 1.02321, 1.03019, 1.00779, 1.03381, 1.01886, 1.01161, 1.02968, 1.00544, 1.02822, 1.00689, 1.02119, 1.0105, 1.01044, 1.01572, 1.02972, 1.0094, 1.02368, 1.0226, 0.98965, 1.01588, 1.02146, 1.0018, 1.01687, 0.99436, 1.00283, 1.01139, 0.97122, 0.98251, 0.94513, 0.97656, 0.90943, 0.85703, 0.75456])
    if args.video_size[0]==544:
        hunyuan_video_sampler.pipeline.transformer.__class__.mag_ratios = np.array([1.0]+[1.06971, 1.29073, 1.11245, 1.09596, 1.05233, 1.01415, 1.05672, 1.00848, 1.03632, 1.02974, 1.00984, 1.03028, 1.00681, 1.06614, 1.05022, 1.02592, 1.01776, 1.02985, 1.00726, 1.03727, 1.01502, 1.00992, 1.03371, 0.9976, 1.02742, 1.0093, 1.01869, 1.00815, 1.01461, 1.01152, 1.03082, 1.0061, 1.02162, 1.01999, 0.99063, 1.01186, 1.0217, 0.99947, 1.01711, 0.9904, 1.00258, 1.00878, 0.97039, 0.97686, 0.94315, 0.97728, 0.91154, 0.86139, 0.76592])
    hunyuan_video_sampler.pipeline.transformer.__class__.retention_ratio = args.retention_ratio
    hunyuan_video_sampler.pipeline.transformer.__class__.forward = magcache_forward # change to `magcache_calibration` when calibrating with your custom prompt and model_type 
    hunyuan_video_sampler.pipeline.transformer.__class__.accumulated_ratio = 1
    hunyuan_video_sampler.pipeline.transformer.__class__.accumulated_err = 0
    hunyuan_video_sampler.pipeline.transformer.__class__.accumulated_steps = 0
    
    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    samples = outputs['samples']
    
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{save_path}/magcache_{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f'Sample save to: {save_path}')

if __name__ == "__main__":
    main()
