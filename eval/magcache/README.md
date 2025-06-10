## Installation

Prerequisites:

- Python >= 3.10
- PyTorch >= 1.13 (We recommend to use a >2.0 version)
- CUDA >= 11.6

We strongly recommend using Anaconda to create a new environment (Python >= 3.10) to run our examples:

```shell
conda create -n magcache python=3.10 -y
conda activate magcache
```

Install MagCache:

```shell
git clone https://github.com/Zehong-Ma/MagCache
cd MagCache
pip install -e .
```
## Evaluation of MagCache

We first generate videos according to VBench's prompts.

And then calculate PSNR, LPIPS and SSIM based on the video generated.

1. Generate video
```
cd eval/magcache
# modify the hyper-parameters in line 420-422
python experiments/opensora.py
```
2. Calculate metrics
```
# these metrics are calculated compared with original model
# gt video is the video of original model
# generated video is our methods's results
python common_metrics/eval.py --gt_video_dir aa --generated_video_dir bb
```

## Demo of MagCache
- **OpenSora**: 
  Set `magcache_thresh=0.12`, `magcache_K=3`, `retention_ratio=0.2` to achieve the latency 21.21s. Set `magcache_thresh=0.24`, `magcache_K=5`, `retention_ratio=0.2` to achieve the latency 16.86s. The `teacache_thresh` is set to 0.2 to achieve the latency 21.67s.

<div align="center">
  <video src="https://github.com/user-attachments/assets/fbb444db-320e-4108-9410-9e36cd6b0b0e" width="100%" poster=""> </video>
</div>
<div class="content has-text-centered">
  <img src="../../assets/OpenSora_header.jpg" style="width: 100%"><br>
</div>
<details style="width: 100%; margin: auto;">
<summary>Prompt: A tranquil tableau of an ornate Victorian streetlamp....</summary>
A tranquil tableau of an ornate Victorian streetlamp standing on a cobblestone street corner, illuminating the empty night
</details>



## Citation
If you find MagCache is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

<!-- ```
@article{liu2024timestep,
  title={Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model},
  author={Liu, Feng and Zhang, Shiwei and Wang, Xiaofeng and Wei, Yujie and Qiu, Haonan and Zhao, Yuzhong and Zhang, Yingya and Ye, Qixiang and Wan, Fang},
  journal={arXiv preprint arXiv:2411.19108},
  year={2024}
}
``` -->

## Acknowledgements
We would like to thank the contributors to the [Open-Sora](https://github.com/hpcaitech/Open-Sora), [TeaCache](https://github.com/ali-vilab/TeaCache), and [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys).
