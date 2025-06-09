## Installation

Prerequisites:

- Python >= 3.10
- PyTorch >= 1.13 (We recommend to use a >2.0 version)
- CUDA >= 11.6

We strongly recommend using Anaconda to create a new environment (Python >= 3.10) to run our examples:

```shell
conda create -n teacache python=3.10 -y
conda activate teacache
```

Install TeaCache:

```shell
git clone https://github.com/LiewFeng/TeaCache
cd TeaCache
pip install -e .
```


## Evaluation of TeaCache

We first generate videos according to VBench's prompts.

And then calculate Vbench, PSNR, LPIPS and SSIM based on the video generated.

1. Generate video
```
cd eval/teacache
python experiments/latte.py
python experiments/opensora.py
python experiments/open_sora_plan.py
python experiments/cogvideox.py
```

2. Calculate Vbench score
```
# vbench is calculated independently
# get scores for all metrics
python vbench/run_vbench.py --video_path aaa --save_path bbb
# calculate final score
python vbench/cal_vbench.py --score_dir bbb
```

3. Calculate other metrics
```
# these metrics are calculated compared with original model
# gt video is the video of original model
# generated video is our methods's results
python common_metrics/eval.py --gt_video_dir aa --generated_video_dir bb
```



## Citation
If you find TeaCache is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```
@article{liu2024timestep,
  title={Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model},
  author={Liu, Feng and Zhang, Shiwei and Wang, Xiaofeng and Wei, Yujie and Qiu, Haonan and Zhao, Yuzhong and Zhang, Yingya and Ye, Qixiang and Wan, Fang},
  journal={arXiv preprint arXiv:2411.19108},
  year={2024}
}
```

## Acknowledgements
We would like to thank the contributors to the [Open-Sora](https://github.com/hpcaitech/Open-Sora), [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [Latte](https://github.com/Vchitect/Latte), [CogVideoX](https://github.com/THUDM/CogVideo) and [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys).
