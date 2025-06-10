# MagCache: Fast Video Generation with Magnitude-Aware Cache

<div class="is-size-5 publication-authors">
            <span class="author-block">
              <a href="https://zehong-ma.github.io/" target="_blank">Zehong Ma</a><sup>1,2</sup><sup>†</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://joinwei-pku.github.io/longhuiwei.github.io/" target="_blank">Longhui Wei</a><sup>2</sup><sup>‡</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://scholar.google.com/citations?user=bKG4Un8AAAAJ&hl=zh-CN" target="_blank">Feng Wang</a><sup>2</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://www.pkuvmc.com/" target="_blank">Shiliang Zhang</a><sup>1</sup><sup>*</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://www.qitian1987.com/" target="_blank">Qi Tian</a><sup>2</sup><sup>*</sup>
            </span>
          </div>

  <div class="is-size-5 publication-authors">
    <span class="author-block"><sup>1</sup>Peking University,&nbsp;</span>
    <span class="author-block"><sup>2</sup>Huawei Inc.</span>
  </div>


  <div class="is-size-5 publication-authors">
    († Work was done during internship at Huawei Inc., ‡ Project Leader. * Corresponding author.)
  </div>

<h5 align="center">

<!-- [![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](---) -->
<!-- [![arXiv](https://img.shields.io/badge/Arxiv-2411.19108-b31b1b.svg?logo=arXiv)](---)  -->
[![Home Page](https://img.shields.io/badge/Project-<Website>-blue.svg)](https://zehong-ma.github.io/MagCache/) 
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](./LICENSE) 
[![github](https://img.shields.io/github/stars/Zehong-Ma/MagCache.svg?style=social)](https://github.com/Zehong-Ma/MagCache/)

</h5>


<!-- ![visualization](./docs/static/images/intro_demo_image.png) -->
<div class="content has-text-centered">
          <img src="./docs/static/images/intro_demo_image.png" style="width: 100%;"><br>
          <span style="font-size: 0.8em; width: 100%; display: inline-block;">Figure 1: Visualization of MagCache on Open-Sora and Wan2.1. MagCache can achieve better visual quality with a comparable or less latency. More demos are shown in the subfolders of each model.</span>
        </div>

## 🫖 Introduction 
We introduce Magnitude-aware Cache (MagCache), a training-free caching approach that estimates and leverages the fluctuating differences among model outputs across timesteps based on the robust magnitude observations, thereby accelerating the inference. MagCache works well for Video Diffusion Models, Image Diffusion models. For more details and results, please visit our [project page](https://zehong-ma.github.io/MagCache/).

<div class="content has-text-centered">
            <img src="./docs/static/images/final_intro.png" style="width: 95%;"><br>
            <span style="font-size: 0.8em; width: 100%; display: inline-block;">Figure 2: Relationships between output residuals across diffusion timesteps. Magnitude ratio serves as both an accurate and stable criterion for measuring the difference between residuals.</span>
          </div>

## 🔥 Latest News 
- **If you like our project, please give us a star ⭐ on GitHub for the latest update.**

- [2025/6/10] 🔥 Support [Wan2.1](https://github.com/Wan-Video/Wan2.1), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [FLUX](https://github.com/black-forest-labs/flux), [Open-Sora](https://github.com/hpcaitech/Open-Sora).
- [2025/6/10] 🎉 Release the [code](https://github.com/LiewFeng/MagCache) of MagCache.
- [2025/6/10] 🎉 Release the [paper](https://arxiv.org/abs/2411.19108) of MagCache.

## 🧩 Community Contributions  
If you develop/use MagCache in your projects and you would like more people to see it, please inform us.(zehongma@stu.pku.edu.cn)
  
## 🎉 Supported Models 
**Text to Video**
- [MagCache4Wan2.1](./MagCache4Wan2.1/README.md)
- [MagCache4HunyuanVideo](./MagCache4HunyuanVideo/README.md)
- [MagCache4Open-Sora](./eval/MagCache/README.md)

 **Image to Video** 
- [MagCache4Wan2.1](./MagCache4Wan2.1/README.md)

 **Text to Image**
- [MagCache4FLUX](./MagCache4FLUX/README.md)
- [MagCache4FLUX-schnell](./MagCache4FLUX_schnell/README.md)

 **Demos**

|                            |                            |                            |
|:--------------------------:|:--------------------------:|:--------------------------:|
| **HunyuanVideo T2V** <br> *(54min05s)* | **TeaCache** <br> *(23min49s)*  <br> PSNR: 22.80, 2.3× speedup | **MagCache** <br> *(19min33s)*  <br> PSNR: 26.76, **2.8×** speedup |

<video class="video" autoplay controls muted loop playsinline>
  <source src="https://private-user-images.githubusercontent.com/80336167/453056257-a3cc8f04-9d1a-4517-98b9-55a2438a2f5a.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDk0ODEyMjQsIm5iZiI6MTc0OTQ4MDkyNCwicGF0aCI6Ii84MDMzNjE2Ny80NTMwNTYyNTctYTNjYzhmMDQtOWQxYS00NTE3LTk4YjktNTVhMjQzOGEyZjVhLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA2MDklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNjA5VDE0NTUyNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNhMDhjZGNkODQ1MTc1NmQ0NzViYzhkNTZlMTRhYWJlYTM5Nzk1YzY3ODRjZDhiMjJhZjM4NTFiNWE1ZGUyYmUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.cxVkj1E4cC1FHCvg3mnfrRj88AA6MvNmWZ7ZuLocKdM" type="video/mp4">
</video>
<p align="center" style="font-size:1.25em; margin:0.5em 0;">
HunyuanVideo T2V, 5s, 720P.
</p>

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; place-items: center; font-size: 1.0em;">
      <div style="text-align: center;">Wan2.1 14B I2V (30min40s)</div>
      <div style="text-align: center;">TeaCache (13min04s) <br> PSNR: 13.67, 2.3x speedup</div>
      <div style="text-align: center;">MagCache (10min03s) <br> PSNR: 23.67, <b>3.0x</b> speedup</div>
      <!-- <div style="text-align: center;">MagCache (--) <br> PSNR: 22.58, 3.0x speedup</div> -->
    </div>
    <video class="video" autoplay controls muted loop playsinline>
      <source src="https://private-user-images.githubusercontent.com/80336167/453056422-7800305a-1efd-49e6-8087-00c583b171a3.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDk0ODEyMjQsIm5iZiI6MTc0OTQ4MDkyNCwicGF0aCI6Ii84MDMzNjE2Ny80NTMwNTY0MjItNzgwMDMwNWEtMWVmZC00OWU2LTgwODctMDBjNTgzYjE3MWEzLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA2MDklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNjA5VDE0NTUyNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQ5MGIyNDVjMDFhYTM3NGQzOGM2YTNmMDJlNmNhMWFlYTIzNjViMGI0NmI0MjRiMTMyOTI1ZTVlZmRkYTdlNmQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.PYlR4DMMMBOnUHv0p8RvkO4xzfF7TIjpQJeP_nSjIyM" type="video/mp4">
    </video>
    <p align="center" style="font-size:1.25em; margin:0.5em 0;">
    Wan2.1 I2V, 3s, 720P.
    </p>

  <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; place-items: center; font-size: 1.0em;">
    <div style="text-align: center;">Wan2.1 14B T2V (60min04s)</div>
    <div style="text-align: center;">TeaCache (30min01s) <br> PSNR: 17.39, 2.0x speedup</div>
    <div style="text-align: center;">MagCache (21min40s) <br> PSNR: 24.39, <b>2.8x</b> speedup</div>
  </div>
  <video class="video" controls>
    <source src="https://private-user-images.githubusercontent.com/80336167/453056581-dedd4ba3-3bc2-4d3f-9f6a-a70d14047ab3.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDk0ODEyMjQsIm5iZiI6MTc0OTQ4MDkyNCwicGF0aCI6Ii84MDMzNjE2Ny80NTMwNTY1ODEtZGVkZDRiYTMtM2JjMi00ZDNmLTlmNmEtYTcwZDE0MDQ3YWIzLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA2MDklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNjA5VDE0NTUyNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTgwMTBmYzVjYjlkYzk4YTI1ZjBlZjlkZmEyMzAwYjA4ZDAwYzRlMmYwYjQ5M2M3YzBhZmZhYmZlODEzYzc2ZjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.6p9dN0elYJIErZEwxszMSzcC4kDYsZgMrYBrR0RfkjY" type="video/mp4">
  </video>
  <p align="center" style="font-size:1.25em; margin:0.5em 0;">
    Wan2.1 14B T2V, 5s, 720P.
  </p>
  

  <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; place-items: center; font-size: 1.0em;">
    <div style="text-align: center;">Wan2.1 14B T2V (60min04s)</div>
    <div style="text-align: center;">TeaCache (30min01s), <br> PSNR: 14.32, 2.0x speedup</div>
    <div style="text-align: center;">MagCache (21min40s), <br> PSNR: 21.82, <b>2.8x</b> speedup</div>
  </div>
  <video class="video" autoplay controls muted loop playsinline>
    <source src="https://private-user-images.githubusercontent.com/80336167/453056622-66a5b5e4-d2e7-42ec-aef3-ae8a74d229a9.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDk0ODE0OTYsIm5iZiI6MTc0OTQ4MTE5NiwicGF0aCI6Ii84MDMzNjE2Ny80NTMwNTY2MjItNjZhNWI1ZTQtZDJlNy00MmVjLWFlZjMtYWU4YTc0ZDIyOWE5Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA2MDklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNjA5VDE0NTk1NlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTY0NjdiNWJiZGUzZGI3MWM1OGYyNzkyNDNjMmY5OTIyNzhmYzA0MjNkYzFiZDVlNDNlOGFmYjU5ODViNTk3M2QmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.fWgQViCpMxhxef8Tqadqib9nhpBUTGMuAbKrRJGTYBY" type="video/mp4">
  </video>
  <p align="center" style="font-size:1.25em; margin:0.5em 0;">
    Wan2.1 14B T2V, 5s, 720P.
  </p>

  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; place-items: center; font-size: 1.0em;">
    <div style="text-align: center;">Wan2.1 1.3B T2V (189s)</div>
    <div style="text-align: center;">TeaCache (95s) <br> PSNR: 14.86, 2.0x speedup</div>
    <div style="text-align: center;">MagCache (87s) <br> PRNR: 20.51, 2.2x speedup</div>
    <div style="text-align: center;">MagCache (68s) <br> PSNR: 18.93, <b>2.8x</b> speedup</div>
  </div>
  <video class="video" autoplay controls muted loop playsinline>
    <source src="https://github.com/user-attachments/assets/cd20fcc3-b5ff-4eb5-8d99-181a3f862b24" type="video/mp4">
  </video>
  <span style="font-size: 1.5em; width: 100%; display: inline-block; text-align: center;">Wan2.1 1.3B T2V, 5s, 480P.</span>

    
  <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; place-items: center; font-size: 1.0em;">
    <div style="text-align: center;">FLUX-dev (14.26s)</div>
    <div style="text-align: center;">TeaCache (5.65s), 2.5x</div>
    <div style="text-align: center;">MagCache (5.05s), 2.8x</div>
  </div>
  <div class="content has-text-centered">
    <img src="./docs/static/images/flux_dev_bicycle.png" style="width: 100%"><br>
    <span style="font-size: 1.5em; width: 100%; display: inline-block; text-align: center;">FLUX-dev, bicycle.</span>
  </div>

## 🤖 Instructions for Supporting Other Models 
- **Welcome for PRs to support other models.**
- Please implement the `magcache_calibration` function by referring the code in [MagCache4Wan2.1](./MagCache4Wan2.1/magcache_generate.py)
- Utilize a random prompt or input to generate the magnitude ratio.
- Implement the `magcache_forward` function by adding a few lines into the original forward function.
- Reuse the default hyper-parameters or set your own hyper-parameters to achieve a trade-off between latency and quality.

## 💐 Acknowledgement 

This repository is built based on [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys), [Diffusers](https://github.com/huggingface/diffusers), [Open-Sora](https://github.com/hpcaitech/Open-Sora), [FLUX](https://github.com/black-forest-labs/flux), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), and [TeaCache](https://github.com/ali-vilab/TeaCache). Thanks for their contributions!

## 🔒 License 

* The core code of this project is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file. The core code includes the calibration and forward code of MagCache.
* For [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys), [Diffusers](https://github.com/huggingface/diffusers), [Open-Sora](https://github.com/hpcaitech/Open-Sora), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [FLUX](https://github.com/black-forest-labs/flux), and [Wan2.1](https://github.com/Wan-Video/Wan2.1) please follow their LICENSE.

## 📖 Citation 
If you find MagCache is useful in your research or applications, please consider giving us a star ⭐ and citing it by the following BibTeX entry.

<!-- ```
@article{liu2024timestep,
  title={Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model},
  author={Liu, Feng and Zhang, Shiwei and Wang, Xiaofeng and Wei, Yujie and Qiu, Haonan and Zhao, Yuzhong and Zhang, Yingya and Ye, Qixiang and Wan, Fang},
  journal={arXiv preprint arXiv:2411.19108},
  year={2024}
}
``` -->


