<!-- ## **MagCache4FLUX** -->
# MagCache4FLUX-Kontext

[MagCache](https://github.com/Zehong-Ma/MagCache) can speedup [FLUX-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) 2.0x without much visual quality degradation, in a training-free manner.

![Image](https://github.com/user-attachments/assets/79d5f654-5828-442d-b1a1-9b754c17e457)


## 📈 Inference Latency Comparisons on a Single L20 GPU


|      FLUX.1 Kontext [dev]       |       MagCache (E005K4R02)  |   
|:-----------------------:|:--------------------:|
|         ~30 s           |     ~15 s   <b>2.0x</b> speedup          |


## Usage

Please try MagCache-Flux-Kontext in [ComfyUI-MagCache](https://github.com/Zehong-Ma/ComfyUI-MagCache). You can modify the '`magcache_thresh`', '`magcache_K`', and '`retention_ratio`' to obtain your desired trade-off between latency and visul quality.

## Citation
If you find MagCache is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```
@misc{ma2025magcachefastvideogeneration,
      title={MagCache: Fast Video Generation with Magnitude-Aware Cache}, 
      author={Zehong Ma and Longhui Wei and Feng Wang and Shiliang Zhang and Qi Tian},
      year={2025},
      eprint={2506.09045},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.09045}, 
}
```

## Acknowledgements

We would like to thank the contributors to the [FLUX](https://github.com/black-forest-labs/flux), [TeaCache](https://github.com/ali-vilab/TeaCache), and [Diffusers](https://github.com/huggingface/diffusers).