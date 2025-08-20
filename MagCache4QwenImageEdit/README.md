<!-- ## **MagCache4Qwen-Image** -->
# MagCache4Qwen-Image

[MagCache](https://github.com/Zehong-Ma/MagCache) can speedup [Qwen-Image-Edit](https://github.com/QwenLM/Qwen-Image) 1.5x without visual quality degradation, in a training-free manner.


## 📈 Inference Latency Comparisons on a Single A800 GPU


|      Qwen-Image [dev]       |       MagCache (E006K2R02)  |   
|:-----------------------:|:--------------------:|
|         ~120 s           |     ~80 s   <b>1.5x</b> speedup          |


## Usage

Please try MagCache-Qwen-Image. You can modify the '`magcache_thresh`', '`magcache_K`', and '`retention_ratio`' to obtain your desired trade-off between latency and visul quality. For diffusers inference, you can use the following command:

```bash
# please install the latest diffusers and transformers first.
python magcache_generate.py
```

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

We would like to thank the contributors to the [Qwen-Image-Edit](https://github.com/QwenLM/Qwen-Image) and [Diffusers](https://github.com/huggingface/diffusers).