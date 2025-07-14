<!-- ## **MagCache4OmniGen2** -->
# MagCache4OmniGen2

[MagCache](https://github.com/Zehong-Ma/MagCache) can speedup [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) 1.8x without much visual quality degradation, in a training-free manner.

## 📈 Inference Latency Comparisons on a Single L20

![Image](https://github.com/user-attachments/assets/12d76948-8730-401e-bcda-e69cfbcd6154)

![Image](https://github.com/user-attachments/assets/165ef220-5fd6-4f2e-9f96-101ff5d9aaa9)

![Image](https://github.com/user-attachments/assets/77f3a1ce-e0a2-4b12-bf92-a16a69756d2f)



## Installation

```shell
git clone git@github.com:VectorSpaceLab/OmniGen2.git
cd OmniGen2
# ... install the enviroment required by OmniGen2
cp /path/to/MagCache4OmniGen2/magcache -r ./omnigen2/
cp /path/to/MagCache4OmniGen2/inference.py ./
cp /path/to/MagCache4OmniGen2/magcache_eval.sh ./


```

## Usage

You can modify the '`magcache_thresh`' to obtain your desired trade-off between latency and visul quality. For single-gpu inference, you can use the provided script:

```bash
sh magcache_eval.sh
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

We would like to thank the contributors to the [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) and [Diffusers](https://github.com/huggingface/diffusers).