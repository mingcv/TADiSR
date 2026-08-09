---
language:
- en
- zh
license: other
tags:
- image-super-resolution
- diffusion
- text-super-resolution
- text-segmentation
pipeline_tag: image-to-image
---

# TADiSR Models

This repository hosts adapter checkpoints for **TADiSR: Text-Aware Real-World
Image Super-Resolution via Diffusion Model with Joint Segmentation Decoders**.
Each checkpoint contains LoRA parameters for the diffusion backbone, VAE skip
adapters, and the jointly trained text-segmentation decoder. It is not a
standalone base model.

## Models

| File | Base model | Training data | SHA256 |
| --- | --- | --- | --- |
| `tadisr_cogview4_realce_17500.pkl` | `zai-org/CogView4-6B` | FTSR + Real-CE | `7cec8d0917305037cbafcb1b1bf81062add1613b561f5d7e2d1e599fe0baf5e0` |
| `tadisr_kolors_ftsr_526000.pkl` | `Kwai-Kolors/Kolors` | FTSR | `4baef8b1fad319caf7d606c82539035c6fc9483c81f94601377c64813251e7f7` |

The CogView4 release is recommended for Chinese text. See the
[code repository](https://github.com/mingcv/TADiSR) for installation,
checkpoint validation, and inference.

## License

The adapter checkpoints are released for research use. They remain subject to
the terms of their respective base models, including the commercial-use terms
of Kolors. Do not treat this model card as a replacement for the base model
licenses.

## Citation

```bibtex
@article{hu2025tadisr,
  title={Text-Aware Real-World Image Super-Resolution via Diffusion Model with Joint Segmentation Decoders},
  author={Hu, Qiming and Fan, Linlong and Luo, Yiyan and Yu, Yuhang and Guo, Xiaojie and Fan, Qingnan},
  journal={arXiv preprint arXiv:2506.04641},
  year={2025}
}
```
