# SamPO
[Eliminating Biased Length Reliance of Direct Preference Optimization via Down-Sampled KL Divergence](https://arxiv.org/abs/2406.10957)

## Environment
We provide [requirements.txt](requirements.txt) for your convenience.

## Fine-tuning
```
Run `bash tasks.sh` for all DPO and all variants, including our SamPO.
```

## Model Weights & Performance

| Name | Share Link |
| --- | --- |
| Pythia-2.8B-HH-RLHF-Iterative-SamPO | [HF Link](https://huggingface.co/robinlee99/Pythia-2.8B-HH-RLHF-Iterative-SamPO) |
| Pythia-2.8B-TLDR-Iterative-SamPO | [HF Link](https://huggingface.co/robinlee99/Pythia-2.8B-TLDR-Iterative-SamPO) |
| Llama-3-8B-Instruct-Iterative-SamPO | [HF Link](https://huggingface.co/Junrulu/Llama-3-8B-Instruct-Iterative-SamPO) |

## Acknowledgment
This code is built upon the [TRL](https://github.com/huggingface/trl) repository.

## Citation
```
@article{LUandLI2024SamPO,
  title={Eliminating Biased Length Reliance of Direct Preference Optimization via Down-Sampled KL Divergence},
  author={Lu, Junru and Li, Jiazheng and An, Siyu and Zhao, Meng and He, Yulan and Yin, Di and Sun, Xing},
  journal={arXiv preprint arXiv:2406.10957},
  year={2024}
}
```
