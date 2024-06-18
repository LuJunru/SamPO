# SamPO
We provide codes and models for SamPO in this repository. Please refer to the paper for details: [Eliminating Biased Length Reliance of Direct Preference Optimization via Down-Sampled KL Divergence](https://arxiv.org/abs/2406.10957). In short, we suggest that the discrepancy between sequence-level KL divergences between chosen and rejected sequences, used in DPO, results in overestimated or underestimated rewards due to varying token lengths, leading to the verbosity issue. We then introduce an effective downsampling regularization approach, named SamPO.

<img src="./introduction.png" width="988px"></img>

## Environment
We provide [requirements.txt](requirements.txt) for your convenience.

## Key Difference between SamPO and DPO
A quick check of the [Key Difference](https://github.com/LuJunru/SamPO/blob/main/dpo_trainer.py#L1023-L1051).

## Fine-tuning
```
Run `bash tasks.sh` for all DPO and all variants, including our SamPO.
```

## Evaluation
For five conditional benchmarks, we use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):
- GSM8K: 8-shot, report strict match
- IFEval: 3-shot, report instruction-level strict accuracy
- PiQA: 3-shot, report accuracy
- MMLU: 0-shot, report normalized accuracy
- TruthfulQA: 3-shot, report accuracy of single-true mc1 setting

For AlpacaEval2, we use [official alpaca_eval](https://github.com/tatsu-lab/alpaca_eval):
- AlpacaEval2: win rate (%)
- LC AlpacaEval2: length-debiased win rate (%) of AlpacaEval2

For HH-RLHF & TL;DR, we use the same GPT-4 Win rate prompt template proposed by the [DPO](https://arxiv.org/abs/2305.18290): 
- Win rate (%): a win rate between fine-tune models vs. SFT basis

## Model Weights & Performance

| Name | Share Link |
| --- | --- |
| Pythia-2.8B-HH-RLHF-Iterative-SamPO | [HF Link](https://huggingface.co/robinlee99/Pythia-2.8B-HH-RLHF-Iterative-SamPO) |
| Pythia-2.8B-TLDR-Iterative-SamPO | [HF Link](https://huggingface.co/robinlee99/Pythia-2.8B-TLDR-Iterative-SamPO) |
| Llama-3-8B-Instruct-Iterative-SamPO | [HF Link](https://huggingface.co/Junrulu/Llama-3-8B-Instruct-Iterative-SamPO) |

Note: test sets of HH-RLHF and TLDR are released in the above link as well.

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
