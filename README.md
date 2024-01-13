# Reproduction scripts for unbiased learning-to-rank methods

Based on [Towards Disentangling Relevance and Bias in Unbiased Learning to Rank](https://arxiv.org/abs/2212.13937)

## Dependencies
- numpy==1.26.2
- scikit-learn==1.3.2
- torch==2.1.2
- xgboost==2.0.3

Scripts are developed on Mac (M1, 8GB).

## Dataset generation
Download [MSLR-WEB30K](https://www.microsoft.com/en-us/research/project/mslr/) dataset and unzip Fold1 into `datasets/MSLR-WEB30K-Fold1/` directory.

To generate synthetic click dataset
```
python3 src/datasets.py
```

## Evaluate
Evaluation scripts are in `eval_scripts` directory. To use it, specify module path as
```
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
python3 eval_scripts/eval_linear.py
```

## Results
### NDCG@5 for ground truth label

| $w$ |   Linear |   Debiased Linear |   XGBoost |   Debiased XGBoost |   Single Tower |   PAL (additive) |   Observation Dropout |   Gradient Reversal |
|-|-|-|-|-|-|-|-|-|
| 1   | 0.2933 | 0.3242 | 0.3059 | 0.1712 | 0.3220 | 0.3465 | 0.3532 | **0.3600** |
| 0.8 | 0.3009 | 0.3270 | 0.3044 | 0.2763 | 0.3068 | **0.3646** | 0.3548 | 0.3479 |
| 0.6 | 0.2945 | 0.3251 | 0.2915 | 0.1829 | 0.3110 | 0.3520 | 0.3519 | **0.3551** |
| 0.2 | 0.2828 | 0.3134 | 0.3189 | 0.1592 | 0.2743 | 0.3242 | 0.3358 | **0.3364** |
| 0   | 0.2787 | 0.3145 | 0.2924 | 0.1595 | 0.2628 | 0.3239 | **0.3355** | 0.3254 |
