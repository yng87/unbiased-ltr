# Reproduction scripts for unbiased learning-to-rank methods

Based on [Towards Disentangling Relevance and Bias in Unbiased Learning to Rank](https://arxiv.org/abs/2212.13937)

## Dependencies
- numpy==1.26.2
- scikit-learn==1.3.2
- torch==2.1.2
- xgboost==2.0.3

## Dataset generation
Download [MSLR-WEB30K](https://www.microsoft.com/en-us/research/project/mslr/) dataset and unzip Fold1 into `datasets/MSLR-WEB30K-Fold1/` directory.

To generate synthetic click dataset
```
python3 src/datasets.py
```

## Evaluate
Evaluation scripts are in `eval_scripts` directory. To use it, specify module path as
```
PYTHONPATH="$(pwd)/src:${PYTHONPATH}" python3 eval_scripts/eval_linear.py
```

## Results
### NDCG@5 for ground truth label

| Oracle weight | Linear | Debiased Linear | XGBoost | Debaised XGBoost | Single Tower | Two Tower | Observation Dropout | Gradient Reversal |
|-|-|-|-|-|-|-|-|-|
| 1.0 | 0.2952 | 0.3304 | 0.3095 | 0.1584 | 0.3104 | **0.3552** | 0.3360 | 0.3304 |
| 0.8 | 0.2873 | 0.3172 | 0.2865 | 0.1914 | 0.3423 | 0.3392 | 0.3395 | **0.3442** |
| 0.6 | 0.2812 | 0.3187 | 0.3062 | 0.1624 | 0.3091 | **0.3384** | 0.3366 | 0.3321 |
| 0.2 | 0.2817 | 0.3170 | 0.2766 | 0.1590 | 0.2749 | 0.3195 | 0.3192 | **0.3328** |
| 0.0 | 0.2740 | 0.3121 | 0.3006 | 0.1758 | 0.2867 | 0.3367 | 0.3353 | **0.3406** |
