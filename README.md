# Reproduction scripts for unbiased learning-to-rank methods

Based on [Towards Disentangling Relevance and Bias in Unbiased Learning to Rank](https://arxiv.org/abs/2212.13937)

## Dependencies installation
Use `rye sync`

## Dataset generation
Download [MSLR-WEB30K](https://www.microsoft.com/en-us/research/project/mslr/) dataset and unzip Fold1 into `data/01_raw/MSLR-WEB30K-Fold1/` directory.

To generate synthetic click dataset
```
rye run kedro run -p data_processing
```

## Evaluate
```
./train.sh
```

## Results
### NDCG@5 for ground truth label

| $w$    |   linear |   debiased_linear |   xgboost |   debiased_xgboost_ipw |   lightgbm |   debiased_lightgbm |   single_tower |   two_tower |   obs_dropout |   grad_rev |
|:-----|---------:|------------------:|----------:|-----------------------:|-----------:|--------------------:|---------------:|------------:|--------------:|-----------:|
| w0   |   0.2787 |            0.3145 |    0.2683 |                 0.2881 |     0.2748 |              0.3115 |         0.2658 |      0.3465 |        0.2860 |     0.3382 |
| w20  |   0.2851 |            0.3157 |    0.2964 |                 0.2783 |     0.3047 |              0.3212 |         0.2983 |      0.3437 |        0.2984 |     0.3432 |
| w60  |   0.2935 |            0.3227 |    0.3057 |                 0.2963 |     0.3179 |              0.3362 |         0.2436 |      0.3255 |        0.3434 |     0.3341 |
| w80  |   0.2957 |            0.3246 |    0.3124 |                 0.2782 |     0.2997 |              0.3414 |         0.2924 |      0.3532 |        0.3163 |     0.3439 |
| w100 |   0.2880 |            0.3189 |    0.2927 |                 0.2905 |     0.3025 |              0.3414 |         0.3075 |      0.3401 |        0.2976 |     0.3399 |
