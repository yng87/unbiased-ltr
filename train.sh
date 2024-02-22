#!/bin/bash
set -e

models=(
    "linear"
    "debiased_linear"
    "xgboost"
    "debiased_xgboost"
    "single_tower"
    "two_tower"
    "obs_dropout"
    "grad_rev"
)
oracle_weights=(
    "w0"
    "w20"
    "w60"
    "w80"
    "w100"
)

for model in "${models[@]}"; do
    for weight in "${oracle_weights[@]}"; do
        rye run kedro run -p "$model" --namespace="$weight"
    done
done
