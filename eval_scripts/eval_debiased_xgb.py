from evaluator import Evaluator
from models.xgboost import PositionDebiasedXGBoostModel


def main():
    evaluator = Evaluator(
        negative_down_sample_ratio=0.1, save_path="results/debiased_xgb.json"
    )

    model = PositionDebiasedXGBoostModel()

    evaluator.evaluate(model)


if __name__ == "__main__":
    main()
