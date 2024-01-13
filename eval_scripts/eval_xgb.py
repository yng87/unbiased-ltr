from evaluator import Evaluator
from models.xgboost import XGBoostModel


def main():
    evaluator = Evaluator(
        negative_down_sample_ratio=0.1, save_path="results/xgb.json"
    )

    model = XGBoostModel()

    evaluator.evaluate(model)


if __name__ == "__main__":
    main()
