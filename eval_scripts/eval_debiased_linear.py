from evaluator import Evaluator
from models.linear import PositionDebiasedLinearModel


def main():
    evaluator = Evaluator(
        negative_down_sample_ratio=0.1,
        save_path="results/debiased_linear.json",
    )

    model = PositionDebiasedLinearModel(
        penalty="l2",
        C=1.0,
        max_iter=100,
        solver="newton-cholesky",
    )

    evaluator.evaluate(model)


if __name__ == "__main__":
    main()
