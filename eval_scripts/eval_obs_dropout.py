from evaluator import Evaluator
from models.nn import NNModel


def main():
    evaluator = Evaluator(
        negative_down_sample_ratio=0.1, save_path="results/obs_dropout.json"
    )

    model = NNModel(
        model_name="observation_dropout",
        model_params={
            "n_features": 136,
            "max_position": 120,
            "dropout_prob": 0.2,
        },
        epochs=30,
        learning_rate=0.001,
        weight_decay=1e-5,
    )

    evaluator.evaluate(model)


if __name__ == "__main__":
    main()
