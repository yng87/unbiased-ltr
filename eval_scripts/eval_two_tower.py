from evaluator import Evaluator
from models.nn import NNModel


def main():
    evaluator = Evaluator(
        negative_down_sample_ratio=0.1, save_path="results/two_tower.json"
    )

    model = NNModel(
        model_name="two_tower",
        model_params={"n_features": 136, "max_position": 120},
        epochs=30,
        learning_rate=0.001,
        weight_decay=1e-5,
    )

    evaluator.evaluate(model)


if __name__ == "__main__":
    main()
