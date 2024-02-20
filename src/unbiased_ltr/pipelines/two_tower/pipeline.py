from common_nodes import evaluate, predict_by_two_tower
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train


def create_pipeline(**kwargs) -> Pipeline:
    base_pipelie = Pipeline(
        [
            node(
                func=train,
                inputs={
                    "df_train": "web30k_synthetic_click_train",
                    "df_val": "web30k_synthetic_click_vali",
                    "batch_size": "params:model_params_two_tower.batch_size",
                    "max_position": "params:model_params_two_tower.max_position",
                    "learning_rate": "params:model_params_two_tower.learning_rate",
                    "weight_decay": "params:model_params_two_tower.weight_decay",
                    "max_epochs": "params:model_params_two_tower.max_epochs",
                    "k": "params:k",
                },
                outputs=[
                    "preprocessor_two_tower",
                    "checkpoint_path_two_tower",
                ],
                name="train_two_tower_node",
            ),
            node(
                func=predict_by_two_tower,
                inputs=[
                    "preprocessor_two_tower",
                    "checkpoint_path_two_tower",
                    "web30k_fold1_test_preprocessed",
                ],
                outputs="predictions_two_tower",
                name="predict_two_tower_node",
            ),
            node(
                func=evaluate,
                inputs=["predictions_two_tower", "params:k"],
                outputs="metrics_two_tower",
                name="evaluate_two_tower_node",
            ),
        ]
    )

    pipelines = [
        pipeline(
            base_pipelie,
            inputs={"web30k_fold1_test_preprocessed"},
            parameters={
                "params:model_params_two_tower.batch_size",
                "params:model_params_two_tower.max_position",
                "params:model_params_two_tower.learning_rate",
                "params:model_params_two_tower.weight_decay",
                "params:model_params_two_tower.max_epochs",
                "params:k",
            },
            namespace=weight,
        )
        for weight in ["w0", "w20", "w60", "w80", "w100"]
    ]

    return sum(pipelines)
