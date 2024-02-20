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
                    "batch_size": "params:model_params_obs_dropout.batch_size",
                    "max_position": "params:model_params_obs_dropout.max_position",
                    "learning_rate": "params:model_params_obs_dropout.learning_rate",
                    "weight_decay": "params:model_params_obs_dropout.weight_decay",
                    "max_epochs": "params:model_params_obs_dropout.max_epochs",
                    "k": "params:k",
                    "dropout_prob": "params:model_params_obs_dropout.dropout_prob",
                },
                outputs=[
                    "preprocessor_obs_dropout",
                    "checkpoint_path_obs_dropout",
                ],
                name="train_obs_dropout_node",
            ),
            node(
                func=predict_by_two_tower,
                inputs=[
                    "preprocessor_obs_dropout",
                    "checkpoint_path_obs_dropout",
                    "web30k_fold1_test_preprocessed",
                ],
                outputs="predictions_obs_dropout",
                name="predict_obs_dropout_node",
            ),
            node(
                func=evaluate,
                inputs=["predictions_obs_dropout", "params:k"],
                outputs="metrics_obs_dropout",
                name="evaluate_obs_dropout_node",
            ),
        ]
    )

    pipelines = [
        pipeline(
            base_pipelie,
            inputs={"web30k_fold1_test_preprocessed"},
            parameters={
                "params:model_params_obs_dropout.batch_size",
                "params:model_params_obs_dropout.max_position",
                "params:model_params_obs_dropout.learning_rate",
                "params:model_params_obs_dropout.weight_decay",
                "params:model_params_obs_dropout.max_epochs",
                "params:k",
                "params:model_params_obs_dropout.dropout_prob",
            },
            namespace=weight,
        )
        for weight in ["w0", "w20", "w60", "w80", "w100"]
    ]

    return sum(pipelines)
