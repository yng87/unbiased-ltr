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
                    "batch_size": "params:model_params_grad_rev.batch_size",
                    "max_position": "params:model_params_grad_rev.max_position",
                    "learning_rate": "params:model_params_grad_rev.learning_rate",
                    "weight_decay": "params:model_params_grad_rev.weight_decay",
                    "max_epochs": "params:model_params_grad_rev.max_epochs",
                    "k": "params:k",
                    "grad_rev_scale": "params:model_params_grad_rev.grad_rev_scale",
                },
                outputs=[
                    "preprocessor_grad_rev",
                    "checkpoint_path_grad_rev",
                ],
                name="train_grad_rev_node",
            ),
            node(
                func=predict_by_two_tower,
                inputs=[
                    "preprocessor_grad_rev",
                    "checkpoint_path_grad_rev",
                    "web30k_fold1_test_preprocessed",
                    "params:model_params_grad_rev.grad_rev_scale",
                ],
                outputs="predictions_grad_rev",
                name="predict_grad_rev_node",
            ),
            node(
                func=evaluate,
                inputs=["predictions_grad_rev", "params:k"],
                outputs="metrics_grad_rev",
                name="evaluate_grad_rev_node",
            ),
        ]
    )

    pipelines = [
        pipeline(
            base_pipelie,
            inputs={"web30k_fold1_test_preprocessed"},
            parameters={
                "params:model_params_grad_rev.batch_size",
                "params:model_params_grad_rev.max_position",
                "params:model_params_grad_rev.learning_rate",
                "params:model_params_grad_rev.weight_decay",
                "params:model_params_grad_rev.max_epochs",
                "params:k",
                "params:model_params_grad_rev.grad_rev_scale",
            },
            namespace=weight,
        )
        for weight in ["w0", "w20", "w60", "w80", "w100"]
    ]

    return sum(pipelines)
