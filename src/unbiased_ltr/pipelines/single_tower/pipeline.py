from kedro.pipeline import Pipeline, node, pipeline

from common_nodes import evaluate

from .nodes import predict, train


def create_pipeline(**kwargs) -> Pipeline:

    base_pipelie = Pipeline(
        [
            node(
                func=train,
                inputs=[
                    "web30k_synthetic_click_train",
                    "web30k_synthetic_click_vali",
                    "params:model_params_single_tower.batch_size",
                    "params:model_params_single_tower.learning_rate",
                    "params:model_params_single_tower.weight_decay",
                    "params:model_params_single_tower.max_epochs",
                    "params:k",
                ],
                outputs=[
                    "preprocessor_single_tower",
                    "checkpoint_path_single_tower",
                ],
                name="train_single_tower_node",
            ),
            node(
                func=predict,
                inputs=[
                    "preprocessor_single_tower",
                    "checkpoint_path_single_tower",
                    "web30k_fold1_test_preprocessed",
                ],
                outputs="predictions_single_tower",
                name="predict_single_tower_node",
            ),
            node(
                func=evaluate,
                inputs=["predictions_single_tower", "params:k"],
                outputs="metrics_single_tower",
                name="evaluate_single_tower_node",
            ),
        ]
    )

    pipelines = [
        pipeline(
            base_pipelie,
            inputs={"web30k_fold1_test_preprocessed"},
            parameters={
                "params:model_params_single_tower.batch_size",
                "params:model_params_single_tower.learning_rate",
                "params:model_params_single_tower.weight_decay",
                "params:model_params_single_tower.max_epochs",
                "params:k",
            },
            namespace=weight,
        )
        for weight in ["w0", "w20", "w60", "w80", "w100"]
    ]

    return sum(pipelines)
