from common_nodes import evaluate
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import predict, train


def create_pipeline(**kwargs) -> Pipeline:
    base_pipelie = Pipeline(
        [
            node(
                func=train,
                inputs=[
                    "web30k_synthetic_click_train_downsampled",
                    "web30k_synthetic_click_vali",
                    "params:model_params_lightgbm",
                ],
                outputs="model_lightgbm",
                name="train_lightgbm_node",
            ),
            node(
                func=predict,
                inputs=[
                    "model_lightgbm",
                    "web30k_fold1_test_preprocessed",
                ],
                outputs="predictions_lightgbm",
                name="predict_lightgbm_node",
            ),
            node(
                func=evaluate,
                inputs=["predictions_lightgbm", "params:k"],
                outputs="metrics_lightgbm",
                name="evaluate_lightgbm_node",
            ),
        ]
    )

    pipelines = [
        pipeline(
            base_pipelie,
            inputs={"web30k_fold1_test_preprocessed"},
            parameters={"params:model_params_lightgbm", "params:k"},
            namespace=weight,
        )
        for weight in ["w0", "w20", "w60", "w80", "w100"]
    ]

    return sum(pipelines)
