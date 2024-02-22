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
                    "params:model_params_debiased_linear",
                ],
                outputs="model_debiased_linear",
                name="train_debiased_linear_node",
            ),
            node(
                func=predict,
                inputs=[
                    "model_debiased_linear",
                    "web30k_fold1_test_preprocessed",
                ],
                outputs="predictions_debiased_linear",
                name="predict_debiased_linear_node",
            ),
            node(
                func=evaluate,
                inputs=["predictions_debiased_linear", "params:k"],
                outputs="metrics_debiased_linear",
                name="evaluate_debiased_linear_node",
            ),
        ]
    )

    pipelines = [
        pipeline(
            base_pipelie,
            inputs={"web30k_fold1_test_preprocessed"},
            parameters={
                "params:model_params_debiased_linear": "params:model_params_linear",
                "params:k": "params:k",
            },
            namespace=weight,
        )
        for weight in ["w0", "w20", "w60", "w80", "w100"]
    ]

    return sum(pipelines)
