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
                    "params:model_params_xgboost",
                ],
                outputs="model_xgboost",
                name="train_xgboost_node",
            ),
            node(
                func=predict,
                inputs=[
                    "model_xgboost",
                    "web30k_fold1_test_preprocessed",
                ],
                outputs="predictions_xgboost",
                name="predict_xgboost_node",
            ),
            node(
                func=evaluate,
                inputs=["predictions_xgboost", "params:k"],
                outputs="metrics_xgboost",
                name="evaluate_xgboost_node",
            ),
        ]
    )

    pipelines = [
        pipeline(
            base_pipelie,
            inputs={"web30k_fold1_test_preprocessed"},
            parameters={"params:model_params_xgboost", "params:k"},
            namespace=weight,
        )
        for weight in ["w0", "w20", "w60", "w80", "w100"]
    ]

    return sum(pipelines)
