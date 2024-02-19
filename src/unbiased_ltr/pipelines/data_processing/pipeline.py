from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_synthetic_clicks_dataset, preprocess_raw_web30k_dataset


def create_pipeline(**kwargs) -> Pipeline:
    preprocess_pipeline = Pipeline(
        [
            node(
                func=preprocess_raw_web30k_dataset,
                inputs=f"web30k_fold1_{split}",
                outputs=f"web30k_fold1_{split}_preprocessed",
                name=f"preprocess_{split}_node",
                tags=["preprocess"],
            )
            for split in ["train", "vali", "test"]
        ]
    )

    synthetic_pipeline = Pipeline(
        [
            node(
                generate_synthetic_clicks_dataset,
                inputs=[
                    f"web30k_fold1_{split}_preprocessed",
                    "params:click_noise",
                    "params:dataset_oracle_weight",
                    "params:negative_downsample_ratio",
                    "params:random_seed",
                ],
                outputs=f"web30k_synthetic_click_{split}",
                name=f"generate_synthetic_{split}_node",
                tags=["synthetic"],
            )
            for split in ["train", "vali", "test"]
        ]
    )
    synthetic_pipelines = [
        pipeline(
            pipe=synthetic_pipeline,
            inputs={
                f"web30k_fold1_{split}_preprocessed"
                for split in ["train", "vali", "test"]
            },
            parameters={
                "params:click_noise",
                "params:negative_downsample_ratio",
                "params:random_seed",
            },
            namespace=weight,
        )
        for weight in ["w0", "w20", "w60", "w80", "w100"]
    ]

    return preprocess_pipeline + sum(synthetic_pipelines)
