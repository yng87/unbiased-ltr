from typing import Any

from kedro.framework.hooks import hook_impl

import wandb


class WandbHook:
    WANDB_PROJECT = "unbiased_ltr"

    @hook_impl
    def before_pipeline_run(self, run_params: dict[str, Any]):
        wandb.init(
            project=self.WANDB_PROJECT,
            name=f"{run_params['pipeline_name']}-{run_params['session_id']}",
            config={"weight": run_params.get("namespace")},
        )

    @hook_impl
    def after_pipeline_run(self):
        wandb.finish()
