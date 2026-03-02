"""DL stacking: Plan 3: Extreme Hunter -- 4 base models with Huber meta-learner"""

from dl_stack_runner import DLStackConfig, run

CONFIG = DLStackConfig(
    stack_name="dlstack_extreme_hunter",
    base_model_keys=['vime', 'tabr', 'gated_mlp', 'attention_mlp'],
    meta_learner_key="huber",
    description="Plan 3: Extreme Hunter -- 4 base models with Huber meta-learner",
)

if __name__ == "__main__":
    run(CONFIG)
