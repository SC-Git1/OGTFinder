"""DL stacking: Plan 9: Hot/Cold Bridge -- 7 base models with ElasticNet meta-learner"""

from dl_stack_runner import DLStackConfig, run

CONFIG = DLStackConfig(
    stack_name="dlstack_hot_cold_bridge",
    base_model_keys=[
        'vime', 'rtdl_resnet', 'tabr', 'sparse_mlp',
        'attention_mlp', 'baseline_mlp', 'wide_deep',
    ],
    meta_learner_key="elasticnet",
    description="Plan 9: Hot/Cold Bridge -- 7 base models with ElasticNet meta-learner",
)

if __name__ == "__main__":
    run(CONFIG)
