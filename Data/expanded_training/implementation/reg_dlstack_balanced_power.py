"""DL stacking: Plan 7: Balanced Power -- 6 base models with CatBoost meta-learner"""

from dl_stack_runner import DLStackConfig, run

CONFIG = DLStackConfig(
    stack_name="dlstack_balanced_power",
    base_model_keys=['danets', 'sparse_mlp', 'lassonet', 'tabm', 'vime', 'gated_mlp'],
    meta_learner_key="catboost",
    description="Plan 7: Balanced Power -- 6 base models with CatBoost meta-learner",
)

if __name__ == "__main__":
    run(CONFIG)
