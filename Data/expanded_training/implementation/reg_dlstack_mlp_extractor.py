"""DL stacking: Plan 4: Deep MLP Feature Extractor -- 6 base models with Lasso meta-learner"""

from dl_stack_runner import DLStackConfig, run

CONFIG = DLStackConfig(
    stack_name="dlstack_mlp_extractor",
    base_model_keys=['wide_deep', 'gated_mlp', 'danets', 'sparse_mlp', 'baseline_mlp', 'realmlp'],
    meta_learner_key="lasso",
    description="Plan 4: Deep MLP Feature Extractor -- 6 base models with Lasso meta-learner",
)

if __name__ == "__main__":
    run(CONFIG)
