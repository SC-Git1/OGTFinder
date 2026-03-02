"""DL stacking: Plan 5: Kitchen Sink -- 10 base models with XGBoost meta-learner"""

from dl_stack_runner import DLStackConfig, run

CONFIG = DLStackConfig(
    stack_name="dlstack_kitchen_sink",
    base_model_keys=[
        'wide_deep', 'danets', 'rtdl_resnet', 'gated_mlp', 'vime',
        'attention_mlp', 'tabm', 'node', 'ft_transformer', 'grownet',
    ],
    meta_learner_key="xgb",
    description="Plan 5: Kitchen Sink -- 10 base models with XGBoost meta-learner",
)

if __name__ == "__main__":
    run(CONFIG)
