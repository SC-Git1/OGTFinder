"""DL stacking: Plan 2: Architecture Diversity -- 5 base models with LightGBM meta-learner"""

from dl_stack_runner import DLStackConfig, run

CONFIG = DLStackConfig(
    stack_name="dlstack_arch_diversity",
    base_model_keys=['wide_deep', 'rtdl_resnet', 'saint', 'tabnet', 'snn'],
    meta_learner_key="lgbm",
    description="Plan 2: Architecture Diversity -- 5 base models with LightGBM meta-learner",
)

if __name__ == "__main__":
    run(CONFIG)
