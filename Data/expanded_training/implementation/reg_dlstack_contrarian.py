"""DL stacking: Plan 8: Contrarian -- 5 base models with RandomForest meta-learner"""

from dl_stack_runner import DLStackConfig, run

CONFIG = DLStackConfig(
    stack_name="dlstack_contrarian",
    base_model_keys=['wide_deep', 'gandalf', 'grownet', 'resnet_mlp', 'node'],
    meta_learner_key="rf",
    description="Plan 8: Contrarian -- 5 base models with RandomForest meta-learner",
)

if __name__ == "__main__":
    run(CONFIG)
