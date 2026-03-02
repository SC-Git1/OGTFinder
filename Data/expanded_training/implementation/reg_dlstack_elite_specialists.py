"""DL stacking: Plan 1: Elite Specialists -- 3 base models with Ridge meta-learner"""

from dl_stack_runner import DLStackConfig, run

CONFIG = DLStackConfig(
    stack_name="dlstack_elite_specialists",
    base_model_keys=['wide_deep', 'vime', 'gated_mlp'],
    meta_learner_key="ridge",
    description="Plan 1: Elite Specialists -- 3 base models with Ridge meta-learner",
)

if __name__ == "__main__":
    run(CONFIG)
