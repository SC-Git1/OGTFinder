"""DL stacking: Plan 10: Two-Stage Cascade -- 8 base models with SVR meta-learner"""

from dl_stack_runner import DLStackConfig, run

CONFIG = DLStackConfig(
    stack_name="dlstack_two_stage_cascade",
    base_model_keys=[
        'danets', 'hopular', 'realmlp', 'tabnet_inspired',
        'saint', 'snn', 'vime', 'gated_mlp',
    ],
    meta_learner_key="svr",
    description="Plan 10: Two-Stage Cascade -- 8 base models with SVR meta-learner",
)

if __name__ == "__main__":
    run(CONFIG)
