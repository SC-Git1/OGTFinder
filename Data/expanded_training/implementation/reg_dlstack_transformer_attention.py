"""DL stacking: Plan 6: Transformer & Attention Resurgence -- 5 base models with MLP meta-learner"""

from dl_stack_runner import DLStackConfig, run

CONFIG = DLStackConfig(
    stack_name="dlstack_transformer_attention",
    base_model_keys=['ft_transformer', 'saint', 'attention_mlp', 'tabnet', 'wide_deep'],
    meta_learner_key="mlp",
    description="Plan 6: Transformer & Attention Resurgence -- 5 base models with MLP meta-learner",
)

if __name__ == "__main__":
    run(CONFIG)
