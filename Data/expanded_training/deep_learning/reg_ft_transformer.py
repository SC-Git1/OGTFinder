"""D13: FT-Transformer - Feature Tokenizer Transformer for Tabular Data

With Optuna hyperparameter tuning.

Architecture: Each feature becomes a token → small Transformer encoder → pooled → head.
Uses grouped variant for efficiency with high-dimensional inputs.

Paper: "Revisiting Deep Learning Models for Tabular Data" (RTDL)
"""

import os
import sys
import json
import logging
import numpy as np
import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from config import (
    BATCH_SIZE,
    MAX_EPOCHS,
    PATIENCE_ES,
    PATIENCE_LR,
    FEATURE_COLUMNS,
    TARGET,
    RANDOM_STATE,
    N_TRIALS,
    N_CV,
    GROUP_COLUMN,
    EXPERIMENT_NAME_DEFAULT,
    MODEL_PERFORMANCE_NAME,
    MODEL_PERFORMANCE_PATH,
    OPTUNA_TIMEOUT,
)

from utils import (
    setup_logging_for_progress,
    print_phase,
    get_tqdm_keras_callback,
    check_gpu,
    r2_keras,
    load_and_preprocess_test,
    load_full_train_data_grouped,
    evaluate_model_extended,
    log_metrics_extended,
    create_optuna_callbacks,
    create_callbacks,
    create_groupkfold_objective_keras,
    plot_all,
    save_results,
    save_performance_to_csv,
    save_perfold_performance_csv,
    save_cv_performance_csv,
    run_optuna_study,
    suggest_common_hyperparams,
    add_wandb_args,
    init_wandb_from_args,
    plot_train_val_test_metrics,
    plot_train_diagnostics,
    log_wandb_images,
    load_sample_weights,
    get_sample_weights_array,
    get_wandb_epoch_callback,
    log_wandb_final_metrics,
    log_wandb_diagnostics,
    set_global_seeds,
    groupkfold_cross_validate,
)

###############################################################################
# CONSTANTS
###############################################################################
MODEL_NAME = "ft_transformer"
OUTPUT_DIR = './out/ft_transformer'

logger = logging.getLogger(__name__)
###############################################################################
# END CONSTANTS
###############################################################################


class FeatureTokenizer(layers.Layer):
    """
    Tokenizes continuous features into embeddings.
    Each feature gets its own embedding.
    """

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        n_features = input_shape[-1]
        # Per-feature embedding weights and biases
        self.feature_weights = self.add_weight(
            name='feature_weights',
            shape=(n_features, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.feature_biases = self.add_weight(
            name='feature_biases',
            shape=(n_features, self.embed_dim),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        # inputs: (batch, n_features)
        # Output: (batch, n_features, embed_dim)
        x = tf.expand_dims(inputs, -1)  # (batch, n_features, 1)
        tokens = x * self.feature_weights + self.feature_biases
        return tokens


class TransformerBlock(layers.Layer):
    """Standard Transformer block with pre-norm."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embed_dim),
            layers.Dropout(self.dropout_rate)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        # Pre-norm attention
        x = self.layernorm1(inputs)
        attn_output = self.att(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = inputs + attn_output

        # Pre-norm FFN
        y = self.layernorm2(x)
        ffn_output = self.ffn(y, training=training)
        return x + ffn_output


class CLSToken(layers.Layer):
    """Learnable [CLS] token for aggregation."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_broadcasted, inputs], axis=1)


def build_ft_transformer(input_shape,
                         embed_dim=32,
                         num_heads=4,
                         ff_dim=64,
                         num_blocks=2,
                         dropout_rate=0.1,
                         use_cls_token=True):
    """
    Build FT-Transformer for tabular regression.

    Note: For 1000 features, this creates 1000 tokens which is expensive.
    Consider using feature grouping variant below.
    """
    inputs = layers.Input(shape=(input_shape,))

    # Tokenize features
    tokenizer = FeatureTokenizer(embed_dim)
    tokens = tokenizer(inputs)  # (batch, n_features, embed_dim)

    # Add CLS token for pooling
    if use_cls_token:
        cls_layer = CLSToken(embed_dim)
        tokens = cls_layer(tokens)  # (batch, n_features+1, embed_dim)

    # Transformer blocks
    x = tokens
    for i in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate, name=f'transformer_{i}')(x)

    # Final layer norm
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Pooling
    if use_cls_token:
        # Use CLS token
        pooled = x[:, 0, :]
    else:
        # Mean pooling
        pooled = tf.reduce_mean(x, axis=1)

    # Output head
    output = layers.Dense(64, activation='relu')(pooled)
    output = layers.Dropout(dropout_rate)(output)
    output = layers.Dense(1)(output)

    model = Model(inputs, output, name='FT_Transformer')
    return model


def build_ft_transformer_grouped(input_shape,
                                 n_groups=50,
                                 embed_dim=32,
                                 num_heads=4,
                                 ff_dim=64,
                                 num_blocks=2,
                                 dropout_rate=0.1):
    """
    FT-Transformer with feature grouping.

    Groups features first to reduce sequence length from 1000 to ~50 tokens.
    More practical for high-dimensional inputs.
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    # Validate and adjust n_groups to avoid invalid reshape
    effective_n_groups = min(n_groups, input_shape)
    if effective_n_groups < 1:
        effective_n_groups = 1
    group_size = max(1, input_shape // effective_n_groups)
    n_used = effective_n_groups * group_size

    # Group features: reshape and pool within groups
    # Reshape: (batch, n_groups, group_size)
    x_grouped = layers.Reshape((effective_n_groups, group_size))(x[:, :n_used])

    # Embed each group
    group_embed = layers.Dense(embed_dim)(x_grouped)  # (batch, n_groups, embed_dim)

    # CLS token
    cls_layer = CLSToken(embed_dim)
    tokens = cls_layer(group_embed)

    # Transformer
    for i in range(num_blocks):
        tokens = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(tokens)

    tokens = layers.LayerNormalization(epsilon=1e-6)(tokens)

    # CLS output
    pooled = tokens[:, 0, :]

    # Output
    output = layers.Dense(64, activation='relu')(pooled)
    output = layers.Dropout(dropout_rate)(output)
    output = layers.Dense(1)(output)

    model = Model(inputs, output, name='FT_Transformer_Grouped')
    return model


def build_ft_transformer_lite(input_shape,
                              compress_dim=128,
                              embed_dim=32,
                              num_heads=4,
                              num_blocks=2,
                              dropout_rate=0.1):
    """
    Lite FT-Transformer: compress features before transformer.

    Reduces dimensionality first, then applies transformer to
    compressed representation.
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    # Compress to fewer dimensions
    compressed = layers.Dense(compress_dim, activation='relu')(x)
    compressed = layers.LayerNormalization(epsilon=1e-6)(compressed)

    # Tokenize compressed features
    tokenizer = FeatureTokenizer(embed_dim)
    tokens = tokenizer(compressed)

    # CLS token
    cls_layer = CLSToken(embed_dim)
    tokens = cls_layer(tokens)

    # Transformer
    for i in range(num_blocks):
        tokens = TransformerBlock(embed_dim, num_heads, embed_dim * 2, dropout_rate)(tokens)

    tokens = layers.LayerNormalization(epsilon=1e-6)(tokens)
    pooled = tokens[:, 0, :]

    output = layers.Dense(64, activation='relu')(pooled)
    output = layers.Dropout(dropout_rate)(output)
    output = layers.Dense(1)(output)

    model = Model(inputs, output, name='FT_Transformer_Lite')
    return model


###############################################################################
# OPTUNA OBJECTIVE
###############################################################################

def build_model_for_trial(trial, input_dim):
    """Build and compile model for Optuna trial."""
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    embed_dim = trial.suggest_categorical('embed_dim', [16, 32, 64])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    num_blocks = trial.suggest_int('num_blocks', 1, 3)
    n_groups = trial.suggest_categorical('n_groups', [25, 50, 100])

    model = build_ft_transformer_grouped(
        input_shape=input_dim,
        n_groups=n_groups,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=embed_dim * 2,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError(), r2_keras]
    )
    return model


def build_model_from_params(params: dict, input_dim: int):
    """Build and compile model from best parameters."""
    model = build_ft_transformer_grouped(
        input_shape=input_dim,
        n_groups=params['n_groups'],
        embed_dim=params['embed_dim'],
        num_heads=params['num_heads'],
        ff_dim=params['embed_dim'] * 2,
        num_blocks=params['num_blocks'],
        dropout_rate=params['dropout_rate']
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError(), r2_keras]
    )
    return model


def main(train_path: str, test_path: str, weights_path: str = None,
         experiment_name: str = None, args=None):
    """Main training function with Optuna hyperparameter optimization using GroupKFold."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging_for_progress(os.path.join(OUTPUT_DIR, 'training_ft_transformer.log'))
    check_gpu()
    experiment_name = experiment_name or EXPERIMENT_NAME_DEFAULT

    if args is None:
        class _Args:
            pass
        args = _Args()
    wandb_run = init_wandb_from_args(args, model_name=MODEL_NAME, output_dir=OUTPUT_DIR)

    set_global_seeds(RANDOM_STATE)
    # Phase 1: Load data
    print_phase(MODEL_NAME, 1, 4, "Loading data...")
    X, y, X_raw, y_original, groups, feature_scaler, target_scaler, feature_names = \
        load_full_train_data_grouped(
            train_path,
            feature_columns=FEATURE_COLUMNS,
            target_col=TARGET,
            group_col=GROUP_COLUMN
        )

    input_dim = X.shape[1]
    n_features = input_dim
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Number of samples: {len(y)}")
    if groups is not None:
        logger.info(f"Number of unique groups: {len(np.unique(groups))}")

    sample_weights = None
    if weights_path and os.path.exists(weights_path):
        weights_dict = load_sample_weights(weights_path)
        sample_weights = get_sample_weights_array(y_original, weights_dict)
        logger.info(f"Loaded sample weights, range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

    # Phase 2: Optuna hyperparameter optimization
    print_phase(MODEL_NAME, 2, 4, f"Optuna tuning ({N_TRIALS} trials)...")
    logger.info("Starting hyperparameter optimization with GroupKFold CV...")

    objective = create_groupkfold_objective_keras(
        build_model_fn=build_model_for_trial,
        X=X_raw, y=y_original, groups=groups,
        sample_weights=sample_weights,
        n_splits=N_CV,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience_es=PATIENCE_ES,
        patience_lr=PATIENCE_LR
    )

    study, best_params, best_value = run_optuna_study(
        objective_fn=objective,
        study_name=f"{MODEL_NAME}_study",
        n_trials=N_TRIALS,
        timeout=OPTUNA_TIMEOUT
    )

    logger.info(f"Best trial validation loss: {best_value:.6f}")
    logger.info(f"Best hyperparameters: {best_params}")

    # Phase 2b: Cross-validation evaluation with best hyperparameters
    print_phase(MODEL_NAME, "2b", 4, "CV evaluation (GroupKFold)...")
    logger.info("Running GroupKFold CV with optimized hyperparameters...")

    cv_mean, cv_std, cv_all_folds = groupkfold_cross_validate(
        model_builder_fn=build_model_from_params,
        best_params=best_params,
        X=X_raw,
        y=y_original,
        groups=groups,
        sample_weights=sample_weights,
        n_splits=N_CV,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience_es=PATIENCE_ES,
        patience_lr=PATIENCE_LR,
        n_features=n_features
    )

    logger.info(f"CV RMSE: {cv_mean['rmse']:.4f} ± {cv_std['rmse']:.4f}")
    logger.info(f"CV R²:   {cv_mean['r2']:.4f} ± {cv_std['r2']:.4f}")

    # Save CV metrics to JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv_metrics_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_cv_metrics.json")
    with open(cv_metrics_path, 'w') as f:
        json.dump({'mean': cv_mean, 'std': cv_std, 'folds': cv_all_folds}, f, indent=2, default=float)
    logger.info(f"CV metrics saved to {cv_metrics_path}")

    # Save per-fold and CV mean/std tracking CSVs
    save_perfold_performance_csv(
        file_path=os.path.join(MODEL_PERFORMANCE_PATH, 'model_perfold_performance.csv'),
        model_name=MODEL_NAME,
        experiment_name=experiment_name,
        all_fold_metrics=cv_all_folds
    )
    save_cv_performance_csv(
        file_path=os.path.join(MODEL_PERFORMANCE_PATH, 'model_cv_performance.csv'),
        model_name=MODEL_NAME,
        experiment_name=experiment_name,
        mean_metrics=cv_mean,
        std_metrics=cv_std
    )

    # Phase 3: Train final model
    print_phase(MODEL_NAME, 3, 4, "Final training...")
    logger.info("Training final model with best hyperparameters...")
    keras.backend.clear_session()
    set_global_seeds(RANDOM_STATE)
    final_model = build_model_from_params(best_params, input_dim)
    final_model.summary(print_fn=logger.info)

    X_test, y_test, y_test_original = load_and_preprocess_test(
        test_path, feature_scaler, target_scaler,
        feature_columns=FEATURE_COLUMNS, target_col=TARGET
    )

    test_sample_weights = None
    if weights_path and os.path.exists(weights_path):
        test_sample_weights = get_sample_weights_array(y_test_original, weights_dict)

    # Build callbacks list with W&B epoch logging and tqdm progress
    callbacks = create_callbacks(PATIENCE_ES, PATIENCE_LR)
    wandb_cb = get_wandb_epoch_callback(wandb_run)
    if wandb_cb is not None:
        callbacks.append(wandb_cb)
    tqdm_cb = get_tqdm_keras_callback(total_epochs=MAX_EPOCHS)
    if tqdm_cb is not None:
        callbacks.append(tqdm_cb)

    history = final_model.fit(
        X, y,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        sample_weight=sample_weights,
        callbacks=callbacks,
        verbose=0
    )

    # Phase 4: Evaluation
    print_phase(MODEL_NAME, 4, 4, "Evaluation & saving...")
    logger.info("Evaluating on test data...")
    metrics = evaluate_model_extended(
        final_model, X_test, y_test, target_scaler,
        n_features=n_features,
        sample_weights=test_sample_weights
    )
    log_metrics_extended(metrics, prefix="Test")

    # Log final test metrics to W&B (including weighted/unweighted)
    log_wandb_final_metrics(wandb_run, metrics, sample_weights=test_sample_weights, prefix='test')

    # Generate training predictions
    logger.info("Generating training predictions...")
    train_pred_scaled = final_model.predict(X, verbose=0)
    train_pred = target_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
    train_actual = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    # Generate training diagnostic plots
    plot_train_diagnostics(train_actual, train_pred, OUTPUT_DIR, MODEL_NAME)

    # Log diagnostic plots to W&B
    log_wandb_diagnostics(wandb_run, metrics['y_actual'], metrics['y_pred'], OUTPUT_DIR, MODEL_NAME, suffix='_test')

    metric_plot_paths = plot_all(history, metrics, OUTPUT_DIR, MODEL_NAME)
    log_wandb_images(wandb_run, metric_plot_paths)
    save_results(
        final_model, metrics, best_params, OUTPUT_DIR, MODEL_NAME,
        train_predictions=(train_actual, train_pred)
    )

    os.makedirs(MODEL_PERFORMANCE_PATH, exist_ok=True)
    csv_path = os.path.join(MODEL_PERFORMANCE_PATH, MODEL_PERFORMANCE_NAME)
    save_performance_to_csv(
        file_path=csv_path,
        model_name=MODEL_NAME,
        experiment_name=experiment_name,
        dataset_type='cv',
        overall_metrics={
            'rmse': cv_mean.get('rmse'),
            'mae': cv_mean.get('mae'),
            'r2': cv_mean.get('r2'),
            'adj_r2': cv_mean.get('adj_r2'),
            'weighted_rmse': cv_mean.get('weighted_rmse')
        },
        binned_metrics={k: v for k, v in cv_mean.items()
                        if '_bottom_' in k or '_top_' in k}
    )
    save_performance_to_csv(
        file_path=csv_path,
        model_name=MODEL_NAME,
        experiment_name=experiment_name,
        dataset_type='test',
        overall_metrics={
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'adj_r2': metrics['adj_r2'],
            'weighted_rmse': metrics.get('weighted_rmse')
        },
        binned_metrics=metrics.get('binned', {})
    )

    logger.info("Training complete!")
    return metrics, best_params


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train FT-Transformer with Optuna tuning')
    parser.add_argument('train_path', type=str, help='Path to training data')
    parser.add_argument('test_path', type=str, help='Path to test data')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to sample weights JSON file')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment')
    add_wandb_args(parser)
    args = parser.parse_args()
    main(args.train_path, args.test_path, args.weights_path, args.experiment_name, args)
