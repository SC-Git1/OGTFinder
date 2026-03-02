"""
TabNet-Inspired Architecture for Tabular Data Regression

Architecture: Attention-based sequential feature selection mechanism.
Uses sparsemax for sparse attention weights to select relevant features.

Features:
- Optuna hyperparameter tuning with GroupKFold CV
- Sequential multi-step processing
- Instance-wise feature selection

Designed for: ~8k samples, ~1000 features, regression task

Usage:
    python reg_tabnet_inspired.py <train_data_path> <test_data_path>
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from keras import ops

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
    adjusted_r2_score,
    calculate_binned_metrics,
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

MODEL_NAME = "tabnet_inspired"
OUTPUT_DIR = './out/tabnet_inspired'

###############################################################################
# MODEL ARCHITECTURE
###############################################################################


def sparsemax(logits, axis=-1):
    """Sparsemax activation function."""
    logits = tf.cast(logits, tf.float32)
    dims = logits.shape.as_list()
    num_dims = len(dims)

    if axis < 0:
        axis = axis + num_dims

    z_sorted = tf.sort(logits, axis=axis, direction='DESCENDING')

    range_val = tf.range(1, tf.shape(logits)[axis] + 1, dtype=tf.float32)
    for _ in range(axis):
        range_val = tf.expand_dims(range_val, 0)
    for _ in range(num_dims - axis - 1):
        range_val = tf.expand_dims(range_val, -1)

    cumsum = tf.cumsum(z_sorted, axis=axis)
    k_check = 1 + range_val * z_sorted > cumsum
    k = tf.reduce_sum(tf.cast(k_check, tf.float32), axis=axis, keepdims=True)

    tau = (tf.reduce_sum(z_sorted * tf.cast(k_check, tf.float32), axis=axis, keepdims=True) - 1) / k

    return tf.maximum(logits - tau, 0)


class GatedLinearUnit(layers.Layer):
    """Gated Linear Unit for feature transformation."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.fc = layers.Dense(self.units * 2, use_bias=False)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x, gate = tf.split(x, 2, axis=-1)
        return x * tf.sigmoid(gate)


class FeatureTransformer(layers.Layer):
    """Feature transformer block with shared and decision-specific layers."""

    def __init__(self, units, num_shared=2, num_decision=2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_shared = num_shared
        self.num_decision = num_decision

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.input_proj = layers.Dense(self.units) if input_dim != self.units else None
        self.shared_layers = [GatedLinearUnit(self.units) for _ in range(self.num_shared)]
        self.decision_layers = [GatedLinearUnit(self.units) for _ in range(self.num_decision)]

    def call(self, inputs, training=False):
        if self.input_proj is not None:
            x = self.input_proj(inputs)
        else:
            x = inputs

        for layer in self.shared_layers:
            x = layer(x, training=training) + x

        for layer in self.decision_layers:
            x = layer(x, training=training) + x

        return x


class AttentionTransformer(layers.Layer):
    """Attention mechanism for feature selection."""

    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim

    def build(self, input_shape):
        self.fc = layers.Dense(self.input_dim, use_bias=False)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, priors, training=False):
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x = x * priors
        return sparsemax(x)


class TabNetInspiredBlock(layers.Layer):
    """Single TabNet-inspired decision step."""

    def __init__(self, feature_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        self.attention = AttentionTransformer(input_shape[-1])
        self.feature_transformer = FeatureTransformer(self.feature_dim)
        self.output_fc = layers.Dense(self.output_dim)

    def call(self, inputs, priors, training=False):
        mask = self.attention(inputs, priors, training=training)
        masked_input = mask * inputs
        transformed = self.feature_transformer(masked_input, training=training)
        output = self.output_fc(transformed)
        new_priors = priors * (1 - mask)
        return output, mask, new_priors


def build_tabnet_inspired(
    input_shape: int,
    num_steps: int = 3,
    feature_dim: int = 128,
    output_dim: int = 64,
    dropout_rate: float = 0.2
) -> Model:
    """
    Build TabNet-inspired model for tabular regression.

    Args:
        input_shape: Number of input features
        num_steps: Number of decision steps
        feature_dim: Dimension of feature transformer
        output_dim: Output dimension of each step
        dropout_rate: Dropout rate

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)
    priors = layers.Lambda(lambda t: ops.ones_like(t))(x)

    step_outputs = []
    attention_masks = []

    for step in range(num_steps):
        block = TabNetInspiredBlock(feature_dim, output_dim, name=f'step_{step}')
        step_output, mask, priors = block(x, priors)
        step_outputs.append(step_output)
        attention_masks.append(mask)

    aggregated = layers.Add()(step_outputs)
    aggregated = layers.ReLU()(aggregated)
    aggregated = layers.Dropout(dropout_rate)(aggregated)

    outputs = layers.Dense(1)(aggregated)

    model = Model(inputs, outputs, name='TabNet_Inspired')
    return model


###############################################################################
# MODEL BUILDER (for GroupKFold CV compatibility)
###############################################################################


def build_model_from_params(params: dict, input_dim: int) -> Model:
    """
    Build and compile TabNet-Inspired model from hyperparameter dict.

    Args:
        params: Dict with keys: learning_rate, weight_decay, dropout_rate,
                num_steps, feature_dim, output_dim
        input_dim: Number of input features

    Returns:
        Compiled Keras model
    """
    model = build_tabnet_inspired(
        input_shape=input_dim,
        num_steps=params['num_steps'],
        feature_dim=params['feature_dim'],
        output_dim=params['output_dim'],
        dropout_rate=params['dropout_rate']
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=[keras.metrics.RootMeanSquaredError(),
                 keras.metrics.MeanAbsoluteError(), r2_keras]
    )
    return model


###############################################################################
# OPTUNA OBJECTIVE
###############################################################################


def define_search_space(trial):
    """Define TabNet-Inspired hyperparameter search space for Optuna."""
    common_params = suggest_common_hyperparams(trial)

    params = {
        'learning_rate': common_params['learning_rate'],
        'dropout_rate': common_params['dropout_rate'],
        'weight_decay': common_params['weight_decay'],
        'num_steps': trial.suggest_int('num_steps', 2, 5),
        'feature_dim': trial.suggest_categorical('feature_dim', [64, 128, 256]),
        'output_dim': trial.suggest_categorical('output_dim', [32, 64, 128]),
    }
    return params


def build_model_for_trial(trial, input_dim):
    """Build and compile model for Optuna trial."""
    params = define_search_space(trial)
    return build_model_from_params(params, input_dim)


###############################################################################
# MAIN
###############################################################################


def main(train_path: str, test_path: str, weights_path: str = None,
         experiment_name: str = None, args=None):
    """Main training function with Optuna hyperparameter optimization using GroupKFold."""
    logger = setup_logging_for_progress(f'{MODEL_NAME}_training.log')
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
    X, y, X_raw, y_original, groups, feature_scaler, target_scaler, \
        feature_names = load_full_train_data_grouped(
            train_path,
            feature_columns=FEATURE_COLUMNS,
            target_col=TARGET,
            group_col=GROUP_COLUMN
        )

    n_features = X.shape[1]
    input_dim = X.shape[1]
    logger.info(f"Training data: {X.shape[0]} samples, {n_features} features")
    logger.info(f"Number of groups: {len(np.unique(groups))}")

    sample_weights = None
    if weights_path and os.path.exists(weights_path):
        weights_dict = load_sample_weights(weights_path)
        sample_weights = get_sample_weights_array(y_original, weights_dict)
        logger.info(f"Loaded sample weights, range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

    # Phase 2: Optuna hyperparameter optimization with GroupKFold
    print_phase(MODEL_NAME, 2, 4, f"Optuna tuning ({N_TRIALS} trials)...")
    logger.info("Starting hyperparameter optimization with GroupKFold CV...")

    objective = create_groupkfold_objective_keras(
        build_model_fn=build_model_for_trial,
        X=X_raw,
        y=y_original,
        groups=groups,
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
        callbacks=callbacks,
        sample_weight=sample_weights,
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

    # Log final test metrics to W&B
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

    # Save
    metric_plot_paths = plot_all(history, metrics, OUTPUT_DIR, MODEL_NAME)
    log_wandb_images(wandb_run, metric_plot_paths)
    save_results(
        final_model, metrics, best_params, OUTPUT_DIR, MODEL_NAME,
        train_predictions=(train_actual, train_pred)
    )

    # Save performance to CSV - CV metrics
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
    # Save performance to CSV - test metrics
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

    parser = argparse.ArgumentParser(
        description='Train TabNet-Inspired with Optuna hyperparameter optimization'
    )
    parser.add_argument('train_path', type=str, help='Path to training data file')
    parser.add_argument('test_path', type=str, help='Path to test data file')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to sample weights JSON file')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment')
    add_wandb_args(parser)

    args = parser.parse_args()
    main(args.train_path, args.test_path, args.weights_path, args.experiment_name, args)
