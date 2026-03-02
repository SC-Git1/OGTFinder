"""
B7: TabNet - Sequential Attentive Feature Selection for Tabular Data Regression

Architecture: Decision steps with sparse masks selecting subsets of features.

Key TabNet concepts:
- Sequential decision steps
- Sparse attention masks (sparsemax)
- Feature reuse penalty
- Instance-wise feature selection
- Interpretable feature importance

Features:
- Optuna hyperparameter tuning
- Full TabNet with GLU and sparsemax

Paper: "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, 2019)

Designed for: ~8k samples, ~1000 features, regression task

Usage:
    python reg_tabnet.py <train_data_path> <test_data_path>
"""

import os
import json
import numpy as np
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
    create_callbacks,
    create_optuna_callbacks,
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

MODEL_NAME = "tabnet"
OUTPUT_DIR = './out/tabnet'

###############################################################################
# MODEL ARCHITECTURE
###############################################################################


def sparsemax(logits, axis=-1):
    """
    Sparsemax activation function.

    Projects onto the probability simplex, producing sparse outputs.
    Many outputs become exactly zero.
    """
    logits = tf.cast(logits, tf.float32)

    z_sorted = tf.sort(logits, axis=axis, direction='DESCENDING')
    z_cumsum = tf.cumsum(z_sorted, axis=axis)

    k = tf.range(1, tf.shape(logits)[axis] + 1, dtype=tf.float32)
    k = tf.reshape(k, [1, -1])

    condition = 1.0 + k * z_sorted > z_cumsum
    k_z = tf.reduce_sum(tf.cast(condition, tf.float32), axis=axis, keepdims=True)

    z_k_sum = tf.reduce_sum(z_sorted * tf.cast(condition, tf.float32), axis=axis, keepdims=True)
    tau = (z_k_sum - 1.0) / k_z

    output = tf.maximum(logits - tau, 0.0)
    return output


class GatedLinearUnit(layers.Layer):
    """GLU: splits input and applies sigmoid gating."""

    def __init__(self, units, momentum=0.98, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.momentum = momentum

    def build(self, input_shape):
        self.fc = layers.Dense(self.units * 2, use_bias=False)
        self.bn = layers.BatchNormalization(momentum=self.momentum)

    def call(self, inputs, training=False):
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x1, x2 = tf.split(x, 2, axis=-1)
        return x1 * tf.sigmoid(x2)


class FeatureTransformer(layers.Layer):
    """Feature transformer: shared + decision-step-specific layers."""

    def __init__(self, n_independent, n_shared, n_output, momentum=0.98, **kwargs):
        super().__init__(**kwargs)
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.n_output = n_output
        self.momentum = momentum

    def build(self, input_shape):
        self.shared_layers = []
        for i in range(self.n_shared):
            self.shared_layers.append(GatedLinearUnit(self.n_output, self.momentum))

        self.step_layers = []
        for i in range(self.n_independent):
            self.step_layers.append(GatedLinearUnit(self.n_output, self.momentum))

    def call(self, inputs, training=False):
        x = inputs

        for i, layer in enumerate(self.shared_layers):
            h = layer(x, training=training)
            if i > 0:
                x = (x + h) * np.sqrt(0.5)
            else:
                x = h

        for i, layer in enumerate(self.step_layers):
            h = layer(x, training=training)
            x = (x + h) * np.sqrt(0.5)

        return x


class AttentiveTransformer(layers.Layer):
    """Attentive transformer: computes sparse attention mask."""

    def __init__(self, n_output, momentum=0.98, **kwargs):
        super().__init__(**kwargs)
        self.n_output = n_output
        self.momentum = momentum

    def build(self, input_shape):
        self.fc = layers.Dense(self.n_output, use_bias=False)
        self.bn = layers.BatchNormalization(momentum=self.momentum)

    def call(self, inputs, priors, training=False):
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x = x * priors
        mask = sparsemax(x)
        return mask


class TabNetEncoder(layers.Layer):
    """TabNet encoder with multiple decision steps."""

    def __init__(self,
                 n_steps=3,
                 n_features=None,
                 n_shared=2,
                 n_independent=2,
                 feature_dim=64,
                 output_dim=64,
                 relaxation_factor=1.5,
                 sparsity_coefficient=1e-5,
                 momentum=0.98,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_shared = n_shared
        self.n_independent = n_independent
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.momentum = momentum

    def build(self, input_shape):
        n_features = input_shape[-1]
        self.n_features = n_features

        self.initial_bn = layers.BatchNormalization(momentum=self.momentum)

        self.initial_transformer = FeatureTransformer(
            self.n_independent, self.n_shared, self.feature_dim + self.output_dim,
            momentum=self.momentum
        )

        self.attention_transformers = []
        self.feature_transformers = []

        for step in range(self.n_steps):
            self.attention_transformers.append(
                AttentiveTransformer(n_features, momentum=self.momentum)
            )
            self.feature_transformers.append(
                FeatureTransformer(
                    self.n_independent, self.n_shared, self.feature_dim + self.output_dim,
                    momentum=self.momentum
                )
            )

    def call(self, inputs, training=False):
        x = self.initial_bn(inputs, training=training)
        h = self.initial_transformer(x, training=training)

        attention_part = h[:, :self.feature_dim]
        prior_scales = tf.ones_like(x)

        outputs = []
        attention_masks = []
        total_entropy = 0.0

        for step in range(self.n_steps):
            mask = self.attention_transformers[step](attention_part, prior_scales, training=training)
            attention_masks.append(mask)

            prior_scales = prior_scales * (self.relaxation_factor - mask)
            masked_features = mask * x

            h = self.feature_transformers[step](masked_features, training=training)

            attention_part = h[:, :self.feature_dim]
            step_output = tf.nn.relu(h[:, self.feature_dim:])

            outputs.append(step_output)

            entropy = -tf.reduce_sum(mask * tf.math.log(mask + 1e-10), axis=-1)
            total_entropy += tf.reduce_mean(entropy)

        aggregated = tf.add_n(outputs)

        sparsity_loss = self.sparsity_coefficient * total_entropy / self.n_steps
        self.add_loss(sparsity_loss)

        return aggregated, attention_masks


def build_tabnet(
    input_shape: int,
    n_steps: int = 3,
    feature_dim: int = 64,
    output_dim: int = 64,
    n_shared: int = 2,
    n_independent: int = 2,
    relaxation_factor: float = 1.5,
    sparsity_coefficient: float = 1e-5,
    dropout_rate: float = 0.1
) -> Model:
    """
    Build TabNet model for tabular regression.

    Args:
        input_shape: Number of input features
        n_steps: Number of decision steps
        feature_dim: Dimension for attention
        output_dim: Output dimension per step
        n_shared: Number of shared GLU layers
        n_independent: Number of step-specific GLU layers
        relaxation_factor: Controls feature reuse
        sparsity_coefficient: Weight for sparsity regularization
        dropout_rate: Dropout rate

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(input_shape,))

    encoder = TabNetEncoder(
        n_steps=n_steps,
        n_shared=n_shared,
        n_independent=n_independent,
        feature_dim=feature_dim,
        output_dim=output_dim,
        relaxation_factor=relaxation_factor,
        sparsity_coefficient=sparsity_coefficient
    )

    encoded, attention_masks = encoder(inputs)

    x = layers.BatchNormalization()(encoded)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='TabNet')
    model.encoder = encoder
    return model


###############################################################################
# OPTUNA OBJECTIVE
###############################################################################


def build_model_for_trial(trial, input_dim):
    """Build and compile model for Optuna trial."""
    common_params = suggest_common_hyperparams(trial)
    learning_rate = common_params['learning_rate']
    dropout_rate = common_params['dropout_rate']
    weight_decay = common_params['weight_decay']

    n_steps = trial.suggest_int('n_steps', 2, 5)
    feature_dim = trial.suggest_categorical('feature_dim', [32, 64, 128])
    output_dim = trial.suggest_categorical('output_dim', [32, 64, 128])
    n_shared = trial.suggest_int('n_shared', 1, 3)
    n_independent = trial.suggest_int('n_independent', 1, 3)
    relaxation_factor = trial.suggest_float('relaxation_factor', 1.0, 2.0)
    sparsity_coefficient = trial.suggest_float('sparsity_coefficient', 1e-6, 1e-3, log=True)

    model = build_tabnet(
        input_shape=input_dim,
        n_steps=n_steps,
        feature_dim=feature_dim,
        output_dim=output_dim,
        n_shared=n_shared,
        n_independent=n_independent,
        relaxation_factor=relaxation_factor,
        sparsity_coefficient=sparsity_coefficient,
        dropout_rate=dropout_rate
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError(), r2_keras]
    )
    return model


def build_model_from_params(params: dict, input_dim: int):
    """Build and compile model from best parameters."""
    model = build_tabnet(
        input_shape=input_dim,
        n_steps=params['n_steps'],
        feature_dim=params['feature_dim'],
        output_dim=params['output_dim'],
        n_shared=params['n_shared'],
        n_independent=params['n_independent'],
        relaxation_factor=params['relaxation_factor'],
        sparsity_coefficient=params['sparsity_coefficient'],
        dropout_rate=params['dropout_rate']
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError(), r2_keras]
    )
    return model


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

    parser = argparse.ArgumentParser(
        description='Train TabNet with Optuna hyperparameter optimization'
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
