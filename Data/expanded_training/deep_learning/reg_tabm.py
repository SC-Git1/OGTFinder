"""
A4: TabM - Parameter-Efficient Ensemble of MLPs for Tabular Data Regression

TabM uses a BatchEnsemble-like approach where a single MLP backbone outputs
multiple predictions in one forward pass, then aggregates them.

Features:
- Optuna hyperparameter tuning
- Efficient ensemble: share most weights, only ensemble-specific scaling factors
- Multiple predictions from one forward pass
- Mean/median aggregation for final prediction

Paper: "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"

Designed for: ~8k samples, ~1000 features, regression task

Usage:
    python reg_tabm.py <train_data_path> <test_data_path>
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

MODEL_NAME = "tabm"
OUTPUT_DIR = './out/tabm'

###############################################################################
# MODEL ARCHITECTURE
###############################################################################


class BatchEnsembleDense(layers.Layer):
    """
    BatchEnsemble-style dense layer.

    Shares base weights across ensemble members, with per-member scaling factors.
    """

    def __init__(self, units, num_members=4, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_members = num_members
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

        self.r_factors = self.add_weight(
            name='r_factors',
            shape=(self.num_members, input_dim),
            initializer=keras.initializers.RandomNormal(mean=1.0, stddev=0.1),
            trainable=True
        )
        self.s_factors = self.add_weight(
            name='s_factors',
            shape=(self.num_members, self.units),
            initializer=keras.initializers.RandomNormal(mean=1.0, stddev=0.1),
            trainable=True
        )

    def call(self, inputs, member_idx=None):
        if member_idx is not None:
            r = self.r_factors[member_idx]
            s = self.s_factors[member_idx]
            x = inputs * r
            x = tf.matmul(x, self.kernel) + self.bias
            x = x * s
        else:
            outputs = []
            for i in range(self.num_members):
                r = self.r_factors[i]
                s = self.s_factors[i]
                x = inputs * r
                x = tf.matmul(x, self.kernel) + self.bias
                x = x * s
                outputs.append(x)
            x = tf.stack(outputs, axis=1)

        if self.activation is not None:
            x = self.activation(x)
        return x


class TabMBlock(layers.Layer):
    """TabM block with batch ensemble dense + normalization + dropout."""

    def __init__(self, units, num_members=4, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_members = num_members
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.dense = BatchEnsembleDense(self.units, self.num_members)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        if len(inputs.shape) == 2:
            x = self.dense(inputs, member_idx=None)
        else:
            outputs = []
            for i in range(self.num_members):
                member_input = inputs[:, i, :]
                member_output = self.dense(member_input, member_idx=i)
                outputs.append(member_output)
            x = tf.stack(outputs, axis=1)

        x = self.norm(x)
        x = tf.nn.gelu(x)
        x = self.dropout(x, training=training)
        return x


def build_tabm(
    input_shape: int,
    hidden_units: list = [256, 128, 64],
    num_members: int = 4,
    dropout_rate: float = 0.1,
    aggregation: str = 'mean'
) -> Model:
    """
    Build TabM model.

    Args:
        input_shape: Number of input features
        hidden_units: Hidden layer sizes
        num_members: Number of ensemble members
        dropout_rate: Dropout rate
        aggregation: 'mean' or 'median' for final aggregation

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.LayerNormalization(epsilon=1e-6)(inputs)

    for i, units in enumerate(hidden_units):
        x = TabMBlock(units, num_members, dropout_rate, name=f'tabm_block_{i}')(x)

    # Per-member output heads
    outputs_list = []
    for i in range(num_members):
        member_features = x[:, i, :]
        member_output = layers.Dense(1, name=f'output_member_{i}')(member_features)
        outputs_list.append(member_output)

    stacked_outputs = layers.Lambda(lambda x: ops.stack(x, axis=1))(outputs_list)

    # Aggregation using Lambda for Keras 3.x compatibility
    if aggregation == 'median':
        mid_idx = num_members // 2
        if num_members % 2 == 1:
            def median_agg(x):
                squeezed = ops.squeeze(x, axis=-1)
                sorted_out = ops.sort(squeezed, axis=1)
                return sorted_out[:, mid_idx:mid_idx+1]
        else:
            def median_agg(x):
                squeezed = ops.squeeze(x, axis=-1)
                sorted_out = ops.sort(squeezed, axis=1)
                return (sorted_out[:, mid_idx-1:mid_idx] + sorted_out[:, mid_idx:mid_idx+1]) / 2
        final_output = layers.Lambda(median_agg)(stacked_outputs)
    else:
        final_output = layers.Lambda(lambda x: ops.mean(x, axis=1))(stacked_outputs)

    model = Model(inputs, final_output, name='TabM')
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

    n_layers = trial.suggest_int('n_layers', 2, 4)
    hidden_units = []
    for i in range(n_layers):
        units = trial.suggest_categorical(f'units_layer_{i}', [64, 128, 256])
        hidden_units.append(units)

    num_members = trial.suggest_categorical('num_members', [2, 4, 8])
    aggregation = trial.suggest_categorical('aggregation', ['mean', 'median'])

    model = build_tabm(
        input_shape=input_dim,
        hidden_units=hidden_units,
        num_members=num_members,
        dropout_rate=dropout_rate,
        aggregation=aggregation
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
    n_layers = params['n_layers']
    hidden_units = [params[f'units_layer_{i}'] for i in range(n_layers)]

    model = build_tabm(
        input_shape=input_dim,
        hidden_units=hidden_units,
        num_members=params['num_members'],
        dropout_rate=params['dropout_rate'],
        aggregation=params['aggregation']
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
        description='Train TabM with Optuna hyperparameter optimization'
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
