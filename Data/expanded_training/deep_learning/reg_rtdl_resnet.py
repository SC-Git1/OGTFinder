"""
A2: RTDL ResNet MLP (Tabular ResNet) for Tabular Data Regression

Architecture: Linear(input→d) then stacked residual MLP blocks (pre-norm) then head.

Based on the "Revisiting Deep Learning Models for Tabular Data" (RTDL) paper.
This is explicitly identified as a strong baseline in careful comparisons.

Features:
- Optuna hyperparameter tuning
- Pre-normalization (LayerNorm before dense)
- Residual connections with dimension matching
- Configurable depth and width

Designed for: ~8k samples, ~1000 features, regression task

Usage:
    python reg_rtdl_resnet.py <train_data_path> <test_data_path>
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

MODEL_NAME = "rtdl_resnet"
OUTPUT_DIR = './out/rtdl_resnet'

###############################################################################
# MODEL ARCHITECTURE
###############################################################################


class RTDLResidualBlock(layers.Layer):
    """
    RTDL-style residual block with pre-normalization.

    Architecture per block:
    x → LayerNorm → Dense → ReLU → Dropout → Dense → Dropout → Add(residual)
    """

    def __init__(self, d_hidden, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_hidden = d_hidden
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        d_in = input_shape[-1]

        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = layers.Dense(self.d_hidden, kernel_initializer='he_normal')
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dense2 = layers.Dense(d_in, kernel_initializer='he_normal')
        self.dropout2 = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        x = self.norm(inputs)
        x = self.dense1(x)
        x = tf.nn.relu(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return inputs + x


def build_rtdl_resnet(
    input_shape: int,
    d_model: int = 256,
    d_hidden: int = 512,
    n_blocks: int = 4,
    dropout_rate: float = 0.1,
    l2_reg: float = 1e-5
) -> Model:
    """
    Build RTDL ResNet MLP for tabular regression.

    Args:
        input_shape: Number of input features
        d_model: Model dimension (width of residual stream)
        d_hidden: Hidden dimension within residual blocks
        n_blocks: Number of residual blocks
        dropout_rate: Dropout rate
        l2_reg: L2 regularization for input projection

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(input_shape,))

    # Input projection to d_model
    x = layers.Dense(
        d_model,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='input_projection'
    )(inputs)

    # Stack of residual blocks
    for i in range(n_blocks):
        x = RTDLResidualBlock(d_hidden, dropout_rate, name=f'resblock_{i}')(x)

    # Final normalization
    x = layers.LayerNormalization(epsilon=1e-6, name='final_norm')(x)

    # Output head
    x = layers.Dense(64, activation='relu', name='head_dense')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, name='output')(x)

    model = Model(inputs, outputs, name='RTDL_ResNet')
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

    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    d_hidden = trial.suggest_categorical('d_hidden', [256, 512, 1024])
    n_blocks = trial.suggest_int('n_blocks', 2, 8)

    model = build_rtdl_resnet(
        input_shape=input_dim,
        d_model=d_model,
        d_hidden=d_hidden,
        n_blocks=n_blocks,
        dropout_rate=dropout_rate,
        l2_reg=weight_decay
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
    model = build_rtdl_resnet(
        input_shape=input_dim,
        d_model=params['d_model'],
        d_hidden=params['d_hidden'],
        n_blocks=params['n_blocks'],
        dropout_rate=params['dropout_rate'],
        l2_reg=params['weight_decay']
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

    # Load sample weights if provided
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

    # Save performance to CSV
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
        description='Train RTDL ResNet with Optuna hyperparameter optimization'
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
