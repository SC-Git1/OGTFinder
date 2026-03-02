"""
A6: Sparse / Group-Sparse MLP for Tabular Data Regression

Architecture: Group-sparse regularization on first layer for built-in feature selection.

Key ideas:
- Encouraging sparsity can help generalization in high-d/low-n regimes
- "MLP + built-in feature selection bias"
- Group sparsity: features are grouped, entire groups can be zeroed out
- L2,1 regularization: L2 within groups, L1 across groups

Features:
- Optuna hyperparameter tuning
- Multiple sparsity types: group_l21, group_soft, l1

Designed for: ~8k samples, ~1000 features, regression task

Usage:
    python reg_sparse_mlp.py <train_data_path> <test_data_path>
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

MODEL_NAME = "sparse_mlp"
OUTPUT_DIR = './out/sparse_mlp'

###############################################################################
# MODEL ARCHITECTURE
###############################################################################


class GroupSparseRegularizer(keras.regularizers.Regularizer):
    """
    Group sparsity (L2,1) regularizer.

    Encourages entire groups of weights to be zero.
    Groups are defined along axis 0 (input features).
    """

    def __init__(self, l21_weight=0.01, group_size=10):
        self.l21_weight = l21_weight
        self.group_size = group_size

    def __call__(self, x):
        input_dim = tf.shape(x)[0]

        group_size = tf.maximum(self.group_size, 1)
        n_groups = tf.maximum(input_dim // group_size, 1)

        n_elements = n_groups * group_size
        x_truncated = x[:n_elements, :]
        x_grouped = tf.reshape(x_truncated, (n_groups, group_size, -1))

        group_norms = tf.sqrt(tf.reduce_sum(tf.square(x_grouped), axis=[1, 2]) + 1e-8)
        l21_penalty = self.l21_weight * tf.reduce_sum(group_norms)

        return l21_penalty

    def get_config(self):
        return {'l21_weight': self.l21_weight, 'group_size': self.group_size}


class GroupSparsityLayer(layers.Layer):
    """
    Layer that applies group-wise soft sparsity.

    Features are divided into groups, each group gets a learnable
    importance weight that is encouraged to be sparse.
    """

    def __init__(self, n_groups=100, sparsity_reg=0.01, **kwargs):
        super().__init__(**kwargs)
        self.n_groups = n_groups
        self.sparsity_reg = sparsity_reg

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("GroupSparsityLayer requires a known input feature dimension")

        self.input_dim = int(input_dim)
        requested_n_groups = max(int(self.n_groups), 1)
        self.effective_n_groups = min(requested_n_groups, self.input_dim)
        self.group_size = int(np.ceil(self.input_dim / self.effective_n_groups))

        self.group_weights = self.add_weight(
            name='group_weights',
            shape=(self.effective_n_groups,),
            initializer=keras.initializers.Constant(1.0),
            regularizer=keras.regularizers.l1(self.sparsity_reg),
            trainable=True
        )

    def call(self, inputs):
        importance = tf.nn.softmax(self.group_weights) * tf.cast(self.effective_n_groups, tf.float32)
        importance_expanded = tf.repeat(importance, self.group_size)
        importance_expanded = importance_expanded[:self.input_dim]
        importance_expanded = tf.cast(importance_expanded, inputs.dtype)
        importance_expanded = tf.reshape(importance_expanded, (1, self.input_dim))
        return inputs * importance_expanded


def build_sparse_mlp(
    input_shape: int,
    hidden_units: list = [512, 256, 128],
    sparsity_type: str = 'group_l21',
    n_groups: int = 100,
    l21_weight: float = 0.01,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4
) -> Model:
    """
    Build Sparse MLP with group sparsity regularization.

    Args:
        input_shape: Number of input features
        hidden_units: Hidden layer sizes
        sparsity_type: 'group_l21', 'group_soft', 'l1'
        n_groups: Number of feature groups
        l21_weight: Weight for L2,1 regularization
        dropout_rate: Dropout rate
        l2_reg: L2 regularization

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    # Apply group sparsity mechanism
    if sparsity_type == 'group_soft':
        x = GroupSparsityLayer(n_groups=n_groups, sparsity_reg=l21_weight)(x)
        regularizer = keras.regularizers.l2(l2_reg)
    elif sparsity_type == 'l1':
        regularizer = keras.regularizers.l1_l2(l1=l21_weight, l2=l2_reg)
    else:  # group_l21
        effective_n_groups = min(n_groups, input_shape)
        group_size = max(1, input_shape // effective_n_groups)
        regularizer = GroupSparseRegularizer(l21_weight=l21_weight, group_size=group_size)

    # First layer with sparsity regularization
    x = layers.Dense(
        hidden_units[0],
        kernel_initializer='he_normal',
        kernel_regularizer=regularizer,
        name='sparse_dense_0'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Remaining layers with standard L2
    for i, units in enumerate(hidden_units[1:], 1):
        x = layers.Dense(
            units,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='Sparse_MLP')
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
        units = trial.suggest_categorical(f'units_layer_{i}', [128, 256, 512])
        hidden_units.append(units)

    sparsity_type = trial.suggest_categorical('sparsity_type', ['group_l21', 'group_soft', 'l1'])
    candidate_groups = sorted(set([g for g in [50, 100, 200] if g <= input_dim] + [max(1, int(input_dim))]))
    n_groups = trial.suggest_categorical('n_groups', candidate_groups)
    l21_weight = trial.suggest_float('l21_weight', 1e-4, 1e-1, log=True)

    model = build_sparse_mlp(
        input_shape=input_dim,
        hidden_units=hidden_units,
        sparsity_type=sparsity_type,
        n_groups=n_groups,
        l21_weight=l21_weight,
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
    n_layers = params['n_layers']
    hidden_units = [params[f'units_layer_{i}'] for i in range(n_layers)]

    model = build_sparse_mlp(
        input_shape=input_dim,
        hidden_units=hidden_units,
        sparsity_type=params['sparsity_type'],
        n_groups=params['n_groups'],
        l21_weight=params['l21_weight'],
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
        description='Train Sparse MLP with Optuna hyperparameter optimization'
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
