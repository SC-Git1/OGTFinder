"""C12: Hopular - Modern Hopfield Networks for Tabular Data Regression

With Optuna hyperparameter tuning.

Architecture: Iterative refinement with modern Hopfield blocks that access
stored training data.

Key ideas:
- Modern Hopfield networks with exponential storage capacity
- Training data as associative memory
- Iterative refinement of predictions
- Continuous attention-based retrieval

Paper: "Hopular: Modern Hopfield Networks for Tabular Data"
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
MODEL_NAME = "hopular"
OUTPUT_DIR = './out/hopular'

logger = logging.getLogger(__name__)
###############################################################################
# END CONSTANTS
###############################################################################


class ModernHopfieldLayer(layers.Layer):
    """
    Modern Hopfield layer with continuous attention.

    Uses softmax-based associative memory retrieval.
    """

    def __init__(self, embed_dim, beta=1.0, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.beta = beta  # Inverse temperature
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Query, Key, Value projections
        self.query_proj = layers.Dense(self.embed_dim)
        self.key_proj = layers.Dense(self.embed_dim)
        self.value_proj = layers.Dense(self.embed_dim)

        # Output projection
        self.output_proj = layers.Dense(self.embed_dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, query, memory, training=False):
        """
        Modern Hopfield retrieval.

        Args:
            query: (batch, embed_dim) query state
            memory: (n_memory, embed_dim) stored patterns

        Returns:
            retrieved: (batch, embed_dim) retrieved pattern
        """
        # Project
        Q = self.query_proj(query)  # (batch, embed_dim)
        K = self.key_proj(memory)   # (n_memory, embed_dim)
        V = self.value_proj(memory) # (n_memory, embed_dim)

        # Attention scores with temperature
        scores = tf.matmul(Q, K, transpose_b=True) * self.beta
        scores = scores / tf.sqrt(tf.cast(self.embed_dim, tf.float32))

        # Softmax attention
        attention = tf.nn.softmax(scores, axis=-1)  # (batch, n_memory)

        # Retrieve
        retrieved = tf.matmul(attention, V)  # (batch, embed_dim)

        # Output
        output = self.output_proj(retrieved)
        output = self.dropout(output, training=training)
        output = self.norm(query + output, training=training)

        return output, attention


class HopularBlock(layers.Layer):
    """
    Hopular block: Hopfield layer + FFN with residuals.
    """

    def __init__(self, embed_dim, ff_dim, beta=1.0, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.beta = beta
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.hopfield = ModernHopfieldLayer(self.embed_dim, self.beta, self.dropout_rate)

        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embed_dim),
            layers.Dropout(self.dropout_rate)
        ])
        self.ffn_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, query, memory, training=False):
        # Hopfield retrieval
        hopfield_out, attention = self.hopfield(query, memory, training=training)

        # FFN with residual
        ffn_out = self.ffn(hopfield_out, training=training)
        output = self.ffn_norm(hopfield_out + ffn_out, training=training)

        return output, attention


def build_hopular(input_shape,
                  n_blocks=3,
                  embed_dim=128,
                  ff_dim=256,
                  n_memory=64,
                  beta=1.0,
                  dropout_rate=0.1):
    """
    Build Hopular model for tabular regression.

    Args:
        input_shape: Number of input features
        n_blocks: Number of Hopular blocks
        embed_dim: Embedding dimension
        ff_dim: Feed-forward dimension
        n_memory: Number of learnable memory patterns
        beta: Inverse temperature for softmax
        dropout_rate: Dropout rate

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(input_shape,))

    # Input encoding
    x = layers.BatchNormalization()(inputs)
    query = layers.Dense(embed_dim)(x)
    query = layers.LayerNormalization(epsilon=1e-6)(query)

    # Learnable memory patterns (pseudo training data)
    memory_layer = layers.Dense(n_memory * embed_dim, use_bias=False)
    memory_flat = memory_layer(tf.ones((1, 1)))
    memory = tf.reshape(memory_flat, (n_memory, embed_dim))

    # Stack of Hopular blocks
    all_attentions = []
    for i in range(n_blocks):
        block = HopularBlock(embed_dim, ff_dim, beta, dropout_rate, name=f'hopular_block_{i}')
        query, attention = block(query, memory)
        all_attentions.append(attention)

    # Output head
    output = layers.Dense(64, activation='relu')(query)
    output = layers.Dropout(dropout_rate)(output)
    output = layers.Dense(1)(output)

    model = Model(inputs, output, name='Hopular')
    return model


def build_hopular_iterative(input_shape,
                            n_iterations=4,
                            embed_dim=128,
                            n_memory=32,
                            beta=2.0,
                            dropout_rate=0.1):
    """
    Iterative Hopular: same block applied multiple times.

    More faithful to original Hopfield network dynamics.
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)
    query = layers.Dense(embed_dim)(x)
    query = layers.LayerNormalization(epsilon=1e-6)(query)

    # Learnable memory
    memory_init = layers.Dense(n_memory * embed_dim, use_bias=False)(tf.ones((1, 1)))
    memory = tf.reshape(memory_init, (n_memory, embed_dim))

    # Single block applied iteratively
    hopfield_block = HopularBlock(embed_dim, embed_dim * 2, beta, dropout_rate)

    for _ in range(n_iterations):
        query, _ = hopfield_block(query, memory)

    output = layers.Dense(64, activation='relu')(query)
    output = layers.Dropout(dropout_rate)(output)
    output = layers.Dense(1)(output)

    model = Model(inputs, output, name='Hopular_Iterative')
    return model


def build_hopular_with_raw(input_shape,
                           n_blocks=2,
                           embed_dim=128,
                           n_memory=32,
                           dropout_rate=0.1):
    """
    Hopular with shortcut from raw input.

    Combines memory-based reasoning with direct feature processing.
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    # Direct path
    direct = layers.Dense(64, activation='relu')(x)
    direct = layers.Dropout(dropout_rate)(direct)

    # Hopfield path
    query = layers.Dense(embed_dim)(x)
    query = layers.LayerNormalization(epsilon=1e-6)(query)

    memory_init = layers.Dense(n_memory * embed_dim, use_bias=False)(tf.ones((1, 1)))
    memory = tf.reshape(memory_init, (n_memory, embed_dim))

    for i in range(n_blocks):
        block = HopularBlock(embed_dim, embed_dim * 2, beta=1.0, dropout_rate=dropout_rate)
        query, _ = block(query, memory)

    # Combine paths
    combined = layers.Concatenate()([direct, query])
    combined = layers.Dense(64, activation='relu')(combined)
    combined = layers.Dropout(dropout_rate)(combined)
    output = layers.Dense(1)(combined)

    model = Model(inputs, output, name='Hopular_Raw')
    return model


###############################################################################
# OPTUNA OBJECTIVE
###############################################################################

def build_model_for_trial(trial, input_dim):
    """Build and compile model for Optuna trial."""
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    n_blocks = trial.suggest_int('n_blocks', 2, 4)
    embed_dim = trial.suggest_categorical('embed_dim', [64, 128, 256])
    n_memory = trial.suggest_categorical('n_memory', [32, 64, 128])
    beta = trial.suggest_float('beta', 0.5, 2.0)

    model = build_hopular(
        input_shape=input_dim,
        n_blocks=n_blocks,
        embed_dim=embed_dim,
        ff_dim=embed_dim * 2,
        n_memory=n_memory,
        beta=beta,
        dropout_rate=dropout_rate
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError(), r2_keras]
    )
    return model


def build_model_from_params(params: dict, input_dim: int):
    """Build and compile model from best parameters."""
    model = build_hopular(
        input_shape=input_dim,
        n_blocks=params['n_blocks'],
        embed_dim=params['embed_dim'],
        ff_dim=params['embed_dim'] * 2,
        n_memory=params['n_memory'],
        beta=params['beta'],
        dropout_rate=params['dropout_rate']
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=params['learning_rate'],
        weight_decay=params['weight_decay'],
        clipnorm=1.0
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
    setup_logging_for_progress(os.path.join(OUTPUT_DIR, 'training_hopular.log'))
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
    parser = argparse.ArgumentParser(description='Train Hopular with Optuna tuning')
    parser.add_argument('train_path', type=str, help='Path to training data')
    parser.add_argument('test_path', type=str, help='Path to test data')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to sample weights JSON file')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment')
    add_wandb_args(parser)
    args = parser.parse_args()
    main(args.train_path, args.test_path, args.weights_path, args.experiment_name, args)
