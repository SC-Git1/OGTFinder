"""E15: NODE - Neural Oblivious Decision Ensembles

With Optuna hyperparameter tuning.

Architecture: Layers of differentiable oblivious decision trees (ODTs).

Paper: "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data"
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
MODEL_NAME = "node"
OUTPUT_DIR = './out/node'

logger = logging.getLogger(__name__)
###############################################################################
# END CONSTANTS
###############################################################################


def sparsemax(logits, axis=-1):
    """
    Sparsemax activation: projects onto probability simplex.
    More sparse than softmax.
    """
    logits = logits - tf.reduce_max(logits, axis=axis, keepdims=True)
    
    # Sort in descending order
    sorted_logits = tf.sort(logits, axis=axis, direction='DESCENDING')
    
    # Compute cumsum
    cumsum = tf.cumsum(sorted_logits, axis=axis)
    
    # Find threshold
    k = tf.cast(tf.range(1, tf.shape(logits)[axis] + 1), logits.dtype)
    k = tf.reshape(k, [1] * axis + [-1] + [1] * (len(logits.shape) - axis - 1))
    
    threshold = (cumsum - 1) / k
    support = tf.cast(sorted_logits > threshold, logits.dtype)
    
    k_max = tf.reduce_sum(support, axis=axis, keepdims=True)
    tau = (tf.reduce_sum(sorted_logits * support, axis=axis, keepdims=True) - 1) / k_max
    
    output = tf.maximum(logits - tau, 0)
    return output


def entmax15(logits, axis=-1):
    """
    Entmax 1.5: between softmax (alpha=1) and sparsemax (alpha=2).
    """
    # Simplified entmax approximation using mixture
    softmax_out = tf.nn.softmax(logits, axis=axis)
    sparsemax_out = sparsemax(logits, axis=axis)
    return 0.5 * softmax_out + 0.5 * sparsemax_out


class FeatureSelection(layers.Layer):
    """
    Differentiable feature selection for each tree.
    Learns which features to use for splits.
    """

    def __init__(self, n_features, n_selected, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.n_selected = n_selected

    def build(self, input_shape):
        self.selection_logits = self.add_weight(
            name='selection_logits',
            shape=(self.n_selected, self.n_features),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs, training=False):
        # Soft selection via entmax
        selection_weights = entmax15(self.selection_logits, axis=-1)  # (n_selected, n_features)
        # Select features: (batch, n_features) @ (n_features, n_selected) = (batch, n_selected)
        selected = tf.matmul(inputs, selection_weights, transpose_b=True)
        return selected


class ObliviousDecisionTree(layers.Layer):
    """
    Single differentiable oblivious decision tree.
    
    In an ODT, all nodes at the same depth use the same splitting feature.
    With depth d, we have 2^d leaves.
    """

    def __init__(self, depth=6, output_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.output_dim = output_dim

    def build(self, input_shape):
        n_features = input_shape[-1]

        # Feature indices for each depth (one feature per depth level)
        self.feature_selector = layers.Dense(
            self.depth, use_bias=False,
            kernel_initializer='glorot_uniform'
        )

        # Thresholds for each depth
        self.thresholds = self.add_weight(
            name='thresholds',
            shape=(self.depth,),
            initializer='zeros',
            trainable=True
        )

        # Leaf responses
        self.leaf_responses = self.add_weight(
            name='leaf_responses',
            shape=(self.n_leaves, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )

        # Temperature for soft routing
        self.temperature = self.add_weight(
            name='temperature',
            shape=(),
            initializer=keras.initializers.Constant(1.0),
            trainable=True,
            constraint=keras.constraints.NonNeg()
        )

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        # Get features for each depth
        features = self.feature_selector(inputs)  # (batch, depth)

        # Compute split decisions (soft)
        # decision_i = sigmoid((feature_i - threshold_i) / temperature)
        temp = tf.maximum(self.temperature, 0.1)
        decisions = tf.nn.sigmoid((features - self.thresholds) / temp)  # (batch, depth)

        # Compute leaf probabilities
        # Each leaf is defined by a binary path through the tree
        # Leaf index in binary: bit i = 1 means "go right at depth i"
        leaf_probs = tf.ones((batch_size, 1))

        for d in range(self.depth):
            decision_d = decisions[:, d:d+1]  # (batch, 1)
            # For each existing path, branch left (1-decision) and right (decision)
            left_probs = leaf_probs * (1 - decision_d)
            right_probs = leaf_probs * decision_d
            leaf_probs = tf.concat([left_probs, right_probs], axis=1)

        # leaf_probs: (batch, n_leaves)
        # Weighted sum of leaf responses
        output = tf.matmul(leaf_probs, self.leaf_responses)  # (batch, output_dim)

        return output


class NODELayer(layers.Layer):
    """
    NODE layer: ensemble of oblivious decision trees.
    """

    def __init__(self, n_trees=128, depth=6, output_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.n_trees = n_trees
        self.depth = depth
        self.output_dim = output_dim

    def build(self, input_shape):
        self.trees = [
            ObliviousDecisionTree(self.depth, self.output_dim, name=f'tree_{i}')
            for i in range(self.n_trees)
        ]

    def call(self, inputs, training=False):
        # Run all trees and average
        outputs = [tree(inputs, training=training) for tree in self.trees]
        stacked = tf.stack(outputs, axis=1)  # (batch, n_trees, output_dim)
        return tf.reduce_mean(stacked, axis=1)  # (batch, output_dim)


class DenseODT(layers.Layer):
    """
    Dense Oblivious Decision Tree: more efficient implementation.
    Uses dense operations instead of explicit tree traversal.
    """

    def __init__(self, n_trees=64, depth=4, output_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.n_trees = n_trees
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.output_dim = output_dim

    def build(self, input_shape):
        n_features = input_shape[-1]

        # Feature weights for tree split features
        # Each tree uses 'depth' features (one per level)
        self.feature_weights = self.add_weight(
            name='feature_weights',
            shape=(self.n_trees, self.depth, n_features),
            initializer='glorot_uniform',
            trainable=True
        )

        # Thresholds
        self.thresholds = self.add_weight(
            name='thresholds',
            shape=(self.n_trees, self.depth),
            initializer='zeros',
            trainable=True
        )

        # Leaf values
        self.leaf_values = self.add_weight(
            name='leaf_values',
            shape=(self.n_trees, self.n_leaves, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )

        # Learnable temperature
        self.log_temp = self.add_weight(
            name='log_temperature',
            shape=(self.n_trees,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        # Compute split features: (batch, n_trees, depth)
        # inputs: (batch, features), weights: (n_trees, depth, features)
        features = tf.einsum('bf,tdf->btd', inputs, self.feature_weights)

        # Temperature
        temperature = tf.exp(self.log_temp) + 0.1  # (n_trees,)
        temperature = tf.reshape(temperature, (1, self.n_trees, 1))

        # Soft decisions
        decisions = tf.nn.sigmoid((features - self.thresholds) / temperature)  # (batch, n_trees, depth)

        # Compute leaf probabilities
        # Binary expansion for leaf indices
        leaf_probs = tf.ones((batch_size, self.n_trees, 1))

        for d in range(self.depth):
            dec = decisions[:, :, d:d+1]  # (batch, n_trees, 1)
            leaf_probs = tf.concat([
                leaf_probs * (1 - dec),
                leaf_probs * dec
            ], axis=-1)

        # leaf_probs: (batch, n_trees, n_leaves)
        # Weighted leaf values
        # leaf_values: (n_trees, n_leaves, output_dim)
        output = tf.einsum('btl,tlo->bto', leaf_probs, self.leaf_values)  # (batch, n_trees, output_dim)

        # Average over trees
        return tf.reduce_mean(output, axis=1)  # (batch, output_dim)


def build_node(input_shape,
               n_trees=64,
               depth=4,
               hidden_dim=64,
               n_layers=2,
               dropout_rate=0.1):
    """
    Build NODE model for tabular regression.
    
    Uses stacked NODE layers with residual connections.
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    # Initial projection
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Stacked NODE layers with residuals
    for i in range(n_layers):
        node_out = DenseODT(n_trees, depth, hidden_dim, name=f'node_layer_{i}')(x)
        node_out = layers.LayerNormalization(epsilon=1e-6)(node_out)
        node_out = layers.Dropout(dropout_rate)(node_out)
        x = x + node_out  # Residual

    # Output head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='NODE')
    return model


def build_node_simple(input_shape,
                      n_trees=128,
                      depth=6,
                      dropout_rate=0.1):
    """
    Simple NODE: single layer of ODTs.
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    # Feature preprocessing
    x = layers.Dense(256, activation='relu')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(dropout_rate)(x)

    # NODE layer
    node_out = DenseODT(n_trees, depth, 64)(x)
    node_out = layers.LayerNormalization(epsilon=1e-6)(node_out)

    # Output
    x = layers.Dense(32, activation='relu')(node_out)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='NODE_Simple')
    return model


def build_node_deep(input_shape,
                    n_trees=32,
                    depth=4,
                    hidden_dim=128,
                    n_layers=4,
                    dropout_rate=0.1):
    """
    Deep NODE: more layers, fewer trees per layer.
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    # Feature preprocessing
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Deep stack with dense connections
    features = [x]

    for i in range(n_layers):
        # Concatenate all previous features (DenseNet-style)
        if len(features) > 1:
            concat = layers.Concatenate()(features)
            proj = layers.Dense(hidden_dim, activation='relu')(concat)
        else:
            proj = x

        node_out = DenseODT(n_trees, depth, hidden_dim, name=f'node_{i}')(proj)
        node_out = layers.LayerNormalization(epsilon=1e-6)(node_out)
        node_out = layers.Dropout(dropout_rate)(node_out)

        features.append(node_out)
        x = node_out

    # Final aggregation
    final_concat = layers.Concatenate()(features)
    x = layers.Dense(128, activation='relu')(final_concat)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='NODE_Deep')
    return model


###############################################################################
# OPTUNA OBJECTIVE
###############################################################################

def build_model_for_trial(trial, input_dim):
    """Build and compile model for Optuna trial."""
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    n_trees = trial.suggest_categorical('n_trees', [32, 64, 128])
    depth = trial.suggest_int('depth', 3, 6)
    n_layers = trial.suggest_int('n_layers', 1, 4)
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])

    model = build_node(
        input_shape=input_dim,
        n_trees=n_trees,
        depth=depth,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
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
    model = build_node(
        input_shape=input_dim,
        n_trees=params['n_trees'],
        depth=params['depth'],
        hidden_dim=params['hidden_dim'],
        n_layers=params['n_layers'],
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
    setup_logging_for_progress(os.path.join(OUTPUT_DIR, 'training_node.log'))
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
    parser = argparse.ArgumentParser(description='Train NODE with Optuna tuning')
    parser.add_argument('train_path', type=str, help='Path to training data')
    parser.add_argument('test_path', type=str, help='Path to test data')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to sample weights JSON file')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment')
    add_wandb_args(parser)
    args = parser.parse_args()
    main(args.train_path, args.test_path, args.weights_path, args.experiment_name, args)
