"""Attention MLP for Tabular Data Regression

With Optuna hyperparameter tuning.

Architecture: Multi-head self-attention mechanism combined with MLP.
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
MODEL_NAME = "attention_mlp"
OUTPUT_DIR = './out/attention_mlp'

logger = logging.getLogger(__name__)
###############################################################################
# END CONSTANTS
###############################################################################


class FeatureEmbedding(layers.Layer):
    """Embed each feature into a higher-dimensional space."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.num_features = input_shape[-1]
        self.embedding = layers.Dense(self.embed_dim)

    def call(self, inputs):
        # Reshape: (batch, features) -> (batch, features, 1)
        x = tf.expand_dims(inputs, -1)
        # Embed each feature: (batch, features, embed_dim)
        x = self.embedding(x)
        return x


class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention for tabular features."""

    def __init__(self, embed_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

    def build(self, input_shape):
        self.query = layers.Dense(self.embed_dim)
        self.key = layers.Dense(self.embed_dim)
        self.value = layers.Dense(self.embed_dim)
        self.combine = layers.Dense(self.embed_dim)
        self.dropout = layers.Dropout(self.dropout_rate)

    def split_heads(self, x, batch_size):
        """Split last dim into (num_heads, head_dim)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, heads, seq, head_dim)

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        # Linear projections
        q = self.query(inputs)  # (batch, seq, embed_dim)
        k = self.key(inputs)
        v = self.value(inputs)

        # Split heads
        q = self.split_heads(q, batch_size)  # (batch, heads, seq, head_dim)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        attention_scores = tf.matmul(q, k, transpose_b=True) / scale
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention to values
        attended = tf.matmul(attention_weights, v)  # (batch, heads, seq, head_dim)

        # Concatenate heads
        attended = tf.transpose(attended, perm=[0, 2, 1, 3])  # (batch, seq, heads, head_dim)
        attended = tf.reshape(attended, (batch_size, -1, self.embed_dim))

        # Final projection
        output = self.combine(attended)
        return output, attention_weights


class AttentionBlock(layers.Layer):
    """Transformer-style attention block with FFN."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.attention = MultiHeadSelfAttention(
            self.embed_dim, self.num_heads, self.dropout_rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embed_dim),
            layers.Dropout(self.dropout_rate)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        # Multi-head attention with residual
        attn_output, attn_weights = self.attention(inputs, training=training)
        x = self.layernorm1(inputs + attn_output)

        # Feed-forward with residual
        ffn_output = self.ffn(x, training=training)
        x = self.layernorm2(x + ffn_output)

        return x, attn_weights


def build_attention_mlp(input_shape,
                        embed_dim=32,
                        num_heads=4,
                        ff_dim=64,
                        num_attention_blocks=2,
                        mlp_units=[256, 128, 64],
                        dropout_rate=0.2,
                        l2_reg=1e-4):
    """
    Build Attention MLP for tabular regression.

    Args:
        input_shape: Number of input features
        embed_dim: Embedding dimension per feature
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension in attention blocks
        num_attention_blocks: Number of attention blocks
        mlp_units: Units in final MLP layers
        dropout_rate: Dropout rate
        l2_reg: L2 regularization

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(input_shape,))

    # Embed features
    x = FeatureEmbedding(embed_dim)(inputs)  # (batch, features, embed_dim)

    # Attention blocks
    attention_weights_list = []
    for i in range(num_attention_blocks):
        x, attn_weights = AttentionBlock(
            embed_dim, num_heads, ff_dim, dropout_rate,
            name=f'attention_block_{i}'
        )(x)
        attention_weights_list.append(attn_weights)

    # Aggregate: mean pooling over features
    x = layers.GlobalAveragePooling1D()(x)

    # MLP head
    for units in mlp_units:
        x = layers.Dense(
            units,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='Attention_MLP')
    return model


def build_feature_attention_mlp(input_shape,
                                attention_dim=64,
                                hidden_dims=[512, 256, 128],
                                dropout_rate=0.3,
                                l2_reg=1e-4):
    """
    Simplified attention MLP with feature-wise attention scores.

    Uses attention mechanism to weight features before MLP processing.
    Simpler and more efficient than full transformer attention.
    """
    inputs = layers.Input(shape=(input_shape,))

    # Batch normalization
    x = layers.BatchNormalization()(inputs)

    # Compute feature attention scores
    attention = layers.Dense(
        attention_dim,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    attention = layers.Dense(
        attention_dim,
        activation='relu'
    )(attention)
    attention_scores = layers.Dense(input_shape)(attention)
    attention_weights = layers.Softmax(name='feature_attention')(attention_scores)

    # Apply attention
    attended = layers.Multiply()([x, attention_weights])

    # Combine original and attended features
    combined = layers.Concatenate()([x, attended])

    # MLP processing
    h = combined
    for units in hidden_dims:
        h = layers.Dense(
            units,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(h)
        h = layers.BatchNormalization()(h)
        h = layers.ReLU()(h)
        h = layers.Dropout(dropout_rate)(h)

    outputs = layers.Dense(1)(h)

    model = Model(inputs, outputs, name='Feature_Attention_MLP')
    return model


def create_callbacks(patience_es=30, patience_lr=15):
    """Create training callbacks."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience_es,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience_lr,
        min_lr=1e-6,
        verbose=1
    )

    return [early_stopping, reduce_lr]


def plot_results(history, metrics, output_dir='.'):
    """Create and save training plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Attention MLP: Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'attention_mlp_loss_curve.png'), dpi=150)
    plt.close()

    # Predictions scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(metrics['y_actual'], metrics['y_pred'], alpha=0.5, s=10)
    min_val = min(metrics['y_actual'].min(), metrics['y_pred'].min())
    max_val = max(metrics['y_actual'].max(), metrics['y_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.title(f"Attention MLP: Actual vs Predicted (R²={metrics['r2']:.4f})")
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(os.path.join(output_dir, 'attention_mlp_predictions.png'), dpi=150)
    plt.close()

    # Residuals
    residuals = metrics['y_actual'] - metrics['y_pred']
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True)
    plt.title('Attention MLP: Residuals Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'attention_mlp_residuals.png'), dpi=150)
    plt.close()


###############################################################################
# OPTUNA OBJECTIVE
###############################################################################

def build_model_for_trial(trial, input_dim):
    """Build and compile model for Optuna trial."""
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    attention_dim = trial.suggest_categorical('attention_dim', [32, 64, 128])
    hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512])

    model = build_feature_attention_mlp(
        input_shape=input_dim,
        attention_dim=attention_dim,
        hidden_dims=[hidden_dim, hidden_dim // 2, hidden_dim // 4],
        dropout_rate=dropout_rate,
        l2_reg=weight_decay
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
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
    hidden_dim = params['hidden_dim']
    model = build_feature_attention_mlp(
        input_shape=input_dim,
        attention_dim=params['attention_dim'],
        hidden_dims=[hidden_dim, hidden_dim // 2, hidden_dim // 4],
        dropout_rate=params['dropout_rate'],
        l2_reg=params['weight_decay']
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=params['learning_rate'],
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
    setup_logging_for_progress(os.path.join(OUTPUT_DIR, 'training_attention_mlp.log'))
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
    parser = argparse.ArgumentParser(description='Train Attention MLP with Optuna tuning')
    parser.add_argument('train_path', type=str, help='Path to training data')
    parser.add_argument('test_path', type=str, help='Path to test data')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to sample weights JSON file')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment')
    add_wandb_args(parser)
    args = parser.parse_args()
    main(args.train_path, args.test_path, args.weights_path, args.experiment_name, args)
