"""D14: SAINT - Self-Attention and Intersample Attention Transformer

With Optuna hyperparameter tuning.

Architecture: Attention over columns (features) AND rows (samples).
Uses grouped variant for efficiency with high-dimensional inputs.

Paper: "SAINT: Improved Neural Networks for Tabular Data via Row Attention"
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
MODEL_NAME = "saint"
OUTPUT_DIR = './out/saint'

logger = logging.getLogger(__name__)
###############################################################################
# END CONSTANTS
###############################################################################


class FeatureEmbedding(layers.Layer):
    """Embed each feature into a d-dimensional space."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        n_features = input_shape[-1]
        self.embeddings = self.add_weight(
            name='feature_embeddings',
            shape=(n_features, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.biases = self.add_weight(
            name='feature_biases',
            shape=(n_features, self.embed_dim),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        # inputs: (batch, n_features)
        x = tf.expand_dims(inputs, -1)  # (batch, n_features, 1)
        embedded = x * self.embeddings + self.biases  # (batch, n_features, embed_dim)
        return embedded


class ColumnAttention(layers.Layer):
    """Self-attention over columns (features)."""

    def __init__(self, embed_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate
        )
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, x, training=False):
        # x: (batch, n_features, embed_dim)
        attn_out = self.mha(x, x, training=training)
        attn_out = self.dropout(attn_out, training=training)
        return self.layernorm(x + attn_out)


class RowAttention(layers.Layer):
    """
    Intersample attention: attention over rows (samples) within a batch.
    Each feature position attends to the same feature across samples.
    """

    def __init__(self, embed_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate
        )
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, x, training=False):
        # x: (batch, n_features, embed_dim)
        # Transpose to (n_features, batch, embed_dim) for row attention
        x_t = tf.transpose(x, [1, 0, 2])
        attn_out = self.mha(x_t, x_t, training=training)
        attn_out = self.dropout(attn_out, training=training)
        out = self.layernorm(x_t + attn_out)
        # Transpose back to (batch, n_features, embed_dim)
        return tf.transpose(out, [1, 0, 2])


class SAINTBlock(layers.Layer):
    """
    SAINT block: Column attention → Row attention → FFN
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.col_attn = ColumnAttention(
            self.embed_dim, self.num_heads, self.dropout_rate
        )
        self.row_attn = RowAttention(
            self.embed_dim, self.num_heads, self.dropout_rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embed_dim),
            layers.Dropout(self.dropout_rate)
        ])
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        # Column attention
        x = self.col_attn(x, training=training)
        # Row attention (intersample)
        x = self.row_attn(x, training=training)
        # FFN with residual
        ffn_out = self.ffn(x, training=training)
        return self.layernorm(x + ffn_out)


class SAINTEncoder(layers.Layer):
    """SAINT encoder: stacked SAINT blocks."""

    def __init__(self, embed_dim, num_heads, ff_dim, num_blocks,
                 dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.blocks = [
            SAINTBlock(self.embed_dim, self.num_heads, self.ff_dim,
                       self.dropout_rate, name=f'saint_block_{i}')
            for i in range(self.num_blocks)
        ]
        self.final_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        return self.final_norm(x)


def build_saint(input_shape,
                embed_dim=32,
                num_heads=4,
                ff_dim=64,
                num_blocks=2,
                dropout_rate=0.1):
    """
    Build SAINT model for tabular regression.

    For 1000 features, we use grouped variant to manage memory.
    """
    inputs = layers.Input(shape=(input_shape,))

    # Input normalization
    x = layers.BatchNormalization()(inputs)

    # Feature embedding
    embedder = FeatureEmbedding(embed_dim)
    x = embedder(x)  # (batch, n_features, embed_dim)

    # SAINT encoder
    encoder = SAINTEncoder(embed_dim, num_heads, ff_dim, num_blocks, dropout_rate)
    x = encoder(x)  # (batch, n_features, embed_dim)

    # Global pooling
    x = layers.Lambda(lambda t: ops.mean(t, axis=1))(x)  # (batch, embed_dim)

    # Output head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='SAINT')
    return model


def build_saint_grouped(input_shape,
                        n_groups=50,
                        embed_dim=32,
                        num_heads=4,
                        ff_dim=64,
                        num_blocks=2,
                        dropout_rate=0.1):
    """
    SAINT with feature grouping for high-dimensional inputs.
    Groups features to reduce sequence length.
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    # Group features
    group_size = input_shape // n_groups
    n_used = n_groups * group_size
    x_grouped = layers.Reshape((n_groups, group_size))(x[:, :n_used])

    # Embed groups
    x_embed = layers.Dense(embed_dim)(x_grouped)  # (batch, n_groups, embed_dim)

    # SAINT encoder on grouped features
    encoder = SAINTEncoder(embed_dim, num_heads, ff_dim, num_blocks, dropout_rate)
    x = encoder(x_embed)

    # Pool
    x = layers.Lambda(lambda t: ops.mean(t, axis=1))(x)

    # Output
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='SAINT_Grouped')
    return model


def cutmix_mixup_augment(x, y, alpha=0.2, cutmix_prob=0.5):
    """
    CutMix + MixUp augmentation for SAINT pretraining.
    
    CutMix: swaps random feature subsets between samples
    MixUp: interpolates between samples
    """
    batch_size = tf.shape(x)[0]
    n_features = tf.shape(x)[1]

    # Random lambda
    lam = tf.random.uniform([], 0, alpha)

    # Random permutation for mixing
    indices = tf.random.shuffle(tf.range(batch_size))
    x_shuffled = tf.gather(x, indices)
    y_shuffled = tf.gather(y, indices)

    # Decide CutMix or MixUp
    if tf.random.uniform([]) < cutmix_prob:
        # CutMix: binary mask for features
        mask = tf.cast(tf.random.uniform([1, n_features]) > lam, tf.float32)
        x_mixed = x * mask + x_shuffled * (1 - mask)
    else:
        # MixUp: interpolation
        x_mixed = lam * x + (1 - lam) * x_shuffled

    y_mixed = lam * y + (1 - lam) * y_shuffled

    return x_mixed, y_mixed, lam


class SAINTPretrainModel(keras.Model):
    """
    SAINT with contrastive pretraining.
    
    Pretraining tasks:
    1. Denoising: reconstruct original from corrupted input
    2. Contrastive: distinguish augmented views
    """

    def __init__(self, input_dim, embed_dim=32, num_heads=4, ff_dim=64,
                 num_blocks=2, n_groups=50, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_groups = n_groups

        # Feature grouping
        group_size = input_dim // n_groups
        self.n_used = n_groups * group_size

        self.input_norm = layers.BatchNormalization()
        self.group_embed = layers.Dense(embed_dim)
        self.encoder = SAINTEncoder(embed_dim, num_heads, ff_dim, num_blocks, dropout_rate)

        # Projection head for contrastive learning
        self.projector = keras.Sequential([
            layers.Dense(embed_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])

        # Reconstruction head
        self.reconstructor = layers.Dense(input_dim)

        # Regression head (for fine-tuning)
        self.regressor = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(1)
        ])

    def encode(self, x, training=False):
        x = self.input_norm(x, training=training)
        x_grouped = tf.reshape(x[:, :self.n_used], (-1, self.n_groups, self.n_used // self.n_groups))
        x_embed = self.group_embed(x_grouped)
        encoded = self.encoder(x_embed, training=training)
        pooled = tf.reduce_mean(encoded, axis=1)
        return pooled

    def call(self, x, training=False):
        encoded = self.encode(x, training=training)
        return self.regressor(encoded)

    def pretrain_step(self, x, noise_ratio=0.3):
        """Single pretraining step with denoising + contrastive objectives."""
        batch_size = tf.shape(x)[0]

        # Create corrupted view
        noise_mask = tf.cast(tf.random.uniform(tf.shape(x)) < noise_ratio, tf.float32)
        noise = tf.random.normal(tf.shape(x)) * 0.1
        x_corrupted = x * (1 - noise_mask) + noise * noise_mask

        with tf.GradientTape() as tape:
            # Encode both views
            z_clean = self.encode(x, training=True)
            z_corrupted = self.encode(x_corrupted, training=True)

            # Project
            p_clean = self.projector(z_clean)
            p_corrupted = self.projector(z_corrupted)

            # Contrastive loss (InfoNCE-style)
            p_clean_norm = tf.nn.l2_normalize(p_clean, axis=1)
            p_corrupted_norm = tf.nn.l2_normalize(p_corrupted, axis=1)
            similarity = tf.matmul(p_clean_norm, p_corrupted_norm, transpose_b=True)
            labels = tf.range(batch_size)
            contrastive_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels, similarity / 0.1)
            )

            # Reconstruction loss
            x_reconstructed = self.reconstructor(z_corrupted)
            recon_loss = tf.reduce_mean(tf.square(x - x_reconstructed))

            total_loss = contrastive_loss + recon_loss

        return total_loss, tape


def build_saint_lite(input_shape,
                     compress_dim=128,
                     embed_dim=32,
                     num_heads=4,
                     ff_dim=64,
                     num_blocks=2,
                     dropout_rate=0.1):
    """
    Lite SAINT: compress features first, then apply SAINT.
    More memory-efficient for very high-dimensional inputs.
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    # Compress
    x = layers.Dense(compress_dim, activation='relu')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Embed compressed features
    embedder = FeatureEmbedding(embed_dim)
    x = embedder(x)

    # SAINT encoder (column attention only for compressed representation)
    for i in range(num_blocks):
        col_attn = ColumnAttention(embed_dim, num_heads, dropout_rate)
        x = col_attn(x)

        ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim)
        ])
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn(x))

    # Pool and output
    x = layers.Lambda(lambda t: ops.mean(t, axis=1))(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='SAINT_Lite')
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
    num_blocks = trial.suggest_int('num_blocks', 1, 4)
    n_groups = trial.suggest_categorical('n_groups', [25, 50, 100])

    model = build_saint_grouped(
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
    model = build_saint_grouped(
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
    setup_logging_for_progress(os.path.join(OUTPUT_DIR, 'training_saint.log'))
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
    parser = argparse.ArgumentParser(description='Train SAINT with Optuna tuning')
    parser.add_argument('train_path', type=str, help='Path to training data')
    parser.add_argument('test_path', type=str, help='Path to test data')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to sample weights JSON file')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment')
    add_wandb_args(parser)
    args = parser.parse_args()
    main(args.train_path, args.test_path, args.weights_path, args.experiment_name, args)
