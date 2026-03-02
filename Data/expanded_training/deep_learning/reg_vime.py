"""F17: VIME - Value Imputation and Mask Estimation

With Optuna hyperparameter tuning.

Architecture: Self-supervised pretraining for tabular data.

Paper: "VIME: Extending the Success of Self- and Semi-supervised Learning
        to Tabular Domain"
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
MODEL_NAME = "vime"
OUTPUT_DIR = './out/vime'

logger = logging.getLogger(__name__)
###############################################################################
# END CONSTANTS
###############################################################################


def mask_generator(p_mask, x):
    """
    Generate random mask for corruption.
    
    Args:
        p_mask: probability of masking each feature
        x: input data (batch, n_features)
    
    Returns:
        mask: binary mask (batch, n_features), 1 = corrupted
    """
    mask = tf.cast(tf.random.uniform(tf.shape(x)) < p_mask, tf.float32)
    return mask


def pretext_generator(x, p_mask=0.3):
    """
    Generate corrupted input and mask for VIME pretraining.
    
    Corruption strategy: replace masked values with values from
    random samples (shuffled within batch).
    
    Args:
        x: input data (batch, n_features)
        p_mask: probability of masking
    
    Returns:
        x_tilde: corrupted input
        mask: corruption mask
    """
    batch_size = tf.shape(x)[0]
    n_features = tf.shape(x)[1]
    
    # Generate mask
    mask = mask_generator(p_mask, x)
    
    # Generate corrupted values by shuffling
    x_shuffled = tf.random.shuffle(x)
    
    # Apply corruption
    x_tilde = x * (1 - mask) + x_shuffled * mask
    
    return x_tilde, mask


class VIMEEncoder(layers.Layer):
    """
    VIME encoder: maps input to latent representation.
    Shared between pretraining and fine-tuning.
    """
    
    def __init__(self, hidden_units=[256, 128], latent_dim=64,
                 dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.input_norm = layers.BatchNormalization()
        
        self.hidden_layers = []
        self.norm_layers = []
        self.dropout_layers = []
        
        for units in self.hidden_units:
            self.hidden_layers.append(layers.Dense(units, activation='relu'))
            self.norm_layers.append(layers.LayerNormalization(epsilon=1e-6))
            self.dropout_layers.append(layers.Dropout(self.dropout_rate))
        
        self.latent_layer = layers.Dense(self.latent_dim)
        self.latent_norm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, training=False):
        x = self.input_norm(inputs, training=training)
        
        for dense, norm, dropout in zip(self.hidden_layers, self.norm_layers, self.dropout_layers):
            x = dense(x)
            x = norm(x, training=training)
            x = dropout(x, training=training)
        
        latent = self.latent_layer(x)
        latent = self.latent_norm(latent, training=training)
        
        return latent


class VIMEMaskPredictor(layers.Layer):
    """Predicts which features were masked (corrupted)."""
    
    def __init__(self, n_features, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.hidden_dim = hidden_dim
    
    def build(self, input_shape):
        self.hidden = layers.Dense(self.hidden_dim, activation='relu')
        self.output_layer = layers.Dense(self.n_features, activation='sigmoid')
    
    def call(self, latent, training=False):
        x = self.hidden(latent)
        return self.output_layer(x)


class VIMEReconstructor(layers.Layer):
    """Reconstructs original features from latent representation."""
    
    def __init__(self, n_features, hidden_units=[128, 256], **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.hidden_units = hidden_units
    
    def build(self, input_shape):
        self.hidden_layers = []
        for units in self.hidden_units:
            self.hidden_layers.append(layers.Dense(units, activation='relu'))
        self.output_layer = layers.Dense(self.n_features)
    
    def call(self, latent, training=False):
        x = latent
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


class VIMEPretrainModel(keras.Model):
    """
    VIME pretraining model.
    
    Two objectives:
    1. Reconstruction: decode(encode(corrupt(x))) ≈ x
    2. Mask estimation: predict which features were corrupted
    """
    
    def __init__(self, n_features, hidden_units=[256, 128], latent_dim=64,
                 dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        
        self.encoder = VIMEEncoder(hidden_units, latent_dim, dropout_rate)
        self.mask_predictor = VIMEMaskPredictor(n_features)
        self.reconstructor = VIMEReconstructor(n_features)
    
    def call(self, inputs, training=False):
        latent = self.encoder(inputs, training=training)
        mask_pred = self.mask_predictor(latent, training=training)
        recon = self.reconstructor(latent, training=training)
        return mask_pred, recon
    
    def pretrain_step(self, x, p_mask=0.3):
        """
        Single pretraining step.
        
        Returns loss and gradients.
        """
        # Generate corrupted input
        x_tilde, mask = pretext_generator(x, p_mask)
        
        with tf.GradientTape() as tape:
            mask_pred, recon = self(x_tilde, training=True)
            
            # Mask estimation loss (binary cross-entropy)
            mask_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(mask, mask_pred)
            )
            
            # Reconstruction loss (MSE)
            recon_loss = tf.reduce_mean(tf.square(x - recon))
            
            # Combined loss
            total_loss = mask_loss + recon_loss
        
        grads = tape.gradient(total_loss, self.trainable_variables)
        return total_loss, mask_loss, recon_loss, grads


class VIMEFineTuneModel(keras.Model):
    """
    VIME model for fine-tuning on downstream task.
    Uses pretrained encoder + new regression head.
    """
    
    def __init__(self, pretrained_encoder, hidden_dim=64, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = pretrained_encoder
        
        self.head = keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(hidden_dim // 2, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(1)
        ])
    
    def call(self, inputs, training=False):
        latent = self.encoder(inputs, training=training)
        return self.head(latent, training=training)


def build_vime_pretrain(input_shape, hidden_units=[256, 128], latent_dim=64,
                        dropout_rate=0.1):
    """Build VIME pretraining model."""
    return VIMEPretrainModel(
        n_features=input_shape,
        hidden_units=hidden_units,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate
    )


def build_vime_supervised(input_shape, hidden_units=[256, 128], latent_dim=64,
                          dropout_rate=0.1):
    """
    Build VIME model for supervised training (no pretraining).
    Uses same architecture but trains end-to-end.
    """
    inputs = layers.Input(shape=(input_shape,))
    
    x = layers.BatchNormalization()(inputs)
    
    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    latent = layers.Dense(latent_dim)(x)
    latent = layers.LayerNormalization(epsilon=1e-6)(latent)
    
    # Regression head
    x = layers.Dense(64, activation='relu')(latent)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs, outputs, name='VIME_Supervised')
    return model


def build_vime_semi_supervised(input_shape, hidden_units=[256, 128], latent_dim=64,
                               dropout_rate=0.1):
    """
    VIME with semi-supervised consistency regularization.
    
    Adds consistency loss: predictions should be similar for
    clean and slightly corrupted versions of the same input.
    """
    inputs = layers.Input(shape=(input_shape,))
    
    x = layers.BatchNormalization()(inputs)
    
    # Encoder
    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    latent = layers.Dense(latent_dim)(x)
    latent = layers.LayerNormalization(epsilon=1e-6)(latent)
    
    # Regression head
    x = layers.Dense(64, activation='relu')(latent)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs, outputs, name='VIME_SemiSupervised')
    return model


def pretrain_vime(model, X_train, pretrain_epochs=100, batch_size=256,
                  learning_rate=1e-3, p_mask=0.3):
    """
    Run VIME pretraining.
    
    Args:
        model: VIMEPretrainModel
        X_train: training data (unlabeled)
        pretrain_epochs: number of pretraining epochs
        batch_size: batch size
        learning_rate: learning rate
        p_mask: corruption probability
    
    Returns:
        model: pretrained model
        history: training history
    """
    logger.info("Starting VIME pretraining...")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    dataset = tf.data.Dataset.from_tensor_slices(X_train.astype(np.float32))
    dataset = dataset.shuffle(len(X_train)).batch(batch_size)
    
    history = {'total_loss': [], 'mask_loss': [], 'recon_loss': []}
    
    for epoch in range(pretrain_epochs):
        epoch_total_loss = 0
        epoch_mask_loss = 0
        epoch_recon_loss = 0
        n_batches = 0
        
        for batch in dataset:
            total_loss, mask_loss, recon_loss, grads = model.pretrain_step(batch, p_mask)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            epoch_total_loss += total_loss.numpy()
            epoch_mask_loss += mask_loss.numpy()
            epoch_recon_loss += recon_loss.numpy()
            n_batches += 1
        
        avg_total = epoch_total_loss / n_batches
        avg_mask = epoch_mask_loss / n_batches
        avg_recon = epoch_recon_loss / n_batches
        
        history['total_loss'].append(avg_total)
        history['mask_loss'].append(avg_mask)
        history['recon_loss'].append(avg_recon)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Pretrain Epoch {epoch + 1}/{pretrain_epochs}: "
                       f"Total={avg_total:.4f}, Mask={avg_mask:.4f}, Recon={avg_recon:.4f}")
    
    logger.info("Pretraining complete.")
    return model, history


def finetune_vime(pretrained_model, X_train, y_train, X_val, y_val,
                  epochs=500, batch_size=256, learning_rate=1e-4,
                  freeze_encoder=False):
    """
    Fine-tune pretrained VIME model for regression.
    
    Args:
        pretrained_model: VIMEPretrainModel with pretrained weights
        X_train, y_train: training data
        X_val, y_val: validation data
        epochs: fine-tuning epochs
        batch_size: batch size
        learning_rate: learning rate
        freeze_encoder: if True, freeze encoder weights
    
    Returns:
        model: fine-tuned model
        history: training history
    """
    # Build fine-tune model
    finetune_model = VIMEFineTuneModel(
        pretrained_encoder=pretrained_model.encoder,
        hidden_dim=64,
        dropout_rate=0.1
    )
    
    # Optionally freeze encoder
    if freeze_encoder:
        finetune_model.encoder.trainable = False
        logger.info("Encoder frozen during fine-tuning")
    
    # Build model
    finetune_model(X_train[:1].astype(np.float32))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    finetune_model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError(), r2_keras]
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=30,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=15, min_lr=1e-6, verbose=1
        )
    ]
    
    logger.info("Starting fine-tuning...")
    history = finetune_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return finetune_model, history


class VIMEConsistencyModel(keras.Model):
    """
    VIME with consistency regularization for semi-supervised learning.
    
    During training, adds consistency loss between predictions on
    clean and corrupted inputs.
    """
    
    def __init__(self, base_model, p_mask=0.1, consistency_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.p_mask = p_mask
        self.consistency_weight = consistency_weight
    
    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            # Clean prediction
            y_pred_clean = self.base_model(x, training=True)
            
            # Corrupted prediction
            x_corrupted, _ = pretext_generator(x, self.p_mask)
            y_pred_corrupted = self.base_model(x_corrupted, training=True)
            
            # Supervised loss
            supervised_loss = tf.reduce_mean(tf.square(y - y_pred_clean))
            
            # Consistency loss
            consistency_loss = tf.reduce_mean(tf.square(y_pred_clean - y_pred_corrupted))
            
            # Total loss
            total_loss = supervised_loss + self.consistency_weight * consistency_loss
        
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return {
            'loss': total_loss,
            'supervised_loss': supervised_loss,
            'consistency_loss': consistency_loss
        }


def create_callbacks(patience_es=30, patience_lr=15):
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience_es,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=patience_lr, min_lr=1e-6, verbose=1
        )
    ]


def plot_results(pretrain_history, finetune_history, metrics, output_dir='.', prefix='vime'):
    os.makedirs(output_dir, exist_ok=True)

    # Pretraining loss
    if pretrain_history is not None:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(pretrain_history['total_loss'], label='Total')
        plt.title('VIME Pretraining: Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(pretrain_history['mask_loss'], label='Mask')
        plt.title('Mask Prediction Loss')
        plt.xlabel('Epoch')
        
        plt.subplot(1, 3, 3)
        plt.plot(pretrain_history['recon_loss'], label='Recon')
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}_pretrain_loss.png'), dpi=150)
        plt.close()

    # Fine-tuning loss
    if hasattr(finetune_history, 'history'):
        plt.figure(figsize=(10, 5))
        plt.plot(finetune_history.history['loss'], label='Train Loss')
        plt.plot(finetune_history.history['val_loss'], label='Validation Loss')
        plt.title('VIME: Fine-tuning Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{prefix}_finetune_loss.png'), dpi=150)
        plt.close()

    # Predictions
    plt.figure(figsize=(8, 8))
    plt.scatter(metrics['y_actual'], metrics['y_pred'], alpha=0.5, s=10)
    min_val = min(metrics['y_actual'].min(), metrics['y_pred'].min())
    max_val = max(metrics['y_actual'].max(), metrics['y_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.title(f"VIME: Actual vs Predicted (R²={metrics['r2']:.4f})")
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(os.path.join(output_dir, f'{prefix}_predictions.png'), dpi=150)
    plt.close()


###############################################################################
# OPTUNA OBJECTIVE
###############################################################################

def build_model_for_trial(trial, input_dim):
    """Build and compile model for Optuna trial."""
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128])
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])

    model = build_vime_supervised(
        input_shape=input_dim,
        hidden_units=[hidden_dim, hidden_dim // 2],
        latent_dim=latent_dim,
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
    model = build_vime_supervised(
        input_shape=input_dim,
        hidden_units=[params['hidden_dim'], params['hidden_dim'] // 2],
        latent_dim=params['latent_dim'],
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
    setup_logging_for_progress(os.path.join(OUTPUT_DIR, 'training_vime.log'))
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
    parser = argparse.ArgumentParser(description='Train VIME with Optuna tuning')
    parser.add_argument('train_path', type=str, help='Path to training data')
    parser.add_argument('test_path', type=str, help='Path to test data')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to sample weights JSON file')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment')
    add_wandb_args(parser)
    args = parser.parse_args()
    main(args.train_path, args.test_path, args.weights_path, args.experiment_name, args)
