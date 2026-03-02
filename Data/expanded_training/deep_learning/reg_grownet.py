"""E16: GrowNet - Gradient Boosting Neural Network

With Optuna hyperparameter tuning.

Architecture: Sequentially fit residuals with shallow neural networks.

Paper: "Gradient Boosting Neural Networks: GrowNet"
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
MODEL_NAME = "grownet"
OUTPUT_DIR = './out/grownet'

logger = logging.getLogger(__name__)
###############################################################################
# END CONSTANTS
###############################################################################


class WeakLearner(keras.Model):
    """
    Shallow neural network as a weak learner.
    Takes input features + current prediction, outputs residual correction.
    """

    def __init__(self, hidden_units=[64, 32], dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = []
        self.norm_layers = []
        self.dropout_layers = []

        for units in hidden_units:
            self.hidden_layers.append(layers.Dense(units, activation='relu'))
            self.norm_layers.append(layers.LayerNormalization(epsilon=1e-6))
            self.dropout_layers.append(layers.Dropout(dropout_rate))

        self.output_layer = layers.Dense(1)

    def call(self, inputs, training=False):
        x = inputs
        for dense, norm, dropout in zip(self.hidden_layers, self.norm_layers, self.dropout_layers):
            x = dense(x)
            x = norm(x, training=training)
            x = dropout(x, training=training)
        return self.output_layer(x)


class GrowNet(keras.Model):
    """
    GrowNet: Gradient Boosting with Neural Network weak learners.
    
    Training procedure:
    1. Initialize prediction to 0 (or mean)
    2. For each stage:
       - Compute residuals
       - Train weak learner on (features, current_pred) -> residual
       - Update prediction: pred += learning_rate * weak_learner output
    3. Periodically run corrective step (fine-tune all learners together)
    """

    def __init__(self, n_learners=10, hidden_units=[64, 32],
                 boost_rate=0.1, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_learners = n_learners
        self.hidden_units = hidden_units
        self.boost_rate = boost_rate
        self.dropout_rate = dropout_rate

        self.input_norm = layers.BatchNormalization()
        self.feature_proj = layers.Dense(128, activation='relu')

        # Initialize weak learners
        self.weak_learners = [
            WeakLearner(hidden_units, dropout_rate, name=f'weak_learner_{i}')
            for i in range(n_learners)
        ]

        # Learnable combination weights (for corrective step)
        self.combination_weights = None

    def build(self, input_shape):
        self.combination_weights = self.add_weight(
            name='combination_weights',
            shape=(self.n_learners,),
            initializer=keras.initializers.Constant(self.boost_rate),
            trainable=True
        )

    def call(self, inputs, training=False):
        x = self.input_norm(inputs, training=training)
        features = self.feature_proj(x)

        batch_size = tf.shape(inputs)[0]
        current_pred = tf.zeros((batch_size, 1))

        # Apply each weak learner
        for i, learner in enumerate(self.weak_learners):
            # Concatenate features with current prediction
            learner_input = tf.concat([features, current_pred], axis=-1)
            residual = learner(learner_input, training=training)
            current_pred = current_pred + self.combination_weights[i] * residual

        return current_pred


class GrowNetTrainer:
    """
    Trainer for GrowNet with proper boosting procedure.
    """

    def __init__(self, input_dim, n_learners=10, hidden_units=[64, 32],
                 boost_rate=0.1, dropout_rate=0.1):
        self.input_dim = input_dim
        self.n_learners = n_learners
        self.hidden_units = hidden_units
        self.boost_rate = boost_rate
        self.dropout_rate = dropout_rate

        self.input_norm = layers.BatchNormalization()
        self.feature_proj = layers.Dense(128, activation='relu')
        self.weak_learners = []
        self.trained_learners = 0

    def _build_weak_learner(self, idx):
        """Build a single weak learner."""
        # Input: features + current prediction
        inputs = layers.Input(shape=(129,))  # 128 features + 1 pred
        x = inputs

        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = layers.Dropout(self.dropout_rate)(x)

        outputs = layers.Dense(1)(x)
        return Model(inputs, outputs, name=f'weak_learner_{idx}')

    def _compute_features(self, X):
        """Compute normalized/projected features."""
        x = self.input_norm(X, training=False)
        return self.feature_proj(x)

    def train_stage(self, X_train, residuals, current_pred,
                    epochs=50, batch_size=256, learning_rate=1e-3):
        """Train one weak learner on residuals."""
        learner = self._build_weak_learner(self.trained_learners)

        # Prepare input: features + current prediction
        features = self._compute_features(X_train)
        learner_input = np.concatenate([features.numpy(), current_pred], axis=-1)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        learner.compile(optimizer=optimizer, loss='mse')

        # Train
        learner.fit(
            learner_input, residuals,
            epochs=epochs, batch_size=batch_size,
            verbose=0
        )

        self.weak_learners.append(learner)
        self.trained_learners += 1

        return learner

    def predict(self, X):
        """Make predictions using all trained learners."""
        features = self._compute_features(X)
        current_pred = np.zeros((len(X), 1))

        for learner in self.weak_learners:
            learner_input = np.concatenate([features.numpy(), current_pred], axis=-1)
            residual = learner.predict(learner_input, verbose=0)
            current_pred = current_pred + self.boost_rate * residual

        return current_pred

    def build_full_model(self, X_sample):
        """Build a single Keras model for inference."""
        # Initialize normalization
        _ = self.input_norm(X_sample[:1])
        _ = self.feature_proj(self.input_norm(X_sample[:1]))

        inputs = layers.Input(shape=(self.input_dim,))
        x = self.input_norm(inputs)
        features = self.feature_proj(x)

        current_pred = tf.zeros((tf.shape(inputs)[0], 1))

        for i, learner in enumerate(self.weak_learners):
            learner_input = tf.concat([features, current_pred], axis=-1)
            residual = learner(learner_input)
            current_pred = current_pred + self.boost_rate * residual

        return Model(inputs, current_pred, name='GrowNet_Full')


def build_grownet_functional(input_shape,
                             n_learners=10,
                             hidden_units=[64, 32],
                             boost_rate=0.1,
                             dropout_rate=0.1):
    """
    Build GrowNet as functional model (for standard Keras training).
    All learners are trained simultaneously rather than sequentially.
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)
    features = layers.Dense(128, activation='relu')(x)
    features = layers.LayerNormalization(epsilon=1e-6)(features)

    # Use Lambda to create zeros tensor compatible with Keras 3.x
    current_pred = layers.Lambda(lambda x: ops.zeros((ops.shape(x)[0], 1)))(inputs)

    for i in range(n_learners):
        # Learner input: features + current prediction
        learner_input = layers.Concatenate()([features, current_pred])

        residual = learner_input
        for units in hidden_units:
            residual = layers.Dense(units, activation='relu')(residual)
            residual = layers.LayerNormalization(epsilon=1e-6)(residual)
            residual = layers.Dropout(dropout_rate)(residual)

        residual = layers.Dense(1)(residual)
        current_pred = current_pred + boost_rate * residual

    model = Model(inputs, current_pred, name='GrowNet_Functional')
    return model


def build_grownet_with_attention(input_shape,
                                 n_learners=8,
                                 hidden_units=[64, 32],
                                 dropout_rate=0.1):
    """
    GrowNet variant with attention-based learner aggregation.
    Learners' outputs are combined using learned attention weights.
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)
    features = layers.Dense(128, activation='relu')(x)
    features = layers.LayerNormalization(epsilon=1e-6)(features)

    # Collect all learner outputs
    learner_outputs = []

    for i in range(n_learners):
        # Each learner sees features only (no sequential dependency)
        residual = features
        for units in hidden_units:
            residual = layers.Dense(units, activation='relu')(residual)
            residual = layers.LayerNormalization(epsilon=1e-6)(residual)
            residual = layers.Dropout(dropout_rate)(residual)

        residual = layers.Dense(1)(residual)
        learner_outputs.append(residual)

    # Stack outputs: (batch, n_learners, 1)
    stacked = layers.Lambda(lambda x: ops.stack(x, axis=1))(learner_outputs)

    # Attention-based aggregation
    # Query: from features, Keys/Values: learner outputs
    query = layers.Dense(32)(features)
    query = layers.Lambda(lambda x: ops.expand_dims(x, 1))(query)  # (batch, 1, 32)

    keys = layers.Dense(32)(stacked)  # (batch, n_learners, 32)
    values = stacked  # (batch, n_learners, 1)

    # Compute attention using Lambda + ops for Keras 3.x compatibility
    def compute_attention(inputs):
        q, k, v = inputs
        scores = ops.matmul(q, ops.transpose(k, [0, 2, 1]))  # (batch, 1, n_learners)
        scores = scores / ops.sqrt(32.0)
        weights = ops.softmax(scores, axis=-1)
        out = ops.matmul(weights, v)  # (batch, 1, 1)
        out = ops.squeeze(out, axis=[1, 2])
        return ops.expand_dims(out, -1)

    output = layers.Lambda(compute_attention)([query, keys, values])

    model = Model(inputs, output, name='GrowNet_Attention')
    return model


def build_grownet_residual(input_shape,
                           n_learners=10,
                           hidden_dim=64,
                           dropout_rate=0.1):
    """
    GrowNet with explicit residual connections.
    Each learner predicts a correction to the previous prediction.
    """
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    # Initial prediction from simple linear model
    initial_pred = layers.Dense(1)(x)

    # Feature extraction
    features = layers.Dense(hidden_dim, activation='relu')(x)
    features = layers.LayerNormalization(epsilon=1e-6)(features)

    current_pred = initial_pred

    for i in range(n_learners):
        # Concatenate features with current prediction error signal
        combined = layers.Concatenate()([features, current_pred])

        # Residual prediction
        residual = layers.Dense(hidden_dim // 2, activation='relu')(combined)
        residual = layers.Dropout(dropout_rate)(residual)
        residual = layers.Dense(1)(residual)

        # Learnable shrinkage
        shrinkage = layers.Dense(1, activation='sigmoid', use_bias=False)(combined)
        current_pred = current_pred + shrinkage * residual

    model = Model(inputs, current_pred, name='GrowNet_Residual')
    return model


def train_grownet_boosting(X_train, y_train, X_val, y_val,
                           n_learners=10, hidden_units=[64, 32],
                           boost_rate=0.1, dropout_rate=0.1,
                           epochs_per_stage=50, corrective_epochs=20,
                           batch_size=256, learning_rate=1e-3):
    """
    Train GrowNet with proper gradient boosting procedure.
    """
    input_dim = X_train.shape[1]
    trainer = GrowNetTrainer(
        input_dim, n_learners, hidden_units, boost_rate, dropout_rate
    )

    # Initialize with dummy forward pass
    X_sample = tf.constant(X_train[:1], dtype=tf.float32)
    _ = trainer._compute_features(X_sample)

    current_pred_train = np.zeros((len(X_train), 1))
    current_pred_val = np.zeros((len(X_val), 1))

    history = {'train_rmse': [], 'val_rmse': []}

    for stage in range(n_learners):
        # Compute residuals
        residuals = y_train.reshape(-1, 1) - current_pred_train

        # Train weak learner
        logger.info(f"Training stage {stage + 1}/{n_learners}...")
        learner = trainer.train_stage(
            tf.constant(X_train, dtype=tf.float32),
            residuals,
            current_pred_train,
            epochs=epochs_per_stage,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # Update predictions
        features_train = trainer._compute_features(tf.constant(X_train, dtype=tf.float32))
        learner_input_train = np.concatenate([features_train.numpy(), current_pred_train], axis=-1)
        stage_pred_train = learner.predict(learner_input_train, verbose=0)
        current_pred_train = current_pred_train + boost_rate * stage_pred_train

        features_val = trainer._compute_features(tf.constant(X_val, dtype=tf.float32))
        learner_input_val = np.concatenate([features_val.numpy(), current_pred_val], axis=-1)
        stage_pred_val = learner.predict(learner_input_val, verbose=0)
        current_pred_val = current_pred_val + boost_rate * stage_pred_val

        # Evaluate
        train_rmse = np.sqrt(mean_squared_error(y_train, current_pred_train.flatten()))
        val_rmse = np.sqrt(mean_squared_error(y_val, current_pred_val.flatten()))
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)

        logger.info(f"  Stage {stage + 1}: Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}")

    return trainer, history


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


def evaluate_trainer(trainer, X_test, y_test, target_scaler):
    """Evaluate GrowNetTrainer."""
    y_pred_scaled = trainer.predict(tf.constant(X_test, dtype=tf.float32)).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)
    mae = np.mean(np.abs(y_test_actual - y_pred))

    return {
        'rmse': rmse, 'r2': r2, 'mae': mae,
        'y_pred': y_pred, 'y_actual': y_test_actual
    }


def plot_results(history, metrics, output_dir='.', prefix='grownet'):
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(history, dict) and 'train_rmse' in history:
        # Boosting history
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_rmse'], label='Train RMSE')
        plt.plot(history['val_rmse'], label='Validation RMSE')
        plt.title('GrowNet: Training Progress by Stage')
        plt.xlabel('Stage')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{prefix}_stage_curve.png'), dpi=150)
        plt.close()
    else:
        # Standard Keras history
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('GrowNet: Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{prefix}_loss_curve.png'), dpi=150)
        plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(metrics['y_actual'], metrics['y_pred'], alpha=0.5, s=10)
    min_val = min(metrics['y_actual'].min(), metrics['y_pred'].min())
    max_val = max(metrics['y_actual'].max(), metrics['y_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.title(f"GrowNet: Actual vs Predicted (R²={metrics['r2']:.4f})")
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

    n_learners = trial.suggest_int('n_learners', 5, 15)
    boost_rate = trial.suggest_float('boost_rate', 0.05, 0.2)
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])

    model = build_grownet_functional(
        input_shape=input_dim,
        n_learners=n_learners,
        hidden_units=[hidden_dim, hidden_dim // 2],
        boost_rate=boost_rate,
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
    model = build_grownet_functional(
        input_shape=input_dim,
        n_learners=params['n_learners'],
        hidden_units=[params['hidden_dim'], params['hidden_dim'] // 2],
        boost_rate=params['boost_rate'],
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
    setup_logging_for_progress(os.path.join(OUTPUT_DIR, 'training_grownet.log'))
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
    parser = argparse.ArgumentParser(description='Train GrowNet with Optuna tuning')
    parser.add_argument('train_path', type=str, help='Path to training data')
    parser.add_argument('test_path', type=str, help='Path to test data')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to sample weights JSON file')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment')
    add_wandb_args(parser)
    args = parser.parse_args()
    main(args.train_path, args.test_path, args.weights_path, args.experiment_name, args)
