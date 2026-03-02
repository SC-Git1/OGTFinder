"""
Consolidated Deep Learning Model Registry for Ensemble Stacking.

Contains all 22 deep learning model architectures extracted from
dl_ogt/deep_learning/reg_*.py files. Each model provides:
  - Custom Keras layers/classes (if any)
  - Architecture build function(s)
  - build_model_from_params_{model_name}(params, input_dim) -> compiled Model

Naming collisions resolved:
  - FeatureEmbedding  -> FeatureEmbedding_Attn (attention_mlp), FeatureEmbedding_SAINT (saint)
  - GatedLinearUnit   -> GatedLinearUnit (tabnet), GatedLinearUnit_Inspired (tabnet_inspired)
  - FeatureTransformer -> FeatureTransformer (tabnet), FeatureTransformer_Inspired (tabnet_inspired)
  - AttentionTransformer -> AttentionTransformer_Inspired (tabnet_inspired)
  - sparsemax         -> sparsemax (tabnet), sparsemax_inspired (tabnet_inspired), sparsemax_node (node)

Usage:
    from dl_model_registry import MODEL_REGISTRY
    builder = MODEL_REGISTRY['baseline_mlp']
    model = builder(best_params, input_dim)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from keras import ops

from dl_utils import r2_keras


###############################################################################
# 1. BASELINE MLP
###############################################################################


def build_baseline_mlp(
    input_shape: int,
    hidden_units: list,
    activation: str = 'relu',
    normalization: str = 'batchnorm',
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4
) -> Model:
    """Build a well-regularized baseline MLP."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f'dense_{i}'
        )(x)

        if normalization == 'batchnorm':
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
        elif normalization == 'layernorm':
            x = layers.LayerNormalization(epsilon=1e-6, name=f'ln_{i}')(x)

        if activation == 'gelu':
            x = layers.Activation('gelu', name=f'act_{i}')(x)
        else:
            x = layers.ReLU(name=f'relu_{i}')(x)

        x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)

    outputs = layers.Dense(1, name='output')(x)

    model = Model(inputs, outputs, name='Baseline_MLP')
    return model


def build_model_from_params_baseline_mlp(params: dict, input_dim: int):
    """Build and compile Baseline MLP from best parameters."""
    n_layers = params['n_layers']
    hidden_units = [params[f'units_layer_{i}'] for i in range(n_layers)]

    model = build_baseline_mlp(
        input_shape=input_dim,
        hidden_units=hidden_units,
        activation=params['activation'],
        normalization=params['normalization'],
        dropout_rate=params['dropout_rate'],
        l2_reg=params['weight_decay']
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=params['learning_rate'],
        clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError(), r2_keras]
    )
    return model


###############################################################################
# 2. SPARSE MLP
###############################################################################


class GroupSparseRegularizer(keras.regularizers.Regularizer):
    """Group sparsity (L2,1) regularizer."""

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
    """Layer that applies group-wise soft sparsity."""

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
    """Build Sparse MLP with group sparsity regularization."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    if sparsity_type == 'group_soft':
        x = GroupSparsityLayer(n_groups=n_groups, sparsity_reg=l21_weight)(x)
        regularizer = keras.regularizers.l2(l2_reg)
    elif sparsity_type == 'l1':
        regularizer = keras.regularizers.l1_l2(l1=l21_weight, l2=l2_reg)
    else:  # group_l21
        effective_n_groups = min(n_groups, input_shape)
        group_size = max(1, input_shape // effective_n_groups)
        regularizer = GroupSparseRegularizer(l21_weight=l21_weight, group_size=group_size)

    x = layers.Dense(
        hidden_units[0],
        kernel_initializer='he_normal',
        kernel_regularizer=regularizer,
        name='sparse_dense_0'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

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


def build_model_from_params_sparse_mlp(params: dict, input_dim: int):
    """Build and compile Sparse MLP from best parameters."""
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
# 3. GATED MLP
###############################################################################


class FeatureGate(layers.Layer):
    """Feature-wise gating layer with network-based gates."""

    def __init__(self, gate_hidden_dim=64, sparsity_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.gate_hidden_dim = gate_hidden_dim
        self.sparsity_reg = sparsity_reg

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.gate_dense1 = layers.Dense(
            self.gate_hidden_dim,
            activation='relu',
            kernel_initializer='he_normal',
            name='gate_hidden'
        )
        self.gate_dense2 = layers.Dense(
            input_dim,
            activation='sigmoid',
            kernel_regularizer=keras.regularizers.l1(self.sparsity_reg),
            bias_initializer=keras.initializers.Constant(1.0),
            name='gate_output'
        )

    def call(self, inputs):
        gate_hidden = self.gate_dense1(inputs)
        gate_values = self.gate_dense2(gate_hidden)
        gated_input = inputs * gate_values
        return gated_input, gate_values


class FeatureGateSimple(layers.Layer):
    """Simpler feature gate: direct learnable per-feature weights."""

    def __init__(self, sparsity_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.sparsity_reg = sparsity_reg

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.gate_weights = self.add_weight(
            name='gate_weights',
            shape=(input_dim,),
            initializer=keras.initializers.Constant(1.0),
            regularizer=keras.regularizers.l1(self.sparsity_reg),
            trainable=True
        )

    def call(self, inputs):
        gate_values = tf.sigmoid(self.gate_weights)
        gated_input = inputs * gate_values
        return gated_input, gate_values


class ContextualFeatureGate(layers.Layer):
    """Contextual feature gate: gate values depend on input context."""

    def __init__(self, context_dim=32, sparsity_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.context_dim = context_dim
        self.sparsity_reg = sparsity_reg

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.context_encoder = keras.Sequential([
            layers.Dense(self.context_dim, activation='relu'),
            layers.Dense(self.context_dim, activation='relu'),
        ])

        self.gate_generator = layers.Dense(
            input_dim,
            activation='sigmoid',
            kernel_regularizer=keras.regularizers.l1(self.sparsity_reg),
            bias_initializer=keras.initializers.Constant(1.0)
        )

    def call(self, inputs):
        context = self.context_encoder(inputs)
        gate_values = self.gate_generator(context)
        gated_input = inputs * gate_values
        return gated_input, gate_values


def build_gated_mlp(
    input_shape: int,
    hidden_units: list = [512, 256, 128],
    gate_type: str = 'network',
    gate_hidden_dim: int = 64,
    sparsity_reg: float = 1e-4,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4
) -> Model:
    """Build Gated MLP for tabular regression."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    if gate_type == 'simple':
        gate_layer = FeatureGateSimple(sparsity_reg=sparsity_reg, name='feature_gate')
    elif gate_type == 'contextual':
        gate_layer = ContextualFeatureGate(
            context_dim=gate_hidden_dim,
            sparsity_reg=sparsity_reg,
            name='feature_gate'
        )
    else:  # 'network'
        gate_layer = FeatureGate(
            gate_hidden_dim=gate_hidden_dim,
            sparsity_reg=sparsity_reg,
            name='feature_gate'
        )

    gated_x, gate_values = gate_layer(x)

    h = gated_x
    for i, units in enumerate(hidden_units):
        h = layers.Dense(
            units,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f'dense_{i}'
        )(h)
        h = layers.BatchNormalization(name=f'bn_{i}')(h)
        h = layers.ReLU(name=f'relu_{i}')(h)
        h = layers.Dropout(dropout_rate, name=f'dropout_{i}')(h)

    outputs = layers.Dense(1, name='output')(h)

    model = Model(inputs, outputs, name='Gated_MLP')
    model.gate_layer = gate_layer
    return model


def build_model_from_params_gated_mlp(params: dict, input_dim: int):
    """Build and compile Gated MLP from best parameters."""
    n_layers = params['n_layers']
    hidden_units = [params[f'units_layer_{i}'] for i in range(n_layers)]

    model = build_gated_mlp(
        input_shape=input_dim,
        hidden_units=hidden_units,
        gate_type=params['gate_type'],
        gate_hidden_dim=params['gate_hidden_dim'],
        sparsity_reg=params['sparsity_reg'],
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
# 4. TabR
###############################################################################


def build_tabr(
    input_shape: int,
    hidden_units: list = [256, 128],
    embed_dim: int = 64,
    n_heads: int = 4,
    dropout_rate: float = 0.1
) -> Model:
    """Build TabR model for tabular regression."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)
    for units in hidden_units:
        x = layers.Dense(units)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)

    embed = layers.Dense(embed_dim)(x)
    embed = layers.LayerNormalization(epsilon=1e-6)(embed)

    attention = layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=embed_dim // n_heads, dropout=dropout_rate
    )
    embed_reshaped = layers.Lambda(lambda t: ops.expand_dims(t, 1))(embed)
    attended = attention(embed_reshaped, embed_reshaped)
    attended = layers.Lambda(lambda t: ops.squeeze(t, 1))(attended)

    combined = layers.Concatenate()([embed, attended])
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.LayerNormalization(epsilon=1e-6)(combined)
    combined = layers.Dropout(dropout_rate)(combined)

    outputs = layers.Dense(64, activation='relu')(combined)
    outputs = layers.Dropout(dropout_rate)(outputs)
    outputs = layers.Dense(1)(outputs)

    model = Model(inputs, outputs, name='TabR')
    return model


def build_model_from_params_tabr(params: dict, input_dim: int):
    """Build and compile TabR from best parameters."""
    n_layers = params['n_layers']
    hidden_units = []
    for i in range(n_layers):
        units = params[f'units_layer_{i}']
        hidden_units.append(units)

    model = build_tabr(
        input_shape=input_dim,
        hidden_units=hidden_units,
        embed_dim=params['embed_dim'],
        n_heads=params['n_heads'],
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
# 5. ATTENTION MLP
###############################################################################


class FeatureEmbedding_Attn(layers.Layer):
    """Embed each feature into a higher-dimensional space (attention_mlp variant)."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.num_features = input_shape[-1]
        self.embedding = layers.Dense(self.embed_dim)

    def call(self, inputs):
        x = tf.expand_dims(inputs, -1)
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
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        attention_scores = tf.matmul(q, k, transpose_b=True) / scale
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        attended = tf.matmul(attention_weights, v)

        attended = tf.transpose(attended, perm=[0, 2, 1, 3])
        attended = tf.reshape(attended, (batch_size, -1, self.embed_dim))

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
        attn_output, attn_weights = self.attention(inputs, training=training)
        x = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(x, training=training)
        x = self.layernorm2(x + ffn_output)

        return x, attn_weights


def build_feature_attention_mlp(input_shape,
                                attention_dim=64,
                                hidden_dims=[512, 256, 128],
                                dropout_rate=0.3,
                                l2_reg=1e-4):
    """Simplified attention MLP with feature-wise attention scores."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

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

    attended = layers.Multiply()([x, attention_weights])

    combined = layers.Concatenate()([x, attended])

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


def build_model_from_params_attention_mlp(params: dict, input_dim: int):
    """Build and compile Attention MLP from best parameters."""
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


###############################################################################
# 6. SAINT
###############################################################################


class FeatureEmbedding_SAINT(layers.Layer):
    """Embed each feature into a d-dimensional space (SAINT variant)."""

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
        x = tf.expand_dims(inputs, -1)
        embedded = x * self.embeddings + self.biases
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
        attn_out = self.mha(x, x, training=training)
        attn_out = self.dropout(attn_out, training=training)
        return self.layernorm(x + attn_out)


class RowAttention(layers.Layer):
    """Intersample attention: attention over rows (samples) within a batch."""

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
        x_t = tf.transpose(x, [1, 0, 2])
        attn_out = self.mha(x_t, x_t, training=training)
        attn_out = self.dropout(attn_out, training=training)
        out = self.layernorm(x_t + attn_out)
        return tf.transpose(out, [1, 0, 2])


class SAINTBlock(layers.Layer):
    """SAINT block: Column attention -> Row attention -> FFN."""

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
        x = self.col_attn(x, training=training)
        x = self.row_attn(x, training=training)
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


def build_saint_grouped(input_shape,
                        n_groups=50,
                        embed_dim=32,
                        num_heads=4,
                        ff_dim=64,
                        num_blocks=2,
                        dropout_rate=0.1):
    """SAINT with feature grouping for high-dimensional inputs."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    group_size = input_shape // n_groups
    n_used = n_groups * group_size
    x_grouped = layers.Reshape((n_groups, group_size))(x[:, :n_used])

    x_embed = layers.Dense(embed_dim)(x_grouped)

    encoder = SAINTEncoder(embed_dim, num_heads, ff_dim, num_blocks, dropout_rate)
    x = encoder(x_embed)

    x = layers.Lambda(lambda t: ops.mean(t, axis=1))(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='SAINT_Grouped')
    return model


def build_model_from_params_saint(params: dict, input_dim: int):
    """Build and compile SAINT from best parameters."""
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


###############################################################################
# 7. GROWNET
###############################################################################


def build_grownet_functional(input_shape,
                             n_learners=10,
                             hidden_units=[64, 32],
                             boost_rate=0.1,
                             dropout_rate=0.1):
    """Build GrowNet as functional model (all learners trained simultaneously)."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)
    features = layers.Dense(128, activation='relu')(x)
    features = layers.LayerNormalization(epsilon=1e-6)(features)

    current_pred = layers.Lambda(lambda x: ops.zeros((ops.shape(x)[0], 1)))(inputs)

    for i in range(n_learners):
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


def build_model_from_params_grownet(params: dict, input_dim: int):
    """Build and compile GrowNet from best parameters."""
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


###############################################################################
# 8. SNN (Self-Normalizing Neural Network)
###############################################################################


def build_snn(input_shape,
              hidden_units=[512, 256, 128, 64],
              dropout_rate=0.1,
              l2_reg=1e-5):
    """Build Self-Normalizing Neural Network for tabular regression."""
    inputs = layers.Input(shape=(input_shape,))

    x = inputs
    for units in hidden_units:
        x = layers.Dense(
            units,
            kernel_initializer='lecun_normal',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(x)
        x = layers.Activation('selu')(x)
        x = layers.AlphaDropout(dropout_rate)(x)

    outputs = layers.Dense(1, kernel_initializer='lecun_normal')(x)

    model = Model(inputs, outputs, name='SNN')
    return model


def build_model_from_params_snn(params: dict, input_dim: int):
    """Build and compile SNN from best parameters."""
    n_layers = params['n_layers']
    hidden_dim = params['hidden_dim']
    hidden_units = [hidden_dim // (2 ** i) for i in range(n_layers)]
    hidden_units = [max(64, u) for u in hidden_units]

    model = build_snn(
        input_shape=input_dim,
        hidden_units=hidden_units,
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


###############################################################################
# 9. TABNET
###############################################################################


def sparsemax(logits, axis=-1):
    """Sparsemax activation function (TabNet variant)."""
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
    """GLU: splits input and applies sigmoid gating (TabNet variant)."""

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
    """Feature transformer: shared + decision-step-specific layers (TabNet variant)."""

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
    """Attentive transformer: computes sparse attention mask (TabNet variant)."""

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
    """Build TabNet model for tabular regression."""
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


def build_model_from_params_tabnet(params: dict, input_dim: int):
    """Build and compile TabNet from best parameters."""
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
# 10. TABNET INSPIRED
###############################################################################


def sparsemax_inspired(logits, axis=-1):
    """Sparsemax activation function (TabNet-Inspired variant)."""
    logits = tf.cast(logits, tf.float32)
    dims = logits.shape.as_list()
    num_dims = len(dims)

    if axis < 0:
        axis = axis + num_dims

    z_sorted = tf.sort(logits, axis=axis, direction='DESCENDING')

    range_val = tf.range(1, tf.shape(logits)[axis] + 1, dtype=tf.float32)
    for _ in range(axis):
        range_val = tf.expand_dims(range_val, 0)
    for _ in range(num_dims - axis - 1):
        range_val = tf.expand_dims(range_val, -1)

    cumsum = tf.cumsum(z_sorted, axis=axis)
    k_check = 1 + range_val * z_sorted > cumsum
    k = tf.reduce_sum(tf.cast(k_check, tf.float32), axis=axis, keepdims=True)

    tau = (tf.reduce_sum(z_sorted * tf.cast(k_check, tf.float32), axis=axis, keepdims=True) - 1) / k

    return tf.maximum(logits - tau, 0)


class GatedLinearUnit_Inspired(layers.Layer):
    """Gated Linear Unit for feature transformation (TabNet-Inspired variant)."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.fc = layers.Dense(self.units * 2, use_bias=False)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x, gate = tf.split(x, 2, axis=-1)
        return x * tf.sigmoid(gate)


class FeatureTransformer_Inspired(layers.Layer):
    """Feature transformer block with shared and decision-specific layers (TabNet-Inspired variant)."""

    def __init__(self, units, num_shared=2, num_decision=2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_shared = num_shared
        self.num_decision = num_decision

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.input_proj = layers.Dense(self.units) if input_dim != self.units else None
        self.shared_layers = [GatedLinearUnit_Inspired(self.units) for _ in range(self.num_shared)]
        self.decision_layers = [GatedLinearUnit_Inspired(self.units) for _ in range(self.num_decision)]

    def call(self, inputs, training=False):
        if self.input_proj is not None:
            x = self.input_proj(inputs)
        else:
            x = inputs

        for layer in self.shared_layers:
            x = layer(x, training=training) + x

        for layer in self.decision_layers:
            x = layer(x, training=training) + x

        return x


class AttentionTransformer_Inspired(layers.Layer):
    """Attention mechanism for feature selection (TabNet-Inspired variant)."""

    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim

    def build(self, input_shape):
        self.fc = layers.Dense(self.input_dim, use_bias=False)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, priors, training=False):
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x = x * priors
        return sparsemax_inspired(x)


class TabNetInspiredBlock(layers.Layer):
    """Single TabNet-inspired decision step."""

    def __init__(self, feature_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        self.attention = AttentionTransformer_Inspired(input_shape[-1])
        self.feature_transformer = FeatureTransformer_Inspired(self.feature_dim)
        self.output_fc = layers.Dense(self.output_dim)

    def call(self, inputs, priors, training=False):
        mask = self.attention(inputs, priors, training=training)
        masked_input = mask * inputs
        transformed = self.feature_transformer(masked_input, training=training)
        output = self.output_fc(transformed)
        new_priors = priors * (1 - mask)
        return output, mask, new_priors


def build_tabnet_inspired(
    input_shape: int,
    num_steps: int = 3,
    feature_dim: int = 128,
    output_dim: int = 64,
    dropout_rate: float = 0.2
) -> Model:
    """Build TabNet-inspired model for tabular regression."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)
    priors = layers.Lambda(lambda t: ops.ones_like(t))(x)

    step_outputs = []
    attention_masks = []

    for step in range(num_steps):
        block = TabNetInspiredBlock(feature_dim, output_dim, name=f'step_{step}')
        step_output, mask, priors = block(x, priors)
        step_outputs.append(step_output)
        attention_masks.append(mask)

    aggregated = layers.Add()(step_outputs)
    aggregated = layers.ReLU()(aggregated)
    aggregated = layers.Dropout(dropout_rate)(aggregated)

    outputs = layers.Dense(1)(aggregated)

    model = Model(inputs, outputs, name='TabNet_Inspired')
    return model


def build_model_from_params_tabnet_inspired(params: dict, input_dim: int):
    """Build and compile TabNet-Inspired from best parameters."""
    model = build_tabnet_inspired(
        input_shape=input_dim,
        num_steps=params['num_steps'],
        feature_dim=params['feature_dim'],
        output_dim=params['output_dim'],
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
# 11. TABM
###############################################################################


class BatchEnsembleDense(layers.Layer):
    """BatchEnsemble-style dense layer."""

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
    """Build TabM model."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.LayerNormalization(epsilon=1e-6)(inputs)

    for i, units in enumerate(hidden_units):
        x = TabMBlock(units, num_members, dropout_rate, name=f'tabm_block_{i}')(x)

    outputs_list = []
    for i in range(num_members):
        member_features = x[:, i, :]
        member_output = layers.Dense(1, name=f'output_member_{i}')(member_features)
        outputs_list.append(member_output)

    stacked_outputs = layers.Lambda(lambda x: ops.stack(x, axis=1))(outputs_list)

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


def build_model_from_params_tabm(params: dict, input_dim: int):
    """Build and compile TabM from best parameters."""
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
# 12. LASSONET
###############################################################################


class LassoNetLayer(layers.Layer):
    """LassoNet layer with hierarchical sparsity constraint."""

    def __init__(self, hidden_units, M=10.0, lambda_reg=0.01, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.M = M
        self.lambda_reg = lambda_reg

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.skip_weights = self.add_weight(
            name='skip_weights',
            shape=(input_dim,),
            initializer='glorot_uniform',
            trainable=True
        )

        self.hidden_kernel = self.add_weight(
            name='hidden_kernel',
            shape=(input_dim, self.hidden_units[0]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.hidden_bias = self.add_weight(
            name='hidden_bias',
            shape=(self.hidden_units[0],),
            initializer='zeros',
            trainable=True
        )

        self.additional_layers = []
        prev_units = self.hidden_units[0]
        for units in self.hidden_units[1:]:
            self.additional_layers.append(
                layers.Dense(units, activation='relu', kernel_initializer='he_normal')
            )
            prev_units = units

    def call(self, inputs, training=False):
        skip_abs = ops.abs(self.skip_weights)

        w_norms = ops.norm(self.hidden_kernel, axis=1, keepdims=True)
        max_norms = self.M * ops.expand_dims(skip_abs, 1)
        scale = ops.minimum(1.0, max_norms / (w_norms + 1e-8))
        constrained_kernel = self.hidden_kernel * scale

        skip_out = inputs * self.skip_weights
        skip_sum = ops.sum(skip_out, axis=1, keepdims=True)

        hidden = ops.matmul(inputs, constrained_kernel) + self.hidden_bias
        hidden = ops.relu(hidden)

        for layer in self.additional_layers:
            hidden = layer(hidden)

        l1_loss = self.lambda_reg * ops.sum(ops.abs(self.skip_weights))
        self.add_loss(l1_loss)

        return hidden, skip_sum, self.skip_weights


def build_lassonet(
    input_shape: int,
    hidden_units: list = [256, 128, 64],
    M: float = 10.0,
    lambda_reg: float = 0.01,
    dropout_rate: float = 0.2
) -> Model:
    """Build LassoNet model for tabular regression."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    lassonet_layer = LassoNetLayer(hidden_units, M=M, lambda_reg=lambda_reg, name='lassonet')
    hidden, skip_out, skip_weights = lassonet_layer(x)

    hidden = layers.Dropout(dropout_rate)(hidden)
    hidden_out = layers.Dense(1, name='hidden_output')(hidden)

    outputs = hidden_out + skip_out

    model = Model(inputs, outputs, name='LassoNet')
    model.lassonet_layer = lassonet_layer
    return model


def build_model_from_params_lassonet(params: dict, input_dim: int):
    """Build and compile LassoNet from best parameters."""
    n_layers = params['n_layers']
    hidden_units = [params[f'units_layer_{i}'] for i in range(n_layers)]

    model = build_lassonet(
        input_shape=input_dim,
        hidden_units=hidden_units,
        M=params['M'],
        lambda_reg=params['lambda_reg'],
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
# 13. WIDE & DEEP
###############################################################################


def build_wide_deep(input_shape,
                    deep_units=[512, 256, 128, 64],
                    dropout_rate=0.3,
                    l2_reg=1e-4,
                    wide_l1_reg=1e-5):
    """Build Wide & Deep network for tabular regression."""
    inputs = layers.Input(shape=(input_shape,))

    # Wide component
    wide = layers.Dense(
        1,
        kernel_regularizer=keras.regularizers.l1(wide_l1_reg),
        name='wide_output'
    )(inputs)

    # Deep component
    deep = inputs
    for i, units in enumerate(deep_units):
        deep = layers.Dense(
            units,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f'deep_dense_{i}'
        )(deep)
        deep = layers.BatchNormalization(name=f'deep_bn_{i}')(deep)
        deep = layers.ReLU(name=f'deep_relu_{i}')(deep)
        deep = layers.Dropout(dropout_rate, name=f'deep_dropout_{i}')(deep)

    deep_output = layers.Dense(1, name='deep_output')(deep)

    combined = layers.Add(name='wide_deep_add')([wide, deep_output])

    model = Model(inputs, combined, name='Wide_Deep')
    return model


def build_model_from_params_wide_deep(params: dict, input_dim: int):
    """Build and compile Wide & Deep from best parameters."""
    n_layers = params['n_layers']
    hidden_dim = params['hidden_dim']
    deep_units = [hidden_dim // (2 ** i) for i in range(n_layers)]
    deep_units = [max(64, u) for u in deep_units]

    model = build_wide_deep(
        input_shape=input_dim,
        deep_units=deep_units,
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


###############################################################################
# 14. VIME
###############################################################################


def build_vime_supervised(input_shape, hidden_units=[256, 128], latent_dim=64,
                          dropout_rate=0.1):
    """Build VIME model for supervised training (no pretraining)."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout_rate)(x)

    latent = layers.Dense(latent_dim)(x)
    latent = layers.LayerNormalization(epsilon=1e-6)(latent)

    x = layers.Dense(64, activation='relu')(latent)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='VIME_Supervised')
    return model


def build_model_from_params_vime(params: dict, input_dim: int):
    """Build and compile VIME from best parameters."""
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


###############################################################################
# 15. RESNET MLP
###############################################################################


class ResidualBlock(layers.Layer):
    """Residual block with skip connection for tabular data."""

    def __init__(self, units, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.dense1 = layers.Dense(self.units, kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(self.units, kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(self.dropout_rate)

        if input_shape[-1] != self.units:
            self.projection = layers.Dense(self.units, kernel_initializer='he_normal')
        else:
            self.projection = None

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        x = self.dense2(x)
        x = self.bn2(x, training=training)

        if self.projection is not None:
            shortcut = self.projection(inputs)
        else:
            shortcut = inputs

        x = layers.Add()([x, shortcut])
        x = tf.nn.relu(x)
        return x


def build_resnet_mlp(input_shape,
                     block_units=[512, 256, 128],
                     dropout_rate=0.3,
                     l2_reg=1e-4):
    """Build ResNet-style MLP for tabular regression."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.Dense(block_units[0],
                     kernel_initializer='he_normal',
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    for units in block_units:
        x = ResidualBlock(units, dropout_rate=dropout_rate)(x)

    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='ResNet_MLP')
    return model


def build_model_from_params_resnet_mlp(params: dict, input_dim: int):
    """Build and compile ResNet MLP from best parameters."""
    n_blocks = params['n_blocks']
    hidden_dim = params['hidden_dim']
    block_units = [hidden_dim // (2 ** i) for i in range(n_blocks)]
    block_units = [max(64, u) for u in block_units]

    model = build_resnet_mlp(
        input_shape=input_dim,
        block_units=block_units,
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


###############################################################################
# 16. RTDL RESNET
###############################################################################


class RTDLResidualBlock(layers.Layer):
    """RTDL-style residual block with pre-normalization."""

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
    """Build RTDL ResNet MLP for tabular regression."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.Dense(
        d_model,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='input_projection'
    )(inputs)

    for i in range(n_blocks):
        x = RTDLResidualBlock(d_hidden, dropout_rate, name=f'resblock_{i}')(x)

    x = layers.LayerNormalization(epsilon=1e-6, name='final_norm')(x)

    x = layers.Dense(64, activation='relu', name='head_dense')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, name='output')(x)

    model = Model(inputs, outputs, name='RTDL_ResNet')
    return model


def build_model_from_params_rtdl_resnet(params: dict, input_dim: int):
    """Build and compile RTDL ResNet from best parameters."""
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
# 17. FT-TRANSFORMER
###############################################################################


class FeatureTokenizer(layers.Layer):
    """Tokenizes continuous features into embeddings."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        n_features = input_shape[-1]
        self.feature_weights = self.add_weight(
            name='feature_weights',
            shape=(n_features, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.feature_biases = self.add_weight(
            name='feature_biases',
            shape=(n_features, self.embed_dim),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        x = tf.expand_dims(inputs, -1)
        tokens = x * self.feature_weights + self.feature_biases
        return tokens


class TransformerBlock(layers.Layer):
    """Standard Transformer block with pre-norm."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embed_dim),
            layers.Dropout(self.dropout_rate)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        x = self.layernorm1(inputs)
        attn_output = self.att(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = inputs + attn_output

        y = self.layernorm2(x)
        ffn_output = self.ffn(y, training=training)
        return x + ffn_output


class CLSToken(layers.Layer):
    """Learnable [CLS] token for aggregation."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_broadcasted, inputs], axis=1)


def build_ft_transformer_grouped(input_shape,
                                 n_groups=50,
                                 embed_dim=32,
                                 num_heads=4,
                                 ff_dim=64,
                                 num_blocks=2,
                                 dropout_rate=0.1):
    """FT-Transformer with feature grouping."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    effective_n_groups = min(n_groups, input_shape)
    if effective_n_groups < 1:
        effective_n_groups = 1
    group_size = max(1, input_shape // effective_n_groups)
    n_used = effective_n_groups * group_size

    x_grouped = layers.Reshape((effective_n_groups, group_size))(x[:, :n_used])

    group_embed = layers.Dense(embed_dim)(x_grouped)

    cls_layer = CLSToken(embed_dim)
    tokens = cls_layer(group_embed)

    for i in range(num_blocks):
        tokens = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(tokens)

    tokens = layers.LayerNormalization(epsilon=1e-6)(tokens)

    pooled = tokens[:, 0, :]

    output = layers.Dense(64, activation='relu')(pooled)
    output = layers.Dropout(dropout_rate)(output)
    output = layers.Dense(1)(output)

    model = Model(inputs, output, name='FT_Transformer_Grouped')
    return model


def build_model_from_params_ft_transformer(params: dict, input_dim: int):
    """Build and compile FT-Transformer from best parameters."""
    model = build_ft_transformer_grouped(
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


###############################################################################
# 18. NODE (Neural Oblivious Decision Ensembles)
###############################################################################


def sparsemax_node(logits, axis=-1):
    """Sparsemax activation (NODE variant)."""
    logits = logits - tf.reduce_max(logits, axis=axis, keepdims=True)

    sorted_logits = tf.sort(logits, axis=axis, direction='DESCENDING')

    cumsum = tf.cumsum(sorted_logits, axis=axis)

    k = tf.cast(tf.range(1, tf.shape(logits)[axis] + 1), logits.dtype)
    k = tf.reshape(k, [1] * axis + [-1] + [1] * (len(logits.shape) - axis - 1))

    threshold = (cumsum - 1) / k
    support = tf.cast(sorted_logits > threshold, logits.dtype)

    k_max = tf.reduce_sum(support, axis=axis, keepdims=True)
    tau = (tf.reduce_sum(sorted_logits * support, axis=axis, keepdims=True) - 1) / k_max

    output = tf.maximum(logits - tau, 0)
    return output


def entmax15(logits, axis=-1):
    """Entmax 1.5: between softmax (alpha=1) and sparsemax (alpha=2)."""
    softmax_out = tf.nn.softmax(logits, axis=axis)
    sparsemax_out = sparsemax_node(logits, axis=axis)
    return 0.5 * softmax_out + 0.5 * sparsemax_out


class FeatureSelection(layers.Layer):
    """Differentiable feature selection for each tree."""

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
        selection_weights = entmax15(self.selection_logits, axis=-1)
        selected = tf.matmul(inputs, selection_weights, transpose_b=True)
        return selected


class ObliviousDecisionTree(layers.Layer):
    """Single differentiable oblivious decision tree."""

    def __init__(self, depth=6, output_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.output_dim = output_dim

    def build(self, input_shape):
        n_features = input_shape[-1]

        self.feature_selector = layers.Dense(
            self.depth, use_bias=False,
            kernel_initializer='glorot_uniform'
        )

        self.thresholds = self.add_weight(
            name='thresholds',
            shape=(self.depth,),
            initializer='zeros',
            trainable=True
        )

        self.leaf_responses = self.add_weight(
            name='leaf_responses',
            shape=(self.n_leaves, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )

        self.temperature = self.add_weight(
            name='temperature',
            shape=(),
            initializer=keras.initializers.Constant(1.0),
            trainable=True,
            constraint=keras.constraints.NonNeg()
        )

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        features = self.feature_selector(inputs)

        temp = tf.maximum(self.temperature, 0.1)
        decisions = tf.nn.sigmoid((features - self.thresholds) / temp)

        leaf_probs = tf.ones((batch_size, 1))

        for d in range(self.depth):
            decision_d = decisions[:, d:d+1]
            left_probs = leaf_probs * (1 - decision_d)
            right_probs = leaf_probs * decision_d
            leaf_probs = tf.concat([left_probs, right_probs], axis=1)

        output = tf.matmul(leaf_probs, self.leaf_responses)

        return output


class NODELayer(layers.Layer):
    """NODE layer: ensemble of oblivious decision trees."""

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
        outputs = [tree(inputs, training=training) for tree in self.trees]
        stacked = tf.stack(outputs, axis=1)
        return tf.reduce_mean(stacked, axis=1)


class DenseODT(layers.Layer):
    """Dense Oblivious Decision Tree: more efficient implementation."""

    def __init__(self, n_trees=64, depth=4, output_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.n_trees = n_trees
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.output_dim = output_dim

    def build(self, input_shape):
        n_features = input_shape[-1]

        self.feature_weights = self.add_weight(
            name='feature_weights',
            shape=(self.n_trees, self.depth, n_features),
            initializer='glorot_uniform',
            trainable=True
        )

        self.thresholds = self.add_weight(
            name='thresholds',
            shape=(self.n_trees, self.depth),
            initializer='zeros',
            trainable=True
        )

        self.leaf_values = self.add_weight(
            name='leaf_values',
            shape=(self.n_trees, self.n_leaves, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )

        self.log_temp = self.add_weight(
            name='log_temperature',
            shape=(self.n_trees,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        features = tf.einsum('bf,tdf->btd', inputs, self.feature_weights)

        temperature = tf.exp(self.log_temp) + 0.1
        temperature = tf.reshape(temperature, (1, self.n_trees, 1))

        decisions = tf.nn.sigmoid((features - self.thresholds) / temperature)

        leaf_probs = tf.ones((batch_size, self.n_trees, 1))

        for d in range(self.depth):
            dec = decisions[:, :, d:d+1]
            leaf_probs = tf.concat([
                leaf_probs * (1 - dec),
                leaf_probs * dec
            ], axis=-1)

        output = tf.einsum('btl,tlo->bto', leaf_probs, self.leaf_values)

        return tf.reduce_mean(output, axis=1)


def build_node(input_shape,
               n_trees=64,
               depth=4,
               hidden_dim=64,
               n_layers=2,
               dropout_rate=0.1):
    """Build NODE model for tabular regression."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    for i in range(n_layers):
        node_out = DenseODT(n_trees, depth, hidden_dim, name=f'node_layer_{i}')(x)
        node_out = layers.LayerNormalization(epsilon=1e-6)(node_out)
        node_out = layers.Dropout(dropout_rate)(node_out)
        x = x + node_out

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='NODE')
    return model


def build_model_from_params_node(params: dict, input_dim: int):
    """Build and compile NODE from best parameters."""
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


###############################################################################
# 19. GANDALF
###############################################################################


class GFLU(layers.Layer):
    """Gated Feature Learning Unit (GFLU)."""

    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.fc1 = layers.Dense(self.units * 2)
        self.bn1 = layers.BatchNormalization()
        self.gate_fc = layers.Dense(self.units)
        self.gate_bn = layers.BatchNormalization()
        self.fc_out = layers.Dense(self.units)
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        h = self.fc1(inputs)
        h = self.bn1(h, training=training)
        h1, h2 = tf.split(h, 2, axis=-1)
        h = h1 * tf.sigmoid(h2)

        gate = self.gate_fc(inputs)
        gate = self.gate_bn(gate, training=training)
        gate = tf.sigmoid(gate)

        out = h * gate
        out = self.fc_out(out)
        out = self.dropout(out, training=training)

        return out, gate


class GFLUBlock(layers.Layer):
    """GFLU Block with residual connection."""

    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.gflu = GFLU(self.units, self.dropout_rate)
        if input_shape[-1] != self.units:
            self.residual_proj = layers.Dense(self.units)
        else:
            self.residual_proj = None

    def call(self, inputs, training=False):
        out, gate = self.gflu(inputs, training=training)
        if self.residual_proj is not None:
            residual = self.residual_proj(inputs)
        else:
            residual = inputs
        return out + residual, gate


class FeatureSelectionGate(layers.Layer):
    """Soft feature selection gate for GANDALF."""

    def __init__(self, temperature=1.0, sparsity_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.sparsity_reg = sparsity_reg

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.attention_fc1 = layers.Dense(input_dim // 4, activation='relu')
        self.attention_fc2 = layers.Dense(input_dim)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, training=False):
        att = self.attention_fc1(inputs)
        att = self.attention_fc2(att)
        att = self.bn(att, training=training)
        selection = tf.sigmoid(att / self.temperature)

        entropy_loss = self.sparsity_reg * tf.reduce_mean(
            -selection * tf.math.log(selection + 1e-10) -
            (1 - selection) * tf.math.log(1 - selection + 1e-10)
        )
        self.add_loss(entropy_loss)

        return inputs * selection, selection


def build_gandalf(
    input_shape: int,
    n_gflu_blocks: int = 4,
    gflu_units: int = 128,
    dropout_rate: float = 0.1,
    use_feature_selection: bool = True,
    sparsity_reg: float = 1e-4
) -> Model:
    """Build GANDALF model for tabular regression."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    if use_feature_selection:
        feature_gate = FeatureSelectionGate(sparsity_reg=sparsity_reg)
        x, selection_weights = feature_gate(x)

    gates = []
    for i in range(n_gflu_blocks):
        gflu_block = GFLUBlock(gflu_units, dropout_rate, name=f'gflu_block_{i}')
        x, gate = gflu_block(x)
        gates.append(gate)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='GANDALF')
    model.gates = gates
    return model


def build_model_from_params_gandalf(params: dict, input_dim: int):
    """Build and compile GANDALF from best parameters."""
    model = build_gandalf(
        input_shape=input_dim,
        n_gflu_blocks=params['n_gflu_blocks'],
        gflu_units=params['gflu_units'],
        dropout_rate=params['dropout_rate'],
        use_feature_selection=params['use_feature_selection'],
        sparsity_reg=params['sparsity_reg']
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
# 20. HOPULAR
###############################################################################


class ModernHopfieldLayer(layers.Layer):
    """Modern Hopfield layer with continuous attention."""

    def __init__(self, embed_dim, beta=1.0, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.beta = beta
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.query_proj = layers.Dense(self.embed_dim)
        self.key_proj = layers.Dense(self.embed_dim)
        self.value_proj = layers.Dense(self.embed_dim)

        self.output_proj = layers.Dense(self.embed_dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, query, memory, training=False):
        Q = self.query_proj(query)
        K = self.key_proj(memory)
        V = self.value_proj(memory)

        scores = tf.matmul(Q, K, transpose_b=True) * self.beta
        scores = scores / tf.sqrt(tf.cast(self.embed_dim, tf.float32))

        attention = tf.nn.softmax(scores, axis=-1)

        retrieved = tf.matmul(attention, V)

        output = self.output_proj(retrieved)
        output = self.dropout(output, training=training)
        output = self.norm(query + output, training=training)

        return output, attention


class HopularBlock(layers.Layer):
    """Hopular block: Hopfield layer + FFN with residuals."""

    def __init__(self, embed_dim, ff_dim, beta=1.0, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.beta = beta
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.hopfield = ModernHopfieldLayer(self.embed_dim, self.beta, self.dropout_rate)

        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embed_dim),
            layers.Dropout(self.dropout_rate)
        ])
        self.ffn_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, query, memory, training=False):
        hopfield_out, attention = self.hopfield(query, memory, training=training)

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
    """Build Hopular model for tabular regression."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)
    query = layers.Dense(embed_dim)(x)
    query = layers.LayerNormalization(epsilon=1e-6)(query)

    memory_layer = layers.Dense(n_memory * embed_dim, use_bias=False)
    memory_flat = memory_layer(tf.ones((1, 1)))
    memory = tf.reshape(memory_flat, (n_memory, embed_dim))

    all_attentions = []
    for i in range(n_blocks):
        block = HopularBlock(embed_dim, ff_dim, beta, dropout_rate, name=f'hopular_block_{i}')
        query, attention = block(query, memory)
        all_attentions.append(attention)

    output = layers.Dense(64, activation='relu')(query)
    output = layers.Dropout(dropout_rate)(output)
    output = layers.Dense(1)(output)

    model = Model(inputs, output, name='Hopular')
    return model


def build_model_from_params_hopular(params: dict, input_dim: int):
    """Build and compile Hopular from best parameters."""
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


###############################################################################
# 21. REALMLP
###############################################################################


def build_realmlp(
    input_shape: int,
    hidden_units: list = [512, 256, 128],
    dropout_rate: float = 0.15,
    use_skip: bool = True
) -> Model:
    """Build RealMLP with modern defaults."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.LayerNormalization(epsilon=1e-6, name='input_norm')(inputs)

    prev_x = None
    for i, units in enumerate(hidden_units):
        h = layers.Dense(
            units,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            bias_initializer='zeros',
            name=f'dense_{i}'
        )(x)

        h = layers.LayerNormalization(epsilon=1e-6, name=f'norm_{i}')(h)
        h = layers.Activation('gelu', name=f'gelu_{i}')(h)
        h = layers.Dropout(dropout_rate, name=f'dropout_{i}')(h)

        if use_skip and prev_x is not None and prev_x.shape[-1] == units:
            h = layers.Add(name=f'skip_{i}')([h, prev_x])

        prev_x = h
        x = h

    outputs = layers.Dense(
        1,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        name='output'
    )(x)

    model = Model(inputs, outputs, name='RealMLP')
    return model


def build_model_from_params_realmlp(params: dict, input_dim: int, n_samples: int = None):
    """Build and compile RealMLP from best parameters."""
    n_layers = params['n_layers']
    hidden_units = [params[f'units_layer_{i}'] for i in range(n_layers)]

    model = build_realmlp(
        input_shape=input_dim,
        hidden_units=hidden_units,
        dropout_rate=params['dropout_rate'],
        use_skip=params['use_skip']
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
# 22. DANETS
###############################################################################


class AbstractLayer(layers.Layer):
    """Abstract Layer: learns to group features into higher-level abstractions."""

    def __init__(self, n_abstracts, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_abstracts = n_abstracts
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.assignment_fc = layers.Dense(self.n_abstracts, activation='softmax')
        self.transform_fc = layers.Dense(self.n_abstracts)
        self.bn = layers.BatchNormalization()
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        assignment = self.assignment_fc(inputs)
        transformed = self.transform_fc(inputs)
        transformed = self.bn(transformed, training=training)
        transformed = tf.nn.relu(transformed)
        abstract = transformed * assignment
        abstract = self.dropout(abstract, training=training)
        return abstract, assignment


class DANetBlock(layers.Layer):
    """DANet Block: Abstract layer + residual connection."""

    def __init__(self, n_abstracts, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_abstracts = n_abstracts
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.abstract_layer = AbstractLayer(self.n_abstracts, self.dropout_rate)
        if input_shape[-1] != self.n_abstracts:
            self.residual_proj = layers.Dense(self.n_abstracts)
        else:
            self.residual_proj = None

    def call(self, inputs, training=False):
        abstract, assignment = self.abstract_layer(inputs, training=training)
        if self.residual_proj is not None:
            residual = self.residual_proj(inputs)
        else:
            residual = inputs
        return abstract + residual, assignment


def build_danets(
    input_shape: int,
    n_abstracts: list = [256, 128, 64],
    dropout_rate: float = 0.1,
    use_shortcut: bool = True
) -> Model:
    """Build DANets model for tabular regression."""
    inputs = layers.Input(shape=(input_shape,))

    x = layers.BatchNormalization()(inputs)

    all_assignments = []
    for i, n_abs in enumerate(n_abstracts):
        block = DANetBlock(n_abs, dropout_rate, name=f'danet_block_{i}')
        x, assignment = block(x)
        all_assignments.append(assignment)

    x = layers.LayerNormalization(epsilon=1e-6)(x)

    if use_shortcut:
        shortcut = layers.Dense(64, activation='relu')(inputs)
        shortcut = layers.Dropout(dropout_rate)(shortcut)
        x = layers.Concatenate()([x, shortcut])

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs, name='DANets')
    model.assignments = all_assignments
    return model


def build_model_from_params_danets(params: dict, input_dim: int):
    """Build and compile DANets from best parameters."""
    n_layers = params['n_layers']
    n_abstracts = [params[f'n_abstracts_{i}'] for i in range(n_layers)]

    model = build_danets(
        input_shape=input_dim,
        n_abstracts=n_abstracts,
        dropout_rate=params['dropout_rate'],
        use_shortcut=params['use_shortcut']
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
# MODEL REGISTRY
###############################################################################

MODEL_REGISTRY = {
    'baseline_mlp':     build_model_from_params_baseline_mlp,
    'sparse_mlp':       build_model_from_params_sparse_mlp,
    'gated_mlp':        build_model_from_params_gated_mlp,
    'tabr':             build_model_from_params_tabr,
    'attention_mlp':    build_model_from_params_attention_mlp,
    'saint':            build_model_from_params_saint,
    'grownet':          build_model_from_params_grownet,
    'snn':              build_model_from_params_snn,
    'tabnet':           build_model_from_params_tabnet,
    'tabnet_inspired':  build_model_from_params_tabnet_inspired,
    'tabm':             build_model_from_params_tabm,
    'lassonet':         build_model_from_params_lassonet,
    'wide_deep':        build_model_from_params_wide_deep,
    'vime':             build_model_from_params_vime,
    'resnet_mlp':       build_model_from_params_resnet_mlp,
    'rtdl_resnet':      build_model_from_params_rtdl_resnet,
    'ft_transformer':   build_model_from_params_ft_transformer,
    'node':             build_model_from_params_node,
    'gandalf':          build_model_from_params_gandalf,
    'hopular':          build_model_from_params_hopular,
    'realmlp':          build_model_from_params_realmlp,
    'danets':           build_model_from_params_danets,
}
