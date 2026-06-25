"""
LTC-AN: Lightweight Temporal Convolutional Attention Network
Engineered Model — F-Operation (FABRICATE)

Researcher: Philip Opoku Brako | KNUST CS | Student ID: 20916855
Date: June 2026

Architecture:
    - Dual-stage residual block replacing standard dense TCN block
    - Stage 1: Depthwise Separable Dilated Convolution
    - Stage 2: Causal Multi-Head Self-Attention Pooling Head
    - Deployment target: TensorFlow Lite Micro (ARM Cortex-M / Raspberry Pi 4)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import random

# ── Reproducibility ──────────────────────────────────────────────────────────
def enforce_reproducibility(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

enforce_reproducibility(42)


# ── Stage 1: Depthwise Separable Dilated Convolution Block ───────────────────
class DepthwiseSeparableDilatedConv(keras.layers.Layer):
    """
    Replaces standard dense 1D dilated convolution.
    Reduces parameters from O(K x C^2) to O(K x C + C^2).
    Chollet (2017) — Xception: https://doi.org/10.1109/CVPR.2017.195
    """
    def __init__(self, filters, kernel_size, dilation_rate, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.depthwise = keras.layers.DepthwiseConv1D(
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )
        self.pointwise = keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            activation='relu'
        )
        self.dropout = keras.layers.Dropout(dropout)
        self.norm = keras.layers.LayerNormalization()

    def call(self, x, training=False):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.dropout(x, training=training)
        return self.norm(x)


# ── Stage 2: Causal Multi-Head Self-Attention Pooling Head ───────────────────
class CausalMultiHeadAttentionPooling(keras.layers.Layer):
    """
    Linear complexity O(N x d_model) causal attention.
    Replaces O(N^2) vanilla attention (Vaswani et al., 2017).
    Causal mask ensures position t attends only to positions <= t.
    """
    def __init__(self, num_heads=2, d_model=32, **kwargs):
        super().__init__(**kwargs)
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model
        )
        self.norm = keras.layers.LayerNormalization()

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        # Causal mask — upper triangular, blocks future positions
        causal_mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len)), -1, 0
        )
        attn_output = self.attention(
            x, x,
            attention_mask=causal_mask,
            training=training
        )
        return self.norm(x + attn_output)  # Residual connection


# ── Dual-Stage LTC-AN Residual Block ─────────────────────────────────────────
class LTCANResidualBlock(keras.layers.Layer):
    """
    Fabricated dual-stage block:
        Stage 1 — DepthwiseSeparableDilatedConv
        Stage 2 — CausalMultiHeadAttentionPooling
    Skip connection added from input to output.
    """
    def __init__(self, filters, kernel_size, dilation_rate,
                 num_heads=2, d_model=32, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.conv_stage = DepthwiseSeparableDilatedConv(
            filters, kernel_size, dilation_rate, dropout
        )
        self.attn_stage = CausalMultiHeadAttentionPooling(num_heads, d_model)
        self.skip_conv = keras.layers.Conv1D(filters, kernel_size=1)

    def call(self, x, training=False):
        residual = self.skip_conv(x)
        out = self.conv_stage(x, training=training)
        out = self.attn_stage(out, training=training)
        return out + residual  # Skip connection


# ── Full LTC-AN Model ─────────────────────────────────────────────────────────
def build_ltcan(
    input_shape,        # (timesteps, n_channels) e.g. (50, 9)
    num_classes=4,      # Normal, SEU, Thermal, Bus-Degrade
    filters=64,
    kernel_size=3,
    dilations=[1, 2, 4],
    num_heads=2,
    d_model=32,
    dropout=0.2
):
    """
    Builds the full LTC-AN model.
    3 dual-stage residual blocks with increasing dilation rates.
    Output: softmax classification over num_classes anomaly types.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for dilation in dilations:
        x = LTCANResidualBlock(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            num_heads=num_heads,
            d_model=d_model,
            dropout=dropout
        )(x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='LTC-AN')
    return model


# ── Focal Loss (handles class imbalance) ─────────────────────────────────────
def focal_loss(gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))
    return loss_fn


# ── Quick model summary (run this file directly to verify) ───────────────────
if __name__ == '__main__':
    enforce_reproducibility(42)
    model = build_ltcan(input_shape=(50, 9), num_classes=4)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss(gamma=2.0),
        metrics=['accuracy']
    )
    model.summary()
