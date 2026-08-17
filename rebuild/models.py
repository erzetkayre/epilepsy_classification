"""Model definitions for the FORTEI-ICEE 2026 revision.

Architectures follow the descriptions in the submitted manuscript so that the
comparison stays faithful to the original study. What changes is the training
protocol, not the topology.
"""

from tensorflow import keras
from tensorflow.keras import layers


def _head(x, n_classes):
    if n_classes == 2:
        return layers.Dense(1, activation="sigmoid")(x)
    return layers.Dense(n_classes, activation="softmax")(x)


def build_cnn(input_shape, n_classes, head="gap", pool=2):
    """1D-CNN: three conv blocks, kernels 3/5/7, filters 32/64/128.

    head='flatten' reproduces the submitted manuscript (Flatten -> Dense(128),
    ~4M parameters for 500 training samples). head='gap' replaces it with global
    average pooling. The difference is one of the ablation factors.
    """
    inp = keras.Input(shape=input_shape)
    x = inp
    for filters, k, drop in ((32, 3, 0.1), (64, 5, 0.2), (128, 7, 0.3)):
        x = layers.Conv1D(filters, k, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool)(x)
        x = layers.Dropout(drop)(x)
    x = layers.Flatten()(x) if head == "flatten" else layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    return keras.Model(inp, _head(x, n_classes), name="cnn")


def build_gru(input_shape, n_classes):
    """Two stacked GRU layers, 128 and 64 units."""
    inp = keras.Input(shape=input_shape)
    x = layers.GRU(128, return_sequences=True)(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.GRU(64)(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(inp, _head(x, n_classes), name="gru")


def build_cnn_gru(input_shape, n_classes):
    """Hybrid: two conv blocks for feature extraction, then two GRU layers."""
    inp = keras.Input(shape=input_shape)
    x = inp
    for filters, k in ((32, 3), (64, 5)):
        x = layers.Conv1D(filters, k, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(4)(x)
    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GRU(64)(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(inp, _head(x, n_classes), name="cnn_gru")


BUILDERS = {"cnn": build_cnn, "gru": build_gru, "cnn_gru": build_cnn_gru}
