"""
Dual-input cyclone intensity network.

Image branch: a squeeze-excite residual stack over depthwise-separable
convolutions, followed by a spatial attention gate that learns to weight the
storm eye/core region before pooling. Metadata branch: a small dense encoder
for the human-entered parameters (eye temperature, cloud-top temperature, eye
diameter, symmetry score, latitude, SST). The two branches fuse into a shared
trunk with two heads: a 6-way intensity-category classifier and a continuous
wind-speed regressor, trained jointly so each head regularizes the other.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

NUM_CATEGORIES = 6
IMAGE_SIZE = (256, 256, 1)
METADATA_FEATURES = 6


def squeeze_excite(x, ratio=16, name="se"):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name}_pool")(x)
    se = layers.Dense(max(filters // ratio, 4), activation="relu", name=f"{name}_reduce")(se)
    se = layers.Dense(filters, activation="sigmoid", name=f"{name}_expand")(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])


def separable_residual_block(x, filters, stride, name):
    shortcut = x
    y = layers.SeparableConv2D(filters, 3, strides=stride, padding="same", name=f"{name}_sep1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("swish", name=f"{name}_act1")(y)

    y = layers.SeparableConv2D(filters, 3, padding="same", name=f"{name}_sep2")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    y = squeeze_excite(y, name=f"{name}_se")

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", name=f"{name}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    y = layers.Add(name=f"{name}_add")([y, shortcut])
    return layers.Activation("swish", name=f"{name}_out")(y)


def spatial_attention(x, name="spatial_attention"):
    avg_pool = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True), name=f"{name}_avg")(x)
    max_pool = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True), name=f"{name}_max")(x)
    concat = layers.Concatenate(name=f"{name}_concat")([avg_pool, max_pool])
    attention_map = layers.Conv2D(1, 7, padding="same", activation="sigmoid", name=f"{name}_conv")(concat)
    return layers.Multiply(name=f"{name}_apply")([x, attention_map])


def build_image_branch(input_shape=IMAGE_SIZE):
    inputs = layers.Input(shape=input_shape, name="ir_image")

    x = layers.Conv2D(32, 3, strides=2, padding="same", name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("swish", name="stem_act")(x)

    stage_filters = [48, 96, 192, 320]
    for i, filters in enumerate(stage_filters):
        x = separable_residual_block(x, filters, stride=2, name=f"stage{i+1}_block1")
        x = separable_residual_block(x, filters, stride=1, name=f"stage{i+1}_block2")

    x = spatial_attention(x, name="eye_attention")

    avg = layers.GlobalAveragePooling2D(name="gap")(x)
    mx = layers.GlobalMaxPooling2D(name="gmp")(x)
    features = layers.Concatenate(name="image_features")([avg, mx])

    return inputs, features


def build_metadata_branch(num_features=METADATA_FEATURES):
    inputs = layers.Input(shape=(num_features,), name="storm_parameters")
    x = layers.Dense(32, activation="relu", name="meta_dense1")(inputs)
    x = layers.BatchNormalization(name="meta_bn1")(x)
    x = layers.Dropout(0.2, name="meta_dropout1")(x)
    x = layers.Dense(16, activation="relu", name="meta_dense2")(x)
    return inputs, x


def build_intensity_model(image_shape=IMAGE_SIZE, num_metadata=METADATA_FEATURES, num_categories=NUM_CATEGORIES):
    image_input, image_features = build_image_branch(image_shape)
    metadata_input, metadata_features = build_metadata_branch(num_metadata)

    fused = layers.Concatenate(name="fusion")([image_features, metadata_features])
    fused = layers.Dense(128, activation="relu", name="fusion_dense1")(fused)
    fused = layers.Dropout(0.3, name="fusion_dropout")(fused)
    fused = layers.Dense(64, activation="relu", name="fusion_dense2")(fused)

    category_output = layers.Dense(num_categories, activation="softmax", name="intensity_category")(fused)
    wind_speed_output = layers.Dense(1, activation="linear", name="wind_speed_knots")(fused)

    model = Model(
        inputs=[image_input, metadata_input],
        outputs=[category_output, wind_speed_output],
        name="cyclone_intensity_net",
    )
    return model


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "intensity_category": "sparse_categorical_crossentropy",
            "wind_speed_knots": tf.keras.losses.Huber(),
        },
        loss_weights={"intensity_category": 1.0, "wind_speed_knots": 0.5},
        metrics={
            "intensity_category": "accuracy",
            "wind_speed_knots": "mae",
        },
    )
    return model
