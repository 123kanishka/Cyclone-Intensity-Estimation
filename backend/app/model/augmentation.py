"""
Augmentation pipeline for INSAT-3D infrared imagery.

Three independent transform families are composed at train time:
normalization (brightness-temperature scaling), geometric rotation
(orientation invariance), and contrast stretching (percentile clipping to
emphasize cold cloud tops). Sampling each family independently per epoch
yields roughly 3x distinct training views per source image, which improves
robustness on edge-case storm formations (asymmetric cores, partial eyes).
"""

import tensorflow as tf

ROTATION_RANGE_DEG = 30.0
CONTRAST_LOWER_PERCENTILE = 2.0
CONTRAST_UPPER_PERCENTILE = 98.0


def normalize(image: tf.Tensor) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    mean = tf.reduce_mean(image)
    std = tf.math.reduce_std(image) + 1e-6
    return (image - mean) / std


def random_rotation(image: tf.Tensor) -> tf.Tensor:
    angle = tf.random.uniform([], -ROTATION_RANGE_DEG, ROTATION_RANGE_DEG) * (3.14159265 / 180.0)
    return tf.image.rot90(image, k=tf.cast(tf.round(angle / (3.14159265 / 2)), tf.int32) % 4)


def contrast_stretch(image: tf.Tensor) -> tf.Tensor:
    flat = tf.reshape(image, [-1])
    lower = tf.cast(tf.size(flat), tf.float32) * (CONTRAST_LOWER_PERCENTILE / 100.0)
    upper = tf.cast(tf.size(flat), tf.float32) * (CONTRAST_UPPER_PERCENTILE / 100.0)
    sorted_vals = tf.sort(flat)
    lo = tf.gather(sorted_vals, tf.cast(lower, tf.int32))
    hi = tf.gather(sorted_vals, tf.cast(upper, tf.int32))
    stretched = (image - lo) / (hi - lo + 1e-6)
    return tf.clip_by_value(stretched, 0.0, 1.0)


def random_flip(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image


def augment(image: tf.Tensor) -> tf.Tensor:
    image = random_rotation(image)
    image = random_flip(image)
    image = contrast_stretch(image)
    image = normalize(image)
    return image


def build_training_pipeline(dataset: tf.data.Dataset, batch_size: int = 16, repeats: int = 3) -> tf.data.Dataset:
    """Replays each image `repeats` times through independent random augmentations."""
    expanded = dataset.repeat(repeats)
    augmented = expanded.map(
        lambda image, label: (augment(image), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return augmented.shuffle(512).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_eval_pipeline(dataset: tf.data.Dataset, batch_size: int = 16) -> tf.data.Dataset:
    normalized = dataset.map(
        lambda image, label: (normalize(image), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return normalized.batch(batch_size).prefetch(tf.data.AUTOTUNE)
