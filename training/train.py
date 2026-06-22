"""
Trains the dual-input cyclone intensity model on INSAT-3D infrared imagery.

Usage:
    python -m training.train --csv path/to/insat_3d_ds.csv --images path/to/CYCLONE_DATASET_INFRARED

Metadata inputs (eye temperature, cloud-top temperature, eye diameter,
symmetry, latitude, SST) are not present in the public label CSV, so this
script trains the image branch with zeroed metadata placeholders. Swap in
real per-storm metadata columns once available for the full dual-input gain.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.model.architecture import build_intensity_model, compile_model  # noqa: E402
from app.model import augmentation  # noqa: E402

from .dataset import load_split


def attach_metadata(dataset: tf.data.Dataset, num_features: int = 6) -> tf.data.Dataset:
    def add_metadata(image, labels):
        category_id, wind_speed = labels
        metadata = tf.zeros((num_features,), dtype=tf.float32)
        return (image, metadata), (category_id, wind_speed)

    return dataset.map(add_metadata, num_parallel_calls=tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", default="cyclone_intensity_model.weights.h5")
    args = parser.parse_args()

    train_ds, test_ds = load_split(args.csv, args.images)

    train_ds = augmentation.build_training_pipeline(train_ds, batch_size=args.batch_size, repeats=3)
    test_ds = augmentation.build_eval_pipeline(test_ds, batch_size=args.batch_size)

    train_ds = attach_metadata(train_ds)
    test_ds = attach_metadata(test_ds)

    model = build_intensity_model()
    compile_model(model)
    model.summary()

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        args.checkpoint, monitor="val_intensity_category_accuracy", save_best_only=True, save_weights_only=True
    )
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_intensity_category_accuracy", patience=6, restore_best_weights=True)

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
    )

    metrics = model.evaluate(test_ds)
    print(dict(zip(model.metrics_names, metrics)))


if __name__ == "__main__":
    main()
