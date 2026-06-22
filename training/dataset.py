import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .labels_bridge import wind_speed_to_category_id

IMAGE_SIZE = (256, 256)


def load_split(csv_path: str, image_dir: str, test_size: float = 0.2, random_state: int = 42):
    frame = pd.read_csv(csv_path)
    frame["category_id"] = frame["label"].apply(wind_speed_to_category_id)
    train_df, test_df = train_test_split(frame, test_size=test_size, random_state=random_state)
    return (
        _build_dataset(train_df, image_dir),
        _build_dataset(test_df, image_dir),
    )


def _build_dataset(frame: pd.DataFrame, image_dir: str) -> tf.data.Dataset:
    paths = [f"{image_dir}/{name}" for name in frame["img_name"]]
    wind_speeds = frame["label"].astype("float32").to_numpy()
    category_ids = frame["category_id"].astype("int32").to_numpy()

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    image_ds = path_ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

    labels_ds = tf.data.Dataset.from_tensor_slices((category_ids, wind_speeds))
    return tf.data.Dataset.zip((image_ds, labels_ds))


def _load_image(path: str) -> tf.Tensor:
    raw = tf.io.read_file(path)
    image = tf.io.decode_jpeg(raw, channels=1)
    return tf.image.resize(image, IMAGE_SIZE)
