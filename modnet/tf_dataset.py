import tensorflow as tf
import os
from typing import Tuple

class VOSDataset:
    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 img_size: Tuple[int, int] = (256, 256),
                 batch_size: int = 8,
                 shuffle: bool = True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _load_image_and_mask(self, image_path, mask_path):
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        image = tf.cast(image, tf.float32) / 255.0

        # Load mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, self.img_size)
        mask = tf.cast(mask, tf.float32) / 255.0

        return image, mask

    def get_dataset(self):
        # Get all image and mask paths
        image_paths = sorted(tf.io.gfile.glob(os.path.join(self.image_dir, "*.jpg")))
        mask_paths = sorted(tf.io.gfile.glob(os.path.join(self.mask_dir, "*.png")))

        assert len(image_paths) == len(mask_paths), "Mismatch between number of images and masks."

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(self._load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=512)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset
