import tensorflow as tf
import numpy as np
import os
import random
from PIL import Image
from visualize import visualize
from model import build_unet

# Config
IMG_SIZE = (256, 256)
VAL_IMG_DIR = "./data/youtube_vos_flat/valid/images"
VAL_MASK_DIR = "./data/youtube_vos_flat/valid/masks"
MODEL_PATH = "./checkpoints/modnet_unet_best.h5"
NUM_SAMPLES = 8

# Load model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Get validation dataset
image_files = sorted(os.listdir(VAL_IMG_DIR))
mask_files = sorted(os.listdir(VAL_MASK_DIR))

# Get random sample indices
sample_indices = random.sample(range(len(image_files)), NUM_SAMPLES)

for idx in sample_indices:
    img_path = os.path.join(VAL_IMG_DIR, image_files[idx])
    mask_path = os.path.join(VAL_MASK_DIR, mask_files[idx])

    # Load and preprocess image and mask
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    mask = Image.open(mask_path).convert("L").resize(IMG_SIZE)

    img_np = np.array(img) / 255.0
    mask_np = np.array(mask) / 255.0
    img_tensor = tf.expand_dims(img_np, axis=0)  # [1, H, W, 3]

    # Predict
    pred = model.predict(img_tensor)
    pred = (pred > 0.5).astype(np.float32)

    # Visualize
    visualize(img_np, np.expand_dims(mask_np, axis=-1), pred[0])
