import tensorflow as tf
import os
from tf_dataset import VOSDataset
from model import build_unet

# Define constants
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 15

TRAIN_IMG_DIR = "./data/youtube_vos_flat/train/images"
TRAIN_MASK_DIR = "./data/youtube_vos_flat/train/masks"
VAL_IMG_DIR = "./data/youtube_vos_flat/valid/images"
VAL_MASK_DIR = "./data/youtube_vos_flat/valid/masks"
MODEL_SAVE_PATH = "./checkpoints/modnet_unet_best.h5"

# Load dataset
print("Loading dataset...")
train_loader = VOSDataset(
    image_dir=TRAIN_IMG_DIR,
    mask_dir=TRAIN_MASK_DIR,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_loader = VOSDataset(
    image_dir=VAL_IMG_DIR,
    mask_dir=VAL_MASK_DIR,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

train_ds = train_loader.get_dataset()
val_ds = val_loader.get_dataset()

# Create model
print("Creating model...")
model = build_unet(input_shape=IMG_SIZE + (3,))
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', mode='min'
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# Start training
print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# Save final model
model.save("./checkpoints/modnet_unet_final.h5")
print("Training complete. Model saved to ./checkpoints/modnet_unet_final.h5")
