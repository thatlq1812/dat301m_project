import matplotlib.pyplot as plt
import numpy as np

def visualize(image, mask, pred=None):
    plt.figure(figsize=(12, 4))

    # Image
    plt.subplot(1, 3 if pred is not None else 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis('off')

    # Ground Truth Mask
    plt.subplot(1, 3 if pred is not None else 2, 2)
    plt.imshow(mask[..., 0], cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    # Predicted Mask (if any)
    if pred is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(pred[..., 0], cmap='gray')
        plt.title("Prediction")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
