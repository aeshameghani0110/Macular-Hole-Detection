import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# Constants
CLASS_MAPPING = {
    "background": (0, 0, 0),  # Black
    "macular_holes": (255, 0, 0),  # Red
    "retina": (0, 255, 0),  # Green
    "irc": (0, 0, 255),  # Blue
    "choroid": (255, 255, 0),  # Yellow
}

CLASS_COLORS = {i: color for i, (_, color) in enumerate(CLASS_MAPPING.items())}
CLASS_NAMES = list(CLASS_MAPPING.keys())
MODEL = tf.keras.models.load_model("model.h5")


def predict(original_image):
    np.random.seed(42)
    tf.random.set_seed(42)

    # Create color map for visualization
    colors = (
        np.array(
            [
                [0, 0, 0],  # Background - Black
                [255, 0, 0],  # Macular Holes - Red
                [0, 255, 0],  # Retina - Green
                [0, 0, 255],  # IRC - Blue
                [255, 255, 0],  # Choroid - Yellow
            ]
        )
        / 255.0
    )

    image = tf.convert_to_tensor(np.array(original_image))  # Convert to tensor
    image = tf.image.resize(image, (128, 128))  # Resize to (128, 128)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize image

    # Add batch dimension for the model input (the model expects a batch of images)
    image = tf.expand_dims(image, axis=0)  # Shape should now be (1, 128, 128, 3)

    # Predict segmentation mask
    predictions = MODEL.predict(image)

    # Get original image and predicted mask
    image = image[0]  # Shape (128, 128, 3)
    pred_mask = np.argmax(predictions[0], axis=-1)  # Shape (128, 128)

    # Create color mask for the prediction
    pred_mask_color = colors[pred_mask]

    # Plot results
    plt.figure(figsize=(10, 4))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    # Prediction Mask
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask_color)
    plt.title("Prediction")
    plt.axis("off")

    # Create the legend using custom lines for each class
    legend_elements = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=tuple(x), markersize=10
        )
        for x in colors
    ]

    # Add legend to the figure
    plt.figlegend(
        legend_elements, CLASS_NAMES, loc="center right", bbox_to_anchor=(0.98, 0.5)
    )

    plt.savefig("static/results.png", bbox_inches="tight", dpi=300)

    return predictions[0][0][0].max() * 100
