from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def get_attention_model(original_model, transformer_layer_name):
    """
    Creates a new model that outputs the original predictions and attention weights.

    Args:
    - original_model: The trained ViT model.
    - transformer_layer_name: Name of the transformer layer to extract attention from.

    Returns:
    - A new model that outputs both predictions and attention weights.
    """
    # Input layer from the original model
    model_input = original_model.input

    # Output layer for predictions
    model_output = original_model.output

    # Extracting the attention layer output
    _, attention_layer_output = original_model.get_layer(transformer_layer_name).output

    # Creating a new model that outputs both predictions and attention weights
    attention_model = Model(model_input, [model_output, attention_layer_output])

    return attention_model

def get_feature_model(original_model, transformer_layer_name):
    """
    Creates a new model that outputs the original predictions and attention weights.

    Args:
    - original_model: The trained ViT model.
    - transformer_layer_name: Name of the transformer layer to extract attention from.

    Returns:
    - A new model that outputs both predictions and attention weights.
    """
    # Input layer from the original model
    model_input = original_model.input

    # Output layer for predictions
    model_output = original_model.output

    # Extracting the attention layer output
    attention_layer_output = original_model.get_layer(transformer_layer_name).output

    # Creating a new model that outputs both predictions and attention weights
    attention_model = Model(model_input, [model_output, attention_layer_output])

    return attention_model


"""

Given the attention scores' dimension is (1, 256, 68), and understanding now that 68 refers to your projection_dim and not the traditional attention scores shape, it seems there was a miscommunication about what these values represent. Typically, attention scores in the context of visualization refer to the weights that indicate how much attention each part of the input (each patch, in the case of ViT) pays to every other part. However, the shape (1, 256, 68) suggests you're looking at the output features of the MultiHeadAttention layer for each patch, not the attention scores themselves.

If your goal is to visualize attention in a manner that highlights which areas of an image the model focuses on, but you have access only to the output features from the MultiHeadAttention layer, you'll need a different approach since the direct attention weights are not available in the shape you've provided.

Given the situation and your data structure, here's an alternative approach to visualize the influence or "attention" implied by the output features:

Aggregate Feature Importance: One way to interpret the "attention" without direct attention scores is to aggregate the feature importance across the projection_dim. For example, you could calculate the norm (magnitude) of the feature vector for each patch to get a single importance score per patch.

Visualization Based on Aggregated Feature Importance: Use these importance scores to generate a heatmap. This method doesn't use attention scores per se but provides a way to visualize which patches the model finds most informative based on the output features' magnitude.

Adjusted Visualization Function:
Here's how you might adjust the visualization function to work with the feature vectors instead of attention scores:

"""


def plot_feature_importance_heatmap(image, feature_vectors, threshold=0.95):
    """
    Plots a heatmap over the image based on the magnitude of feature vectors, adjusting the color range
    according to max and min values after applying a threshold.

    Args:
    - image: The input image to the model. Assume shape is (H, W, C).
    - feature_vectors: The output feature vectors from MultiHeadAttention with shape (1, 256, 68).
    - threshold: A float value representing the threshold for filtering feature importance.
    """
    # Calculate the norm of each feature vector (i.e., the magnitude)
    feature_importance = np.linalg.norm(feature_vectors.squeeze(0), axis=1)

    # Normalize for better visualization
    feature_importance_normalized = feature_importance / np.max(feature_importance)

    # Apply threshold and then rescale the values to [0, 1] based on the new min and max
    feature_importance_normalized = np.maximum(feature_importance_normalized - threshold, 0)
    if np.max(feature_importance_normalized) > 0:  # Avoid division by zero
        feature_importance_normalized /= np.max(feature_importance_normalized)

    # Assuming an 16x16 grid of patches based on your patch_size and image dimensions
    num_patches_side = int(np.sqrt(feature_importance_normalized.shape[0]))

    # Reshape to a square form for visualization
    importance_map = feature_importance_normalized.reshape((num_patches_side, num_patches_side))

    # Resize the importance map to the image size for overlaying
    importance_map_resized = np.array(
        Image.fromarray((importance_map * 255).astype(np.uint8)).resize(image.shape[:2], Image.BILINEAR))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.9)
    plt.imshow(importance_map_resized, cmap='jet', alpha=0.6)  # Overlaying the importance map
    plt.title('Feature Importance Heatmap (Threshold: {:.2f})'.format(threshold))
    plt.axis('off')
    plt.show()




def plot_aggregated_attention_heatmap(image, attention_weights):
    """
    Plots a heatmap over the image based on aggregated attention weights.

    Args:
    - image: The input image to the model. Assume shape is (H, W, C).
    - attention_weights: The attention weights for the image with shape (1, num_heads, seq_length, seq_length).
    """
    # Normalize and average the attention weights across heads and tokens
    attention_weights = attention_weights.squeeze(0)  # Remove batch dimension
    attention_aggregated = np.mean(attention_weights, axis=(0, 2))  # Aggregate across heads and tokens

    # Normalize the aggregated attention scores to range [0, 1]
    attention_aggregated -= attention_aggregated.min()
    attention_aggregated /= attention_aggregated.max()

    # Reshape into a square grid
    num_patches_side = int(np.sqrt(len(attention_aggregated)))
    attention_map = attention_aggregated.reshape((num_patches_side, num_patches_side))

    # Upsample to match the image size
    attention_map_resized = np.array(
        Image.fromarray((attention_map * 255).astype(np.uint8)).resize(image.shape[:2], Image.BILINEAR))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.6)
    plt.imshow(attention_map_resized, cmap='jet', alpha=0.4)  # Overlaying the attention map
    plt.title('Aggregated Attention Heatmap')
    plt.axis('off')
    plt.show()


def plot_attention_heatmap(image, attention_weights, token_index=0):
    """
    Plots a heatmap over the image based on normalized attention weights.

    Args:
    - image: The input image to the model. Assume shape is (H, W, C).
    - attention_weights: The attention weights for the image with shape (1, num_heads, seq_length, seq_length).
    - token_index: The index of the token for which to visualize attention weights.
    """
    # Averaging across the attention heads dimension and normalizing
    attention_weights_mean = np.mean(attention_weights[0], axis=0)  # Shape: (seq_length, seq_length)
    attention_token = attention_weights_mean[token_index]  # Shape: (seq_length,)

    # Normalize the attention weights to range [0, 1]
    attention_token -= attention_token.min()
    attention_token /= attention_token.max()

    # Reshape attention to a square grid that matches the layout of the image patches
    num_patches_side = int(np.sqrt(attention_token.shape[0]))
    attention_map = attention_token.reshape((num_patches_side, num_patches_side))

    # Upsample attention map to match the image size
    attention_map_resized = np.array(Image.fromarray((attention_map * 255).astype(np.uint8)).resize(image.shape[:2], Image.BILINEAR))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.6)
    plt.imshow(attention_map_resized, cmap='jet', alpha=0.4)  # Overlaying the normalized attention map
    plt.title('Attention Heatmap')
    plt.axis('off')
    plt.show()

def plot_attention_heatmap_default(image, attention_weights, token_index=0):
    """
    Plots a heatmap over the image based on attention weights.

    Args:
    - image: The input image to the model. Assume shape is (H, W, C).
    - attention_weights: The attention weights for the image with shape (1, num_heads, seq_length, seq_length).
    - token_index: The index of the token for which to visualize attention weights.
    """
    # Averaging across the attention heads dimension
    attention_weights_mean = np.mean(attention_weights[0], axis=0)  # Shape: (seq_length, seq_length)
    print('Tokens Count', len(attention_weights_mean))

    # Selecting the attention weights for a specific token
    attention_token = attention_weights_mean[token_index]  # Shape: (seq_length,)

    # Reshape attention to a square grid that matches the layout of the image patches
    num_patches_side = int(np.sqrt(attention_token.shape[0]))
    attention_map = attention_token.reshape((num_patches_side, num_patches_side))

    attention_map = attention_map / np.max(attention_map)

    # Upsample attention map to match the image size
    attention_map_resized = np.array(
        Image.fromarray((attention_map * 255).astype(np.uint8)).resize(image.shape[:2], Image.BILINEAR))

    # Normalize the upsampled attention map for better visualization
    attention_map_resized = (attention_map_resized - np.min(attention_map_resized)) / np.ptp(attention_map_resized)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.6)
    plt.imshow(attention_map_resized, cmap='jet', alpha=0.4)  # Overlaying the attention map
    plt.title('Attention Heatmap for Token ' + str(token_index))
    plt.axis('off')
    plt.show()


def plot_aggregated_attention_heatmap_default(image, attention_weights):
    """
    Plots a heatmap over the image based on aggregated attention weights.

    Args:
    - image: The input image to the model. Assume shape is (H, W, C).
    - attention_weights: The attention weights for the image with shape (1, num_heads, seq_length, seq_length).
    """
    # Normalize and average the attention weights across heads and tokens
    attention_weights = np.mean(attention_weights, axis=(1, 2))  # Shape: (1, seq_length)
    attention_token = np.mean(attention_weights, axis=0)  # Shape: (seq_length,)

    # Reshape attention to a square grid that matches the layout of the image patches
    num_patches_side = int(np.sqrt(attention_token.shape[0]))
    attention_map = attention_token.reshape((num_patches_side, num_patches_side))

    attention_map = attention_map / np.max(attention_map)

    # Upsample attention map to match the image size
    attention_map_resized = np.array(Image.fromarray((attention_map * 255).astype(np.uint8)).resize(image.shape[:2], Image.BILINEAR))

    # Normalize the upsampled attention map for better visualization
    #attention_map_resized = (attention_map_resized - np.min(attention_map_resized)) / np.ptp(attention_map_resized)

    print(attention_map_resized)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.6)
    plt.imshow(attention_map_resized, cmap='jet', alpha=0.5) # Overlaying the attention map
    plt.title('Aggregated Attention Heatmap')
    plt.axis('off')
    plt.show()
