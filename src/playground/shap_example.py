import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import shap

# Load the ResNet50 model pre-trained on ImageNet
model = ResNet50(weights='imagenet')

# Function to preprocess the image and return both preprocessed and original images
def preprocess_and_get_original_img(img_path, target_size=(224, 224)):
    original_img = image.load_img(img_path, target_size=target_size)
    original_img_array = image.img_to_array(original_img) / 255.0  # Normalize to [0, 1] for displaying
    img_array = image.img_to_array(original_img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims), original_img_array

# Load and preprocess your image
img_path = 'C:\\Users\\Felipe\\PycharmProjects\\vit-cnn-lidc-idri-studies\\src\\playground\\cat.png'  # Update this path
preprocessed_img, original_img_array = preprocess_and_get_original_img(img_path)

# Use a batch (or subset) of your data as a background distribution
# Here, we use a single image, but for better explanations, use a more representative set of background examples
background = preprocessed_img[:1]

# Instead of DeepExplainer, try using GradientExplainer
explainer = shap.GradientExplainer(model, preprocessed_img)


# Generate SHAP values for the preprocessed image
shap_values = explainer.shap_values(preprocessed_img)

# Plot the SHAP values for the top prediction
# Adjust the index [0] for different classes according to your model's output
shap.image_plot(shap_values, preprocessed_img)

