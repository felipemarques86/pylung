import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Load the ResNet50 model pre-trained on ImageNet
model = ResNet50(weights='imagenet')

# Function to preprocess the image and return both preprocessed and original images
def preprocess_and_get_original_img(img_path, target_size=(224, 224)):
    original_img = image.load_img(img_path, target_size=target_size)
    original_img_array = image.img_to_array(original_img) / 255.0  # Normalize to [0, 1] for displaying
    img_array = image.img_to_array(original_img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims), original_img_array

# Prediction function for the model
def model_predict(img_array):
    preds = model.predict(img_array)
    return preds

# Load and preprocess your image (update the path to your image)
img_path = 'C:\\Users\\Felipe\\PycharmProjects\\vit-cnn-lidc-idri-studies\\src\\playground\\cat.png'  # Update this path
preprocessed_img, original_img_array = preprocess_and_get_original_img(img_path)

# Initialize LIME Image Explainer
explainer = lime_image.LimeImageExplainer()

# Explain the prediction
explanation = explainer.explain_instance(preprocessed_img[0].astype('double'), model_predict, top_labels=1, hide_color=0, num_samples=1000)

# Get image and mask for the top prediction
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)

# Overlay the LIME explanation on the original image
plt.figure(figsize=(8, 8))
plt.imshow(original_img_array)  # Show the original image
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask), alpha=0.8)  # Overlay the explanation
plt.axis('off')
plt.show()
