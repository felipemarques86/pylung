import configparser
import io
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from bottle import response
from keras import Model
from keras.layers import Conv2D
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from matplotlib import patches
import tensorflow as tf

import cv2
from matplotlib.colors import LinearSegmentedColormap
from skimage.color import gray2rgb
from skimage.segmentation import mark_boundaries
from sklearn.tree import DecisionTreeClassifier, export_text

import pylung
from main.experiment.experiment_utilities import get_model
from main.model_registry.classification.vit import ModelDefinition
from main.utilities.utilities_lib import get_data_transformer_function, img_transformer
from main.utilities.vit_util import get_attention_model, plot_feature_importance_heatmap, get_feature_model, \
    plot_attention_heatmap, plot_attention_heatmap_default, plot_aggregated_attention_heatmap, \
    plot_aggregated_attention_heatmap_default

import shap

config = configparser.ConfigParser()

# check config.ini is in the root folder
config_file = config.read('config.ini')
if len(config_file) == 0:
    raise Exception("config.ini file not found")


def load_your_dataset(directory, ds_name, start_index=0, end_index=1000):
    X = []  # To store images
    y = []  # To store labels

    for index in range(start_index, end_index + 1):
        image_path = os.path.join(directory, ds_name, f'image-{index}.raw')
        annotation_path = os.path.join(directory, ds_name, f'annotation-{index}.txt')

        # Load the image and annotation
        try:
            originalImage = load_data(image_path)
            with open(annotation_path, 'r') as file:
                annotation = file.read().strip()  # Assuming annotation is directly usable as a label

            X.append(originalImage)
            y.append(annotation)
        except FileNotFoundError:
            print(f"Files for index {index} not found, skipping.")
        except Exception as e:
            print(f"Error loading data for index {index}: {e}")

    return X, y

def get_image_ds(ds_type, ds_name, index):
    bbox = True
    crop = False
    data = None

    ret = None
    annotations = None
    directory = config['DATASET'][f'processed_{ds_type}_location']
    with open(directory + f'/{ds_name}/image-{index}.raw', 'rb') as file:
        ret = pickle.load(file)
    if bbox or crop:
        with open(directory + f'/{ds_name}/annotation-{index}.txt', 'rb') as file:
            annotations = pickle.load(file)

    ret[ret < -1000] = -1000
    ret[ret > 600] = 600
    ret = (ret + 1000) / (600 + 1000)
    ret = ret * 255
    buf = io.BytesIO()

    fig, ax = plt.subplots(figsize=(5.12, 5.12))
    ax.axis('off')

    ax.imshow(ret, cmap=plt.cm.gray)
    if bbox and (annotations[0] != 0 or annotations[1] != 0 or annotations[2] != 0 or annotations[3] != 0):
        rect = patches.Rectangle(
            (int(annotations[2]), int(annotations[0])),
            int(annotations[3] - annotations[2]),
            int(annotations[1] - annotations[0]),
            facecolor="none",
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rect)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show()



#models = pylung.models()
#print(models)
#trials = pylung.trials()
#print(trials)


ds_name = 'DS_Charlie'
ds_type = 'lidc_idri'
index = 43364 # com tumor
#index = 43365 # sem tumor
trial = 'results$resnet50'

#prediction = pylung.predict_nodule('results$vit', 'lidc_idri', 'sample', 43298)
#print(prediction)

ModelDefinition

directory = config['DATASET'][f'processed_{ds_type}_location']


# Utilize a function to load data to potentially reduce memory usage
def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


image_path = os.path.join(directory, ds_name, f'image-{index}.raw')
annotation_path = os.path.join(directory, ds_name, f'annotation-{index}.txt')
originalImage = load_data(image_path)
annotation = load_data(annotation_path)

print(annotation)

trial_path = 'weights/' + trial.replace('$', os.sep) + '.json'
with open(trial_path, 'r') as json_fp:
    json_data = json.load(json_fp)

data_transformer = get_data_transformer_function(json_data['data_transformer_name'])

def shap_xai():
    # Assuming get_model() is a function that returns your TensorFlow model
    m_model = get_model(model_type=json_data['model_type'], image_size=json_data['image_size'], static_params=True,
                        metrics=[], code_name=json_data['code_name'],
                        data_transformer_name=json_data['data_transformer_name'],
                        params=json_data['learning_params'], return_model_only=True, batch_size=json_data['batch_size'],
                        epochs=json_data['epochs'], num_classes=json_data['num_classes'], loss=json_data['loss'],
                        data=None, detection=json_data['detection'],
                        isolate_nodule_image=json_data['isolate_nodule_image'], attention=True
                        )

    m = m_model(None)

    # Prepare the image
    image = img_transformer(json_data['image_size'], json_data['image_channels'], json_data['isolate_nodule_image'])(
        originalImage, annotation, None, None)
    img = np.expand_dims(image, axis=0)

    # Create a SHAP explainer object
    # Note: You might need to adapt this to use DeepExplainer or GradientExplainer depending on your model
    explainer = shap.DeepExplainer(m, img)

    # Generate SHAP values for the input image
    shap_values = explainer.shap_values(img)
    print(np.array(shap_values).shape)

    # Plot the SHAP values for the first prediction
    # Adjust the index [0] if you want to visualize explanations for a different class
    shap.image_plot(shap_values[0], img)

def lime_xai():

    m_model = get_model(model_type=json_data['model_type'], image_size=json_data['image_size'], static_params=True,
                        metrics=[], code_name=json_data['code_name'],
                        data_transformer_name=json_data['data_transformer_name'],
                        params=json_data['learning_params'], return_model_only=True, batch_size=json_data['batch_size'],
                        epochs=json_data['epochs'], num_classes=json_data['num_classes'], loss=json_data['loss'],
                        data=None, detection=json_data['detection'],
                        isolate_nodule_image=json_data['isolate_nodule_image'], attention=True
                        )

    m = m_model(None)

    def extract_most_important_segment(image, explanation, N):
        """
        Extract and color the top-N segments from an image that have the highest contributions
        according to the LIME explanation, each with a different color.

        :param image: The original image.
        :param explanation: The LIME explanation object.
        :param N: The number of top segments to highlight.
        """
        # Extract the list of feature weights from the explanation for the top label
        top_label = explanation.top_labels[0]
        feature_weights = explanation.local_exp[top_label]

        # Sort the features (segments) based on their contribution weight and get the top-N
        sorted_weights = sorted(feature_weights, key=lambda x: x[1], reverse=True)

        # Start with a copy of the original image to paint the segments
        painted_image = image.copy()

        # Create a colormap with N unique colors
        colormap = plt.cm.get_cmap('hsv', N)

        for index, (feature, weight) in enumerate(sorted_weights):
            # Generate a unique color for each segment
            color = colormap(index)[:3]  # Get RGB values from the colormap
            color = (np.array(color) * 255).astype(int)  # Scale to 0-255 for image

            # Find the mask for the current segment
            segment_mask = explanation.segments == feature

            if weight > 0:
                for i in range(3):  # Apply the color to each channel
                    painted_image[segment_mask, i] = color[i]
            #else:
                # Apply white color to each channel for negative weights
             #   for i in range(3):  # This loop ensures each channel is set individually
              #      painted_image[segment_mask, i] = 255


        # Display the original image with the top-N segments colored
        plt.figure(figsize=(8, 8))
        plt.imshow(painted_image)


        for index, (feature, weight) in enumerate(sorted_weights):
            # Find the mask for the current segment
            segment_mask = explanation.segments == feature

            # Optionally, overlay the segment's index or weight as text
            y, x = np.argwhere(segment_mask).mean(axis=0).astype(int)
            if index == 0:
                plt.text(x, y, str(feature), color='white', fontsize=20, ha='center', va='center')
            elif index < N:
                plt.text(x, y, str(feature), color='black', fontsize=20, ha='center', va='center')


        plt.axis('off')
        plt.show()

    def model_predict(img_array):
        preds = m.predict(img_array)
        return preds

    image = img_transformer(json_data['image_size'], json_data['image_channels'], json_data['isolate_nodule_image'])(
        originalImage, annotation, None, None)

    img = np.expand_dims(image, axis=0)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img[0].astype('double'),
                                             model_predict,  # prediction function
                                             num_features=5,  # Number of superpixels to consider
                                             top_labels=1,
                                             hide_color=0,
                                             num_samples=1000)
    top_label = explanation.top_labels[0]

    # Extracting the weights for the top label
    weights = explanation.local_exp[top_label]

    # Sorting the features (superpixels) by their contribution (by absolute value)
    sorted_weights = sorted(weights, key=lambda x: x[1], reverse=False)


    # Generating a textual explanation
    textual_explanation = f"Model's prediction was influenced by the following regions, ranked by importance:\n"
    for feature, weight in sorted_weights:
        # Assuming positive weights contribute towards the prediction and negative weights against
        contribution = "positively" if weight > 0 else "negatively"
        textual_explanation += f"- Region {feature} contributed {contribution} with a weight of {weight:.4f}\n"

    print(textual_explanation)

    extract_most_important_segment(img[0], explanation, 10)


def lime_xaixx():

    m_model = get_model(model_type=json_data['model_type'], image_size=json_data['image_size'], static_params=True,
                        metrics=[], code_name=json_data['code_name'],
                        data_transformer_name=json_data['data_transformer_name'],
                        params=json_data['learning_params'], return_model_only=True, batch_size=json_data['batch_size'],
                        epochs=json_data['epochs'], num_classes=json_data['num_classes'], loss=json_data['loss'],
                        data=None, detection=json_data['detection'],
                        isolate_nodule_image=json_data['isolate_nodule_image'], attention=True
                        )

    m = m_model(None)

    def model_predict(img_array):
        # Assume 'model' is a preloaded TensorFlow model, e.g., ResNet50
        preds = m.predict(img_array)
        return preds

    image = img_transformer(json_data['image_size'], json_data['image_channels'], json_data['isolate_nodule_image'])(
        originalImage, annotation, None, None)

    img = np.expand_dims(image, axis=0)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img[0].astype('double'),
                                             model_predict,  # prediction function
                                             top_labels=1,
                                             hide_color=0,
                                             num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=20,
                                                hide_rest=True)
    #plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    #plt.title("LIME Explanation overlaid on Original Image")
    #plt.axis('off')  # Hide the axis to focus on the image
    #plt.show()

    original_img_display = image  # Ensure the original image is in [0, 1] for display

    # Overlay the LIME explanation mask onto the original image
    plt.imshow(img[0])
    plt.imshow(mark_boundaries(original_img_display, mask, color=(1, 0, 0)),
               alpha=0.8)  # alpha controls the transparency
    #plt.axis('off')
    plt.show()

    top_label = explanation.top_labels[0]

    # Extracting the weights for the top label
    weights = explanation.local_exp[top_label]

    # Sorting the features (superpixels) by their contribution (by absolute value)
    sorted_weights = sorted(weights, key=lambda x: abs(x[1]), reverse=True)

    # Generating a textual explanation
    textual_explanation = f"Model's prediction was influenced by the following regions, ranked by importance:\n"
    for feature, weight in sorted_weights:
        # Assuming positive weights contribute towards the prediction and negative weights against
        contribution = "positively" if weight > 0 else "negatively"
        textual_explanation += f"- Region {feature + 1} contributed {contribution} with a weight of {weight:.2f}\n"

    print(textual_explanation)

def global_surrogate_xai():
    # Assuming m_model is your complex model as defined above
    m_model = get_model(...)

    # Load your dataset (features and labels)
    # For demonstration, X would be your features, and y would be the labels
    X, y = load_your_dataset()  # Implement dataset loading as appropriate

    # Predict probabilities with the complex model for the dataset
    # Here, we assume a binary classification for simplicity. Adjust as needed.
    probs = m_model.predict(X)

    # For binary classification, select the probability of the positive class
    prob_positive_class = probs[:, 1]

    # Binarize the probabilities for surrogate model training
    # This step might vary depending on your specific use case and model output
    y_pred = (prob_positive_class > 0.5).astype(int)

    # Train a global surrogate model - here we use a decision tree for simplicity
    surrogate_model = DecisionTreeClassifier(max_depth=5)
    surrogate_model.fit(X, y_pred)

    # Now you can inspect the surrogate model to understand the complex model's behavior
    tree_rules = export_text(surrogate_model, feature_names=['Your', 'Feature', 'Names', 'Here'])
    print(tree_rules)

    # Optionally, visualize the decision tree (if you're in an environment that supports it)
    # plot_tree(surrogate_model)

# Call the function
#global_surrogate_xai()

def calculate_heatmap():
    # Load model weights only once if possible or use a caching mechanism
    m_model = get_model(model_type=json_data['model_type'], image_size=json_data['image_size'], static_params=True,
                        metrics=[], code_name=json_data['code_name'],
                        data_transformer_name=json_data['data_transformer_name'],
                        params=json_data['learning_params'], return_model_only=True, batch_size=json_data['batch_size'],
                        epochs=json_data['epochs'], num_classes=json_data['num_classes'], loss=json_data['loss'],
                        data=None, detection=json_data['detection'],
                        isolate_nodule_image=json_data['isolate_nodule_image'], attention=True
                        )

    m = m_model(None)

    get_image_ds(ds_type, ds_name, index)

    image = img_transformer(json_data['image_size'], json_data['image_channels'], json_data['isolate_nodule_image'])(
        originalImage, annotation, None, None)

    x = np.expand_dims(image, axis=0)

    last_conv_layer = m.get_layer('resnet50').get_layer('conv5_block3_out')
    out = m.get_layer('resnet50').output
    input = m.get_layer('resnet50').inputs
    grad_model = tf.keras.models.Model([input], [last_conv_layer.output, out])

    fig, ax = plt.subplots(figsize=(5.12, 5.12))
    ax.axis('off')
    rect = patches.Rectangle(
        (int((annotation[2] / originalImage.shape[0]) * json_data['image_size']),
         int((annotation[0] / originalImage.shape[0]) * json_data['image_size'])),
        int((annotation[3] / originalImage.shape[0]) * json_data['image_size'] - (
                annotation[2] / originalImage.shape[0]) * json_data['image_size']),
        int((annotation[1] / originalImage.shape[0]) * json_data['image_size'] - (
                annotation[0] / originalImage.shape[0]) * json_data['image_size']),
        facecolor="none",
        edgecolor="black",
        linewidth=2,
    )



    # Get the predicted class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, np.argmax(predictions[0])]

    # Get the gradients with respect to the last convolutional layer
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Compute the guided gradients
    positive_grads = tf.cast(grads > 0, 'float32')
    negative_grads = tf.cast(grads < 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * positive_grads * grads + tf.cast(output <= 0,
                                                                                     'float32') * negative_grads * grads

    # Compute the weights using global average pooling
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    # Get the feature map from the last convolutional layer
    cam = np.zeros((output.shape[0], output.shape[1]), dtype=np.float32)
    # for i, w in enumerate(weights):
    #    cam += w * output[:, :, i]

    for i, w in enumerate(weights):
        cam += w * np.array(output[:, :, i])

    # Resize the heatmap to match the input image size
    cam = cv2.resize(cam.numpy(), (
        json_data['image_size'], json_data['image_size']))  # use size attribute to get image dimensions
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    heatmap = np.uint8(255 * heatmap)

    # Apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

    # Plot the original image and the heatmap

    ax.add_patch(rect)
    ax.axis('off')
    ax.imshow(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))
    plt.show()


def find_last_conv_layer(model):
    """Returns the name of the last convolutional layer in the model."""
    """Returns the name of the last convolutional layer in the base ResNet50 model."""
    # Assuming the first layer of the Sequential model is the ResNet50 base model
    base_model = model.layers[0]
    last_conv_layer_name = None
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
    if last_conv_layer_name is not None:
        return last_conv_layer_name
    else:
        raise ValueError("No convolutional layer found in the base model.")

#calculate_heatmap()
lime_xai()
lime_xai()
lime_xai()
lime_xai()
lime_xai()
lime_xai()
lime_xai()
lime_xai()
lime_xai()
lime_xai()
#shap_xai()
