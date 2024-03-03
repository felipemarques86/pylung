import configparser
import io
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from bottle import response
from matplotlib import patches

import pylung
from main.experiment.experiment_utilities import get_model
from main.model_registry.classification.vit import ModelDefinition
from main.utilities.utilities_lib import get_data_transformer_function, img_transformer
from main.utilities.vit_util import get_attention_model, plot_feature_importance_heatmap, get_feature_model, \
    plot_attention_heatmap, plot_attention_heatmap_default, plot_aggregated_attention_heatmap, \
    plot_aggregated_attention_heatmap_default

config = configparser.ConfigParser()

# check config.ini is in the root folder
config_file = config.read('config.ini')
if len(config_file) == 0:
    raise Exception("config.ini file not found")


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


ds_name = 'sample'
ds_type = 'lidc_idri'
index = 43364 # com tumor
#index = 43365 # sem tumor
trial = 'results$vit'

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
image = load_data(image_path)
annotation = load_data(annotation_path)

trial_path = 'weights/' + trial.replace('$', os.sep) + '.json'
with open(trial_path, 'r') as json_fp:
    json_data = json.load(json_fp)

data_transformer = get_data_transformer_function(json_data['data_transformer_name'])



attention_heads = ['modified_multi_head_attention_2']

def calculate_attention_map():
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

    img_original = get_image_ds(ds_type, ds_name, index)

    attention_model = get_attention_model(m.original.model, 'modified_multi_head_attention_2')

    attention_model.load_weights('weights/' + trial.replace('$', os.sep) + '.h5')
    im = img_transformer(json_data['image_size'], json_data['image_channels'], json_data['isolate_nodule_image'])(
        image, annotation, None, None)
    vectorized_image = np.expand_dims(im, axis=0)
    output, attention_values = attention_model.predict(vectorized_image)
    #print(output)

    class_token_attention = attention_values[0, :, 0, 1:]  # Shape: (num_heads, seq_length-1)

    # Aggregate attention across heads for the class token
    class_token_attention_aggregated = np.mean(class_token_attention, axis=0)

    # Rank tokens by their influence on the class token
    influential_tokens = np.argsort(class_token_attention_aggregated)[::-1]

    # Print the most influential tokens
    print("Most influential tokens for the class token output:")
    print(influential_tokens)

    plot_aggregated_attention_heatmap_default(im, attention_values)
    #for token in influential_tokens[:5]:  # Plot the top 5 most influential tokens
    #    print('Token:', token)
    #    plot_attention_heatmap_default(im, attention_values, token)


def calculate_feature_map():

    # Load model weights only once if possible or use a caching mechanism
    m_model = get_model(model_type=json_data['model_type'], image_size=json_data['image_size'], static_params=True,
                        metrics=[], code_name=json_data['code_name'],
                        data_transformer_name=json_data['data_transformer_name'],
                        params=json_data['learning_params'], return_model_only=True, batch_size=json_data['batch_size'],
                        epochs=json_data['epochs'], num_classes=json_data['num_classes'], loss=json_data['loss'],
                        data=None, detection=json_data['detection'],
                        isolate_nodule_image=json_data['isolate_nodule_image'], attention=False
                        )

    m = m_model(None)

    global output
    for head in attention_heads:
        print(head)
        attention_model = get_feature_model(m.original.model, 'multi_head_attention_2')

        attention_model.load_weights('weights/' + trial.replace('$', os.sep) + '.h5')
        im = img_transformer(json_data['image_size'], json_data['image_channels'], json_data['isolate_nodule_image'])(
            image, annotation, None, None)
        vectorized_image = np.expand_dims(im, axis=0)
        output, attention_values = attention_model.predict(vectorized_image)

        max_value = np.max(output[0])
        predicted_int = [1 if value == max_value else 0 for value in output[0]]

        # Construct response
        ret = {
            'predicted': output[0].tolist(),
            'predicted_int': predicted_int,
            'annotation': str(annotation),
            'transformed_annotation': data_transformer(annotation, None, None, None),
            'textual': pylung.get_textual(output[0], json_data['data_transformer_name']),
            'expected_textual': pylung.get_textual(data_transformer(annotation, None, None, None),
                                                   json_data['data_transformer_name'])
        }

        plot_feature_importance_heatmap(im, attention_values, 0.0)
    # attention_model = get_attention_model(m.original.model, 'multi_head_attention_2')
    # attention_model.load_weights('weights/' + trial.replace('$', os.sep) + '.h5')
    # im = img_transformer(json_data['image_size'], json_data['image_channels'], json_data['isolate_nodule_image'])(image, annotation, None, None)
    # vectorized_image = np.expand_dims(im, axis=0)
    # output, attention_values = attention_model.predict(vectorized_image)
    #
    # print(attention_values)
    # print(attention_values.shape)
    #
    # plot_feature_importance_heatmap(im, attention_values)


#calculate_feature_map()
calculate_attention_map()







