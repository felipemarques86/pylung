import ast
import configparser
import io
import json
import multiprocessing
import os
import pickle
import re
import subprocess
import time
import traceback
from importlib import reload
from json import dumps
from urllib.parse import parse_qs
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna_dashboard
import tensorflow
import typer
from bottle import response, request
from bottle import route, run, static_file
from colorama import init as colorama_init
from matplotlib import patches
from optuna.samplers import TPESampler
from prettytable import PrettyTable

from main.dataset_utilities.dataset_reader_classes import DatasetTransformer
from main.dataset_utilities.lidc_dataset_reader_classes import LidcDatasetReader, CustomLidcDatasetReader
from main.dataset_utilities.pylung_ui import display_dataset_images
from main.experiment.experiment_utilities import get_ds, get_model, load_module
from main.models.ml_model import CustomModelDefinition
from main.utilities.utilities_lib import warning, error, info, get_data_transformer, display_original_image_bbox, \
    get_experiment_codename, get_channels, img_transformer, LIDC_ANN_Y0, LIDC_ANN_Y1, LIDC_ANN_X0, LIDC_ANN_X1, \
    get_list_database_transformers, bounding_box_intersection_over_union

config = configparser.ConfigParser()
app = typer.Typer()
colorama_init()
dashboards = []
current_studies = []
CONDA_ENV = os.environ['CONDA_DEFAULT_ENV']

STUDY_RUNNING = False
# check config.ini is in the root folder
config_file = config.read('config.ini')
if len(config_file) == 0:
    raise Exception("config.ini file not found")


def print_config(conf):
    table = PrettyTable(['Section', 'Property', 'Value'])
    dict = {}
    for section in conf:
        for prop in conf[section]:
            dict[prop] = conf[section][prop]
            table.add_row([section, prop, conf[section][prop]])
    print(table)
    return dict

@app.command("shuffle_database")
def shuffle_database(name: str, _type: str, new_name: str):
    ds_root_folder = config['DATASET'][f'processed_{_type}_location']
    dataset_reader = CustomLidcDatasetReader(location=ds_root_folder + f'/{name}/')
    dataset_reader.load_custom()
    dataset_reader.shuffle_data()
    dataset_reader.name = new_name
    os.mkdir(ds_root_folder + f'/{new_name}/')
    dataset_reader.location = ds_root_folder + f'/{new_name}/'
    dataset_reader.part = 0
    dataset_reader.part_size = 100000
    dataset_reader.save()


@app.command("deflate_ds")
def deflate_ds(name: str, _type: str, new_name: str):
    directory = config['DATASET'][f'processed_{_type}_location']
    dataset_reader = CustomLidcDatasetReader(location=directory + '/' + name + '/')
    dataset_reader.load_custom()
    path = config['DATASET'][f'processed_{_type}_location'] + '/' + new_name + '/'
    if os.path.exists(path):
        info(f"The dataset '{new_name}' already exist. Some files might be replaced")
    else:
        os.makedirs(path)
        info(f"The dataset '{new_name}' created")

    original_ds_config = configparser.ConfigParser()
    original_ds_config.read(directory + '/' + name + '/config.ini')

    dataset_info_config = configparser.ConfigParser()



    if _type == 'lidc_idri':
        dataset_reader.zip = False
        dataset_reader.name = new_name

        dataset_info_config['DEFAULT'] = {
            'name': new_name,
            'image_size': original_ds_config['DEFAULT']['image_size'],
            'consensus_level': original_ds_config['DEFAULT']['consensus_level'],
            'type': _type,
            'deflated': 'True',
            'count': str(dataset_reader.save(other_location=path))
        }

        with open(path + 'config.ini', 'w') as configfile:
            dataset_info_config.write(configfile)
    else:
        error(f'Type {_type} is currently not supported')


# This is a Python function that creates a dataset for machine learning tasks. The function takes in several input parameters including the name of the dataset, the type of the dataset, the size of the images, and other parameters related to the dataset creation process.
#
# The function first checks if a dataset with the specified name already exists, and if it does, it provides a warning message. If the dataset does not exist, it creates a new directory for it and saves the dataset's configuration in a file named config.ini.
#
# If the dataset type is lidc_idri, the function creates a LidcDatasetReader object with the specified parameters and loads the dataset. It then saves the dataset and if stop_after_current_part is not set to True, it iteratively loads the next part of the dataset until there are no more parts to load. It also updates the num_parts parameter in the config.ini file.
#
# If the dataset type is not lidc_idri, the function raises an error stating that the dataset type is not currently supported.
#
# In summary, this function creates a dataset and saves its configuration file, and if the dataset type is lidc_idri, it loads and saves the dataset parts iteratively.
@app.command("create_dataset")
def create_dataset(name: str, type: str, image_size: int = 512, consensus_level: float = 0.5, pad: int = 0,
                   current_part: int = 0, part_size: int = 100, stop_after_current_part: bool = False):
    warning("This process might take several minutes or even hours depending on your system")
    path = config['DATASET'][f'processed_{type}_location'] + '/' + name + '/'
    if os.path.exists(path):
        info(f"The dataset '{name}' already exist. Some files might be replaced")
    else:
        os.makedirs(path)
        info(f"The dataset '{name}' created")

    dataset_info_config = configparser.ConfigParser()
    ret = dataset_info_config.read(path + 'config.ini')
    if len(ret) == 0:
        dataset_info_config['DEFAULT'] = {
            'name': name,
            'image_size': image_size,
            'consensus_level': consensus_level,
            'pad': pad,
            'part_size': part_size,
            'type': type
        }
        with open(path + 'config.ini', 'w') as configfile:
            dataset_info_config.write(configfile)
    elif dataset_info_config['DEFAULT']['name'] != name or dataset_info_config['DEFAULT']['type'] != type or int(
            dataset_info_config['DEFAULT']['image_size']) != image_size or \
            float(dataset_info_config['DEFAULT']['consensus_level']) != consensus_level or int(
        dataset_info_config['DEFAULT']['pad']) != pad or \
            int(dataset_info_config['DEFAULT']['part_size']) != part_size:
        error(
            f"Current config.ini for the dataset '{name}' file is not consistent with the parameters specified. Please change the parameters or the config.ini file")
        exit(-1)

    if type == 'lidc_idri':
        lidc_dataset_reader = LidcDatasetReader(
            location=path,
            image_size=image_size,
            consensus_level=consensus_level,
            pad=pad,
            part=current_part,
            part_size=part_size
        )
        lidc_dataset_reader.load()
        lidc_dataset_reader.save()
        n = 1

        if not stop_after_current_part:
            lidc_dataset_reader = lidc_dataset_reader.next()

            while lidc_dataset_reader.load():
                lidc_dataset_reader.save()
                n = n + 1
                lidc_dataset_reader = lidc_dataset_reader.next()

            dataset_info_config['DEFAULT']['num_parts'] = n
            with open(path + 'config.ini', 'w') as configfile:
                dataset_info_config.write(configfile)
    else:
        error(f'Type {type} is currently not supported')


@app.command("datasets")
def datasets(_type: str):
    directory = config['DATASET'][f'processed_{_type}_location']
    _list = []
    for folder in os.listdir(directory):
        print('')
        ds_config = configparser.ConfigParser()
        ds_config.read(directory + '/' + folder + '/config.ini')
        _list.append(print_config(ds_config))
    return {'directory:': directory, 'datasets': _list}

@app.command("navigate_dataset")
def navigate_dataset(dataset_name: str, _type: str, data_transformer_name=None, image_size: int = -1, isolate_nodule_image: bool = False, channels: int = -1):
    directory = config['DATASET'][f'processed_{_type}_location']
    dataset_reader = CustomLidcDatasetReader(location=directory + '/' + dataset_name + '/')
    ds_config = configparser.ConfigParser()
    ds_config.read(directory + '/' + dataset_name + '/config.ini')
    info('Dataset config info')
    print_config(ds_config)

    if isolate_nodule_image is True:
        info('Images will be nodule only')
        dataset_reader.filter_out(lambda data: data[LIDC_ANN_Y0] == 0 and data[LIDC_ANN_Y1] == 0 and
                                               data[LIDC_ANN_X0] == 0 and data[LIDC_ANN_X1] == 0)

    if data_transformer_name is not None:
        _, data_transformer, _, metrics = get_data_transformer(data_transformer_name)
        info(f'Data transformer {data_transformer} will be applied')
        dataset_reader.dataset_data_transformers.append(DatasetTransformer(function=data_transformer))

    if image_size > 0 and channels > 0:
        info(f'Images will be changed to size {image_size}x{image_size}x{channels} ')
        dataset_reader.dataset_image_transformers.append(
            DatasetTransformer(function=img_transformer(image_size, channels, isolate_nodule_image)))

    dataset_reader.load_custom()
    display_dataset_images(dataset_reader)

@app.command("dataset_details")
# This is a Python function that takes in several parameters related to a dataset and applies various data and image transformations to it. The input parameters are:
#
# dataset_name: the name of the dataset to be processed
# _type: the type of the dataset, e.g. 'train' or 'test'
# data_transformer_name: the name of a data transformer to be applied to the dataset
# image_size: the size to which images in the dataset will be resized
# isolate_nodule_image: a boolean flag indicating whether to isolate only the nodule in the images
# channels: the number of channels in the images
# train_size: the proportion of the dataset to be used for training (default is 0.8)
# dump_annotations_to_file: a boolean flag indicating whether to dump the annotations to a file
# The function then reads the configuration information for the dataset and uses it to create a custom dataset reader. It loads and shuffles the data and then applies the requested data and image transformations. Finally, it loads the transformed data and outputs statistics on the dataset before and after the transformations. If dump_annotations_to_file is True, it also saves the annotations to a file.

def dataset_details(dataset_name: str, _type: str, data_transformer_name=None, image_size: int = -1, isolate_nodule_image: bool = False, channels: int = -1, train_size: float = 0.8, dump_annotations_to_file: bool=False):
    directory = config['DATASET'][f'processed_{_type}_location']
    dataset_reader = CustomLidcDatasetReader(location=directory + '/' + dataset_name + '/')
    ds_config = configparser.ConfigParser()
    ds_config.read(directory + '/' + dataset_name + '/config.ini')
    info('Dataset config info')
    print_config(ds_config)

    info(f'Loading the statistics (this might take several minutes)...')
    info('Statistics before data and image transformation')
    dataset_reader.load_custom()
    dataset_reader.shuffle_data()
    dataset_reader.statistics()

    if isolate_nodule_image is True:
        info('Images will be nodule only')
        dataset_reader.filter_out(lambda data: data[LIDC_ANN_Y0] == 0 and data[LIDC_ANN_Y1] == 0 and
                                               data[LIDC_ANN_X0] == 0 and data[LIDC_ANN_X1] == 0)

    if data_transformer_name is not None:
        _, data_transformer, _, metrics = get_data_transformer(data_transformer_name)
        info(f'Data transformer {data_transformer} will be applied')
        dataset_reader.dataset_data_transformers.append(DatasetTransformer(function=data_transformer))

    if image_size > 0 and channels > 0:
        info(f'Images will be changed to size {image_size}x{image_size}x{channels} ')
        dataset_reader.dataset_image_transformers.append(
            DatasetTransformer(function=img_transformer(image_size, channels, isolate_nodule_image)))

    info('Statistics after data and image transformation')
    dataset_reader.load_custom()
    dataset_reader.statistics(first=train_size, dump_annotations_to_file=dump_annotations_to_file)

@app.command("get_image")
def get_image_data(dataset_name: str, _type: str, index: int = -1):
    directory = config['DATASET'][f'processed_{_type}_location']
    dataset_reader = CustomLidcDatasetReader(location=directory + '/' + dataset_name + '/')
    dataset_reader.load_custom()
    if index < 0:
        print(dataset_reader.export_as_table())
        print(f'Count: {len(dataset_reader.images)}')

    else:
        image = dataset_reader.images[index]
        annotations = dataset_reader.annotations[index]

        display_original_image_bbox(image, annotations)


def get_heatmap(vectorized_image, model, last_conv_layer, pred_index=None):
    '''
    Function to visualize grad-cam heatmaps
    '''
    gradient_model = tensorflow.keras.models.Model(
        [model.inputs], [model.get_layer(index=last_conv_layer).output, model.output]
    )

    # Gradient Computations
    with tensorflow.GradientTape() as tape:
        last_conv_layer_output, preds = gradient_model(vectorized_image)
        if pred_index is None:
            pred_index = tensorflow.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tensorflow.newaxis]
    heatmap = tensorflow.squeeze(heatmap)
    heatmap = tensorflow.maximum(heatmap, 0) / tensorflow.math.reduce_max(heatmap) # Normalize the heatmap
    return heatmap.numpy()




@app.command("predict_detection")
def predict_detection(dataset_name: str, _type: str, weight_file_name, index: int = -1):
    directory = config['DATASET'][f'processed_{_type}_location']
    dataset_reader = CustomLidcDatasetReader(location=directory + '/' + dataset_name + '/')
    dataset_reader.load_custom()
    if index < 0:
        print(dataset_reader.export_as_table())
        print(f'Count: {len(dataset_reader.images)}')

    else:
        image = dataset_reader.images[index]
        annotations = dataset_reader.annotations[index]
        json_data = None
        with open(weight_file_name + '.json', 'r') as json_fp:
            json_data = json.load(json_fp)

        _, data_transformer, _, metrics = get_data_transformer(json_data['data_transformer_name'])

        model_ = get_model(model_type=json_data['model_type'], image_size=json_data['image_size'], static_params=True,
                                       metrics=metrics, code_name=json_data['code_name'], data_transformer_name=json_data['data_transformer_name'],
                                       params=json_data['learning_params'], return_model_only=True, batch_size=json_data['batch_size'],
                                       epochs=json_data['epochs'], num_classes=json_data['num_classes'], loss=json_data['loss'],
                                       data=None
        )

        model = model_(None)
        model.load_weights(weight_file_name + '.h5')
        im = img_transformer(json_data['image_size'], json_data['image_channels'], False)(image, annotations, None)
        vectorized_image = np.expand_dims(im, axis=0)
        output = model.predict(vectorized_image)
        output_scaled = [int(x * 512) for x in output[0]]
        annotation_transformed = data_transformer(annotations, None, None)
        annotation_scaled = [x * 512 for x in annotation_transformed]
        #miou = tf.keras.metrics.MeanIoU(num_classes=4)
        #miou.update_state(annotation_transformed, output_scaled)
        #miou = miou.result().numpy()


        display_original_image_bbox(image, annotations, output_scaled,
                                    'Predicted Value scaled = ' + str(output_scaled) + ', \n'
                                    'Actual Value transformed = ' + str(annotation_transformed) + ', \n'
                                    'Actual Value scaled = ' + str(annotation_scaled) + ', \n'
                                    'Predicted Value raw = ' + str(output[0]) + ', \n' +
                                    #'MeanIoU = ' + str(miou) + ', \n' +
                                    'IoU BBox = ' + str(bounding_box_intersection_over_union(annotation_transformed, output[0])))


        del dataset_reader
        tensorflow.keras.backend.clear_session()

@app.command("drill_down")
def drill_down(batch_size: int, epochs: int, study_file_path: str, train_size: float, data_set: str, load_weights: bool = False, isolate_nodule_image: bool = True):
    json_object = None
    weights_file = None
    with open(study_file_path + '.json', 'r') as json_fp:
        json_object = json.load(json_fp)

    if load_weights:
        weights_file = study_file_path + '.h5'

    num_classes, data_transformer, loss, metrics = get_data_transformer(json_object['data_transformer_name'])

    code_name = str(time.time_ns())

    data = get_ds(config=config, data_transformer=data_transformer, image_size=json_object['image_size'],
                  train_size=train_size, ds=data_set, isolate_nodule_image=isolate_nodule_image,
                  channels=get_channels(json_object['model_type']))


    table = PrettyTable(['Parameter', 'Value'])
    table.add_row(['Code Name', code_name])
    table.add_row(['Model Type', json_object['model_type']])
    table.add_row(['Batch Size', str(batch_size)])
    table.add_row(['Epochs', str(epochs)])
    table.add_row(['Num Classes', str(num_classes)])
    table.add_row(['Train Size', str(train_size)])
    table.add_row(['Image Size', str(json_object['image_size'])])
    table.add_row(['Problem reduction function', json_object['data_transformer_name']])
    table.add_row(['Dataset Name', data_set])
    table.add_row(['Isolate Nodule image', str(isolate_nodule_image)])
    for i in json_object['learning_params']:
        table.add_row([i,  json_object['learning_params'][i]])

    print(table)

    objective = get_model(model_type=json_object['model_type'], image_size=json_object['image_size'],
                                               batch_size=batch_size,num_classes=num_classes, loss=loss, epochs=epochs, data=data,
                                               metrics=metrics, save_weights=True, code_name=code_name,
                                               static_params=True, params=json_object['learning_params'],
                                               data_transformer_name=json_object['data_transformer_name'], weights_file=weights_file)

    print(f'Accuracy = {objective(None)}')



def save_model(obj):
    print(obj)
    filename = os.path.dirname(os.path.realpath(__file__)) + '/main/model_registry/classification/%s.py' % obj['name']
    error = None
    try:
        f = open(filename, "w")
        f.write(obj['code'].strip())
        f.close()
    except Exception as e:
        traceback.print_exc()
        error = str(e)
    return {'filename': filename, 'error': error}

@app.command("databases")
def databases():
    directory = os.path.dirname(os.path.realpath(__file__))
    print('Looking for databases inside %s' % directory)
    names = []
    for entry in os.listdir(directory):
        if entry.endswith('.sqlite3'):
            names.append(os.path.splitext(entry)[0])
    print(names)
    return names

@app.command("trials")
def trials():
    root_dir = "weights/"
    pattern = re.compile(r'\\')

    unique_paths = set()

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = pattern.sub('$', os.path.relpath(file_path, root_dir))
            file_name, file_extension = os.path.splitext(file)
            unique_paths.add(relative_path.replace(file_extension, ''))

    # Sort the unique relative paths alphabetically
    sorted_paths = sorted(unique_paths)

    # Print the sorted unique relative paths
    for path in sorted_paths:
        print(path)

    return sorted_paths


@app.command("get_trial")
def get_trial(trial: str):
    root = "weights/" + trial.replace('$', '\\')

    with open(root + '.json', 'r') as json_fp:
        json_data = json.load(json_fp)

    return json_data

# This function, called models(), lists and inspects all the models in a specific directory, with the goal of extracting relevant information from each one of them.
#
# First, the function sets the directory path for the model registry. Then it initializes three empty lists: details_list, incomplete_models_list, and error_list, which will be populated with the model information.
#
# The function then iterates over each file in the directory and attempts to load it as a module. If it succeeds, it checks whether the module contains a ModelDefinition object. If it does, it calls the details() method of the object to get additional information about the model and appends the details to details_list. If it does not, it appends the module name to the incomplete_models_list.
#
# If the module fails to load, the function appends an error message to the error_list. Two types of errors are handled: syntax errors and other runtime exceptions. Syntax errors contain information about the line number and the error message. Other exceptions only contain a general error message.
#
# Finally, the function returns a dictionary containing the populated details_list, incomplete_models_list, and error_list.
@app.command("list_models")
def models():
    directory = os.path.dirname(os.path.realpath(__file__)) + '/main/model_registry/classification'
    details_list = []
    incomplete_models_list = []
    error_list = []
    for folder in os.listdir(directory):
        name = os.path.splitext(folder)[0]
        if name == '__init__' or name == '__pycache__':
            continue
        try:
            m = load_module(name)
            m = reload(m)
            with open(m.__file__, mode='r', encoding="utf8") as f:
                source = f.read()
            ast.parse(source)
            if hasattr(m, 'ModelDefinition'):
                modelDef: CustomModelDefinition = m.ModelDefinition()
                details, table = modelDef.details()
                details_list.append(details)
                print(table)
            else:
                incomplete_models_list.append(name)
                warning(f'Model {name} is not valid. Missing details function')
        except SyntaxError as e:
            traceback.print_exc()
            error_list.append({'model': name, 'error': f'Syntax error line {e.lineno} in the file {name}.py: {e.msg}'})
        except Exception as e:
            traceback.print_exc()
            error_list.append({'model': name, 'error': f'Error: {e}'})

    return {'details_list': details_list, 'incomplete_models_list': incomplete_models_list, 'error_list': error_list}


@app.command("train")
def train(batch_size: int, epochs: int, train_size: float, image_size: int, model_type: str,
          data_transformer_name: str, data_set: str, params: str, save_weights=True, isolate_nodule_image: bool=True):

    parsed_url = urlparse('?' + params)
    params_arr_ = parse_qs(parsed_url.query)
    params_arr = []
    for i in params_arr_:
        params_arr[i] = params_arr_[i][0]

    num_classes, data_transformer, loss, metrics = get_data_transformer(data_transformer_name)

    code_name = str(time.time_ns())

    data = get_ds(config=config, data_transformer=data_transformer, image_size=image_size, train_size=train_size,
                  ds=data_set, isolate_nodule_image=isolate_nodule_image, channels=get_channels(model_type))

    table = PrettyTable(['Parameter', 'Value'])
    table.add_row(['Code Name', code_name])
    table.add_row(['Model Type', model_type])
    table.add_row(['Batch Size', str(batch_size)])
    table.add_row(['Epochs', str(epochs)])
    table.add_row(['Num Classes', str(num_classes)])
    table.add_row(['Train Size', str(train_size)])
    table.add_row(['Image Size', str(image_size)])
    table.add_row(['Problem reduction function', data_transformer_name])
    table.add_row(['Dataset Name', data_set])
    table.add_row(['Isolate Nodule image', str(isolate_nodule_image)])
    for i in params_arr:
        table.add_row([i, params_arr[i]])

    print(table)

    objective = get_model(model_type=model_type, image_size=image_size, batch_size=batch_size,
                                               num_classes=num_classes, loss=loss, epochs=epochs, data=data,
                                               metrics=metrics, save_weights=save_weights, code_name=code_name,
                                               static_params=True, params=params_arr, data_transformer_name=data_transformer_name)
    print(f'Accuracy = {objective(None)}')

# This function is used to conduct a hyperparameter optimization study for a machine learning model. It takes in several arguments such as batch size, epochs, train size, image size, model type, number of trials, and others.
#
# The function first creates an Optuna study with the specified study name and storage location. Then, it sets the user attributes for the study, such as batch size, epochs, and model type.
#
# It then creates a table to display the hyperparameters that will be optimized during the study. It loads the dataset and gets the necessary information such as the number of classes, data transformer, loss, and metrics.
#
# Next, it creates an objective function that takes in the hyperparameters to optimize and trains the model. The objective function returns the validation accuracy of the model.
#
# Finally, the function calls the Optuna study's optimize method to run the study and find the best hyperparameters. It prints out the number of finished trials and sets the STUDY_RUNNING variable to false.
@app.command("study")
def study(batch_size: int, epochs: int, train_size: float, image_size: int, model_type: str, n_trials: int,
          data_transformer_name: str, data_set: str, db_name: str, centroid_only: bool = False, isolate_nodule_image: bool = True, detection: bool = False):

    STUDY_RUNNING = True
    study_counter = config['STUDY']['study_counter']
    config['STUDY']['study_counter'] = str(int(config['STUDY']['study_counter']) + 1)
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    if detection:
        directions = ["minimize", "maximize"]
    else:
        directions = ["maximize", "minimize", "maximize"]

    code_name = str(get_experiment_codename(int(study_counter)+1))
    optuna_study = optuna.create_study(storage=f'sqlite:///{db_name}.sqlite3', directions=directions,
                                       study_name=f'{code_name}',
                                       sampler=TPESampler())

    optuna_study.set_user_attr('batch_size', batch_size)
    optuna_study.set_user_attr('epochs', epochs)
    optuna_study.set_user_attr('train_size', train_size)
    optuna_study.set_user_attr('model_type', model_type)
    optuna_study.set_user_attr('n_trials', n_trials)
    optuna_study.set_user_attr('data_transformer_name', data_transformer_name)
    optuna_study.set_user_attr('data_set', data_set)
    optuna_study.set_user_attr('isolate_nodule_image', isolate_nodule_image)
    optuna_study.set_user_attr('pylung_version', config['VERSION']['pylung_version'])
    optuna_study.set_user_attr('detection', detection)
    optuna_study.set_user_attr('centroid_only', centroid_only)

    table = PrettyTable(['Parameter', 'Value'])

    table.add_row(['Model Type', model_type])
    table.add_row(['Batch Size', str(batch_size)])
    table.add_row(['Epochs', str(epochs)])
    table.add_row(['Train Size', str(train_size)])
    table.add_row(['Image Size', str(image_size)])
    table.add_row(['N Trials', str(n_trials)])
    table.add_row(['Problem reduction function', data_transformer_name])
    table.add_row(['Dataset Name', data_set])
    table.add_row(['Isolate Nodule image', str(isolate_nodule_image)])
    table.add_row(['Centroid Only', str(centroid_only)])
    table.add_row(['Detection?', str(detection)])
    table.add_row(['Code name', code_name])

    print(table)

    num_classes, data_transformer, loss, metrics = get_data_transformer(data_transformer_name, detection)

    info(f'Loading dataset...')
    data = get_ds(config=config, data_transformer=data_transformer, image_size=image_size, train_size=train_size, ds=data_set, centroid_only=centroid_only, isolate_nodule_image=isolate_nodule_image, channels=get_channels(model_type))
    info(f'Dataset loaded with {len(data[0])} images for training and {len(data[1])} images for validation.')



    objective = get_model(model_type=model_type, image_size=image_size, batch_size=batch_size,
                                               num_classes=num_classes, loss=loss, epochs=epochs, data=data,
                                               metrics=metrics, save_weights=True, code_name=code_name, isolate_nodule_image=isolate_nodule_image,
                                               data_transformer_name=data_transformer_name, detection=detection)

    optuna_study.optimize(objective, n_trials=n_trials) #, timeout=600)

    print("Number of finished trials: {}".format(len(optuna_study.trials)))

    STUDY_RUNNING = False


def start_optuna(name, port):
    print(f'Optuna dashboard will start with port {port} for the database {name}')
    optuna_dashboard.run_server(port=port, storage=f'sqlite:///{name}')

@app.command("dashboard")
def dashboard(port=8088):
    run(host='localhost', port=port, debug=True, reloader=True)
@route('/')
def dashboard_index():
    return static_file('index.html', root='./html/dist')

@route('/static/<model_name:path>')
def static(model_name):
    return static_file(model_name, root='./static')

@route('/rest/datasets/<ds_type:path>')
def rest_datasets(ds_type):
    ret = datasets(ds_type)
    response.content_type = 'application/json'
    return dumps(ret)

@route('/rest/navigate/<ds_type:path>/<ds_name:path>/image-<index:path>.png')
def get_image_ds(ds_type, ds_name, index):
    query = request.query
    bbox = False
    crop = False
    data = None

    try:
        bbox = query['bbox'] == 'True' or query['bbox'] == 'true'
    except:
        bbox = False

    try:
        crop = query['crop'] == 'True' or query['crop'] == 'true'
    except:
        crop = False

    try:
        data = query['data'].split(',')
        data = [float(val) for val in data]
    except:
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

    if not bbox and data is None and crop and (annotations[0] != 0 or annotations[1] != 0 or annotations[2] !=0 or annotations[3] != 0):
        ax.imshow(ret[int(annotations[0]):int(annotations[1]), int(annotations[2]):int(annotations[3])], cmap=plt.cm.gray)
    elif data is not None and crop:
        print(f'ret[{int(data[0] * 512)}:{int(data[1] * 512)}, {int(data[2] * 512)}:{int(data[3] * 512)}]')
        ax.imshow(ret[int(data[0] * 512):int(data[1] * 512), int(data[2] * 512):int(data[3] * 512)], cmap=plt.cm.gray)
    else:
        ax.imshow(ret, cmap=plt.cm.gray)
    if not crop and bbox and (annotations[0] != 0 or annotations[1] != 0 or annotations[2] !=0 or annotations[3] != 0):
        rect = patches.Rectangle(
            (int(annotations[2]), int(annotations[0])),
            int(annotations[3] - annotations[2]),
            int(annotations[1] - annotations[0]),
            facecolor="none",
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rect)

    if not crop and data is not None:
        rect = patches.Rectangle(
            (int(data[2] * 512), int(data[0] * 512)),
            int(data[3] * 512 - data[2] * 512),
            int(data[1] * 512 - data[0] * 512),
            facecolor="none",
            edgecolor="green",
            linewidth=2,
        )
        ax.add_patch(rect)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(buf, dpi=100, bbox_inches='tight', pad_inches=0)

    response.content_type = 'image/x-png'
    return buf.getvalue()

@route('/rest/navigate/<ds_type:path>/<ds_name:path>/annotation-<index:path>.txt')
def get_annotation_ds(ds_type, ds_name, index):
    ret = ''
    directory = config['DATASET'][f'processed_{ds_type}_location']
    with open(directory + f'/{ds_name}/annotation-{index}.txt', 'rb') as file:
        ret = pickle.load(file)
    response.content_type = 'text/plain'
    return str(ret)


@route('/rest/predict/<trial:path>/<ds_type:path>/<ds_name:path>/<index:path>')
def predict_nodule(trial, ds_type, ds_name, index):
    image = None
    annotation = None

    directory = config['DATASET'][f'processed_{ds_type}_location']
    with open(directory + f'/{ds_name}/image-{index}.raw', 'rb') as file:
        image = pickle.load(file)
    with open(directory + f'/{ds_name}/annotation-{index}.txt', 'rb') as file:
        annotation = pickle.load(file)

    with open('weights/' + trial.replace('$', '/') + '.json', 'r') as json_fp:
        json_data = json.load(json_fp)

    _, data_transformer, _, metrics = get_data_transformer(json_data['data_transformer_name'])

    model_ = get_model(model_type=json_data['model_type'], image_size=json_data['image_size'], static_params=True,
                       metrics=metrics, code_name=json_data['code_name'],
                       data_transformer_name=json_data['data_transformer_name'],
                       params=json_data['learning_params'], return_model_only=True, batch_size=json_data['batch_size'],
                       epochs=json_data['epochs'], num_classes=json_data['num_classes'], loss=json_data['loss'],
                       data=None,detection=json_data['detection'], isolate_nodule_image=json_data['isolate_nodule_image']
                       )

    model = model_(None)
    model.load_weights('weights/' + trial.replace('$', '/') + '.h5')
    im = img_transformer(json_data['image_size'], json_data['image_channels'], json_data['isolate_nodule_image'])(image, annotation, None, None)
    vectorized_image = np.expand_dims(im, axis=0)
    start = time.time()
    output = model.predict(vectorized_image)
    end = time.time()

    ret = {
        'predicted': output[0].tolist(),
        'predicted_int': output[0].round().astype(int).tolist(),
        'annotation': str(annotation),
        'transformed_annotation': data_transformer(annotation, None, None, None),
        'timespent': end-start
    }

    response.content_type = 'application/json'
    return dumps(ret)

@route('/rest/models', method='GET')
def rest_models():
    ret = models()
    response.content_type = 'application/json'
    return dumps(ret)

@route('/rest/trials', method='GET')
def rest_trials():
    ret = trials()
    response.content_type = 'application/json'
    return dumps(ret)

@route('/rest/trials/<trial:path>', method='GET')
def rest_trials_details(trial):
    ret = get_trial(trial)
    response.content_type = 'application/json'
    return dumps(ret)

@route('/rest/datatransformers', method='GET')
def rest_data_transformers():
    ret = get_list_database_transformers()
    response.content_type = 'application/json'
    return dumps({'list': ret})

@route('/rest/models', method='POST')
def rest_save_model():
    ret = save_model(request.json)
    response.content_type = 'application/json'
    return dumps(ret)

@route('/rest/studies', method='POST')
def rest_start_study():

    batch_size = int(request.json['batch_size'])
    epochs = int(request.json['epochs'])
    train_size = float(request.json['train_size'])
    image_size = int(request.json['image_size'])
    model_type = request.json['model_type']
    n_trials = int(request.json['n_trials'])
    data_transformer_name = request.json['data_transformer_name']
    data_set = request.json['data_set']
    db_name = request.json['db_name']
    isolate_nodule_image = bool(request.json['isolate_nodule_image'])
    if not isolate_nodule_image:
        isolate_nodule_image = '--no-isolate-nodule-image'
    else:
        isolate_nodule_image = ''
    d = run_study_cmd(batch_size, data_set, data_transformer_name, db_name, epochs, image_size, isolate_nodule_image,
                  model_type, n_trials, train_size)
    return {}


def run_study_cmd(batch_size, data_set, data_transformer_name, db_name, epochs, image_size, isolate_nodule_image,
                  model_type, n_trials, train_size):
    command = 'python pylung.py study'
    args = f' {batch_size} {epochs} {train_size} {image_size} {model_type} {n_trials} {data_transformer_name} {data_set} {db_name} {isolate_nodule_image}'
    print(command + args)
    d = multiprocessing.Process(target=run_cmd_subprocess, args=(command, args))
    current_studies.append({'study': args, 'pid': d.pid, 'process': d})
    if len(current_studies) == 1:
        d.start()
    return d.pid


def run_cmd_subprocess(command, args):
    print(f'New process started for command {command} with args {args}')
    s = subprocess.run(command + args, shell=True)
    print(f'Process finished')


@route('/rest/databases')
def rest_databases():
    ret = databases()
    response.content_type = 'application/json'
    return dumps(ret)

@route('/rest/optuna/start/<name:path>')
def rest_open_optuna(name):
    global dashboards
    port = -1
    if dashboards.count(name) == 0:
        port = 9980 + len(dashboards)
        od = multiprocessing.Process(target=start_optuna, args=(name,port))
        dashboards.append(name)
        od.start()

    if port == -1:
        port = 9980 + dashboards.index(name)
    print(f'Returning port {port} to the client')
    return str(port)

@route('/<filename:path>')
def dashboard_resources(filename):
    return static_file(filename, root='./html/dist')

if __name__ == "__main__":
    app()
