import configparser
import json
import os
import time
from urllib.parse import parse_qs
from urllib.parse import urlparse

import numpy as np
import optuna
import typer
from colorama import init as colorama_init
from optuna.samplers import TPESampler
from prettytable import PrettyTable

from main.dataset_utilities.dataset_reader_classes import DatasetTransformer
from main.dataset_utilities.lidc_dataset_reader_classes import LidcDatasetReader, CustomLidcDatasetReader
from main.dataset_utilities.pylung_ui import display_dataset_images
from main.experiment.experiment_utilities import build_classification_objective, get_ds
from main.utilities.utilities_lib import warning, error, info, get_data_transformer, display_original_image_bbox, \
    get_experiment_codename, get_channels, img_transformer, LIDC_ANN_Y0, LIDC_ANN_Y1, LIDC_ANN_X0, LIDC_ANN_X1

config = configparser.ConfigParser()
app = typer.Typer()
colorama_init()

# check config.ini is in the root folder
config_file = config.read('config.ini')
if len(config_file) == 0:
    raise Exception("config.ini file not found")


def print_config(conf):
    table = PrettyTable(['Section', 'Property', 'Value'])
    for section in conf:
        for prop in conf[section]:
            table.add_row([section, prop, conf[section][prop]])
    print(table)

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
    for folder in os.listdir(directory):
        print('')
        ds_config = configparser.ConfigParser()
        ds_config.read(directory + '/' + folder + '/config.ini')
        print_config(ds_config)


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

@app.command("predict")
def predict(dataset_name: str, _type: str, weight_file_name, index: int = -1):
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

        model_ = build_classification_objective(model_type=json_data['model_type'], image_size=json_data['image_size'], static_params=True,
                                       metrics=metrics, code_name=json_data['code_name'], data_transformer_name=json_data['data_transformer_name'],
                                       params=json_data['learning_params'], return_model_only=True, batch_size=json_data['batch_size'],
                                       epochs=json_data['epochs'], num_classes=json_data['num_classes'], loss=json_data['loss'],
                                       data=None
        )

        model = model_(None)
        model.load_weights(weight_file_name + '.h5')
        im = img_transformer(json_data['image_size'], json_data['image_channels'], True)(image, annotations)
        print(im.shape)
        output = model.predict(np.expand_dims(im, axis=0))

        display_original_image_bbox(image, annotations,
                                    'Predicted Value = ' + str(output[0]) + ', '
                                    'Actual Value = ' + str(data_transformer(annotations, None)))


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

    objective = build_classification_objective(model_type=json_object['model_type'], image_size=json_object['image_size'],
                                               batch_size=batch_size,num_classes=num_classes, loss=loss, epochs=epochs, data=data,
                                               metrics=metrics, save_weights=True, code_name=code_name,
                                               static_params=True, params=json_object['learning_params'],
                                               data_transformer_name=json_object['data_transformer_name'], weights_file=weights_file)

    print(f'Accuracy = {objective(None)}')


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

    objective = build_classification_objective(model_type=model_type, image_size=image_size, batch_size=batch_size,
                                               num_classes=num_classes, loss=loss, epochs=epochs, data=data,
                                               metrics=metrics, save_weights=save_weights, code_name=code_name,
                                               static_params=True, params=params_arr, data_transformer_name=data_transformer_name)
    print(f'Accuracy = {objective(None)}')


@app.command("study")
def study(batch_size: int, epochs: int, train_size: float, image_size: int, model_type: str, n_trials: int,
          data_transformer_name: str, data_set: str, db_name: str, isolate_nodule_image: bool=True):

    study_counter = config['STUDY']['study_counter']
    config['STUDY']['study_counter'] = str(int(config['STUDY']['study_counter']) + 1)
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    code_name = str(get_experiment_codename(int(study_counter)+1))
    optuna_study = optuna.create_study(storage=f'sqlite:///{db_name}.sqlite3', directions=["maximize", "minimize", "maximize"],
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

    print(table)

    num_classes, data_transformer, loss, metrics = get_data_transformer(data_transformer_name)

    info(f'Loading dataset...')
    data = get_ds(config=config, data_transformer=data_transformer, image_size=image_size, train_size=train_size, ds=data_set, isolate_nodule_image=isolate_nodule_image, channels=get_channels(model_type))
    info(f'Dataset loaded with {len(data[0])} images for training and {len(data[1])} images for validation.')



    objective = build_classification_objective(model_type=model_type, image_size=image_size, batch_size=batch_size,
                                               num_classes=num_classes, loss=loss, epochs=epochs, data=data,
                                               metrics=metrics, save_weights=True, code_name=code_name,
                                               data_transformer_name=data_transformer_name)

    optuna_study.optimize(objective, n_trials=n_trials) #, timeout=600)

    print("Number of finished trials: {}".format(len(optuna_study.trials)))

    print("Best trial:")
    trial = optuna_study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


@app.command("detection_study")
def detection_study(batch_size: int, epochs: int, train_size: float, image_size: int, model_type: str, n_trials: int,
          data_transformer_name: str, data_set: str, db_name: str):

    isolate_nodule_image = False
    study_counter = config['STUDY']['study_counter']
    config['STUDY']['study_counter'] = str(int(config['STUDY']['study_counter']) + 1)
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    code_name = str(get_experiment_codename(int(study_counter)+1))
    optuna_study = optuna.create_study(storage=f'sqlite:///{db_name}.sqlite3', direction="maximize",
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

    print(table)

    num_classes, data_transformer, loss, metrics = get_data_transformer(data_transformer_name, True)

    info(f'Loading dataset...')
    data = get_ds(config=config, data_transformer=data_transformer, image_size=image_size, train_size=train_size, ds=data_set, isolate_nodule_image=isolate_nodule_image, channels=get_channels(model_type))
    info(f'Dataset loaded with {len(data[0])} images for training and {len(data[1])} images for validation.')



    objective = build_classification_objective(model_type=model_type, image_size=image_size, batch_size=batch_size,
                                               num_classes=num_classes, loss=loss, epochs=epochs, data=data,
                                               metrics=metrics, save_weights=True, code_name=code_name,
                                               data_transformer_name=data_transformer_name, detection=True)

    optuna_study.optimize(objective, n_trials=n_trials) #, timeout=600)

    print("Number of finished trials: {}".format(len(optuna_study.trials)))

    print("Best trial:")
    trial = optuna_study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    app()


if __name__ == "__main__":
    app()


