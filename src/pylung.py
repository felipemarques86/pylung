import configparser
import os
import pickle
import time

import optuna
import typer
from colorama import init as colorama_init
from optuna.samplers import NSGAIISampler
from prettytable import PrettyTable

from main.dataset_utilities.dataset_reader_classes import DatasetTransformer
from main.dataset_utilities.lidc_dataset_reader_classes import LidcDatasetReader, CustomLidcDatasetReader
from main.experiment.experiment import ClassificationExperiment
from main.experiment.experiment_utilities import build_classification_objective, get_ds
from main.models.resnet_model import ResNet50Model
from main.models.vit_model import VitModel
from main.utilities.utilities_lib import warning, error, info, binary_malignancy, five_one_hot_malignancy, \
    two_one_hot_malignancy, bbox, get_data_transformer

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


@app.command("create_vit_model")
def create_vit_model(name: str, version: int, patch_size: int, projection_dim: int, num_heads: int, dropout1: float,
                     dropout2: float, dropout3: float, image_size: int, activation: str, num_clases: int,
                     transformer_layers: int,
                     image_channels: int):
    vit_model = VitModel(
        name=name,
        version=version,
        patch_size=patch_size,
        projection_dim=projection_dim,
        num_heads=num_heads,
        mlp_head_units=[2048, 1024, 512, 64, 32],
        dropout1=dropout1,
        dropout2=dropout2,
        dropout3=dropout3,
        image_size=image_size,
        activation=activation,
        num_classes=num_clases,
        transformer_layers=transformer_layers,
        image_channels=image_channels
    )

    vit_model.save_model()
    print(vit_model.export_as_table())
    info(f'{name} model saved successfully')

@app.command("create_restnet50_model")
def create_restnet50_model(name: str, version: int, activation: str, num_clases: int, dropout: float,
                     pooling: str, weights: str, image_channels: int):
    model = ResNet50Model(
            name=name,
            version=version,
            activation=activation,
            num_classes=num_clases,
            image_size=224,
            dropout=dropout,
            pooling=pooling,
            weights=weights,
            image_channels=image_channels
        )

    model.save_model()
    print(model.export_as_table())
    info(f'{name} model saved successfully')


@app.command("datasets")
def datasets(_type: str):
    directory = config['DATASET'][f'processed_{_type}_location']
    for folder in os.listdir(directory):
        print('')
        ds_config = configparser.ConfigParser()
        ds_config.read(directory + '/' + folder + '/config.ini')
        print_config(ds_config)


@app.command("models")
def models():
    directory = config['DEFAULT']['models_location']

    def load_model(file):
        with open(directory + '/' + file, 'rb') as filePointer:
            return pickle.load(filePointer)

    models = map(load_model, [fi for fi in os.listdir(directory) if fi.endswith(".model")])

    for model in models:
        print(model.export_as_table())
        print('')


@app.command("train_classification")
def train_classification(dataset_name: str, dataset_type: str, model_name: str, learning_rate: float,
                         weight_decay: float,
                         batch_size: int, num_epochs: int, optimizer: str, validation_split: float, train_size: float,
                         loss: str, labels: str = '', data_transformer: str = 'binary_malignancy'):
    ds_root_folder = config['DATASET'][f'processed_{dataset_type}_location']
    models_root_folder = config['DEFAULT']['models_location']

    def load_model(file):
        with open(models_root_folder + '/' + file, 'rb') as filePointer:
            return pickle.load(filePointer)

    models_files = [fi for fi in os.listdir(models_root_folder) if fi.endswith(".model")]
    model = None
    for model_file in models_files:
        model_ = load_model(model_file)
        if model_.name == model_name:
            model = model_
            break

    if model is None:
        error(f'Model not found with the specified name {model_name}')
        exit(-1)

    info(f'Model being used')
    print(model.export_as_table())

    ds_folder = None
    for folder in os.listdir(ds_root_folder):
        if folder == dataset_name:
            ds_folder = folder
            break
    if ds_folder is None:
        error(f'No dataset was found with name {dataset_name}')
        exit(-1)

    dataset_reader = None
    if dataset_type == 'lidc_idri':
        dataset_reader = CustomLidcDatasetReader(location=ds_root_folder + '/' + ds_folder + '/')
        data_transformer_func = None
        if data_transformer == 'binary_malignancy':
            if model.num_classes != 1:
                error(f'Number of classes must be 1 for this type of data transformer')
                exit(-1)
            data_transformer_func = DatasetTransformer(function=binary_malignancy)
        elif data_transformer == 'five_one_hot_malignancy':
            if model.num_classes != 2:
                error(f'Number of classes must be 2 for this type of data transformer')
                exit(-1)
            data_transformer_func = DatasetTransformer(function=five_one_hot_malignancy)
        elif data_transformer == 'two_one_hot_malignancy':
            if model.num_classes != 2:
                error(f'Number of classes must be 2 for this type of data transformer')
                exit(-1)
            data_transformer_func = DatasetTransformer(function=two_one_hot_malignancy)
        elif data_transformer == 'bbox':
            if model.num_classes != 4:
                error(f'Number of classes must be 4 for this type of data transformer')
                exit(-1)
            data_transformer_func = DatasetTransformer(function=bbox)

        dataset_reader.dataset_data_transformers.append(data_transformer_func)
        info('Data set being used')
        print(dataset_reader.export_as_table())
        dataset_reader.load_custom()

    else:
        warning(f'{dataset_type} currently not supported')
        exit(0)

    experiment = ClassificationExperiment(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        num_epochs=num_epochs,
        train_size=train_size,
        learning_rate=learning_rate,
        validation_split=validation_split,
        weight_decay=weight_decay,
        x=dataset_reader.images,
        y=dataset_reader.annotations,
        loss=loss
    )
    info(f'Experiment data')
    print(experiment.export_as_table())

    experiment.train()

@app.command("classify")
def classify(dataset_name: str, dataset_type: str, model_name: str, weights_file_name: str, n_images: int = 10,
             train_size: float = 0.8):
    ds_root_folder = config['DATASET'][f'processed_{dataset_type}_location']
    models_root_folder = config['DEFAULT']['models_location']

    model = get_model(model_name, models_root_folder)

    ds_folder = None
    for folder in os.listdir(ds_root_folder):
        if folder == dataset_name:
            ds_folder = folder
            break
    if ds_folder is None:
        error(f'No dataset was found with name {dataset_name}')
        exit(-1)

    dataset_reader = None
    if dataset_type == 'lidc_idri':
        dataset_reader = CustomLidcDatasetReader(location=ds_root_folder + '/' + ds_folder + '/')
        if model.num_classes == 2:
            def reduce_classes(data):
                clazz = 0
                if data[4] > 3:
                    clazz = 1
                ret = [0, 0]
                ret[clazz] = 1
                return ret

            dataset_reader.dataset_data_transformers.append(DatasetTransformer(function=reduce_classes))
            dataset_reader.load_custom()

        else:
            warning(f'{model.num_classes} classes are not currently supported')
            exit(0)
    else:
        warning(f'{dataset_type} currently not supported')
        exit(0)

    experiment = ClassificationExperiment(
        model=model,
        train_size=train_size,
        x=dataset_reader.images,
        y=dataset_reader.annotations,
    )

    experiment.print_results_classification(
        weights_location=config['DEFAULT']['weights_location'] + '/' + weights_file_name, n=n_images)


def get_model(model_name, models_root_folder):
    def load_model(file):
        with open(models_root_folder + '/' + file, 'rb') as filePointer:
            return pickle.load(filePointer)

    models_files = [fi for fi in os.listdir(models_root_folder) if fi.endswith(".model")]
    model = None
    for model_file in models_files:
        model_ = load_model(model_file)
        if model_.name == model_name:
            model = model_
            break
    if model is None:
        error(f'Model not found with the specified name {model_name}')
        exit(-1)
    return model


@app.command("study")
def study(batch_size: int, epochs: int, train_size: float, image_size: int, model_type: str, n_trials: int,
          data_transformer_name: str, data_set,  croptumor: bool=True):
    optuna_study = optuna.create_study(storage="sqlite:///01_2023.sqlite3", direction="maximize",
                                       study_name=f'{model_type}-{data_set}-{batch_size}-{epochs}-{train_size}-{image_size}-{n_trials}-{data_transformer_name}-{croptumor}-{time.time()}',
                                       sampler=NSGAIISampler(
        population_size=50,
        mutation_prob=0.001,
        crossover_prob=0.01,
        swapping_prob=0.0001
    ))

    num_classes, data_transformer, loss = get_data_transformer(data_transformer_name)

    data = get_ds(config=config, data_transformer=data_transformer, image_size=image_size, train_size=train_size, ds=data_set, crop_tumor=croptumor, channels=1)

    objective = build_classification_objective(model_type=model_type, image_size=image_size, batch_size=batch_size,
                                               num_classes=num_classes, loss=loss, epochs=epochs, data=data)

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
