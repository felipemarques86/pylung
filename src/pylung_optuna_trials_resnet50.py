import configparser

import cv2
import numpy as np
import optuna
import tensorflow as tf
import tensorflow_addons as tfa
from keras.backend import clear_session
from optuna.integration import KerasPruningCallback
from optuna.samplers import NSGAIISampler

# TODO(crcrpar): Remove the below three lines once everything is ok.
# Register a global custom opener to avoid HTTP Error 403: Forbidden when downloading MNIST.
# opener = urllib.request.build_opener()
# opener.addheaders = [("User-agent", "Mozilla/5.0")]
# urllib.request.install_opener(opener)
from main.dataset_utilities.dataset_reader_classes import DatasetTransformer
from main.dataset_utilities.lidc_dataset_reader_classes import CustomLidcDatasetReader
from main.models.resnet_model import ResNet50Model
from main.models.vit_model import VitModel
from main.utilities.utilities_lib import two_one_hot_malignancy, LIDC_ANN_Y0, LIDC_ANN_Y1, LIDC_ANN_X0, LIDC_ANN_X1

config = configparser.ConfigParser()

# check config.ini is in the root folder
config_file = config.read('config.ini')
if len(config_file) == 0:
    raise Exception("config.ini file not found")

BATCHSIZE = 60
CLASSES = 2
EPOCHS = 35
TRAIN_SIZE = 0.8
IMAGE_SIZE = 98


def resize_old(image):
    im = np.float32(image)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
    return np.int16(im)


def resize(image, annotation):
    image = np.float32(image)
    im = cv2.resize(image[annotation[LIDC_ANN_Y0]:annotation[LIDC_ANN_Y1], annotation[LIDC_ANN_X0]:annotation[LIDC_ANN_X1]], (IMAGE_SIZE, IMAGE_SIZE))
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    return np.int16(im)


def get_optimizer(name, learning_rate, weight_decay, momentum):
    if name == 'AdamW':
        return tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    elif name == 'SGDW':
        return tfa.optimizers.SGDW(learning_rate=learning_rate, weight_decay=weight_decay)
    elif name == 'SGD':
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum, nesterov=False, name='SGD')


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    # We compile our model with a sampled learning rate.
    learning_rate = trial.suggest_float("Learning Rate", 1e-6, 1e-1, log=True)
    drop_out = trial.suggest_float("Drop Out", 0.1, 0.6, log=True)
    activation = trial.suggest_categorical("activation", ['softmax', 'sigmoid', 'softplus', 'tanh'])
    weight = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    momentum = trial.suggest_float("Momentum", 1e-7, 1e-1, log=True)
    optimizer = trial.suggest_categorical("Optimizer", ['AdamW', 'SGDW', 'SGD'])
    augmentation_size = trial.suggest_float("Augmentation size (%)", 0.2, 0.5, log=True)
    augmentation_image_rotation = trial.suggest_float("Augmentation Image rotation (dg)", 0.1, 0.7, log=True)

    ds_root_folder = config['DATASET'][f'processed_lidc_idri_location']
    dataset_reader = CustomLidcDatasetReader(location=ds_root_folder + '/main/')
    dataset_reader.dataset_data_transformers.append(DatasetTransformer(function=two_one_hot_malignancy))
    dataset_reader.dataset_image_transformers.append(DatasetTransformer(function=resize))
    dataset_reader.load_custom()

    x = dataset_reader.images
    y = dataset_reader.annotations

    (x_train), (y_train) = (
        x[: int(len(x) * TRAIN_SIZE)],
        y[: int(len(y) * TRAIN_SIZE)],
    )
    (x_valid), (y_valid) = (
        np.asarray(x[int(len(x) * TRAIN_SIZE):]),
        np.asarray(y[int(len(y) * TRAIN_SIZE):]),
    )

    dataset_reader.images = x_train
    dataset_reader.annotations = y_train

    dataset_reader.augment(augmentation_size, augmentation_image_rotation)

    (x_train), (y_train) = (
        np.asarray(dataset_reader.images),
        np.asarray(dataset_reader.annotations),
    )


    model = ResNet50Model(
        name='rs-model-test0',
        version='1.0',
        activation=activation,
        num_classes=CLASSES,
        image_size=IMAGE_SIZE,
        dropout=drop_out,
        pooling='avg',
        weights='imagenet',
        image_channels=3
    )

    model.build_model()

    model = model.model

    model.compile(
        loss="categorical_crossentropy",
        optimizer=get_optimizer(optimizer, learning_rate=learning_rate, weight_decay=weight, momentum=momentum),
        metrics=["accuracy"],
    )


    model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        shuffle=True,
        callbacks=[KerasPruningCallback(trial, "val_accuracy")],
        batch_size=BATCHSIZE,
        epochs=EPOCHS,
        verbose=False
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(x_valid, y_valid, verbose=0)
    return score[1]


def get_name():
    return f'ResNet50 w/ DA *** Image size {IMAGE_SIZE}x{IMAGE_SIZE}, {CLASSES} classes, {EPOCHS} epochs, {int(TRAIN_SIZE*100)}% train size and Batch Size of {BATCHSIZE}, GA 50/0.001/0.01/0.0001'


if __name__ == "__main__":
    study = optuna.create_study(storage="sqlite:///db.sqlite3", direction="maximize", study_name=get_name(), sampler=NSGAIISampler(
        population_size=50,
        mutation_prob=0.001,
        crossover_prob=0.01,
        swapping_prob=0.0001
    ))
    study.optimize(objective, n_trials=100) #, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
