import configparser
import time
import cv2
import numpy as np
import optuna
from keras.backend import clear_session
from optuna.integration import KerasPruningCallback
from optuna.samplers import NSGAIISampler

from main.dataset_utilities.dataset_reader_classes import DatasetTransformer
from main.dataset_utilities.lidc_dataset_reader_classes import CustomLidcDatasetReader
from main.models.vit_model import VitModel
from main.utilities.utilities_lib import two_one_hot_malignancy, get_optimizer, LIDC_ANN_Y0, LIDC_ANN_Y1, LIDC_ANN_X0, \
    LIDC_ANN_X1

config = configparser.ConfigParser()

# check config.ini is in the root folder
config_file = config.read('config.ini')
if len(config_file) == 0:
    raise Exception("config.ini file not found")

BATCHSIZE = 100
CLASSES = 2
EPOCHS = 15
TRAIN_SIZE = 0.8
IMAGE_SIZE = 32 #98

def resize(image, annotation):
    image = np.float32(image)
    im = cv2.resize(image[annotation[LIDC_ANN_Y0]:annotation[LIDC_ANN_Y1], annotation[LIDC_ANN_X0]:annotation[LIDC_ANN_X1]], (IMAGE_SIZE, IMAGE_SIZE))
    return np.int16(im)


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    # We compile our model with a sampled learning rate.
    learning_rate = trial.suggest_float("Learning Rate", 1e-9, 1e-1, log=True)
    projection_dim_decay = trial.suggest_int("Projection Dimension", 64, 128, 2)
    num_heads_decay = trial.suggest_int("Num. Heads", 2, 8, 2)
    drop_out_1 = trial.suggest_float("Drop Out 1", 0.01, 0.4, log=True)
    drop_out_2 = trial.suggest_float("Drop Out 2", 0.01, 0.3, log=True)
    drop_out_3 = trial.suggest_float("Drop Out 3", 0.01, 0.2, log=True)
    transformer_layers = trial.suggest_int("Num. Transformer layers", 2, 16, 2)
    patch_size = trial.suggest_categorical("Patch Size", [8, 16])
    activation = trial.suggest_categorical("Activation", ['softmax', 'sigmoid', 'softplus', 'tanh'])
    weight = trial.suggest_float("Weight", 1e-8, 1e-1, log=True)
    momentum = trial.suggest_float("Momentum", 1e-8, 1e-1, log=True)
    optimizer = trial.suggest_categorical("Optimizer", ['AdamW', 'SGDW', 'SGD'])
    augmentation_size = trial.suggest_float("Augmentation size (%)", 0.1, 0.4, log=True)
    augmentation_image_rotation = trial.suggest_float("Augmentation Image rotation (dg)", 0.01, 0.3, log=True)


    ds_root_folder = config['DATASET'][f'processed_lidc_idri_location']
    dataset_reader = CustomLidcDatasetReader(location=ds_root_folder + '/DS1/')
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


    vit_model = VitModel(
        name='ignore',
        version=1,
        patch_size=patch_size,
        projection_dim=projection_dim_decay,
        num_heads=num_heads_decay,
        mlp_head_units=[2048, 1024, 512, 64, 32],
        dropout1=drop_out_1,
        dropout2=drop_out_2,
        dropout3=drop_out_3,
        image_size=IMAGE_SIZE,
        activation=activation,
        num_classes=CLASSES,
        transformer_layers=transformer_layers,
        image_channels=1
    )

    vit_model.build_model()

    model = vit_model.model

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
    return f'{IMAGE_SIZE}x{IMAGE_SIZE}, {CLASSES} classes, {EPOCHS} epochs, {int(TRAIN_SIZE*100)}% train size and Batch Size of {BATCHSIZE}, GA 50/0.001/0.01/0.0001 ' + str(time.time())

if __name__ == "__main__":
    study = optuna.create_study(storage="sqlite:///db.sqlite3", direction="maximize", study_name=get_name(), sampler=NSGAIISampler(
        population_size=50,
        mutation_prob=0.001,
        crossover_prob=0.01,
        swapping_prob=0.0001
    ))
    study.optimize(objective, n_trials=50) #, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
