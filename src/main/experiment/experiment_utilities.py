import numpy as np
from keras.backend import clear_session
from optuna.integration import KerasPruningCallback

from main.dataset_utilities.dataset_reader_classes import DatasetTransformer
from main.dataset_utilities.lidc_dataset_reader_classes import CustomLidcDatasetReader
from main.models.resnet_model import ResNet50Model
from main.models.vit_model import VitModel
from main.utilities.utilities_lib import img_transformer, get_optimizer, LIDC_ANN_Y0, LIDC_ANN_Y1, LIDC_ANN_X0, \
    LIDC_ANN_X1
from vit.vit_tensorflow.cait import CaiT


def get_ds(config, isolate_nodule_image, train_size, image_size, channels, data_transformer, ds, shuffle):
    ds_root_folder = config['DATASET'][f'processed_lidc_idri_location']
    dataset_reader = CustomLidcDatasetReader(location=ds_root_folder + f'/{ds}/')
    if isolate_nodule_image:
        dataset_reader.filter_out(lambda data: data[LIDC_ANN_Y0] == 0 and data[LIDC_ANN_Y1] == 0 and
                                               data[LIDC_ANN_X0] == 0 and data[LIDC_ANN_X1] == 0)

    dataset_reader.dataset_data_transformers.append(DatasetTransformer(function=data_transformer))
    dataset_reader.dataset_image_transformers.append(
        DatasetTransformer(function=img_transformer(image_size, channels, isolate_nodule_image)))
    dataset_reader.load_custom()

    if shuffle:
        dataset_reader.shuffle_data()


    x = dataset_reader.images
    y = dataset_reader.annotations
    (x_train), (y_train) = (
        x[: int(len(x) * train_size)],
        y[: int(len(y) * train_size)],
    )
    (x_valid), (y_valid) = (
        np.asarray(x[int(len(x) * train_size):]),
        np.asarray(y[int(len(y) * train_size):]),
    )
    dataset_reader.images = x_train
    dataset_reader.annotations = y_train
    dataset_reader.augment(0.2, 0.3)
    (x_train), (y_train) = (
        np.asarray(dataset_reader.images),
        np.asarray(dataset_reader.annotations),
    )
    return x_train, x_valid, y_train, y_valid


def build_classification_objective(model_type, image_size, batch_size, epochs, num_classes, loss, data):
    def _resnet50_classify(trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()

        # We compile our model with a sampled learning rate.
        learning_rate = trial.suggest_float("Learning Rate", 1e-6, 1e-1, log=True)
        drop_out = trial.suggest_float("Drop Out", 0.1, 0.6, log=True)
        activation = trial.suggest_categorical("activation", ['softmax', 'sigmoid', 'softplus', 'tanh'])
        weight = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        momentum = trial.suggest_float("Momentum", 1e-7, 1e-1, log=True)
        optimizer = trial.suggest_categorical("Optimizer", ['AdamW', 'SGDW', 'SGD'])
        # augmentation_size = trial.suggest_float("Augmentation size (%)", 0.2, 0.5, log=True)
        # augmentation_image_rotation = trial.suggest_float("Augmentation Image rotation (dg)", 0.1, 0.7, log=True)

        x_train, x_valid, y_train, y_valid = data

        model = ResNet50Model(
            name='in-memory',
            version='1.0',
            activation=activation,
            num_classes=num_classes,
            image_size=image_size,
            dropout=drop_out,
            pooling='avg',
            weights='imagenet',
            image_channels=3
        )

        model.build_model()

        model = model.model

        model.compile(
            loss=loss,
            optimizer=get_optimizer(optimizer, learning_rate=learning_rate, weight_decay=weight, momentum=momentum),
            metrics=["accuracy"],
        )

        model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            shuffle=True,
            callbacks=[KerasPruningCallback(trial, "val_accuracy")],
            batch_size=batch_size,
            epochs=epochs,
            verbose=False
        )

        # Evaluate the model accuracy on the validation set.
        score = model.evaluate(x_valid, y_valid, verbose=0)
        return score[1]

    def _vit_classify_objective(trial):
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
        patch_size = trial.suggest_categorical("Patch Size", [3, 4, 8, 16])
        activation = trial.suggest_categorical("Activation", ['softmax', 'sigmoid', 'softplus', 'tanh'])
        weight = trial.suggest_float("Weight", 1e-8, 1e-1, log=True)
        momentum = trial.suggest_float("Momentum", 1e-8, 1e-1, log=True)
        optimizer = trial.suggest_categorical("Optimizer", ['AdamW', 'SGDW', 'SGD'])
        # augmentation_size = trial.suggest_float("Augmentation size (%)", 0.1, 0.4, log=True)
        # augmentation_image_rotation = trial.suggest_float("Augmentation Image rotation (dg)", 0.01, 0.3, log=True)

        x_train, x_valid, y_train, y_valid = data

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
            image_size=image_size,
            activation=activation,
            num_classes=num_classes,
            transformer_layers=transformer_layers,
            image_channels=1
        )

        vit_model.build_model()

        model = vit_model.model

        model.compile(
            loss=loss,
            optimizer=get_optimizer(optimizer, learning_rate=learning_rate, weight_decay=weight, momentum=momentum),
            metrics=["accuracy"],
        )

        model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            shuffle=True,
            callbacks=[KerasPruningCallback(trial, "val_accuracy")],
            batch_size=batch_size,
            epochs=epochs,
            verbose=False
        )

        # Evaluate the model accuracy on the validation set.
        score = model.evaluate(x_valid, y_valid, verbose=0)
        return score[1]

    def cait_classify(trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()

        # We compile our model with a sampled learning rate.
        learning_rate = trial.suggest_float("Learning Rate", 1e-9, 1e-1, log=True)
        mlp_dim = trial.suggest_int("MLP Dimension", 1024, 2048, 2)
        dim = trial.suggest_int("Projection Dimension", 128, 1024, 16)
        num_heads_decay = trial.suggest_int("Num. Heads", 2, 8, 2)
        drop_out_1 = trial.suggest_float("Drop Out 1", 0.01, 0.4, log=True)
        drop_out_2 = trial.suggest_float("Drop Out 2", 0.01, 0.3, log=True)
        drop_out_3 = trial.suggest_float("Drop Out 3", 0.01, 0.2, log=True)
        depth = trial.suggest_int("Depth", 2, 16, 2)
        cls_depth = trial.suggest_int("CLS Depth", 2, 16, 2)
        patch_size = trial.suggest_categorical("Patch Size", [8, 16, 32])
        # augmentation_size = trial.suggest_float("Augmentation size (%)", 0.1, 0.4, log=True)
        # augmentation_image_rotation = trial.suggest_float("Augmentation Image rotation (dg)", 0.01, 0.3, log=True)
        weight = trial.suggest_float("Weight", 1e-8, 1e-1, log=True)
        momentum = trial.suggest_float("Momentum", 1e-8, 1e-1, log=True)
        optimizer = trial.suggest_categorical("Optimizer", ['AdamW', 'SGDW', 'SGD'])

        x_train, x_valid, y_train, y_valid = data

        model = CaiT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,  # depth of transformer for patch to patch attention only
            cls_depth=cls_depth,  # depth of cross attention of CLS tokens to patch
            heads=num_heads_decay,
            mlp_dim=mlp_dim,
            dropout=drop_out_1,
            emb_dropout=drop_out_2,
            layer_dropout=drop_out_3  # randomly dropout 5% of the layers
        )

        model.compile(
            loss=loss,
            optimizer=get_optimizer(optimizer, learning_rate=learning_rate, weight_decay=weight, momentum=momentum),
            metrics=["accuracy"],
        )

        model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            shuffle=True,
            callbacks=[KerasPruningCallback(trial, "val_accuracy")],
            batch_size=batch_size,
            epochs=epochs,
            verbose=False
        )

        # Evaluate the model accuracy on the validation set.
        score = model.evaluate(x_valid, y_valid, verbose=0)
        return score[1]

    if model_type == 'vit':
        return _vit_classify_objective
    if model_type == 'resnet50':
        return _resnet50_classify
    if model_type == 'cait':
        return cait_classify

    return None
