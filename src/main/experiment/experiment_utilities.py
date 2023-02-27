import json

import numpy as np
from keras import Model
from keras.backend import clear_session
from keras.utils.vis_utils import plot_model

from main.dataset_utilities.dataset_reader_classes import DatasetTransformer
from main.dataset_utilities.lidc_dataset_reader_classes import CustomLidcDatasetReader
from main.models.resnet_model import ResNet50Model
from main.models.vit_model import VitModel
from main.utilities.utilities_lib import img_transformer, get_optimizer, LIDC_ANN_Y0, LIDC_ANN_Y1, LIDC_ANN_X0, \
    LIDC_ANN_X1


def get_ds(config, isolate_nodule_image, train_size, image_size, channels, data_transformer, ds):
    ds_root_folder = config['DATASET'][f'processed_lidc_idri_location']
    dataset_reader = CustomLidcDatasetReader(location=ds_root_folder + f'/{ds}/')
    if isolate_nodule_image:
        dataset_reader.filter_out(lambda data: data[LIDC_ANN_Y0] == 0 and data[LIDC_ANN_Y1] == 0 and
                                               data[LIDC_ANN_X0] == 0 and data[LIDC_ANN_X1] == 0)

    dataset_reader.dataset_data_transformers.append(DatasetTransformer(function=data_transformer))
    dataset_reader.dataset_image_transformers.append(
        DatasetTransformer(function=img_transformer(image_size, channels, isolate_nodule_image)))
    dataset_reader.load_custom()

    #dataset_reader.shuffle_data()


    x = dataset_reader.images
    y = dataset_reader.annotations

    if len(x) != len(y):
        raise f"Total images does not match with total annotations {len(x)} != {len(y)}"

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


def save_model(model_name, model, save_weights, code_name, acc, trial, params):
    if save_weights:
        import time
        if trial is not None:
            name = 'trial_' + str(trial.number) + '-' + code_name + '_a' + acc
        else:
            name = model_name + '_' + str(time.time_ns()) + '_a' + acc
            if code_name is None:
                name = model_name + '_' + code_name + '_a' + acc
        model.save_weights('weights/' + name + '.h5')
        json_obj = json.dumps(params, indent=4)
        with open('weights/' + name + '.json', "w") as outfile:
            outfile.write(json_obj)

def build_classification_objective(model_type, image_size, batch_size, epochs, num_classes, loss, data, metrics,
                                   code_name=None, save_weights=False, static_params=False, params=[], data_transformer_name=None,
                                   return_model_only=False, weights_file=None, detection=False):
    def _resnet50_classify(trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()

        if static_params is False:
            learning_rate = trial.suggest_float("Learning Rate", 1e-6, 1e-1, log=True)
            drop_out = trial.suggest_float("Drop Out", 0.1, 0.6, log=True)
            activation = trial.suggest_categorical("activation", ['softmax', 'sigmoid'])
            weight = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
            momentum = trial.suggest_float("Momentum", 1e-7, 1e-1, log=True)
            optimizer = trial.suggest_categorical("Optimizer", ['AdamW', 'SGDW', 'SGD'])
        else:
            learning_rate = float(params['learning_rate'])
            drop_out = float(params['drop_out'])
            activation = params['activation']
            weight = float(params['weight'])
            momentum = params['momentum']
            optimizer = params['optimizer']

        model_ = ResNet50Model(
            name='resnet50',
            version='1.0',
            activation=activation,
            num_classes=num_classes,
            image_size=image_size,
            dropout=drop_out,
            pooling='avg',
            weights='imagenet',
            image_channels=3
        )

        model_.build_model()

        model = model_.model

        # model.summary()
        plot_model(model, to_file='resnet50.png', show_shapes=True, show_layer_names=True)
        # visualizer(model, filename='resnet50-2.png', format='png')

        model.compile(
            loss=loss,
            optimizer=get_optimizer(optimizer, learning_rate=learning_rate, weight_decay=weight, momentum=momentum),
            metrics=metrics,
        )

        if return_model_only:
            return model

        x_train, x_valid, y_train, y_valid = data

        if weights_file is not None:
            model.load_weights(weights_file)

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )


        score = model.evaluate(x_valid, y_valid, verbose=0)
        print(score)
        print(model.metrics_names)

        trial_id = 'N/A'
        if trial is not None:
            trial_id = str(trial.number)

        train_params = {
            'trial_id': trial_id,
            'model_type': model_type,
            'image_size': image_size,
            'batch_size': batch_size,
            'epochs': epochs,
            'num_classes': num_classes,
            'loss': loss,
            'code_name': code_name,
            'save_weights': save_weights,
            'static_params': static_params,
            'score': score,
            'score_names': model.metrics_names,
            'x_train_size': len(x_train),
            'y_train_size': len(y_train),
            'x_valid_size': len(x_valid),
            'y_valid_size': len(y_valid),
            'image_channels': model_.image_channels,
            'version': model_.version,
            'pooling': model_.pooling,
            'weights': model_.weights,
            'data_transformer_name': data_transformer_name,
            'history': history.history,
            'learning_params': {
                'learning_rate': learning_rate,
                'drop_out': drop_out,
                'activation': activation,
                'weight': weight,
                'momentum': momentum,
                'optimizer': optimizer
            }
        }

        save_model('resnet50', model, save_weights, code_name, str(score[1]), trial, train_params)

        if detection:
            return score[1]
        return score[1], score[2], score[5]



    def _vit_classify_objective(trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()

        if static_params is False:
            learning_rate = trial.suggest_float("Learning Rate", 1e-9, 1e-1, log=True)
            projection_dim = trial.suggest_int("Projection Dimension", 64, 128, 2)
            num_heads = trial.suggest_int("Num. Heads", 2, 8, 2)
            drop_out_1 = trial.suggest_float("Drop Out 1", 0.01, 0.4, log=True)
            drop_out_2 = trial.suggest_float("Drop Out 2", 0.01, 0.3, log=True)
            drop_out_3 = trial.suggest_float("Drop Out 3", 0.01, 0.2, log=True)
            transformer_layers = trial.suggest_int("Num. Transformer layers", 2, 16, 2)
            patch_size = trial.suggest_categorical("Patch Size", [16, 32, 64])
            activation = trial.suggest_categorical("Activation", ['softmax', 'sigmoid'])
            weight = trial.suggest_float("Weight", 1e-8, 1e-1, log=True)
            momentum = trial.suggest_float("Momentum", 1e-8, 1e-1, log=True)
            optimizer = trial.suggest_categorical("Optimizer", ['AdamW', 'SGDW', 'SGD'])
        else:
            learning_rate = float(params['learning_rate'])
            projection_dim = int(params['projection_dim'])
            num_heads = int(params['num_heads'])
            drop_out_1 = float(params['drop_out_1'])
            drop_out_2 = float(params['drop_out_2'])
            drop_out_3 = float(params['drop_out_3'])
            transformer_layers = int(params['transformer_layers'])
            patch_size = int(params['patch_size'])
            activation = params['activation']
            weight = float(params['weight'])
            momentum = float(params['momentum'])
            optimizer = params['optimizer']

        vit_model = VitModel(
            name='ignore',
            version=1,
            patch_size=patch_size,
            projection_dim=projection_dim,
            num_heads=num_heads,
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

        model: Model = vit_model.model

        # model.summary()
        plot_model(model, to_file='vit.png', show_shapes=True, show_layer_names=True)
        # visualizer(model, filename='vit-2.png', format='png')

        model.compile(
            loss=loss,
            optimizer=get_optimizer(optimizer, learning_rate=learning_rate, weight_decay=weight, momentum=momentum),
            metrics=metrics,
        )

        if return_model_only:
            return model

        x_train, x_valid, y_train, y_valid = data

        if weights_file is not None:
            model.load_weights(weights_file)

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )

        # Evaluate the model accuracy on the validation set.
        score = model.evaluate(x_valid, y_valid, verbose=0)
        print(score)
        print(model.metrics_names)

        trial_id = 'N/A'
        if trial is not None:
            trial_id = str(trial.number)

        train_params = {
            'trial_id': trial_id,
            'model_type': model_type,
            'image_size': image_size,
            'batch_size': batch_size,
            'epochs': epochs,
            'num_classes': num_classes,
            'loss': loss,
            'code_name': code_name,
            'save_weights': save_weights,
            'static_params': static_params,
            'score': score,
            'score_names': model.metrics_names,
            'x_train_size': len(x_train),
            'y_train_size': len(y_train),
            'x_valid_size': len(x_valid),
            'y_valid_size': len(y_valid),
            'image_channels': vit_model.image_channels,
            'version': vit_model.version,
            'data_transformer_name': data_transformer_name,
            'history': history.history,
            'learning_params': {
                'learning_rate': learning_rate,
                'projection_dim': projection_dim,
                'num_heads': num_heads,
                'drop_out_1': drop_out_1,
                'drop_out_2': drop_out_2,
                'drop_out_3': drop_out_3,
                'transformer_layers': transformer_layers,
                'patch_size': patch_size,
                'activation': activation,
                'weight': weight,
                'momentum': momentum,
                'optimizer': optimizer
            }
        }

        save_model('vit', model, save_weights, code_name, str(score[1]), trial, train_params)
        if detection:
            return score[1]
        return score[1], score[2], score[5]

    def vgg16(trial):
        from keras.applications.vgg16 import VGG16
        from tensorflow import keras

        clear_session()

        if static_params is False:
            learning_rate = trial.suggest_float("Learning Rate", 1e-6, 1e-1, log=True)
            activation = trial.suggest_categorical("activation", ['softmax', 'sigmoid'])
            weight = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
            momentum = trial.suggest_float("Momentum", 1e-7, 1e-1, log=True)
            optimizer = trial.suggest_categorical("Optimizer", ['SGD'])
        else:
            learning_rate = params['learning_rate']
            activation = params['activation']
            weight = params['weight']
            momentum = params['momentum']
            optimizer = params['optimizer']


        shape = (image_size, image_size, 3)

        base_model = VGG16(weights='imagenet', input_shape=shape, include_top=False)
        base_model.trainable = True
        inputs = keras.Input(shape=shape)
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(num_classes, activation=activation)(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            loss=loss,
            optimizer=get_optimizer(optimizer, learning_rate=learning_rate, weight_decay=weight, momentum=momentum),
            metrics=metrics,
        )

        # model.summary()
        plot_model(model, to_file='vgg16.png', show_shapes=True, show_layer_names=True)
        # visualizer(model, filename='vgg16-2.png', format='png')

        if return_model_only:
            return model

        x_train, x_valid, y_train, y_valid = data

        if weights_file is not None:
            model.load_weights(weights_file)

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )


        # Evaluate the model accuracy on the validation set.
        score = model.evaluate(x_valid, y_valid, verbose=0)
        print(score)
        print(model.metrics_names)

        trial_id = 'N/A'
        if trial is not None:
            trial_id = str(trial.number)

        train_params = {
            'trial_id': trial_id,
            'model_type': model_type,
            'image_size': image_size,
            'batch_size': batch_size,
            'epochs': epochs,
            'num_classes': num_classes,
            'loss': loss,
            'code_name': code_name,
            'save_weights': save_weights,
            'static_params': static_params,
            'score': score,
            'score_names': model.metrics_names,
            'x_train_size': len(x_train),
            'y_train_size': len(y_train),
            'x_valid_size': len(x_valid),
            'y_valid_size': len(y_valid),
            'image_channels': 3,
            'data_transformer_name': data_transformer_name,
            'history': history.history,
            'learning_params': {
                'learning_rate': learning_rate,
                'activation': activation,
                'weight': weight,
                'momentum': momentum,
                'optimizer': optimizer
            }
        }
        save_model('vgg16', model, save_weights, code_name, str(score[1]), trial, train_params)
        if detection:
            return score[1]
        return score[1], score[2], score[5]

    if model_type == 'vit':
        return _vit_classify_objective
    if model_type == 'resnet50':
        return _resnet50_classify
    if model_type == 'vgg16':
        return vgg16

    return None
