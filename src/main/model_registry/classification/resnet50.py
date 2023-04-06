from keras import Model

from main.models.ml_model import CustomModelDefinition
from main.models.resnet_model import ResNet50Model


class ModelDefinition(CustomModelDefinition):

    def default_build(self):
        return self.build(512, 1, 1, 1, None, [], [],
                          params={'learning_rate': 0.001,
                                  'drop_out': 0.1,
                                  'activation': 'softmax',
                                  'weight': 0.0001,
                                  'momentum': 0.0001,
                                  'optimizer':  'SGD'},
                          static_params=True, return_model_only=True)(None)

    def _details(self):
        details = {
            'model_name': 'resnet50',
            'parameters': 'Learning Rate, Drop out, Activation, Weight Decay, Momentum, Optimizer',
            'description': 'ResNet-50 is a convolutional neural network that is 50 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database [1]. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.',
            'extra_information': 'Pooling = avg, weigths = imagenet, channels = 3'
        }
        return details

    def build(self, image_size, batch_size, epochs, num_classes, loss, data, metrics, code_name=None,
              save_weights=False, static_params=False, params=[], data_transformer_name=None, return_model_only=False,
              weights_file=None, detection=False, isolate_nodule_image=False):

        model_type = 'resnet50'
        def objective(trial):
            # Clear clutter from previous Keras session graphs.
            self.clear_session()

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

            model: Model = model_.model

            model.compile(
                loss=loss,
                optimizer=self.get_optimizer(optimizer, learning_rate=learning_rate, weight_decay=weight, momentum=momentum),
                metrics=metrics,
            )

            if return_model_only:
                print('Model details')
                model.summary()
                self.save_model_as_image(model, model_type)
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

            train_params = self.train_parameters(model_type, image_size, batch_size, epochs, num_classes, loss, code_name,
                                            save_weights, static_params, data_transformer_name, activation, drop_out,
                                            history, learning_rate, model, model_, momentum,
                                            optimizer, score, trial_id, weight, x_train, x_valid, y_train, y_valid)

            self.save_model('resnet50', model, save_weights, code_name, str(score[1]), trial, train_params)

            if detection:
                return score[1]
            return score[1], score[2], score[5]

        return objective

    def train_parameters(self, model_type, image_size, batch_size, epochs, num_classes, loss, code_name, save_weights,
                         static_params, data_transformer_name, activation, drop_out, history, learning_rate, model,
                         model_,
                         momentum, optimizer, score, trial_id, weight, x_train, x_valid, y_train, y_valid):
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
            'isolate_nodule_image': isolate_nodule_image,
            'detection': detection,
            'learning_params': {
                'learning_rate': learning_rate,
                'drop_out': drop_out,
                'activation': activation,
                'weight': weight,
                'momentum': momentum,
                'optimizer': optimizer
            }
        }
        return train_params
