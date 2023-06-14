from main.models.ml_model import CustomModelDefinition

class ModelDefinition(CustomModelDefinition):

    def default_build(self):
        return self.build(512, 1, 1, 1, None, [], [],
                          params={'learning_rate': 0.001,
                                  'activation': 'softmax',
                                  'weight': 0.0001,
                                  'momentum': 0.0001,
                                  'optimizer':  'SGD'},
                          static_params=True, return_model_only=True)(None)

    def _details(self):
        details = {
            'model_name': 'vgg16',
            'parameters': 'Learning Rate, Activation, Weight Decay, Momentum, Optimizer',
            'description': 'VGG16 refers to the VGG model, also called VGGNet. It is a convolution neural network (CNN) model supporting 16 layers.',
            'extra_information': 'Optimizer is SGD only, channels = 3'
        }
        return details


    def build(self, image_size, batch_size, epochs, num_classes, loss, data, metrics,
                       code_name=None, save_weights=False, static_params=False, params=[],
                       data_transformer_name=None,
                       return_model_only=False, weights_file=None, detection=False, isolate_nodule_image=False):

        model_type = 'vgg16'
        def objective(trial):
            from keras.applications.vgg16 import VGG16
            from tensorflow import keras

            self.clear_session()

            if static_params is False:
                learning_rate = trial.suggest_float("Learning Rate", 1e-6, 1e-1, log=True)
                activation = trial.suggest_categorical("activation", ['sigmoid'])
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
                optimizer=self.get_optimizer(optimizer, learning_rate=learning_rate, weight_decay=weight, momentum=momentum),
                metrics=metrics,
            )

            if return_model_only:
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

            self.save_model('vgg16', model, save_weights, code_name, str(score[1]), trial, train_params, isolate_nodule_image, detection, data_transformer_name)

            if detection:
                return score[1]
            return score[1], score[2], score[5]

        return objective
