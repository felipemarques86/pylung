from optuna.integration import KerasPruningCallback

from main.models.ml_model import CustomModelDefinition
from main.models.vit_model import VitModel


class ModelDefinition(CustomModelDefinition):

    def default_build(self):
        return self.build(512, 1, 1, 1, None, [], [],
                          params={'learning_rate': 0.001,
                                  'drop_out_1': 0.1,
                                  'drop_out_2': 0.1,
                                  'drop_out_3': 0.1,
                                  'transformer_layers': 1,
                                  'num_heads': 6,
                                  'projection_dim': 64,
                                  'patch_size': 16,
                                  'activation': 'softmax',
                                  'weight': 0.0001,
                                  'momentum': 0.0001,
                                  'optimizer':  'SGD'},
                          static_params=True, return_model_only=True)(None)
    def _details(self):
        details = {
            'model_name': 'vit',
            'parameters': 'Learning Rate, Projection Dimension, Num. Heads, Drop Out 1, Drop Out 2, Drop Out 3, Num. Transformer layers, Patch Size, Activation, Weight, Momentum, Optimizer',
            'description': 'The Vision Transformer, or ViT, is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder. In order to perform classification, the standard approach of adding an extra learnable “classification token” to the sequence is used.',
            'extra_information': 'MLP Heads = [2048, 1024, 512, 64, 32], channels = 1'
        }
        return details

    def build(self, image_size, batch_size, epochs, num_classes, loss, data, metrics,
                       code_name=None, save_weights=False, static_params=False, params=[],
                       data_transformer_name=None,
                       return_model_only=False, weights_file=None, detection=False, isolate_nodule_image=False):

        model_type = 'vit'
        def objective(trial):
            # Clear clutter from previous Keras session graphs.
            self.clear_session()

            if static_params is False:
                learning_rate = trial.suggest_float("Learning Rate", 1e-10, 1e-1, log=True)
                projection_dim = trial.suggest_int("Projection Dimension", 64, 128, 2)
                num_heads = trial.suggest_int("Num. Heads", 2, 8, 2)
                drop_out_1 = trial.suggest_float("Drop Out 1", 0.01, 0.4, log=True)
                drop_out_2 = trial.suggest_float("Drop Out 2", 0.01, 0.3, log=True)
                drop_out_3 = trial.suggest_float("Drop Out 3", 0.01, 0.2, log=True)
                transformer_layers = trial.suggest_int("Num. Transformer layers", 2, 6, 1)
                patch_size = trial.suggest_categorical("Patch Size", [int(image_size/32), int(image_size/16), int(image_size/8), int(image_size/4)])
                if detection:
                    activation = 'softmax'
                else:
                    activation = trial.suggest_categorical("Activation", ['sigmoid'])
                weight = trial.suggest_float("Weight", 1e-8, 1e-1, log=True)
                momentum = trial.suggest_float("Momentum", 1e-8, 1e-1, log=True)
                optimizer = trial.suggest_categorical("Optimizer", ['AdamW', 'SGDW', 'SGD', 'RMSProp', 'Adagrad', 'Adam', 'Adadelta', 'Adamax', 'Nadam', 'Ftrl'])
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

            model = vit_model.model

            model.compile(
                loss=loss,
                optimizer=self.get_optimizer(optimizer, learning_rate=learning_rate, weight_decay=weight, momentum=momentum),
                metrics=metrics,
            )

            if weights_file is not None:
                model.load_weights(weights_file)

            if return_model_only:
                self.save_model_as_image(model, model_type)
                return model

            x_train, x_valid, y_train, y_valid = data

            try:

                history = model.fit(
                    x_train,
                    y_train,
                    validation_data=(x_valid, y_valid),
                    shuffle=True,
                    batch_size=batch_size,
                    callbacks=[KerasPruningCallback(trial, "val_loss")],
                    epochs=epochs,
                    verbose=1
                )

            except Exception as e:
                print('Error during fit process')
                print(str(e))
                # If there is a crash, fail the trial and save the error message
                trial.set_user_attr('error', str(e))
                return None

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
                'isolate_nodule_image': isolate_nodule_image,
                'detection': detection,
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

            self.save_model('vit', model, save_weights, code_name, str(score[1]), trial, train_params, isolate_nodule_image, detection, data_transformer_name)

            if detection:
                return score[0], score[1]
            return score[1], score[2], score[5]

        return objective
