from keras import Model
from keras.backend import clear_session
from keras.utils.vis_utils import plot_model

from main.experiment.experiment_utilities import save_model
from main.utilities.utilities_lib import get_optimizer


def details():
    pass

def train_parameters(model_type, image_size, batch_size, epochs, num_classes, loss, code_name, save_weights,
                     static_params, data_transformer_name, history, learning_rate, model, model_,
                     score, trial_id, x_train, x_valid, y_train, y_valid, ): #Add here your learning parameters
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
        'data_transformer_name': data_transformer_name,
        'history': history.history,
        'learning_params': {
            'learning_rate': learning_rate
            # Add more learning parameters
        }
    }
    return train_params
def build(model_type, image_size, batch_size, epochs, num_classes, loss, data, metrics,
                   code_name=None, save_weights=False, static_params=False, params=[],
                   data_transformer_name=None,
                   return_model_only=False, weights_file=None, detection=False):
    def objective(trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()

        if static_params is False:
            learning_rate = trial.suggest_float("Learning Rate", 1e-6, 1e-1, log=True)
            weight = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
            momentum = trial.suggest_float("Momentum", 1e-7, 1e-1, log=True)
            optimizer = trial.suggest_categorical("Optimizer", ['AdamW', 'SGDW', 'SGD'])
            # Add your parameters here
        else:
            learning_rate = float(params['learning_rate'])
            # Add your parameters here

        # Create here your model
        model: Model = None

        model.compile(
            loss=loss,
            optimizer=get_optimizer(optimizer, learning_rate=learning_rate, weight_decay=weight, momentum=momentum),
            metrics=metrics,
        )

        if return_model_only:
            print('Model details')
            model.summary()
            plot_model(model, to_file=model_type + '.png', show_shapes=True, show_layer_names=True)
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

        train_params = train_parameters(model_type, image_size, batch_size, epochs, num_classes, loss, code_name,
                                        save_weights, static_params, data_transformer_name, activation, drop_out,
                                        history, learning_rate, model, model_, momentum,
                                        optimizer, score, trial_id, weight, x_train, x_valid, y_train, y_valid)

        save_model(model_type, model, save_weights, code_name, str(score[1]), trial, train_params)

        if detection:
            return score[1]
        return score[1], score[2], score[5]

    return objective
