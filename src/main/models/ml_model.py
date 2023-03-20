import json
import pickle

from keras import Model
from keras.backend import clear_session
from keras.utils.vis_utils import plot_model
from prettytable import PrettyTable
from tensorflow import keras

from main.common.config_classes import ConfigurableObject
from main.utilities.utilities_lib import info, get_optimizer


class CustomModelDefinition(ConfigurableObject):
    def __init__(self):
        super().__init__()

    def clear_session(self):
        clear_session()

    def _details(self):
        raise Exception('Implement the method details')

    def details(self):
        stringlist =[]
        details = self._details()
        model: Model = self.default_build()
        model.summary(print_fn=lambda x: stringlist.append(x))
        details['build'] = '\n'.join(stringlist)

        table = PrettyTable(['Property', 'Value'])
        table.add_row(['Model Name', details['model_name']])
        table.add_row(['Parameters', details['parameters']])
        table.add_row(['Description', details['description']])
        table.add_row(['Extra Information', details['extra_information']])
        return details, table

    def build(self, image_size, batch_size, epochs, num_classes, loss, data, metrics,
                   code_name=None, save_weights=False, static_params=False, params=[],
                   data_transformer_name=None,
                   return_model_only=False, weights_file=None, detection=False):
        raise Exception('Implement the method build')

    def save_model(self, model_name, model, save_weights, code_name, acc, trial, params):
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

    def get_optimizer(self, optimizer, learning_rate, weight_decay, momentum):
        return get_optimizer(optimizer, learning_rate=learning_rate, weight_decay=weight_decay, momentum=momentum)

    def save_model_as_image(self, model, model_type):
        import os
        return plot_model(model, to_file=os.getcwd() + '\\static\\'+model_type + '.png', show_shapes=True, show_layer_names=True)

    def default_build(self):
        raise Exception('Implement the method default_build')


class MlModel(ConfigurableObject):
    def __init__(self, name, version, num_classes=0, image_size=0, image_channels=0):
        super().__init__()
        self.name = name
        self.version = version
        self.num_classes = num_classes
        self.image_size = image_size
        self.image_channels = image_channels
        self.model = None
        self.type = None

    def set_ml_model_fields(self, model):
        self.name = model.name
        self.version = model.version
        self.num_classes = model.num_classes
        self.image_size = model.image_size
        self.image_channels = model.image_channels

    def save_model(self):
        with open(self.get_models_location() + self.get_model_file_name(), 'wb') as filePointer:
            pickle.dump(self, filePointer, pickle.HIGHEST_PROTOCOL)

    def get_model_file_name(self):
        return self.name + '.v' + str(self.version) + '.model'

    def get_model_image_file_name(self):
        return self.name + '.v' + str(self.version) + '.png'

    def build_model(self):
        pass

    def load_model(self):
        with open(self.get_models_location() + self.get_model_file_name(), 'rb') as filePointer:
            model = pickle.load(filePointer)
            self.set_ml_model_fields(model)
            self.process_loaded_model(model)

    def process_loaded_model(self, model):
        pass

    def export_to_dot(self):
        keras.utils.plot_model(self.model, to_file=self.get_models_location() + self.get_model_image_file_name(), show_shapes=True)
        info(f"Model saved as {self.get_model_file_name()}")

    def get_models_location(self):
        return self.get_config()['DEFAULT']['models_location'] + '/'

    def export_as_table(self):
        table = PrettyTable(['Property', 'Value'])
        table.add_row(["Name", self.name])
        table.add_row(["Version", self.version])
        table.add_row(["N. Classes", self.num_classes])
        table.add_row(["Image Size", self.image_size])
        table.add_row(["Image Channels", self.image_channels])
        return table
