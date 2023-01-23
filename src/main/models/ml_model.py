import pickle

from prettytable import PrettyTable
from tensorflow import keras

from main.common.config_classes import ConfigurableObject
from main.utilities.utilities_lib import info

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
