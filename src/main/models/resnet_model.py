from keras import Sequential
from keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import resnet50
from tensorflow.keras.models import Model
import tensorflow as tf
import cv2
import numpy as np

from main.models.ml_model import MlModel


class ResNet50Model(MlModel):

    def __init__(self, name, version, image_size=0, image_channels=0, pooling=0, num_classes=0, weights=0, dropout=0, activation=''):
        super().__init__(name, version, num_classes, image_size, image_channels)
        self.activation = activation
        self.dropout = dropout
        self.pooling = pooling
        self.weights = weights
        self.type = 'ResNet50Model'

    def process_loaded_model(self, model):
        self.activation = model.activation
        self.dropout = model.dropout
        self.pooling = model.pooling
        self.weights = model.weights

    def build_model(self):
        self.model = Sequential()
        base_model = resnet50.ResNet50(include_top=False,
                                       input_shape=(self.image_size, self.image_size, self.image_channels),
                                       pooling=self.pooling, classes=self.num_classes,
                                       weights=self.weights)
        base_model.trainable = False
        self.model.add(base_model)
        self.model.add(Flatten())
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.image_size, activation='relu'))
        self.model.add(Dense(self.num_classes, activation=self.activation))
        self.model.trainable = True

    def export_as_table(self):
        table = super().export_as_table()
        table.add_row(["Activation", self.activation])
        table.add_row(["Dropout", self.dropout])
        table.add_row(["Pooling", self.pooling])
        table.add_row(["weights", self.weights])
        return table

