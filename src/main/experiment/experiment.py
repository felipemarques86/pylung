import time

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
from prettytable import PrettyTable

from main.common.config_classes import ConfigurableObject
from main.models.ml_model import MlModel


class Experiment(ConfigurableObject):

    def __init__(self, model: MlModel, learning_rate, weight_decay, batch_size, num_epochs, x, y, optimizer, validation_split, train_size, loss) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.validation_split = validation_split
        self.history = None
        self.name = self.generate_name()
        self.loss = loss

        (self.x_train), (self.y_train) = (
            np.asarray(x[: int(len(x) * train_size)]),
            np.asarray(y[: int(len(y) * train_size)]),
        )
        (self.x_test), (self.y_test) = (
            np.asarray(x[int(len(x) * train_size):]),
            np.asarray(y[int(len(y) * train_size):]),
        )

    def export_as_table(self):
        table = PrettyTable(['Property', 'Value'])
        table.add_row(['Model used', self.model.name])
        table.add_row(['Learning Rate', self.learning_rate])
        table.add_row(['Weight Decay', self.weight_decay])
        table.add_row(['Batch Size', self.batch_size])
        table.add_row(['Num Epochs', self.num_epochs])
        table.add_row(['Optimizer', self.optimizer])
        table.add_row(['Validation Split', self.validation_split])
        table.add_row(['X train size', len(self.x_train)])
        table.add_row(['X test size', len(self.x_test)])
        table.add_row(['Loss', self.loss])
        return table


    def generate_name(self):
        print(self.model)
        return f'{self.model.name}-{self.learning_rate}-{self.weight_decay}-{self.batch_size}-{self.num_epochs}-{self.optimizer}-{self.validation_split}'

    def load_model(self, weights_file):
        model = self.model.model
        if model is None:
            self.model.build_model()
            model = self.model.model
        model.load_weights(weights_file, by_name=False, skip_mismatch=False, options=None)
        return model


    def train(self):
        model = self.model.model
        if model is None:
            self.model.build_model()
            model = self.model.model


        if self.optimizer == 'AdamW':
            optimizer = tfa.optimizers.AdamW(
                learning_rate=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer == 'SGDW':
            optimizer = tfa.optimizers.SGDW(
                learning_rate=self.learning_rate, weight_decay=self.weight_decay, momentum=0.001, nesterov=False, name='SGDW'
            )
        elif self.optimizer == 'SGD':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate, momentum=0.001, nesterov=False, name='SGD'
            )

        model.compile(optimizer=optimizer, loss=self.loss, metrics=['accuracy'])

        checkpoint_filepath = "../logs/"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )

        history = model.fit(
            x=self.x_train,
            y=self.y_train,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            validation_split=self.validation_split,
            callbacks=[
                checkpoint_callback
            ],
        )
        self.history = history
        weights_path = self.config['DEFAULT']['weights_location'] + '/' + self.generate_name() + '---ID-' + str(round(time.time()*1000)) + '.h5'
        model.save(weights_path)


class ClassificationExperiment(Experiment):

    def __init__(self, model: MlModel, train_size, x, y, learning_rate=0, weight_decay=0, batch_size=0, num_epochs=0, optimizer=0,
                 validation_split=0, loss='categorical_crossentropy' ) -> None:
        super().__init__(model, learning_rate, weight_decay, batch_size, num_epochs, x, y, optimizer, validation_split,
                         train_size, loss)

    def print_history(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def print_results_classification(self, n=10, weights_location=None, labels=''):

        if weights_location is not None:
            if self.model.model is None:
                self.model.build_model()
            self.model.model.load_weights(weights_location, by_name=False, skip_mismatch=False, options=None)

        for i in range(0, n):
            input_image = self.x_test[i]
            fig, (ax1) = plt.subplots(1, 1, figsize=(15, 15))
            im = input_image

            # Display the image
            ax1.imshow(im, cmap=plt.cm.gray)

            input_image = np.expand_dims(input_image, axis=0)
            p = self.model.model.predict(input_image)

            def convert_to_label(y, label_list_str):
                label_list = label_list_str.split(",")
                if self.model.num_classes == 2:
                    max_pos = -1
                    m = -1
                    for j in range(0, len(y)):
                        if m < y[j]:
                            m = y[j]
                            max_pos = j
                    if max_pos < 0:
                        return '----'
                    return label_list[max_pos]
                elif self.model.num_classes == 1:
                    if y == 0:
                        return label_list[0]
                    if y == 1:
                        return label_list[1]

            predicted = p[0]
            truth = self.y_test[i]
            if len(labels) > 0:

                ax1.set_xlabel(
                    "Predicted: " + str(predicted) + "(" + convert_to_label(predicted, labels) + "), Original: "
                    + str(truth) + "(" + convert_to_label(truth, labels) + ")"
                )
            else:
                ax1.set_xlabel(
                    "Predicted: " + str(predicted) + ", Original: " + str(truth)
                )
        plt.show()


