import logging
import unittest

from main.experiment.experiment import Experiment
from main.models.resnet_model import ResNet50Model

logging.basicConfig(level=logging.DEBUG)

class ExperimentTest(unittest.TestCase):

    def test_create_experiment(self):
        model = ResNet50Model(
            name='rs-model-test0',
            version='1.0',
            activation='softmax',
            num_classes=2,
            image_size=512,
            dropout=0.35,
            pooling='avg',
            weights='imagenet',
            image_channels=3
        )



        experiment = Experiment(
            model=model,
            optimizer='SGD',
            batch_size=16,
            num_epochs=10,
            train_size=0.8,
            weight_decay=0.0001,
            learning_rate=0.001,
            validation_split=0.1,
            x=[1, 2, 3, 4],
            y=[5, 6, 7, 8],
        )

        print(experiment.name)


if __name__ == '__main__':
    unittest.main()

