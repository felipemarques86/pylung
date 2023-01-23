import logging
import unittest

from main.models.resnet_model import ResNet50Model

logging.basicConfig(level=logging.DEBUG)


class ResNet50ModelTest(unittest.TestCase):

    def test_create_model(self):
        model = ResNet50Model(
            name='rs-model-test0',
            version='1.0',
            activation='softmax',
            num_classes=2,
            image_size=224,
            dropout=0.35,
            pooling='avg',
            weights='imagenet',
            image_channels=3
        )

        model.save_model()

        model_2 = ResNet50Model(name='rs-model-test0', version='1.0')
        model_2.load_model()
        model_2.build_model()
        model_2.export_to_dot()


if __name__ == '__main__':
    unittest.main()
