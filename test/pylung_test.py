import logging
import unittest

from main.models.vit_model import VitModel
from pylung import train_classification

logging.basicConfig(level=logging.DEBUG)

class PyLungTest(unittest.TestCase):
    def test_train(self):
        vit_model = VitModel(
            name='vit-model-ut',
            version='1.0',
            patch_size=32,
            projection_dim=64,
            num_heads=8,
            mlp_head_units=[2048, 1024, 512, 64, 32],
            dropout1=0.1,
            dropout2=0.1,
            dropout3=0.1,
            image_size=512,
            activation='softmax',
            num_classes=2,
            transformer_layers=8,
            image_channels=1
        )

        vit_model.save_model()

        train_classification(
            model_name='vit-model-ut',
            dataset_name='main',
            train_size=0.85,
            weight_decay=0.0001,
            learning_rate=0.0001,
            validation_split=0.1,
            batch_size=32,
            num_epochs=3,
            optimizer='AdaW',
            dataset_type='lidc_idri',
            labels='benign,malignant'
        )


if __name__ == '__main__':
    unittest.main()

