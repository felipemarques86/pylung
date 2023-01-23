import logging
import unittest

from main.models.vit_model import VitModel

logging.basicConfig(level=logging.DEBUG)


class VitModelTest(unittest.TestCase):

    def test_create_vit_model(self):
        vit_model = VitModel(
            name='vit-model-test4',
            version='1.0',
            patch_size=16,
            projection_dim=64,
            num_heads=4,
            mlp_head_units=[2048, 1024, 512, 64, 32],
            dropout1=0.1,
            dropout2=0.1,
            dropout3=0.15,
            image_size=512,
            activation='sigmoid',
            num_classes=1,
            transformer_layers=2,
            image_channels=1
        )

        vit_model.save_model()

        # vit_model_2 = VitModel(name='vit-model-test0', version='1.0')
        # vit_model_2.load_model()
        # vit_model_2.build_model()
        # vit_model_2.export_to_dot()


if __name__ == '__main__':
    unittest.main()
