import unittest
from pathlib import Path
import logging
from src.dataset.lidc_idri_loader import load_pre_processed_lidc_idri, build_pre_processed_dataset

logging.basicConfig(level=logging.DEBUG)

class TestDataSetLoader(unittest.TestCase):

    def test_load_basic(self):
        source_path = Path(__file__).resolve()
        source_dir = str(source_path.parent.parent)

        try:
            load_pre_processed_lidc_idri(128, source_dir + '\\data\\datasets\\')
            print('Check fail')
            self.fail('Expect an error message: image_size parameter must have the value: 128, 256, or 512')
        except Exception as e:
            self.assertTrue(str(e).find('imagre_size parameter must have the value: 128, 256, or 512'), 'Unexpected error message: ' + str(e))


        # load_pre_processed_lidc_idri(128, None)

    def test_load(self):
        source_path = Path(__file__).resolve()
        source_dir = str(source_path.parent.parent)

        xtrain, ytrain, xtest, ytest = load_pre_processed_lidc_idri(128, source_dir + '\\data\\datasets\\')
        self.assertIsNotNone(xtrain, "Train images is None")
        self.assertIsNotNone(ytrain, "Train bounding boxes is None")
        self.assertIsNotNone(xtest, "Test images is None")
        self.assertIsNotNone(ytest, "Test bounding boxes is None")
        self.assertTrue(len(xtrain) > 0, "Train image set is empty")
        self.assertTrue(len(ytrain) > 0, "Train bounding box set is empty")
        self.assertTrue(len(xtest) > 0, "Test image set is empty")
        self.assertTrue(len(ytest) > 0, "Train bounding box set is empty")
        self.assertEqual(len(xtrain), 8, "Train set should have 80% of the total images")
        self.assertEqual(len(xtest), 2, "Test set should have 20% of the total images")

    def test_build_pre_processed(self):
        source_path = Path(__file__).resolve()
        source_dir = str(source_path.parent.parent)

        build_pre_processed_dataset(image_size=512, base_path=source_dir + '\\data\\temp\\', image_count=10, pad=5)
        build_pre_processed_dataset(image_size=256, base_path=source_dir + '\\data\\temp\\', image_count=10, pad=5)
        xtrain, ytrain, _, _ = load_pre_processed_lidc_idri(256, source_dir + '\\data\\temp\\')
        self.assertIsNotNone(xtrain, "Train images is None")
        self.assertIsNotNone(ytrain, "Train bounding boxes is None")

if __name__ == '__main__':
    unittest.main()