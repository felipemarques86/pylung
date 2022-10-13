import unittest
from pathlib import Path
import logging
from src.dataset.data_set_loader import load_pre_processed_lidc_idri

logging.basicConfig(level=logging.DEBUG)

class TestDataSetLoader(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()