import time

import numpy as np

from common.experiment import run_experiment_sgd, print_results
from runtime.read_ds import read_lidc_dataset
from vit.model_01 import create_vit_object_detector

TRAIN_SIZE = 0.8
TEST_SIZE = 1 - TRAIN_SIZE
IMAGE_SIZE = 512
SCAN_COUNT_PERC = 1
PAD = 0
patch_size = 32 # Size of the patches to be extracted from the input images
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)  # input image shape
learning_rate = 0.01
weight_decay = 0.00001
batch_size = 32
num_epochs = 6000
num_patches = (IMAGE_SIZE // patch_size) ** 2
projection_dim = 64
num_heads = 8
TRAIN = False
# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 6
mlp_head_units = [2048, 1024, 512, 64, 32]  # Size of the dense layers
history = []
num_patches = (IMAGE_SIZE // patch_size) ** 2

x_train = []
y_train = []
x_test = []
y_test = []

start_time = time.perf_counter()

MIN_BOUND = -1000.0
MAX_BOUND = 600.0

# normalize an LIDC-IDRI image to grayscale
def normalize(im):
    im[im < MIN_BOUND] = -1000
    im[im > MAX_BOUND] = 600
    im = (im + 1000) / (600 + 1000)
    return np.array(255 * im, dtype="uint8")




x_train, y_train, x_test, y_test = read_lidc_dataset(normalize)

vit_object_detector = create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
)

file_name = "vit_model_" + str(learning_rate) + "-" + str(weight_decay) + "-" + str(num_heads) + "-" + str(projection_dim) + \
            "-" + str(transformer_layers) + "-" + str(num_epochs) + ".h5"

if TRAIN:
    history = run_experiment_sgd(
        vit_object_detector, learning_rate, batch_size, num_epochs, x_train, y_train
    )
    vit_object_detector.save(file_name)

vit_object_detector.load_weights(file_name, by_name=False, skip_mismatch=False, options=None) # ('vit_object_detector.h5')

print_results(vit_object_detector, x_test, y_test)

end_time = time.perf_counter()
print(end_time - start_time, "seconds")
