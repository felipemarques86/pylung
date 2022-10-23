import time
from common.experiment import run_experiment, print_results
from dataset.lidc_idri_loader import load_lidc_idri
from vit.model_01 import create_vit_object_detector
import pickle

def save_ds_single_file(filename, obj):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    pass

def load_ds_single_file(image_size, pad, scan_count_perc):
    with open(get_ds_single_file_name(image_size, pad, scan_count_perc), 'rb') as filePointer:
        xtrain, ytrain, _, _ = pickle.load(filePointer)

    with open(get_ds_single_file_name(512, pad, scan_count_perc), 'rb') as filePointer:
        _, _, xtest, ytest = pickle.load(filePointer)

    return xtrain, ytrain, xtest, ytest

def get_ds_single_file_name(image_size, pad, scan_count_perc):
    return 'C:\\temp\\LIDC-' + str(image_size) + '-' + str(pad) + '-' + str(scan_count_perc) + '.pkl'

TRAIN_SIZE = 0.8
TEST_SIZE = 1 - TRAIN_SIZE
IMAGE_SIZE = 512
SCAN_COUNT_PERC = 1
PAD = 20
patch_size = 32  # Size of the patches to be extracted from the input images
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)  # input image shape
learning_rate = 0.005
weight_decay = 0.0001
batch_size = 32
num_epochs = 400
num_patches = (IMAGE_SIZE // patch_size) ** 2
projection_dim = 64
num_heads = 16
# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 32
mlp_head_units = [2048, 1024, 512, 64, 32]  # Size of the dense layers
history = []
num_patches = (IMAGE_SIZE // patch_size) ** 2

x_train = []
y_train = []
x_test = []
y_test = []
save = False

try:
    x_train, y_train, x_test, y_test = load_ds_single_file(IMAGE_SIZE, PAD, SCAN_COUNT_PERC)
except Exception as e:
    print(e)
    x_train, y_train, x_test, y_test = load_lidc_idri(image_size=IMAGE_SIZE, annotation_size_perc=SCAN_COUNT_PERC, pad=PAD)
    save = True

if save:
    try:
        save_ds_single_file(get_ds_single_file_name(IMAGE_SIZE, PAD, SCAN_COUNT_PERC), (x_train, y_train, x_test, y_test))
    except Exception as e:
        print(e)

start_time = time.perf_counter()

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

history = run_experiment(
    vit_object_detector, learning_rate, weight_decay, batch_size, num_epochs, x_train, y_train
)

print_results(vit_object_detector, IMAGE_SIZE, x_test, y_test)

end_time = time.perf_counter()
print(end_time - start_time, "seconds")
