import time

from common.ds_reader import get_ds_single_file_name, save_ds_single_file
from dataset.lidc_idri_loader import load_lidc_idri_per_consensus_no_norm

x_train = []
y_train = []
x_test = []
y_test = []
save = False


start_time = time.perf_counter()

r = [0, 250]
x_train, y_train, x_test, y_test = load_lidc_idri_per_consensus_no_norm(start=r[0], end=r[1], pad=0)
save_ds_single_file(get_ds_single_file_name(512, 0, 1, 'consensus-pt-1'),
                        (x_train, y_train, x_test, y_test))
print(str(r[0]) + ' to ' + str(r[1]) + ' done')

r = [251, 501]
x_train, y_train, x_test, y_test = load_lidc_idri_per_consensus_no_norm(start=r[0], end=r[1], pad=0)
save_ds_single_file(get_ds_single_file_name(512, 0, 1, 'consensus-pt-2'),
                       (x_train, y_train, x_test, y_test))
print(str(r[0]) + ' to ' + str(r[1]) + ' done')

r = [502, 752]
x_train, y_train, x_test, y_test = load_lidc_idri_per_consensus_no_norm(start=r[0], end=r[1], pad=0)
save_ds_single_file(get_ds_single_file_name(512, 0, 1, 'consensus-pt-3'),
                        (x_train, y_train, x_test, y_test))
print(str(r[0]) + ' to ' + str(r[1]) + ' done')

r = [753, 1018]
x_train, y_train, x_test, y_test = load_lidc_idri_per_consensus_no_norm(start=r[0], end=r[1], pad=0)
save_ds_single_file(get_ds_single_file_name(512, 0, 1, 'consensus-pt-4'),
                        (x_train, y_train, x_test, y_test))
print(str(r[0]) + ' to ' + str(r[1]) + ' done')

end_time = time.perf_counter()
print(end_time - start_time, "seconds")
