import time

from common.ds_reader import get_ds_single_file_name, save_ds_single_file
from dataset.lidc_idri_loader import load_lidc_idri_per_consensus_no_norm

start_time = time.perf_counter()

# r = [0, 250]
# images, annotations = load_lidc_idri_per_consensus_no_norm(start=r[0], end=r[1], pad=0)
# save_ds_single_file(get_ds_single_file_name(512, 0, 1, 'img-consensus-pt-1'),
#                     images)
# save_ds_single_file(get_ds_single_file_name(512, 0, 1, 'ann-consensus-pt-1'),
#                     annotations)
#
# print(str(r[0]) + ' to ' + str(r[1]) + ' done')

# r = [251, 501]
# images, annotations = load_lidc_idri_per_consensus_no_norm(start=r[0], end=r[1], pad=0)
# save_ds_single_file(get_ds_single_file_name(512, 0, 1, 'img-consensus-pt-2'),
#                     images)
# save_ds_single_file(get_ds_single_file_name(512, 0, 1, 'ann-consensus-pt-2'),
#                     annotations)
# print(str(r[0]) + ' to ' + str(r[1]) + ' done')

# r = [502, 752]
# images, annotations = load_lidc_idri_per_consensus_no_norm(start=r[0], end=r[1], pad=0)
# save_ds_single_file(get_ds_single_file_name(512, 0, 1, 'img-consensus-pt-3'),
#                     images)
# save_ds_single_file(get_ds_single_file_name(512, 0, 1, 'ann-consensus-pt-3'),
#                     annotations)
# print(str(r[0]) + ' to ' + str(r[1]) + ' done')

r = [753, 1018]
images, annotations = load_lidc_idri_per_consensus_no_norm(start=r[0], end=r[1], pad=0)
save_ds_single_file(get_ds_single_file_name(512, 0, 1, 'img-consensus-pt-4'),
                    images)
save_ds_single_file(get_ds_single_file_name(512, 0, 1, 'ann-consensus-pt-4'),
                    annotations)
print(str(r[0]) + ' to ' + str(r[1]) + ' done')

end_time = time.perf_counter()
print(end_time - start_time, "seconds")
