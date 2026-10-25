"""Train on both CaliperGT bags using the remapped 800x800 OCam images."""

import os

_base_ = ['./custom_fisheye_bevdet_train_caliper.py']

# The image FPN builds two output levels, while the fisheye view transformer
# consumes only the first one.  DDP must therefore tolerate the unused second
# FPN convolution (fpn_convs.1) when training on multiple GPUs.
find_unused_parameters = True

data_root = os.environ.get(
    'FISHEYE_COMBINED_ROOT',
    'data/Custom/combined_caliper_fisheye3dod/')

train_dataloader = dict(dataset=dict(dataset=dict(
    data_root=data_root,
    ann_file=(
        'caliper_annotations_ocam_800x800/fisheye3dod_infos_train.pkl'))))
val_dataloader = dict(dataset=dict(
    data_root=data_root,
    ann_file=(
        'caliper_annotations_ocam_800x800/fisheye3dod_infos_val.pkl')))
test_dataloader = val_dataloader

model = dict(view_transform=dict(
    image_size=[800, 800],
    ocam_path=os.path.join(data_root, 'calib_results.txt'),
    kb4_calibration_path=None))

work_dir = 'work_dirs/custom_fisheye_bevdet_train_combined_ocam'
