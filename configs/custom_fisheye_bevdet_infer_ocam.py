"""Inference-only config for canonical 800x800 OCam images."""

import os

_base_ = ['./custom_fisheye_bevdet.py']

data_root = os.environ.get('FISHEYE_INFERENCE_DATA_ROOT', '')
ocam_calibration = os.environ.get(
    'FISHEYE_OCAM_CALIBRATION',
    'data/Fisheye3DODdataset/calib_results.txt')

test_dataloader = dict(dataset=dict(data_root=data_root))
model = dict(view_transform=dict(
    image_size=[800, 800],
    ocam_path=ocam_calibration,
    kb4_calibration_path=None))
