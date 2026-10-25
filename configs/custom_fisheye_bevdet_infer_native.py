"""Inference-only config for native 1920x1536 KB4 images."""

import os

_base_ = ['./custom_fisheye_bevdet.py']

data_root = os.environ.get('FISHEYE_INFERENCE_DATA_ROOT', '')
calibration = os.environ.get(
    'FISHEYE_NATIVE_CALIBRATION',
    'projects/Fisheye3DOD/calibration.json')

test_dataloader = dict(dataset=dict(data_root=data_root))
model = dict(view_transform=dict(
    image_size=[1536, 1920],
    kb4_calibration_path=calibration,
    kb4_camera_frames=('ARC_F', 'ARC_SL', 'ARC_SR', 'ARC_R')))
