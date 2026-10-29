"""Train Fisheye3DOD on the shared dual-format DetZero export."""

import os

_base_ = ["./custom_fisheye_bevdet_train_caliper.py"]

find_unused_parameters = True

data_root = os.environ.get(
    "DUAL_FISHEYE_ROOT",
    "data/Custom/combined_improved_detzero_rosa/",
)

train_dataloader = dict(dataset=dict(dataset=dict(
    data_root=data_root,
    ann_file="fisheye3dod/fisheye3dod_infos_train.pkl",
)))
val_dataloader = dict(dataset=dict(
    data_root=data_root,
    ann_file="fisheye3dod/fisheye3dod_infos_val.pkl",
))
test_dataloader = val_dataloader

model = dict(view_transform=dict(
    image_size=[800, 800],
    ocam_path=os.path.join(data_root, "calib_results.txt"),
    kb4_calibration_path=None,
))

work_dir = "work_dirs/custom_fisheye_bevdet_train_dual_export"
