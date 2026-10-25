"""Fine-tune Fisheye BEVDet using annotations exported by the manual editor."""

_base_ = ['./custom_fisheye_bevdet_train.py']

train_dataloader = dict(dataset=dict(
    ann_file='manual_annotations/fisheye3dod_infos_manual_train.pkl'))
val_dataloader = dict(dataset=dict(
    ann_file='manual_annotations/fisheye3dod_infos_manual_val.pkl'))
test_dataloader = val_dataloader
work_dir = 'work_dirs/custom_fisheye_bevdet_train_manual'
