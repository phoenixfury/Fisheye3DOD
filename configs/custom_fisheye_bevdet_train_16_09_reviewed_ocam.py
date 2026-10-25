"""Fine-tune canonical 800x800 OCam FishEye BEVDet on the reviewed drive."""

_base_ = ['./custom_fisheye_bevdet_train_caliper.py']

data_root = (
    'data/Custom/BAG_2026_09_16_09_58_56_converted/'
    'synchronized_dual_images/')

train_dataloader = dict(dataset=dict(dataset=dict(
    data_root=data_root,
    ann_file=(
        'caliper_annotations_ocam_800x800/fisheye3dod_infos_train.pkl'))))
val_dataloader = dict(dataset=dict(
    data_root=data_root,
    ann_file='caliper_annotations_ocam_800x800/fisheye3dod_infos_val.pkl'))
test_dataloader = val_dataloader

model = dict(view_transform=dict(
    image_size=[800, 800],
    kb4_calibration_path=None,
    ocam_path='data/Fisheye3DODdataset/calib_results.txt'))

load_from = 'projects/Fisheye3DOD/weights/fisheye_bevdet.pth'

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook', interval=1, save_best='NDS', rule='greater',
        max_keep_ckpts=3))

work_dir = 'work_dirs/custom_fisheye_bevdet_16_09_reviewed_ocam'
