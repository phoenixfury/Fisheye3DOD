"""Fine-tune native-KB4 FishEye BEVDet on the reviewed 16 September drive."""

_base_ = ['./custom_fisheye_bevdet_train_caliper.py']

data_root = (
    'data/Custom/BAG_2026_09_16_09_58_56_converted/'
    'synchronized_dual_images/')

train_dataloader = dict(dataset=dict(dataset=dict(
    data_root=data_root,
    ann_file='caliper_annotations_native/fisheye3dod_infos_train.pkl')))
val_dataloader = dict(dataset=dict(
    data_root=data_root,
    ann_file='caliper_annotations_native/fisheye3dod_infos_val.pkl'))
test_dataloader = val_dataloader

model = dict(view_transform=dict(
    image_size=[1536, 1920],
    kb4_calibration_path=(
        'data/Custom/BAG_2026_09_16_09_58_56_converted/calibration.json'),
    kb4_camera_frames=('ARC_F', 'ARC_SL', 'ARC_SR', 'ARC_R')))

# Avoid validation leakage from native_epoch_20.pth, which was trained on an
# older annotation export of this drive.
load_from = 'projects/Fisheye3DOD/weights/fisheye_bevdet.pth'

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook', interval=1, save_best='NDS', rule='greater',
        max_keep_ckpts=3))

work_dir = 'work_dirs/custom_fisheye_bevdet_16_09_reviewed'
