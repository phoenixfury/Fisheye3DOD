"""Fine-tune Fisheye BEVDet on native KB4 images and CaliperGT labels."""

_base_ = ['./custom_fisheye_bevdet_train.py']

data_root = 'data/Custom/BAG_2026_09_30_10_10_54_converted/fisheye3dod_native_2hz/'
classes = ['Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist']
cam_type = 'cam_fisheye'
camera_names = [
    'fisheye_camera_front', 'fisheye_camera_left',
    'fisheye_camera_right', 'fisheye_camera_rear',
]

train_pipeline = [
    dict(
        type='LoadFisheye3DODMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        load_cam_type=cam_type,
        load_cam_names=camera_names),
    dict(
        type='FisheyePhotoMetricDistortion',
        img_key=cam_type,
        brightness_delta=24,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=12),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectRangeFilter', point_cloud_range=[-48, -48, -5, 48, 48, 5]),
    dict(
        type='Fisheye3DODPackDetInputs',
        keys=[cam_type, 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'cam2lidar', 'lidar2cam', 'lidar2img',
            'box_type_3d', 'sample_idx', 'token',
        ],
        input_img_keys=[cam_type]),
]

val_pipeline = [
    dict(
        type='LoadFisheye3DODMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        load_cam_type=cam_type,
        load_cam_names=camera_names),
    dict(
        type='Fisheye3DODPackDetInputs',
        keys=[cam_type, 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'cam2lidar', 'lidar2cam', 'lidar2img',
            'box_type_3d', 'sample_idx', 'token',
        ],
        input_img_keys=[cam_type]),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=0.05,
        dataset=dict(
            type='Fisheye3DODDataset',
            data_root=data_root,
            ann_file='caliper_annotations/fisheye3dod_infos_train.pkl',
            pipeline=train_pipeline,
            test_mode=False,
            filter_empty_gt=False,
            metainfo=dict(classes=classes))))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='Fisheye3DODDataset',
        data_root=data_root,
        ann_file='caliper_annotations/fisheye3dod_infos_val.pkl',
        pipeline=val_pipeline,
        test_mode=True,
        load_eval_anns=True,
        filter_empty_gt=False,
        metainfo=dict(classes=classes)))

test_dataloader = val_dataloader

model = dict(view_transform=dict(
    image_size=[1536, 1920],
    kb4_calibration_path=(
        'data/Custom/BAG_2026_09_30_10_10_54_converted/calibration.json'),
    kb4_camera_frames=('ARC_F', 'ARC_SL', 'ARC_SR', 'ARC_R')))

# Effective batch size is four with the inherited four-step accumulation.
optim_wrapper = dict(optimizer=dict(lr=5e-5))
auto_scale_lr = dict(enable=False, base_batch_size=8)
load_from = 'projects/Fisheye3DOD/weights/fisheye_bevdet.pth'
work_dir = 'work_dirs/custom_fisheye_bevdet_train_caliper_native'
