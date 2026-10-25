"""Fine-tune Fisheye BEVDet on LiDAR-verified custom pseudo-labels."""

_base_ = ['./fisheye_bevdet.py']

data_root = 'data/Custom/BAG_2026_09_30_10_10_54_converted/fisheye3dod_inference_2hz/'
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

# Do not use CBGSDataset here. Van and Bus are absent from the source proto,
# and this MMDetection3D version's CBGS wrapper misaligns ratios when a class
# has zero examples.
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='Fisheye3DODDataset',
        data_root=data_root,
        ann_file='groundtruth/fisheye3dod_infos_train.pkl',
        pipeline=train_pipeline,
        test_mode=False,
        filter_empty_gt=False,
        metainfo=dict(classes=classes)))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='Fisheye3DODDataset',
        data_root=data_root,
        ann_file='groundtruth/fisheye3dod_infos_val.pkl',
        pipeline=val_pipeline,
        test_mode=True,
        load_eval_anns=True,
        filter_empty_gt=False,
        metainfo=dict(classes=classes)))

test_dataloader = val_dataloader

# Mixed-precision training with gradient accumulation
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.01,
    ),
    loss_scale='dynamic',
    accumulative_counts=4,
    clip_grad=dict(max_norm=35, norm_type=2),
)

# Disable automatic LR scaling while testing
auto_scale_lr = dict(enable=False, base_batch_size=8)

model = dict(view_transform=dict(
    ocam_path='data/Fisheye3DODdataset/calib_results.txt'))
load_from = 'projects/Fisheye3DOD/weights/fisheye_bevdet.pth'
work_dir = 'work_dirs/custom_fisheye_bevdet_train'
