"""Inference-only BEVDet config for the converted custom four-camera bag."""

_base_ = ['./fisheye_bevdet.py']

data_root = 'data/Custom/BAG_2026_09_30_10_10_54_converted/fisheye3dod_native_2hz/'
classes = ['Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist']
cam_type = 'cam_fisheye'

model = dict(view_transform=dict(
    image_size=[1536, 1920],
    kb4_calibration_path=(
        'data/Custom/BAG_2026_09_30_10_10_54_converted/calibration.json'),
    kb4_camera_frames=('ARC_F', 'ARC_SL', 'ARC_SR', 'ARC_R')))

test_pipeline = [
    dict(
        type='LoadFisheye3DODMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        load_cam_type=cam_type,
        load_cam_names=[
            'fisheye_camera_front', 'fisheye_camera_left',
            'fisheye_camera_right', 'fisheye_camera_rear',
        ]),
    dict(
        type='Fisheye3DODPackDetInputs',
        keys=[cam_type],
        meta_keys=[
            'cam2img', 'cam2lidar', 'lidar2img', 'lidar2cam',
            'sample_idx', 'token', 'img_path', 'lidar_path',
            'num_pts_feats', 'box_type_3d',
        ],
        input_img_keys=[cam_type]),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='Fisheye3DODDataset',
        data_root=data_root,
        ann_file='fisheye3dod_infos_test.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        load_eval_anns=False,
        filter_empty_gt=False,
        metainfo=dict(classes=classes)))

test_evaluator = dict(
    type='DumpResults',
    out_file_path='work_dirs/custom_fisheye_bevdet/predictions.pkl')

work_dir = 'work_dirs/custom_fisheye_bevdet'
