_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8-singlecpu.py', '../_base_/datasets/zhuhai15708_3class.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='mmcls://mobilenet_v2',
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        widen_factor=1.,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6)),
    decode_head=dict(in_channels=320, c1_in_channels=24,num_classes=3),
    auxiliary_head=dict(in_channels=96,num_classes=3),
)
1