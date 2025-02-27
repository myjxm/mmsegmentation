# model settings
_base_ = [
    '../_base_/datasets/mastr1325.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
#norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN',requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='CPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        prior_channels=512,
        prior_size=64,
        am_kernel_size=11,
        groups=1,
#        drop_out_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_prior_decode=dict(type='AffinityLoss', loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
#        drop_out_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
