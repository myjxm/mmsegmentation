# model settings
_base_ = [
    '../_base_/datasets/combine_zrdy.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
#norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN',requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='mmcls://mobilenet_v2',
    backbone=dict(
        #_delete_=True,
        type='MobileNetV2',
        widen_factor=1.,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6)),
    decode_head=dict(
        type='CPHeadPlus',
        in_channels=320,
        in_index=3,
        channels=80, #最后分类卷积前的通道数，cpnet与prior_channels一致
        prior_channels=80,
        prior_size=64,  #prior_size与backbone输出特征图大小一致
        am_kernel_size=11,
        groups=1,
#        drop_out_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        c1_in_channels=24,
        c1_channels=4, #decode对第一层降维后的通道数
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_prior_decode=dict(type='AffinityLoss', loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=96,
        in_index=2,
        channels=24,
        num_convs=1,
        concat_input=False,
#        drop_out_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
