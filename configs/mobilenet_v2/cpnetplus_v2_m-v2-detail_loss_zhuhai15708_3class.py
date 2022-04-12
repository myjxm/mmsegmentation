# model settings
_base_ = [
    '../_base_/datasets/zhuhai15708_3class.py', '../_base_/default_runtime.py',
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
        out_indices=(0, 1, 2, 3, 4, 5, 6)),
    decode_head=dict(
        type='CPHeadPlus_V2',
        in_channels=320,
        in_index=6,
        channels=80, #最后分类卷积前的通道数，cpnet与prior_channels一致
        prior_channels=80,
        prior_size=64,  #prior_size与backbone输出特征图大小一致
        am_kernel_size=6,
        aggress_dilation=2,
        groups=1,
#        drop_out_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        c0_in_channels=24,
        c0_channels=24,  # decode对第一层降维后的通道数
        c1_in_channels=-1, #最后一层拼接cpnet输出，若取消输入-1
        c1_channels=0, #decode对最后层降维后的通道数，最后一层拼接cpnet输出，若取消输入-1
        detail_index=1,
        detail_channels=24,
        arm_channels=-1,#不使用输入小于0的值
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, alpha=1, gamma=0,use_pixel_weight=True,pa=2,only_block=True),
        loss_prior_decode=dict(type='AffinityLoss', loss_weight=1.0),
        loss_detail_loss=dict(type='DetailAggregateLoss', loss_weight=1.0, use_x8=True,only_x1=True),
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=3,
        channels=24,
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
