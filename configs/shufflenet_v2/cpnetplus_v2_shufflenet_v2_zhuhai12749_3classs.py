_base_ = [
    '../_base_/datasets/zhuhai12749_3class.py', '../_base_/default_runtime.py',
    '../_base_/schedules/shufflenet_v2.py'
]

# model settings
norm_cfg = dict(type='BN', eps=0.001, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(type='ShuffleNetV2', widen_factor=1.0,out_indices=(0,1,2,3)),
    decode_head=dict(
        type='CPHeadPlus_V2',
        in_channels=464,
        in_index=2,
        channels=116,  # 最后分类卷积前的通道数，cpnet与prior_channels一致
        prior_channels=116,
        prior_size=16,  # prior_size与backbone输出特征图大小一致
        am_kernel_size=6,
        aggress_dilation=2,
        groups=1,
        #        drop_out_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        c0_in_channels=116,  # detail——loss那层的通道数需要和detail_channels一致，如果需要和最终拼接就输入24，如果不拼接就输出-1
        c0_channels=24,  # decode对第一层降维后的通道数
        c1_in_channels=1024,  # 最后一层拼接cpnet输出，若取消输入-1
        c1_channels=512,  # decode对最后层降维后的通道数，最后一层拼接cpnet输出，若取消输入-1
        detail_index=0,  # detail_loss对应的层数
        detail_channels=116,
        arm_channels=-1,  # 不使用输入小于0的值
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, alpha=1, gamma=0, use_pixel_weight=True, pa=2,
            only_block=True, sky=False, sky_a=1, denominator=8),
        loss_prior_decode=dict(type='AffinityLoss', loss_weight=1.0),
        loss_detail_loss=dict(type='DetailAggregateLoss', loss_weight=1.0, use_x8=True, only_x1=False, use_x2=False,
                              only_x8=False, use_x1_x8=False),
        seg_head=True,
        concat_x=False,
        out_index=3,#c1_channel拼接层
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=232,
        in_index=1,
        channels=72,
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