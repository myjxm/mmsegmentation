_base_ = [
    '../_base_/datasets/zhuhai12749_3class.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
# Re-config the data sampler.
#data = dict(samples_per_gpu=4, workers_per_gpu=4)
runner = dict(type='IterBasedRunner', max_iters=320000)

# model settings
norm_cfg = dict(type='BN', eps=0.001, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://contrib/mobilenet_v3_large',
    backbone=dict(
        type='MobileNetV3',
        arch='large',
        out_indices=(6,10,16),
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='CPHeadPlus_V2',
        in_channels=960,
        in_index=2,
        channels=80,  # 最后分类卷积前的通道数，cpnet与prior_channels一致
        prior_channels=80,
        prior_size=64,  # prior_size与backbone输出特征图大小一致
        am_kernel_size=6,
        aggress_dilation=2,
        groups=1,
        #        drop_out_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        c0_in_channels=40,  # detail——loss那层的通道数需要和detail_channels一致，如果需要和最终拼接就输入24，如果不拼接就输出-1
        c0_channels=40,  # decode对第一层降维后的通道数
        c1_in_channels=-1,  # 最后一层拼接cpnet输出，若取消输入-1
        c1_channels=0,  # decode对最后层降维后的通道数，最后一层拼接cpnet输出，若取消输入-1
        detail_index=0,  # detail_loss对应的层数
        detail_channels=40,
        arm_channels=-1,  # 不使用输入小于0的值
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, alpha=1, gamma=0, use_pixel_weight=True, pa=2,
            only_block=True, sky=False, sky_a=1, denominator=8),
        loss_prior_decode=dict(type='AffinityLoss', loss_weight=1.0),
        loss_detail_loss=dict(type='DetailAggregateLoss', loss_weight=1.0, use_x8=True, only_x1=False, use_x2=False,
                              only_x8=False, use_x1_x8=False),
        seg_head=True,
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=80,
        in_index=1,
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


