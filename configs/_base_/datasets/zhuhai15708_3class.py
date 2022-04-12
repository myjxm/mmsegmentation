# dataset settings
dataset_type = 'Zhuhai15708_3class'
data_root = '/home/home2/zrd/data/datasets/zhuhai15708_3class'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),  #img_scale是图片的大小
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),  #cat_max_ratio指单个类别可以在裁剪后的图片中占有的比率
    dict(type='RandomFlip', prob=0.5),   ##随机反转，看随机数是否小于Prob,小于就翻，默认 水平翻转
    dict(type='RandomRotate',prob=0.5,degree=5),  ##随机旋转
    dict(type='PhotoMetricDistortion'), ##做的光度畸变
    dict(type='Normalize', **img_norm_cfg),  ##做的图像归一化
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [   #在训练的评估阶段也用这个
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            #dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),   #后加
            #dict(type='RandomFlip'),
            #dict(type='AdjustGamma',gamma=2.2),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        img_dir='image_train',
        ann_dir='mask_train',
        split='train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        img_dir='image_val',
        ann_dir='mask_val',
        split='val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        img_dir='image_test',
        ann_dir='mask_test',
        split='test.txt',
        pipeline=test_pipeline)
    )


