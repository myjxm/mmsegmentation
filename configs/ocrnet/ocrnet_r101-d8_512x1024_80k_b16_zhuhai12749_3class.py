_base_ = [
    '../_base_/models/ocrnet_r50-d8_3class.py', '../_base_/datasets/zhuhai12749_3class.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
optimizer = dict(lr=0.02)
lr_config = dict(min_lr=2e-4)
