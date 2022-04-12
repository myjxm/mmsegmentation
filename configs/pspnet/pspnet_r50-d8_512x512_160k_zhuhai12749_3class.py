_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/zhuhai12749_3class.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='BN',requires_grad=True)
model = dict(
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=3))
