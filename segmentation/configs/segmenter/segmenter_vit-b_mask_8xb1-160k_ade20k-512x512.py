_base_ = [
    '../_base_/models/segmenter_vit-b16-224_mask.py',
    '../_base_/datasets/ade20k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
backbone = dict(training=False)
model = dict(data_preprocessor=data_preprocessor, backbone=backbone)
optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=8)
val_dataloader = dict(batch_size=1)
model_wrapper = dict(find_unused_parameters=True)