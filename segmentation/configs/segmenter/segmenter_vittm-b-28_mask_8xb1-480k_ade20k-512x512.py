_base_ = [
    "../_base_/models/segmenter_vittm-b28_mask.py",
    "../_base_/datasets/ade20k_28.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]
crop_size = (504, 504)
data_preprocessor = dict(size=crop_size)
backbone = dict(training=False)
model = dict(data_preprocessor=data_preprocessor, backbone=backbone)
# optimizer = dict(lr=0.0040, weight_decay=0.0)
# optimizer = dict(lr=0.0035, weight_decay=0.0)
# optimizer = dict(lr=0.0005, weight_decay=0.0005)
optimizer = dict(lr=0.0005, weight_decay=0.0000)
# optimizer = dict(lr=0.001, weight_decay=0.00)
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=optimizer,
    # clip_grad=dict(max_norm=10, norm_type=2, _delete_=True)
)
train_dataloader = dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=16
)
val_dataloader = dict(batch_size=1)
custom_hooks = [dict(type='EMAHook', ema_type='StochasticWeightAverage')]
model_wrapper = dict(find_unused_parameters=True)