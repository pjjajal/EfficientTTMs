_base_ = [
    "../_base_/models/upernet_vittm-b16.py",
    "../_base_/datasets/ade20k.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150),
    backbone=dict(uper=True, training=False)
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    # optimizer = dict(type='SGD', lr=0.0035, momentum=0.9, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "memory_pos_embed": dict(decay_mult=0.0),
            "process_pos_embed": dict(decay_mult=0.0),
        },
    ),
    accumulative_counts=2,
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type="PolyLR", power=1.0, eta_min=0.0, by_epoch=False),
]

train_dataloader = dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=1,
)

model_wrapper = dict(find_unused_parameters=True)