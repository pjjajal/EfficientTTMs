# model settings
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained="/scratch/gilbreth/pjajal/checkpoints-vttm-new/2024-08-24-232314.027290-vittm-base-im21k-lin-ls4-mlp-16-32-0/best_performing.pth",
    backbone=dict(
        type='ViTTM',
        img_size=(504, 504),
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=[11],
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        global_pool="avg",
        memory_ps=28,
        process_ps=28,
        rw_head_type="lin",
        fusion_type="residual",
        process_embedder_type="patch",
        dynamic_img_size=True,
        dynamic_img_pad=False,
        latent_size_scale=4,
        training=False,
        num_classes = 1000,
        init_cfg=dict(type='Pretrained', checkpoint="/home/pjajal/ema_384.pth"),
        # init_cfg=dict(type='Pretrained', checkpoint="/depot/yunglu/data/pj/best-weights/vittm-b-28/fixed-ema-vittm-b-28-im21k-ft-1k-nov18.pth"),
        # init_cfg=dict(type='Pretrained', checkpoint="/home/pjajal/fixed-ema-vittm-b-28-im21k-ft-1k-82-4.pth"),
        # init_cfg=dict(type='Pretrained', checkpoint="/scratch/gilbreth/pjajal/checkpoints-vttm-new/2024-08-30-140149.930331-vittm-base-im21k-lin-ls4-28-28/best_performing.pth"),
        # init_cfg=dict(type='Pretrained', checkpoint="/home/pjajal/fixed-ema-vittm-b-28-im21k-ft-1k-82.pth"),
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=768 * 2,
        channels=768,
        num_classes=150,
        num_layers=2,
        num_heads=12,
        embed_dims=768,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(504, 504), stride=(480, 480)),
)