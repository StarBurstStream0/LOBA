optimizer = dict(
    type='AdamW',
    lr=0.0002,
    betas=(0.9, 0.999), # consistant with the default setting of AdamW in PyTorch
    weight_decay=0.05
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))