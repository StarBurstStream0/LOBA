optimizer = dict(
    type='AdamW',
    lr=1e-6,
    betas=(0.9, 0.999), # consistant with the default setting of AdamW in PyTorch
    weight_decay=1e-4
)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))