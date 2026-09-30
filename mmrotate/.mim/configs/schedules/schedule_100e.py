# evaluation
evaluation = dict(interval=100, metric='mAP')
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 3,
    step=[60, 80, 95])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=2)
