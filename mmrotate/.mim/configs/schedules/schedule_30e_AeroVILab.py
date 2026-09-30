# evaluation
evaluation = dict(interval=1, metric='mAP', save_best='mAP')
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22, 27])
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
