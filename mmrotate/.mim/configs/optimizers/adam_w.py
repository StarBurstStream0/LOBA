optimizer = dict(
    # _delete_=True, 
    type='AdamW', lr=0.0001, weight_decay=0.05
    # constructor='CustomLayerDecayOptimizerConstructor',
    # constructor='LayerDecayOptimizerConstructor',
    # paramwise_cfg=dict(num_layers=30, layer_decay_rate=1.0,
    #                     depths=[4, 4, 18, 4])
)
optimizer_config = dict(grad_clip=None)