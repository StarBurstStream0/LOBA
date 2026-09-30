optimizer = dict(
                # _delete_=True, 
                type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                constructor='LayerDecayOptimizerConstructor', 
                paramwise_cfg=dict(
                    num_layers=12, 
                    layer_decay_rate=0.75,
                    custom_keys={
                        'bias': dict(decay_multi=0.),
                        'pos_embed': dict(decay_mult=0.),
                        'relative_position_bias_table': dict(decay_mult=0.),
                        'norm': dict(decay_mult=0.),
                        "rel_pos_h": dict(decay_mult=0.),
                        "rel_pos_w": dict(decay_mult=0.),
                        }
                        )
                )
optimizer_config = dict(grad_clip=None)