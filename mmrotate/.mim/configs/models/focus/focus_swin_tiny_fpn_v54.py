angle_version = 'le90'
pretrained = '/home/zzc/archive/weights/pretrained/swin/swin_tiny_patch4_window7_224.pth'
model = dict(
    # type='OrientedRCNN',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5
    ),
    rpn_head=dict(
        type='QualityOrientedRPNHead',
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        scale_angle=False,
        use_fpn_feature=True,
        enable_sa=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.25),
        bbox_coder=dict(
            type='RotatedDistancePointBBoxCoder', angle_version=angle_version),
        loss_bbox=dict(type='PolyGIoULoss', loss_weight=0.25)
    ),
    roi_head=None,
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='RotatedATSSAssigner',
                            topk=9,
                            iou_calculator=dict(type='RBboxOverlaps2D'),
                            ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=None
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=None
    )
)
