angle_version = 'le90'
pretrained = '/home/zzc/archive/weights/pretrained/InternImage/internimage_t_1k_224.pth'
model = dict(
    # type='OrientedRCNN',
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
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
