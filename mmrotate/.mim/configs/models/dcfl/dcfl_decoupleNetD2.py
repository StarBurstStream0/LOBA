angle_version = 'le135'
pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth'
model = dict(
    # type='RotatedRetinaNet',
    backbone=dict(
        type='DecoupleNet',
        in_chans=3,
        embed_dim=64,
        depths=(1, 6, 6, 2),
        att_kernel=(9, 9, 9, 9),
        drop_path_rate=0.1,
        fork_feat=True,
        init_cfg=dict(type='Pretrained',
            checkpoint="/home/zzc/archive/weights/pretrained/DecoupleNet/DecoupleNet_D2.pth"),
        pretrained=None),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RDCFLHead',
        num_classes=16,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        dcn_assign = True,
        dilation_rate = 3,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=1, 
            ratios=[1.0], 
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(
            type='RotatedIoULoss',
            loss_weight=1.0)), 
    train_cfg=dict(
        assigner=dict(
            type='C2FAssigner',
            ignore_iof_thr=-1,
            gpu_assign_thr= 1024,
            iou_calculator=dict(type='RBboxMetrics2D'),
            assign_metric='gjsd',
            topk=16,
            topq=12,
            constraint='dgmm',
            gauss_thr=0.6),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05, 
        nms=dict(iou_thr=0.4), 
        max_per_img=2000))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
checkpoint_config = dict(interval=4)
evaluation = dict(interval=4, metric='mAP')


# work_dir = '/home/zzc/archive/MMRotate_series/dotav1.5_test_dcfl_r50_40e/r1'
work_dir = '/home/zzc/archive/MMRotate_series/dotav1.5_test_dcfl_r50_40e/r2_FalseBN'

'''
CUDA_VISIBLE_DEVICES=1 python ./tools/train.py ./configs/dcfl/dotav1.5_test_dcfl_r50_40e.py

'''