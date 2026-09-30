# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import ROTATED_DETECTORS, build_head
from .two_stage import RotatedTwoStageDetector


@ROTATED_DETECTORS.register_module()
class OrientedRCNN_self_v1(RotatedTwoStageDetector):
    """Implementation of `Oriented R-CNN for Object Detection.`__

    __ https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf  # noqa: E501, E261.
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(OrientedRCNN_self_v1, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmrotate/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 6).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                    #   img,
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels,
                    expected_stages=None,
                    gt_bboxes_ignore=None,
                    gt_masks=None,
                    proposals=None,
                    anchors=None,
                    with_neck=False,
                    **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # x = self.extract_feat(img)
        if expected_stages == None:
            x = self.extract_feat(x)
        elif expected_stages == '0':
            if with_neck:
                x = self.neck(x)
        else:
            [final, x_out, out_hw_shape] = x
            _, x, _ = self.backbone.forward_stages(final, outs=x_out, hw_shape=out_hw_shape, expected_stages=expected_stages)
            if with_neck:
                x = self.neck(x)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                anchors=anchors,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses
    
    def simple_test(self, x, img_metas, refined_anchors, proposals=None, rescale=False, with_neck=False, expected_stages=None):
        """Test without augmentation."""

        # assert self.with_bbox, 'Bbox head must be implemented.'
        # x = self.extract_feat(img)
        if expected_stages == None:
            x = self.extract_feat(x)
        elif expected_stages == '0':
            if with_neck:
                x = self.neck(x)
        else:
            [final, x_out, out_hw_shape] = x
            _, x, _ = self.backbone.forward_stages(final, outs=x_out, hw_shape=out_hw_shape, expected_stages=expected_stages)
            if with_neck:
                x = self.neck(x)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(refined_anchors, x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
        
@ROTATED_DETECTORS.register_module()
class OrientedRCNN_self_v2(RotatedTwoStageDetector):
    """Implementation of `Oriented R-CNN for Object Detection.`__

    __ https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf  # noqa: E501, E261.
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 rrn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(OrientedRCNN_self_v2, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        if rrn_head is not None:
            rrn_train_cfg = train_cfg.rrn if train_cfg is not None else None
            rrn_head_ = rrn_head.copy()
            rrn_head_.update(train_cfg=rrn_train_cfg)
            self.rrn_head = build_head(rrn_head_)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmrotate/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 6).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                    img,
                    img_metas,
                    gt_bboxes,
                    gt_labels,
                    gt_bboxes_ignore=None,
                    gt_masks=None,
                    proposals=None,
                    **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        rrn_losses, anchors, x = self.rrn_head.forward_train_feats(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=gt_bboxes_ignore,
            # proposal_cfg=proposal_cfg,
            **kwargs)
        losses.update(rrn_losses)
            
        # RRN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                            self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            anchors,
            gt_labels=None,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg,
            **kwargs)
        losses.update(rpn_losses)

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                gt_bboxes, gt_labels,
                                                gt_bboxes_ignore, gt_masks,
                                                **kwargs)
        losses.update(roi_losses)

        return losses
    
    def simple_test(self, x, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        # assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        anchors = self.rpn_head.simple_test_rpn(x, img_metas)
        proposal_list = self.rrn_head.simple_test_rrn(anchors, x, img_metas)

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)