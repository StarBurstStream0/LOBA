# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from mmcv.ops import batched_nms
from mmdet.core import anchor_inside_flags, unmap, multi_apply

from mmrotate.core import obb2xyxy
from ..builder import ROTATED_HEADS
from .oriented_rpn_head import OrientedRPNHead

from ...utils import get_root_logger
from collections import OrderedDict
from mmcv.runner import _load_checkpoint, force_fp32
from torchvision.ops import DeformConv2d
import torch.nn.functional as F

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        # 1. 定义一个普通卷积层，用于学习偏移量 (offsets)
        # 输出通道数为 2 * kernel_size * kernel_size，因为每个采样点需要 (x, y) 两个偏移值
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size, 
            kernel_size=kernel_size, 
            padding=padding
        )
        
        # 2. 定义可变形卷积层
        self.deform_conv = DeformConv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )

    def forward(self, x):
        # 前向传播时，先通过 offset_conv 生成偏移量
        offsets = self.offset_conv(x)
        # 然后将输入和偏移量一起传入 deform_conv 进行计算
        output = self.deform_conv(x, offsets)
        return output
    
class DeformableConv2d_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, off_channels=None):
        super().__init__()
        # 1. 定义一个普通卷积层，用于学习偏移量 (offsets)
        # 输出通道数为 2 * kernel_size * kernel_size，因为每个采样点需要 (x, y) 两个偏移值
        self.offset_conv = nn.Conv2d(
            off_channels, 
            2 * kernel_size * kernel_size, 
            kernel_size=kernel_size, 
            padding=padding
        )
        
        # 2. 定义可变形卷积层
        self.deform_conv = DeformConv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
        
        # 3. 参数初始化
        self.init_weights()
        
    def init_weights(self):
        # 偏移卷积初始化：权重和偏置必须全部置 0 
        # 这样初始 offset 为 0，模型行为等同于标准卷积，训练更稳定 
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        # 主卷积层初始化：推荐使用 Kaiming 初始化
        nn.init.kaiming_normal_(self.deform_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.deform_conv.bias is not None:
            nn.init.constant_(self.deform_conv.bias, 0)
        
    def forward(self, x, offset):
        # 前向传播时，先通过 offset_conv 生成偏移量
        offsets = self.offset_conv(offset)
        # 然后将输入和偏移量一起传入 deform_conv 进行计算
        output = self.deform_conv(x, offsets)
        return output

@ROTATED_HEADS.register_module()
class DefOrientedRPNHead(OrientedRPNHead):
    """Oriented RPN head for Oriented R-CNN."""

    def _init_layers(self):
        """Initialize layers of the head."""
        # self.rpn_conv = nn.Conv2d(
        #     self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_conv = DeformableConv2d(self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 6, 1)
        
    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        # print('x: ', x.size())
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred
    
@ROTATED_HEADS.register_module()
class DefOrientedRPNHead_v2(OrientedRPNHead):
    """Oriented RPN head for Oriented R-CNN."""

    def _init_layers(self):
        """Initialize layers of the head."""
        # self.rpn_conv = nn.Conv2d(
        #     self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_conv = DeformableConv2d_v2(self.in_channels, self.feat_channels, 3, padding=1, off_channels=128)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 6, 1)
        
    def forward_single(self, x, offsets):
        """Forward feature map of a single scale level."""
        # print('x: ', x)
        # print('x: ', x.size())
        # print('offsets', offsets.size())
        x = self.rpn_conv(x, offsets)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred
    
    def forward(self, feats, offsets):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # print('feats: ', len(feats))
        # print('offsets: ', len(offsets))
        return multi_apply(self.forward_single, feats, offsets)
    
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      refpoints=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x, refpoints)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list
        
    def simple_test_rpn(self, x, refpoints, img_metas):
        """Test without augmentation, only for ``RPNHead`` and its variants,
        e.g., ``GARPNHead``, etc.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        rpn_outs = self(x, refpoints)
        proposal_list = self.get_bboxes(*rpn_outs, img_metas=img_metas)
        return proposal_list