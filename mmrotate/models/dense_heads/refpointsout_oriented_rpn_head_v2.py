# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from mmcv.ops import batched_nms
from mmdet.core import anchor_inside_flags, unmap, images_to_levels, multi_apply

from mmrotate.core import obb2xyxy
from ..builder import ROTATED_HEADS
from .rotated_rpn_head import RotatedRPNHead

from ...utils import get_root_logger
from collections import OrderedDict
from mmcv.runner import _load_checkpoint, force_fp32


def calculate_horizontal_iou(bboxes1, bboxes2):
    """
    计算两个水平框集合之间的精确IoU
    bboxes1: [N, 4] -> [x1, y1, x2, y2]
    bboxes2: [M, 4] -> [x1, y1, x2, y2]
    返回: [N, M] 的IoU矩阵
    """
    # Expand dimensions for broadcasting
    bboxes1 = bboxes1.unsqueeze(1)  # [N, 1, 4]
    bboxes2 = bboxes2.unsqueeze(0)  # [1, M, 4]
    
    # Calculate intersection coordinates
    inter_x1 = torch.max(bboxes1[..., 0], bboxes2[..., 0])  # [N, M]
    inter_y1 = torch.max(bboxes1[..., 1], bboxes2[..., 1])  # [N, M]
    inter_x2 = torch.min(bboxes1[..., 2], bboxes2[..., 2])  # [N, M]
    inter_y2 = torch.min(bboxes1[..., 3], bboxes2[..., 3])  # [N, M]
    
    # Calculate intersection area
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)  # [N, M]
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)  # [N, M]
    inter_area = inter_w * inter_h  # [N, M]
    
    # Calculate areas of both sets of boxes
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])  # [N, 1]
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])  # [1, M]
    
    # Calculate union area
    union_area = area1 + area2 - inter_area  # [N, M]
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)  # [N, M] 添加小值防止除零
    
    return iou


@ROTATED_HEADS.register_module()
class RPO_OrientedRPNHead(RotatedRPNHead):
    """Oriented RPN head for Oriented R-CNN."""
    
    def __init__(self,
                 use_refpoints=True,
                 use_levels_for_refpoints=1,
                 minus_refpoints=False,
                 **kwargs):
        super(RPO_OrientedRPNHead, self).__init__(**kwargs)
        self.use_refpoints = use_refpoints
        self.use_levels_for_refpoints = use_levels_for_refpoints
        self.minus_refpoints = minus_refpoints
    
    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 6, 1)
        
    def load_weights(self):
        logger = get_root_logger()
        if 'checkpoint' not in self.init_cfg:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
        else:
            # assert 'checkpoint' in self.init_cfg, f'Only support ' \
            #                                       f'specify `Pretrained` in ' \
            #                                       f'`init_cfg` in ' \
            #                                       f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
                
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('rpn_head.'):
                    state_dict[k[9:]] = v
            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # load state_dict
            self.load_state_dict(state_dict, False)
            
    def simple_test_rpn(self, x, img_metas):
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
        
        features, points = x
        
        ###################################################################
        ### TODO: get xyxy offsets
        if self.use_refpoints:
            multi_level_xyxy = []
            for single_level_points in points:
                bs, points_num = single_level_points.size()[:2]
                single_level_points = single_level_points.repeat(1, 1, 2).unsqueeze(2).repeat(1, 1, 3, 1).view(bs, points_num*3, 2*2)
                multi_level_xyxy.append(single_level_points)
        else:
            # print('Not using refpoints, multi_level_xyxy will be set to None')
            multi_level_xyxy = None
        ###################################################################
        
        rpn_outs = self(features)
        proposal_list = self.get_bboxes(*rpn_outs, multi_level_xyxy=multi_level_xyxy, img_metas=img_metas)
        return proposal_list
        
    def get_anchors(self, multi_level_xyxy, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        
        # print('multi_level_xyxy: ', multi_level_xyxy[0][:10])
        # for single_level_xyxy in multi_level_xyxy:
        #     print('single_level_xyxy: ', single_level_xyxy.size())
        
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        # print('featmap_sizes: ', featmap_sizes)
        # print('multi_level_anchors: ', multi_level_anchors[0][:10])
        # for single_level_anchors in multi_level_anchors:
        #     print('single_level_anchors: ', single_level_anchors.size())
            
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
            
        if self.use_refpoints and multi_level_xyxy is not None:
            for img_id in range(num_imgs):
                for level_id in range(len(multi_level_xyxy)):
                    if level_id < self.use_levels_for_refpoints:
                        if self.minus_refpoints:
                            anchor_list[img_id][level_id] -= multi_level_xyxy[level_id][img_id]
                        else:
                            anchor_list[img_id][level_id] += multi_level_xyxy[level_id][img_id]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             multi_level_xyxy,
             gt_bboxes,
             img_metas,
             gt_labels=None,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        # print('featmap_sizes: ', featmap_sizes)

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            multi_level_xyxy, featmap_sizes, img_metas, device=device)
        # print('anchor_list[0][0]: ', anchor_list[0][0][0:10])
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        
        # 计算每个GT分配到的正样本的平均数量和质量（IoU）
        gt_assignment_stats = self.compute_gt_assignment_stats(
            concat_anchor_list, gt_bboxes, img_metas)
        
        loss_dict = dict(
            loss_rpn_cls=losses_cls, 
            loss_rpn_bbox=losses_bbox,
            gt_avg_pos_count=gt_assignment_stats['avg_pos_count'],
            gt_avg_iou_quality=gt_assignment_stats['avg_iou_quality']
        )
        return loss_dict

    def compute_gt_assignment_stats(self, concat_anchor_list, gt_bboxes_list, img_metas):
        """
        计算每个GT分配到的正样本的平均数量和质量（IoU）

        Args:
            concat_anchor_list (list[Tensor]): Concatenated anchors for each image
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
            img_metas (list[dict]): Meta info of each image

        Returns:
            dict: 包含平均正样本数量和平均IoU质量的字典
        """
        all_avg_pos_counts = []
        all_avg_iou_qualities = []
        
        for i, (anchors, gt_bboxes) in enumerate(zip(concat_anchor_list, gt_bboxes_list)):
            if gt_bboxes.numel() == 0 or anchors.numel() == 0:
                continue
                
            # 将旋转框转换为水平框用于精确IoU计算
            gt_hbboxes = obb2xyxy(gt_bboxes, self.version)
            anchor_hbboxes = anchors[:, :4]  # 假设锚框已经是水平框格式
            
            # 计算精确的水平框IoU
            iou_matrix = calculate_horizontal_iou(anchor_hbboxes, gt_hbboxes)  # [num_anchors, num_gt]
            
            # 获取分配结果
            assign_result = self.assigner.assign(
                anchors, gt_hbboxes, None, None)
            assigned_gt_inds = assign_result.gt_inds  # [num_anchors]
            
            # 统计每个GT的正样本数量和IoU质量
            num_gt = gt_bboxes.size(0)
            if num_gt == 0:
                continue
                
            pos_mask = assigned_gt_inds > 0  # 正样本掩码
            if not pos_mask.any():
                continue
                
            # 获取正样本对应的IoU
            pos_assigned_gt_inds = assigned_gt_inds[pos_mask] - 1  # 转换为0索引
            pos_ious = iou_matrix[pos_mask, pos_assigned_gt_inds]
            
            # 按照GT索引分组统计
            avg_pos_counts = []
            avg_iou_qualities = []
            
            for gt_idx in range(num_gt):
                gt_pos_mask = pos_assigned_gt_inds == gt_idx
                if gt_pos_mask.sum() > 0:
                    gt_pos_ious = pos_ious[gt_pos_mask]
                    avg_pos_count = gt_pos_ious.numel()
                    avg_iou_quality = gt_pos_ious.mean()
                else:
                    avg_pos_count = 0
                    avg_iou_quality = 0.0
                
                avg_pos_counts.append(avg_pos_count)
                avg_iou_qualities.append(avg_iou_quality)
            
            if avg_pos_counts:
                all_avg_pos_counts.extend(avg_pos_counts)
                all_avg_iou_qualities.extend(avg_iou_qualities)
        
        # 计算整体平均值
        if all_avg_pos_counts:
            overall_avg_pos_count = torch.tensor(all_avg_pos_counts).float().mean()
            overall_avg_iou_quality = torch.tensor(all_avg_iou_qualities).float().mean()
        else:
            overall_avg_pos_count = torch.tensor(0.0)
            overall_avg_iou_quality = torch.tensor(0.0)
        
        return {
            'avg_pos_count': overall_avg_pos_count,
            'avg_iou_quality': overall_avg_iou_quality
        }

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
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
        features, points = x
        # print('features: ', len(features))
        # print('points: ', len(points))
        
        ###################################################################
        ### TODO: get xyxy offsets
        if self.use_refpoints and points is not None:
            multi_level_xyxy = []
            for single_level_points in points:
                bs, points_num = single_level_points.size()[:2]
                single_level_points = single_level_points.repeat(1, 1, 2).unsqueeze(2).repeat(1, 1, 3, 1).view(bs, points_num*3, 2*2)
                multi_level_xyxy.append(single_level_points)
        else:
            # print('Not using refpoints, multi_level_xyxy will be set to None')
            multi_level_xyxy = None
        ###################################################################
        
        outs = self(features)
        cls_scores, bbox_preds = outs
        losses = self.loss(cls_scores=cls_scores,
                           bbox_preds=bbox_preds,
                           multi_level_xyxy=multi_level_xyxy,
                           gt_bboxes=gt_bboxes,
                           gt_labels=gt_labels,
                           img_metas=img_metas,
                           gt_bboxes_ignore=gt_bboxes_ignore)
        # if gt_labels is None:
        #     loss_inputs = outs + (points, gt_bboxes, img_metas)
        # else:
        #     loss_inputs = outs + (points, gt_bboxes, gt_labels, img_metas)
        # losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, multi_level_xyxy=multi_level_xyxy, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   multi_level_xyxy,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert with_nms, '``with_nms`` in RPNHead should always True'
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        result_list = []
        for img_id, _ in enumerate(img_metas):
            
            # print('num_levels: ', num_levels)
            # print('multi_level_xyxy: ', len(multi_level_xyxy))
            ###################################################################################
            if multi_level_xyxy is not None and self.use_refpoints:
                mlvl_anchors_off = []
                for i in range(num_levels):
                    if i < self.use_levels_for_refpoints:
                        if self.minus_refpoints:
                            mlvl_anchors_off.append(mlvl_anchors[i] - multi_level_xyxy[i][img_id])
                        else:
                            mlvl_anchors_off.append(mlvl_anchors[i] + multi_level_xyxy[i][img_id])
                    else:
                        mlvl_anchors_off.append(mlvl_anchors[i])
            else:
                mlvl_anchors_off = mlvl_anchors
                # print('multi_level_xyxy is None, mlvl_anchors_off will be the same as mlvl_anchors')
            ###################################################################################

            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                mlvl_anchors_off, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list
    
###########################################################################################
    
    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)
    
    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (torch.Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape
                (num_anchors ,4)
            valid_flags (torch.Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (torch.Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (torch.Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (torch.Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple (list[Tensor]):

                - labels_list (list[Tensor]): Labels of each level
                - label_weights_list (list[Tensor]): Label weights of each \
                  level
                - bbox_targets_list (list[Tensor]): BBox targets of each level
                - bbox_weights_list (list[Tensor]): BBox weights of each level
                - num_total_pos (int): Number of positive samples in all images
                - num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        gt_hbboxes = obb2xyxy(gt_bboxes, self.version)

        assign_result = self.assigner.assign(
            anchors, gt_hbboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_hbboxes)

        if gt_bboxes.numel() == 0:
            sampling_result.pos_gt_bboxes = gt_bboxes.new(
                (0, gt_bboxes.size(-1))).zero_()
        else:
            sampling_result.pos_gt_bboxes = \
                gt_bboxes[sampling_result.pos_assigned_gt_inds, :]

        num_valid_anchors = anchors.shape[0]
        bbox_targets = anchors.new_zeros((anchors.size(0), 6))
        bbox_weights = anchors.new_zeros((anchors.size(0), 6))
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (torch.Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (torch.Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            anchors (torch.Tensor): Box reference for each scale level with
                shape (N, num_total_anchors, 4).
            labels (torch.Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (torch.Tensor): Label weights of each anchor with
                shape (N, num_total_anchors)
            bbox_targets (torch.Tensor): BBox regression targets of each anchor
            weight shape (N, num_total_anchors, 5).
            bbox_weights (torch.Tensor): BBox regression loss weights of each
                anchor with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple (torch.Tensor):

                - loss_cls (torch.Tensor): cls. loss for each scale level.
                - loss_bbox (torch.Tensor): reg. loss for each scale level.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 6)
        bbox_weights = bbox_weights.reshape(-1, 6)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 6)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

          Args:
            cls_scores (list[Tensor]): Box scores of all scale level
                each item has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas of all
                scale level, each item has shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx, _ in enumerate(cls_scores):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 6)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2]
            h = proposals[:, 3]
            valid_mask = (w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]
        if proposals.numel() > 0:
            hproposals = obb2xyxy(proposals, self.version)
            _, keep = batched_nms(hproposals, scores, ids, cfg.nms)
            dets = torch.cat([proposals, scores[:, None]], dim=1)
            dets = dets[keep]
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg.max_per_img]
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_rpn_anchors(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert with_nms, '``with_nms`` in RPNHead should always True'
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        result_list = []
        result_old_list = []
        for img_id, _ in enumerate(img_metas):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals, proposals_old = self._get_rpn_anchors_single(cls_score_list, bbox_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
            result_old_list.append(proposals_old)
        return result_list, result_old_list
    
    def _get_rpn_anchors_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

          Args:
            cls_scores (list[Tensor]): Box scores of all scale level
                each item has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas of all
                scale level, each item has shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx, _ in enumerate(cls_scores):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                # scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 6)
            anchors = mlvl_anchors[idx]
            # if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
            #     # sort is faster than topk
            #     ranked_scores, rank_inds = scores.sort(descending=True)
            #     topk_inds = rank_inds[:cfg.nms_pre]
            #     scores = ranked_scores[:cfg.nms_pre]
            #     rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
            #     anchors = anchors[topk_inds, :]
            # mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            # mlvl_valid_anchors.append(anchors)
            # print('anchors: ', anchors.size())
            # print('rpn_bbox_pred: ', rpn_bbox_pred.size())
            #######################################################################
            ### TODO: anchor points or obbs
            # proposals = self.bbox_coder.decode_anchor(
            #     anchors, rpn_bbox_pred, max_shape=img_shape)
            proposals = self.bbox_coder.decode(
                anchors, rpn_bbox_pred, max_shape=img_shape)[:, :4]
            #######################################################################
            
            # print('proposals: ', proposals.size())
            # print('anchors: ', anchors.size())
            scores = scores.unsqueeze(-1)
            # print('scores: ', scores.size())
            # print('max scores: ', scores.max())
            # print('minscores: ', scores.min())
            
            new_proposals = torch.where(scores>0.5, proposals, anchors).contiguous()
            
            mlvl_valid_anchors.append(new_proposals)
            # level_ids.append(
            #     scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        # scores = torch.cat(mlvl_scores)
        # anchors = torch.cat(mlvl_valid_anchors)
        # rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        # print('anchors: ', anchors.size())
        # print('rpn_bbox_pred: ', rpn_bbox_pred.size())
        # proposals = self.bbox_coder.decode_anchor(
        #     anchors, rpn_bbox_pred, max_shape=img_shape)
        # print('proposals: ', proposals.size())
        # ids = torch.cat(level_ids)

        # if cfg.min_bbox_size > 0:
        #     w = proposals[:, 2]
        #     h = proposals[:, 3]
        #     valid_mask = (w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size)
        #     if not valid_mask.all():
        #         proposals = proposals[valid_mask]
        #         scores = scores[valid_mask]
        #         ids = ids[valid_mask]
        # if proposals.numel() > 0:
        #     hproposals = obb2xyxy(proposals, self.version)
        #     _, keep = batched_nms(hproposals, scores, ids, cfg.nms)
        #     dets = torch.cat([proposals, scores[:, None]], dim=1)
        #     dets = dets[keep]
        # else:
        #     return proposals.new_zeros(0, 5)

        # return dets[:cfg.max_per_img]
        return mlvl_valid_anchors, mlvl_anchors
    
    def simple_test_anchor(self, x, img_metas):
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
        rpn_outs = self(x)
        proposal_list = self.get_rpn_anchors(*rpn_outs, img_metas=img_metas)
        return proposal_list 