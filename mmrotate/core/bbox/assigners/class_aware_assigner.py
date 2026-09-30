import torch
import json
import numpy

from ..builder import build_bbox_coder
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from ..builder import ROTATED_BBOX_ASSIGNERS
#from mmcv.utils import build_from_cfg

from mmcv.ops import points_in_polygons
from ..transforms import obb2poly

@ROTATED_BBOX_ASSIGNERS.register_module()
class CAAssigner_v1(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=512,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 assign_metric='gjsd',
                 topk=1,
                 topq=1,
                 constraint=False,
                 gauss_thr = 1.0,
                 bbox_coder=dict(
                     type='DeltaXYWHAOBBoxCoder',
                     target_means=(.0, .0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0, 1.0))):
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk
        self.topq = topq
        self.constraint = constraint
        self.gauss_thr = gauss_thr
        self.bbox_coder = build_bbox_coder(bbox_coder)

    def assign(self, cls_scores, bbox_preds, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.
        """
        
        box_dim = gt_bboxes.size(-1)
        
        assign_on_cpu = True if (self.gpu_assign_thr >= 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        ##################################################################################
        ###TODO: change existing "gjsd" into "iou" calculator
        # print('self.assign_metric: ', self.assign_metric)
        overlaps_1 = self.iou_calculator(gt_bboxes, bboxes, mode=self.assign_metric)
        # overlaps_2 = self.iou_calculator(gt_bboxes, bboxes, mode='gjsd')
        overlaps = overlaps_1

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, _ = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps_1, _ = overlaps_1.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]
        # gt_max_overlaps_2, _ = overlaps_2.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]
        gt_max_overlaps = gt_max_overlaps_1
        # print('gt_max_overlaps_1: ', gt_max_overlaps_1)
        # print('gt_max_overlaps_2: ', gt_max_overlaps_2)
        # print('#####################################################')
        # assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.8)] = 0
        assigned_gt_inds[max_overlaps > 0] = 0

        for i in range(num_gts):
            for j in range(self.topk):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1
        ##################################################################################

        device = bboxes.device
        bbox_preds = bbox_preds.to(device)
        cls_scores = cls_scores.to(device)
        bbox_preds = torch.transpose(bbox_preds, 0, 1)
        bbox_preds = self.bbox_coder.decode(bboxes, bbox_preds)
        
        num_gt = gt_bboxes.size(0)
        num_bboxes = bboxes.size(0)

        can_positive_mask = assigned_gt_inds > 0
        can_positive_inds = torch.nonzero(can_positive_mask)

        poscan = assigned_gt_inds[can_positive_inds].squeeze(-1)
        can_other_mask = assigned_gt_inds <= 0

        can_pos_scores = cls_scores[:,can_positive_inds].squeeze(-1)

        can_pos_scores = torch.transpose(can_pos_scores, 0, 1)
        can_bbox_pred = bbox_preds[can_positive_inds,:].squeeze(-1).squeeze()
        # print('can_bbox_pred: ', can_bbox_pred.shape)

        can_pos_iou = self.iou_calculator(gt_bboxes.to(device), can_bbox_pred, mode ='iou')
        can_pos_iou = can_pos_iou[poscan-1, range(poscan.size(0))]
        can_pos_cls, _ = torch.max(can_pos_scores,1)

        can_pos_quality = can_pos_iou + can_pos_cls.sigmoid() 
        can_pos_quality = can_pos_quality.unsqueeze(0).repeat(num_gt, 1) # size of gt, pos anchors
        
        gt_poscan = torch.zeros_like(can_pos_quality) - 100 # size of gt, pos anchors
        gt_poscan[poscan-1, range(poscan.size(0))] = can_pos_quality[poscan-1, range(poscan.size(0))]

        if self.topq >= can_pos_quality.size(1):
            topq = can_pos_quality.size(1)
        else:
            topq = self.topq
        gt_max_quality, gt_argmax_quality = gt_poscan.topk(topq, dim=1, largest=True, sorted=True)  # gt_argmax_quality [num_gt, q]

        assign_result_pre_gt = assigned_gt_inds

        assigned_gt_inds_init = assign_result_pre_gt * can_other_mask
        assigned_pos_prior = torch.zeros((num_gt, topq, box_dim),device=device)
        
        for i in range(num_gt):
            for j in range(topq):
                index = gt_argmax_quality[i,j]
                remap_inds = can_positive_inds[index,0]
                assigned_gt_inds_init[remap_inds] = assign_result_pre_gt [remap_inds]
                assigned_pos_prior[i,j,:] = bboxes[remap_inds,:] 
        assigned_gt_inds = assigned_gt_inds_init

        if self.constraint == 'dgmm':
            device1 = gt_bboxes.device
            if box_dim == 5:
                xy_gt, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
            elif box_dim == 4:
                xy_gt, sigma_t = self.xy_wh_2_xy_sigma(gt_bboxes)
            # get the mean of the positive samples
            pos_prior_mean = torch.mean(assigned_pos_prior[...,:2], dim=-2)
            if box_dim == 5:
                _, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
            elif box_dim == 4:
                _, sigma_t = self.xy_wh_2_xy_sigma(gt_bboxes)
            xy_pt = pos_prior_mean
            xy_a = bboxes[...,:2]
            xy_gt = xy_gt[...,None,:,:2].unsqueeze(-1)
            xy_pt = xy_pt[...,None,:,:2].unsqueeze(-1)
            xy_a = xy_a[...,:,None,:2].unsqueeze(-1)
            inv_sigma_t = torch.stack((sigma_t[..., 1, 1], -sigma_t[..., 0, 1],
                                      -sigma_t[..., 1, 0], sigma_t[..., 0, 0]),
                                      dim=-1).reshape(-1, 2, 2)
            ###################################################################################################
            inv_sigma_t = inv_sigma_t / sigma_t.det().unsqueeze(-1).unsqueeze(-1)
            # inv_sigma_t = inv_sigma_t / sigma_t.cpu().det().cuda().unsqueeze(-1).unsqueeze(-1)
            # sigma_t_det = sigma_t[:,0,0]*sigma_t[:,1,1] - sigma_t[:,0,1]*sigma_t[:,1,0]
            # inv_sigma_t = inv_sigma_t / sigma_t_det.unsqueeze(-1).unsqueeze(-1)
            ###################################################################################################
            gaussian_gt = torch.exp(-0.5*(xy_a-xy_gt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_gt)).squeeze(-1).squeeze(-1)
            gaussian_pt = torch.exp(-0.5*(xy_a-xy_pt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_pt)).squeeze(-1).squeeze(-1)
            gaussian = 0.7*gaussian_gt + 0.3*gaussian_pt 

            inside_flag = gaussian >= torch.exp(torch.tensor([-self.gauss_thr])).to(device1)
            length = range(assigned_gt_inds.size(0))
            inside_mask = inside_flag[length, (assigned_gt_inds-1).clamp(min=0)]
            assigned_gt_inds *= inside_mask

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        assign_result = AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        
        return assign_result

    def assign_wrt_ranking(self,  overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, _ = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, _ = overlaps.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]


        assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < 0.8)] = 0

        for i in range(num_gts):
            for j in range(self.topk):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def xy_wh_r_2_xy_sigma(self, xywhr):
        """Convert oriented bounding box to 2-D Gaussian distribution.

        Args:
            xywhr (torch.Tensor): rbboxes with shape (N, 5).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = xywhr[..., 4]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma


    def xy_wh_2_xy_sigma(self, xywh):
        """Convert horizontal bounding box to 2-D Gaussian distribution.

        Args:
            xywh (torch.Tensor): bboxes with shape (N, 4).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywh.shape
        assert _shape[-1] == 4
        xy = xywh[..., :2]
        wh = xywh[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = torch.zeros([_shape[0]]).type_as(xywh)
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma
    

@ROTATED_BBOX_ASSIGNERS.register_module()
class CAAssigner_v2(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=512,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 assign_metric='gjsd',
                 topk=1,
                 topq=1,
                 constraint=False,
                 gauss_thr = 1.0,
                 bbox_coder=dict(
                     type='DeltaXYWHAOBBoxCoder',
                     target_means=(.0, .0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0, 1.0))):
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk
        self.topq = topq
        self.constraint = constraint
        self.gauss_thr = gauss_thr
        self.bbox_coder = build_bbox_coder(bbox_coder)

    def assign(self, cls_scores, bbox_preds, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.
        
        The assignment is done in following steps

        1. compute gjsd between all bbox (bbox of all pyramid levels) and gt
        2. on each pyramid level, for each gt, select k bbox whose gjsd
            are largest to the gt center, so we total select k*l bbox as
            candidates for each gt
        3. get corresponding predicted quality[iou, cls] for the these candidates, and compute the
            mean and std, set mean + std as the quality threshold
        4. select these candidates whose quality are greater than or equal to
            the threshold as positive
        5. limit the positive sample's center with dgmm

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 5).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 5).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        
        box_dim = gt_bboxes.size(-1)
        
        assign_on_cpu = True if (self.gpu_assign_thr >= 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        ##################################################################################
        ###TODO: change existing "gjsd" into "iou" calculator
        # print('self.assign_metric: ', self.assign_metric)
        INF = 50000
        overlaps = self.iou_calculator(gt_bboxes, bboxes, mode=self.assign_metric)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
        # print('num_gts: ', num_gts)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                            -1,
                                            dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)
            
        ###################################################################
        ### TODO: step 2: Selecting candidates based on the gjsd
        candidate_idxs = []
        start_idx = 0
        num_level_bboxes = [16384, 4096, 1024, 256, 64]
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose gjsd are largest to the gt center
            end_idx = start_idx + bboxes_per_level
            overlaps_per_level = overlaps[:, start_idx:end_idx]
            _, topk_idxs_per_level = overlaps_per_level.topk(
                self.topk, dim=1, largest=True, sorted=True)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=1)
        # print('candidate_idxs: ', candidate_idxs.size())
        
        ### TODO: step 3: get corresponding quality for the these candidates,
        ### and compute mean and std, set mean + std as the quality threshold
        device = bboxes.device
        # print('device: ', device)
        # gt_device = gt_bboxes.device
        # print('gt_device: ', gt_device)
        bbox_preds = bbox_preds.to(device)
        cls_scores = cls_scores.to(device)
        bbox_preds = torch.transpose(bbox_preds, 0, 1)
        cls_scores = torch.transpose(cls_scores, 0, 1)
        bbox_preds = self.bbox_coder.decode(bboxes, bbox_preds)
        # print('cls_scores: ', cls_scores.size())
        # print('bbox_preds: ', bbox_preds.size())
        # print('gt_bboxes: ', gt_bboxes.size())
        # bbox_preds = torch.transpose(bbox_preds, 0, 1)
        can_quality_list = []
        for i in range(num_gts):
            can_scores = cls_scores[candidate_idxs[i], :]
            can_preds = bbox_preds[candidate_idxs[i], :]
            # print('gt_bboxes[i]: ', gt_bboxes[i])
            # print('can_scores: ', can_scores)
            # print('can_preds: ', can_preds)
            # print('can_preds: ', can_preds.size())
            # print('gt_bboxes[i].unsqueeze(0): ', gt_bboxes[i].squeeze(0).size())
            can_iou = self.iou_calculator(gt_bboxes[i].unsqueeze(0).to(device), can_preds, mode ='iou')
            # can_pos_iou = can_pos_iou[poscan-1, range(poscan.size(0))]
            # print('can_iou: ', can_iou.size())
            can_cls, _ = torch.max(can_scores, 1)
            can_quality = can_iou + can_cls.sigmoid()
            can_quality_list.append(can_quality)
        can_quality = torch.cat(can_quality_list)
        # print('can_quality: ', can_quality.size())
        # print('can_quality: ', can_quality)
        qualitys_mean_per_gt = can_quality.mean(1)
        qualitys_std_per_gt = can_quality.std(1)
        qualitys_thr_per_gt = qualitys_mean_per_gt + qualitys_std_per_gt
        
        ### TODO: step 4: select these candidates whose quality are greater than 
        ### or equal to the threshold as positive
        is_pos = can_quality >= qualitys_thr_per_gt[:, None]
        # print('is_pos: ', is_pos.sum())
        
        #############################################################################################
        ### TODO: step 5: limit the positive sample's center with inside
        # angle_version = 'le135'
        # gt_polys = obb2poly(gt_bboxes, angle_version)
        # gt_polys = gt_polys.to(device)
        # bboxes_points = bboxes[:, :2]
        # bboxes_points = bboxes_points.to(device)
        # # print('bboxes_points device: ', bboxes_points.device)
        # # print('gt_polys device: ', gt_polys.device)
        # inside_flag = points_in_polygons(bboxes_points, gt_polys)
        # # print('inside_flag: ', inside_flag.size())
        # # print('candidate_idxs: ', candidate_idxs.size())
        # is_in_gts = inside_flag[candidate_idxs.t(), torch.arange(num_gts)].t().to(is_pos.dtype)
        # # print('is_in_gts: ', is_in_gts.size())
        # is_pos = is_pos & is_in_gts
        # print('after inside is_pos: ', is_pos.sum())
        #############################################################################################
        
        # print('is_pos: ', is_pos.size())
        # print('is_pos: ', is_pos)
        # print('is_pos: ', is_pos.sum())
        for gt_idx in range(num_gts):
            candidate_idxs[gt_idx, :] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)
        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps, -INF).contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gts, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        # print('max_overlaps: ', max_overlaps)
        # print('argmax_overlaps: ', argmax_overlaps)
        
        
        # ### TODO: step 5: limit the positive sample's center with dgmm
        # if self.constraint == 'dgmm':
        #     device1 = gt_bboxes.device
        #     if box_dim == 5:
        #         xy_gt, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
        #     elif box_dim == 4:
        #         xy_gt, sigma_t = self.xy_wh_2_xy_sigma(gt_bboxes)
        #     # get the mean of the positive samples
        #     pos_prior_mean = torch.mean(assigned_pos_prior[...,:2], dim=-2)
        #     if box_dim == 5:
        #         _, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
        #     elif box_dim == 4:
        #         _, sigma_t = self.xy_wh_2_xy_sigma(gt_bboxes)
        #     xy_pt = pos_prior_mean
        #     xy_a = bboxes[...,:2]
        #     xy_gt = xy_gt[...,None,:,:2].unsqueeze(-1)
        #     xy_pt = xy_pt[...,None,:,:2].unsqueeze(-1)
        #     xy_a = xy_a[...,:,None,:2].unsqueeze(-1)
        #     inv_sigma_t = torch.stack((sigma_t[..., 1, 1], -sigma_t[..., 0, 1],
        #                                 -sigma_t[..., 1, 0], sigma_t[..., 0, 0]),
        #                                 dim=-1).reshape(-1, 2, 2)
        #     ###################################################################################################
        #     inv_sigma_t = inv_sigma_t / sigma_t.det().unsqueeze(-1).unsqueeze(-1)
        #     # inv_sigma_t = inv_sigma_t / sigma_t.cpu().det().cuda().unsqueeze(-1).unsqueeze(-1)
        #     # sigma_t_det = sigma_t[:,0,0]*sigma_t[:,1,1] - sigma_t[:,0,1]*sigma_t[:,1,0]
        #     # inv_sigma_t = inv_sigma_t / sigma_t_det.unsqueeze(-1).unsqueeze(-1)
        #     ###################################################################################################
        #     gaussian_gt = torch.exp(-0.5*(xy_a-xy_gt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_gt)).squeeze(-1).squeeze(-1)
        #     gaussian_pt = torch.exp(-0.5*(xy_a-xy_pt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_pt)).squeeze(-1).squeeze(-1)
        #     gaussian = 0.7*gaussian_gt + 0.3*gaussian_pt 

        #     inside_flag = gaussian >= torch.exp(torch.tensor([-self.gauss_thr])).to(device1)
        #     length = range(assigned_gt_inds.size(0))
        #     inside_mask = inside_flag[length, (assigned_gt_inds-1).clamp(min=0)]
        #     assigned_gt_inds *= inside_mask
        ###################################################################

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        assign_result = AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        
        return assign_result

    def assign_wrt_ranking(self,  overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, _ = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, _ = overlaps.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]


        assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < 0.8)] = 0

        for i in range(num_gts):
            for j in range(self.topk):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def xy_wh_r_2_xy_sigma(self, xywhr):
        """Convert oriented bounding box to 2-D Gaussian distribution.

        Args:
            xywhr (torch.Tensor): rbboxes with shape (N, 5).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = xywhr[..., 4]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma


    def xy_wh_2_xy_sigma(self, xywh):
        """Convert horizontal bounding box to 2-D Gaussian distribution.

        Args:
            xywh (torch.Tensor): bboxes with shape (N, 4).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywh.shape
        assert _shape[-1] == 4
        xy = xywh[..., :2]
        wh = xywh[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = torch.zeros([_shape[0]]).type_as(xywh)
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma
    
@ROTATED_BBOX_ASSIGNERS.register_module()
class CAAssigner_v3(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 angle_version='le135',
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=512,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 assign_metric='gjsd',
                 topk=1,
                 topq=1,
                 constraint=False,
                 gauss_thr = 1.0,
                 bbox_coder=dict(
                     type='DeltaXYWHAOBBoxCoder',
                     target_means=(.0, .0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0, 1.0))):
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk
        self.topq = topq
        self.constraint = constraint
        self.gauss_thr = gauss_thr
        self.bbox_coder = build_bbox_coder(bbox_coder)
        
        self.angle_version = angle_version

    def assign(self, cls_scores, bbox_preds, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.
        
        The assignment is done in following steps

        1. compute gjsd between all bbox (bbox of all pyramid levels) and gt
        2. on each pyramid level, for each gt, select k bbox whose gjsd
            are largest to the gt center, so we total select k*l bbox as
            candidates for each gt
        3. get corresponding predicted quality[iou, cls] for the these candidates, and compute the
            mean and std, set mean + std as the quality threshold
        4. select these candidates whose quality are greater than or equal to
            the threshold as positive
        5. limit the positive sample's center with dgmm

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 5).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 5).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        
        INF = 100000000
        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                            0,
                                            dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        # the center of gt and bbox
        gt_points = gt_bboxes[:, :2]
        bboxes_points = bboxes[:, :2]

        distances = (bboxes_points[:, None, :] -
                    gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        num_level_bboxes = [16384, 4096, 1024, 256, 64]
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(
                self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        gt_bboxes = obb2poly(gt_bboxes, self.angle_version)

        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        inside_flag = points_in_polygons(bboxes_points, gt_bboxes)
        is_in_gts = inside_flag[candidate_idxs,
                                torch.arange(num_gt)].to(is_pos.dtype)

        is_pos = is_pos & is_in_gts
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                        -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def assign_wrt_ranking(self,  overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, _ = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, _ = overlaps.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]


        assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < 0.8)] = 0

        for i in range(num_gts):
            for j in range(self.topk):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def xy_wh_r_2_xy_sigma(self, xywhr):
        """Convert oriented bounding box to 2-D Gaussian distribution.

        Args:
            xywhr (torch.Tensor): rbboxes with shape (N, 5).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = xywhr[..., 4]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma


    def xy_wh_2_xy_sigma(self, xywh):
        """Convert horizontal bounding box to 2-D Gaussian distribution.

        Args:
            xywh (torch.Tensor): bboxes with shape (N, 4).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywh.shape
        assert _shape[-1] == 4
        xy = xywh[..., :2]
        wh = xywh[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = torch.zeros([_shape[0]]).type_as(xywh)
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma

@ROTATED_BBOX_ASSIGNERS.register_module()
class CAAssigner_v41(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 angle_version='le135',
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=512,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 assign_metric='gjsd',
                 topk=1,
                 topq=1,
                 constraint=False,
                 gauss_thr = 1.0,
                 bbox_coder=dict(
                     type='DeltaXYWHAOBBoxCoder',
                     target_means=(.0, .0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0, 1.0))):
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk
        self.topq = topq
        self.constraint = constraint
        self.gauss_thr = gauss_thr
        self.bbox_coder = build_bbox_coder(bbox_coder)
        
        self.angle_version = angle_version

    def assign(self, cls_scores, bbox_preds, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.
        
        The assignment is done in following steps

        1. compute gjsd between all bbox (bbox of all pyramid levels) and gt
        2. on each pyramid level, for each gt, select k bbox whose gjsd
            are largest to the gt center, so we total select k*l bbox as
            candidates for each gt
        3. get corresponding predicted quality[iou, cls] for the these candidates, and compute the
            mean and std, set mean + std as the quality threshold
        4. select these candidates whose quality are greater than or equal to
            the threshold as positive
        5. limit the positive sample's center with dgmm

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 5).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 5).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        
        INF = 100000000
        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]
        box_dim = gt_bboxes.size(-1)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                            0,
                                            dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        # the center of gt and bbox
        gt_points = gt_bboxes[:, :2]
        bboxes_points = bboxes[:, :2]
        distances = (bboxes_points[:, None, :] -
                    gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0

        # print('num_bboxes: ', num_bboxes)
        if num_bboxes == 21824:
            num_level_bboxes = [16384, 4096, 1024, 256, 64]
        elif num_bboxes == 13343:
            num_level_bboxes = [10000, 2500, 625, 169, 49]
        assert sum(num_level_bboxes) == num_bboxes
        
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(
                self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        # gt_bboxes = obb2poly(gt_bboxes, self.angle_version)
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        
        ###########################################################
        #### TODO: get the mean of the positive samples
        device = bboxes.device
        pos_prior_mean = torch.zeros((num_gt, 2), device=device)
        for gt_idx in range(num_gt):
            can_bbox_idxs_per_gt = candidate_idxs[:, gt_idx][is_pos[:, gt_idx]]
            # print('bboxes: ', bboxes.size())
            # print('can_bbox_idxs_per_gt: ', can_bbox_idxs_per_gt.size())
            # print('can_bbox_idxs_per_gt: ', can_bbox_idxs_per_gt)
            pos_prior_mean[gt_idx, :] = torch.mean(bboxes[can_bbox_idxs_per_gt, :2], dim=0)
        ###########################################################
        
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                        -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        
        ### TODO: step 6: limit the positive sample's center with dgmm
        if self.constraint == 'dgmm':
            device1 = gt_bboxes.device
            if box_dim == 5:
                xy_gt, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
            elif box_dim == 4:
                xy_gt, sigma_t = self.xy_wh_2_xy_sigma(gt_bboxes)
            # get the mean of the positive samples
            # pos_prior_mean = torch.mean(assigned_pos_prior[...,:2], dim=-2)
            if box_dim == 5:
                _, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
            elif box_dim == 4:
                _, sigma_t = self.xy_wh_2_xy_sigma(gt_bboxes)
            xy_pt = pos_prior_mean
            xy_a = bboxes[...,:2]
            xy_gt = xy_gt[...,None,:,:2].unsqueeze(-1)
            xy_pt = xy_pt[...,None,:,:2].unsqueeze(-1)
            xy_a = xy_a[...,:,None,:2].unsqueeze(-1)
            inv_sigma_t = torch.stack((sigma_t[..., 1, 1], -sigma_t[..., 0, 1],
                                        -sigma_t[..., 1, 0], sigma_t[..., 0, 0]),
                                        dim=-1).reshape(-1, 2, 2)
            ###################################################################################################
            inv_sigma_t = inv_sigma_t / sigma_t.det().unsqueeze(-1).unsqueeze(-1)
            # inv_sigma_t = inv_sigma_t / sigma_t.cpu().det().cuda().unsqueeze(-1).unsqueeze(-1)
            # sigma_t_det = sigma_t[:,0,0]*sigma_t[:,1,1] - sigma_t[:,0,1]*sigma_t[:,1,0]
            # inv_sigma_t = inv_sigma_t / sigma_t_det.unsqueeze(-1).unsqueeze(-1)
            ###################################################################################################
            gaussian_gt = torch.exp(-0.5*(xy_a-xy_gt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_gt)).squeeze(-1).squeeze(-1)
            gaussian_pt = torch.exp(-0.5*(xy_a-xy_pt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_pt)).squeeze(-1).squeeze(-1)
            gaussian = 0.7*gaussian_gt + 0.3*gaussian_pt 
            inside_flag = gaussian >= torch.exp(torch.tensor([-self.gauss_thr])).to(device1)
            length = range(assigned_gt_inds.size(0))
            inside_mask = inside_flag[length, (assigned_gt_inds-1).clamp(min=0)]
            assigned_gt_inds *= inside_mask
        
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def assign_wrt_ranking(self,  overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, _ = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, _ = overlaps.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]


        assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < 0.8)] = 0

        for i in range(num_gts):
            for j in range(self.topk):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def xy_wh_r_2_xy_sigma(self, xywhr):
        """Convert oriented bounding box to 2-D Gaussian distribution.

        Args:
            xywhr (torch.Tensor): rbboxes with shape (N, 5).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = xywhr[..., 4]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma


    def xy_wh_2_xy_sigma(self, xywh):
        """Convert horizontal bounding box to 2-D Gaussian distribution.

        Args:
            xywh (torch.Tensor): bboxes with shape (N, 4).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywh.shape
        assert _shape[-1] == 4
        xy = xywh[..., :2]
        wh = xywh[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = torch.zeros([_shape[0]]).type_as(xywh)
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma

@ROTATED_BBOX_ASSIGNERS.register_module()
class CAAssigner_v42(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 angle_version='le135',
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=512,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 assign_metric='gjsd',
                 topk=1,
                 topq=1,
                 constraint=False,
                 gauss_thr = 1.0,
                 bbox_coder=dict(
                     type='DeltaXYWHAOBBoxCoder',
                     target_means=(.0, .0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0, 1.0))):
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk
        self.topq = topq
        self.constraint = constraint
        self.gauss_thr = gauss_thr
        self.bbox_coder = build_bbox_coder(bbox_coder)
        
        self.angle_version = angle_version

    def assign(self, cls_scores, bbox_preds, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.
        
        The assignment is done in following steps

        1. compute gjsd between all bbox (bbox of all pyramid levels) and gt
        2. on each pyramid level, for each gt, select k bbox whose gjsd
            are largest to the gt center, so we total select k*l bbox as
            candidates for each gt
        3. get corresponding predicted quality[iou, cls] for the these candidates, and compute the
            mean and std, set mean + std as the quality threshold
        4. select these candidates whose quality are greater than or equal to
            the threshold as positive
        5. limit the positive sample's center with dgmm

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 5).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 5).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        
        INF = 100000000
        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]
        box_dim = gt_bboxes.size(-1)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                            0,
                                            dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        # the center of gt and bbox
        gt_points = gt_bboxes[:, :2]
        bboxes_points = bboxes[:, :2]
        distances = (bboxes_points[:, None, :] -
                    gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        num_level_bboxes = [16384, 4096, 1024, 256, 64]
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(
                self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        # gt_bboxes = obb2poly(gt_bboxes, self.angle_version)
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        
        ###########################################################
        #### TODO: get the mean of the positive samples
        device = bboxes.device
        pos_prior_mean = torch.zeros((num_gt, 2), device=device)
        for gt_idx in range(num_gt):
            can_bbox_idxs_per_gt = candidate_idxs[:, gt_idx][is_pos[:, gt_idx]]
            # print('bboxes: ', bboxes.size())
            # print('can_bbox_idxs_per_gt: ', can_bbox_idxs_per_gt.size())
            # print('can_bbox_idxs_per_gt: ', can_bbox_idxs_per_gt)
            pos_prior_mean[gt_idx, :] = torch.mean(bboxes[can_bbox_idxs_per_gt, :2], dim=0)
        ###########################################################
        
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                        -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        
        ### TODO: step 6: limit the positive sample's center with dgmm
        if self.constraint == 'dgmm':
            device1 = gt_bboxes.device
            if box_dim == 5:
                xy_gt, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
            elif box_dim == 4:
                xy_gt, sigma_t = self.xy_wh_2_xy_sigma(gt_bboxes)
            # get the mean of the positive samples
            # pos_prior_mean = torch.mean(assigned_pos_prior[...,:2], dim=-2)
            # if box_dim == 5:
            #     _, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
            # elif box_dim == 4:
            #     _, sigma_t = self.xy_wh_2_xy_sigma(gt_bboxes)
            xy_pt = pos_prior_mean
            xy_a = bboxes[...,:2]
            xy_gt = xy_gt[...,None,:,:2].unsqueeze(-1)
            xy_pt = xy_pt[...,None,:,:2].unsqueeze(-1)
            xy_a = xy_a[...,:,None,:2].unsqueeze(-1)
            inv_sigma_t = torch.stack((sigma_t[..., 1, 1], -sigma_t[..., 0, 1],
                                        -sigma_t[..., 1, 0], sigma_t[..., 0, 0]),
                                        dim=-1).reshape(-1, 2, 2)
            ###################################################################################################
            inv_sigma_t = inv_sigma_t / sigma_t.det().unsqueeze(-1).unsqueeze(-1)
            # inv_sigma_t = inv_sigma_t / sigma_t.cpu().det().cuda().unsqueeze(-1).unsqueeze(-1)
            # sigma_t_det = sigma_t[:,0,0]*sigma_t[:,1,1] - sigma_t[:,0,1]*sigma_t[:,1,0]
            # inv_sigma_t = inv_sigma_t / sigma_t_det.unsqueeze(-1).unsqueeze(-1)
            ###################################################################################################
            gaussian_gt = torch.exp(-0.5*(xy_a-xy_gt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_gt)).squeeze(-1).squeeze(-1)
            # gaussian_pt = torch.exp(-0.5*(xy_a-xy_pt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_pt)).squeeze(-1).squeeze(-1)
            # gaussian = 0.7*gaussian_gt + 0.3*gaussian_pt 
            gaussian = gaussian_gt
            inside_flag = gaussian >= torch.exp(torch.tensor([-self.gauss_thr])).to(device1)
            length = range(assigned_gt_inds.size(0))
            inside_mask = inside_flag[length, (assigned_gt_inds-1).clamp(min=0)]
            assigned_gt_inds *= inside_mask
        
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def assign_wrt_ranking(self,  overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, _ = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, _ = overlaps.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]


        assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < 0.8)] = 0

        for i in range(num_gts):
            for j in range(self.topk):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def xy_wh_r_2_xy_sigma(self, xywhr):
        """Convert oriented bounding box to 2-D Gaussian distribution.

        Args:
            xywhr (torch.Tensor): rbboxes with shape (N, 5).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = xywhr[..., 4]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma


    def xy_wh_2_xy_sigma(self, xywh):
        """Convert horizontal bounding box to 2-D Gaussian distribution.

        Args:
            xywh (torch.Tensor): bboxes with shape (N, 4).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywh.shape
        assert _shape[-1] == 4
        xy = xywh[..., :2]
        wh = xywh[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = torch.zeros([_shape[0]]).type_as(xywh)
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma

@ROTATED_BBOX_ASSIGNERS.register_module()
class CAAssigner_v50(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 angle_version='le135',
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=512,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 assign_metric='gjsd',
                 topk=1,
                 topq=1,
                 constraint=False,
                 gauss_thr = 1.0,
                 bbox_coder=dict(
                     type='DeltaXYWHAOBBoxCoder',
                     target_means=(.0, .0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0, 1.0))):
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk
        self.topq = topq
        self.constraint = constraint
        self.gauss_thr = gauss_thr
        self.bbox_coder = build_bbox_coder(bbox_coder)
        
        self.angle_version = angle_version

    def assign(self, cls_scores, bbox_preds, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.
        
        The assignment is done in following steps

        1. compute gjsd between all bbox (bbox of all pyramid levels) and gt
        2. on each pyramid level, for each gt, select k bbox whose gjsd
            are largest to the gt center, so we total select k*l bbox as
            candidates for each gt
        3. get corresponding predicted quality[iou, cls] for the these candidates, and compute the
            mean and std, set mean + std as the quality threshold
        4. select these candidates whose quality are greater than or equal to
            the threshold as positive
        5. limit the positive sample's center with dgmm

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 5).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 5).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        
        INF = 100000000
        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]
        box_dim = gt_bboxes.size(-1)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                            0,
                                            dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        # the center of gt and bbox
        gt_points = gt_bboxes[:, :2]
        bboxes_points = bboxes[:, :2]
        distances = (bboxes_points[:, None, :] -
                    gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        num_level_bboxes = [16384, 4096, 1024, 256, 64]
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(
                self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        # gt_bboxes = obb2poly(gt_bboxes, self.angle_version)
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        
        ###########################################################
        #### TODO: get the mean of the positive samples
        device = bboxes.device
        pos_prior_mean = torch.zeros((num_gt, 2), device=device)
        for gt_idx in range(num_gt):
            can_bbox_idxs_per_gt = candidate_idxs[:, gt_idx][is_pos[:, gt_idx]]
            # print('bboxes: ', bboxes.size())
            # print('can_bbox_idxs_per_gt: ', can_bbox_idxs_per_gt.size())
            # print('can_bbox_idxs_per_gt: ', can_bbox_idxs_per_gt)
            pos_prior_mean[gt_idx, :] = torch.mean(bboxes[can_bbox_idxs_per_gt, :2], dim=0)
        ###########################################################
        
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                        -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        
        ### TODO: step 6: limit the positive sample's center with dgmm
        if self.constraint == 'dgmm':
            device1 = gt_bboxes.device
            if box_dim == 5:
                xy_gt, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
            elif box_dim == 4:
                xy_gt, sigma_t = self.xy_wh_2_xy_sigma(gt_bboxes)
            # get the mean of the positive samples
            # pos_prior_mean = torch.mean(assigned_pos_prior[...,:2], dim=-2)
            if box_dim == 5:
                _, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
            elif box_dim == 4:
                _, sigma_t = self.xy_wh_2_xy_sigma(gt_bboxes)
            xy_pt = pos_prior_mean
            xy_a = bboxes[...,:2]
            xy_gt = xy_gt[...,None,:,:2].unsqueeze(-1)
            xy_pt = xy_pt[...,None,:,:2].unsqueeze(-1)
            xy_a = xy_a[...,:,None,:2].unsqueeze(-1)
            inv_sigma_t = torch.stack((sigma_t[..., 1, 1], -sigma_t[..., 0, 1],
                                        -sigma_t[..., 1, 0], sigma_t[..., 0, 0]),
                                        dim=-1).reshape(-1, 2, 2)
            ###################################################################################################
            inv_sigma_t = inv_sigma_t / sigma_t.det().unsqueeze(-1).unsqueeze(-1)
            # inv_sigma_t = inv_sigma_t / sigma_t.cpu().det().cuda().unsqueeze(-1).unsqueeze(-1)
            # sigma_t_det = sigma_t[:,0,0]*sigma_t[:,1,1] - sigma_t[:,0,1]*sigma_t[:,1,0]
            # inv_sigma_t = inv_sigma_t / sigma_t_det.unsqueeze(-1).unsqueeze(-1)
            ###################################################################################################
            gaussian_gt = torch.exp(-0.5*(xy_a-xy_gt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_gt)).squeeze(-1).squeeze(-1)
            gaussian_pt = torch.exp(-0.5*(xy_a-xy_pt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_pt)).squeeze(-1).squeeze(-1)
            gaussian = 0.7*gaussian_gt + 0.3*gaussian_pt 
            inside_flag = gaussian >= torch.exp(torch.tensor([-self.gauss_thr])).to(device1)
            length = range(assigned_gt_inds.size(0))
            inside_mask = inside_flag[length, (assigned_gt_inds-1).clamp(min=0)]
            assigned_gt_inds *= inside_mask
        
        ########################################################################################
        ### TODO: test gt_labels
        # print('gt_labels: ', gt_labels)
        # 使用 torch.unique 获取唯一元素及其出现次数
        unique_elements, counts = torch.unique(gt_labels, return_counts=True)
        # 打印结果
        print("Unique elements:", unique_elements)
        print("Counts:", counts)
        if len(unique_elements) > 1:
            index = torch.argmin(counts)
            min_element = unique_elements[index]
            # print('min_element: ', min_element)
            indices = torch.nonzero(gt_labels==min_element)
            # print('indices: ', indices)
            # print('min gt ind: ', indices[0])
            min_samples = sum(assigned_gt_inds == indices[0]+1)
            print('min_samples: ', min_samples)
            print('num_gt: ', num_gt)
            all_samples = sum(assigned_gt_inds>0)
            print('all_samples: ', all_samples)
            print('samples per gt: ', all_samples/num_gt)
            if all_samples/num_gt > min_samples:
                print('+1')
        print("###############################################################")
        ########################################################################################
        
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def assign_wrt_ranking(self,  overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, _ = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, _ = overlaps.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]


        assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < 0.8)] = 0

        for i in range(num_gts):
            for j in range(self.topk):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def xy_wh_r_2_xy_sigma(self, xywhr):
        """Convert oriented bounding box to 2-D Gaussian distribution.

        Args:
            xywhr (torch.Tensor): rbboxes with shape (N, 5).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = xywhr[..., 4]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma

    def xy_wh_2_xy_sigma(self, xywh):
        """Convert horizontal bounding box to 2-D Gaussian distribution.

        Args:
            xywh (torch.Tensor): bboxes with shape (N, 4).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywh.shape
        assert _shape[-1] == 4
        xy = xywh[..., :2]
        wh = xywh[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = torch.zeros([_shape[0]]).type_as(xywh)
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma
