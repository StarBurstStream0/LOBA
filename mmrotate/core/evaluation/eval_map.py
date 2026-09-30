# Copyright (c) OpenMMLab. All rights reserved.
import os

from multiprocessing import get_context

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmcv.utils import print_log
from mmdet.core import average_precision
from terminaltables import AsciiTable

def combine_iou_info_list(iou_info_list):
    """合并多个IoU信息字典列表，包含尺度掩码
    
    Args:
        iou_info_list: 列表，每个元素是tp_iou_info或fp_iou_info字典
        
    Returns:
        合并后的字典，包含所有字段的拼接数组
    """
    if not iou_info_list:
        return {
            'ious': np.array([], dtype=np.float32),
            'scores': np.array([], dtype=np.float32),
            'areas': np.array([], dtype=np.float32),
            'indices': np.array([], dtype=np.int32),
            'image_indices': np.array([], dtype=np.int32),
            'scale_masks': np.array([], dtype=bool)
        }
    
    # 初始化存储列表
    all_ious = []
    all_scores = []
    all_areas = []
    all_indices = []
    all_image_indices = []
    all_scale_masks = []
    
    # 确定尺度数量（从第一个有效元素获取）
    num_scales = None
    first_valid_info = None
    
    # 首先找到第一个有效元素以确定尺度数量
    for iou_info in iou_info_list:
        if iou_info is not None and 'scale_masks' in iou_info and iou_info['scale_masks'].size > 0:
            num_scales = iou_info['scale_masks'].shape[0]
            first_valid_info = iou_info
            break
    
    # 遍历所有图像
    for img_idx, iou_info in enumerate(iou_info_list):
        if iou_info is None:
            continue
            
        # 检查是否有数据
        if len(iou_info['ious']) == 0:
            continue
            
        # 获取有效掩码 (iou >= 0 表示该检测被考虑)
        valid_mask = iou_info['ious'] >= 0
        valid_count = np.sum(valid_mask)
        
        if valid_count == 0:
            continue
        
        # 收集基础数据
        all_ious.append(iou_info['ious'][valid_mask])
        all_scores.append(iou_info['scores'][valid_mask])
        all_areas.append(iou_info['areas'][valid_mask])
        
        # 处理索引
        if 'indices' in iou_info:
            original_indices = iou_info['indices'][valid_mask]
            # 添加唯一标识
            offset_indices = original_indices + img_idx * 100000
            all_indices.append(offset_indices)
        
        # 记录图像来源
        all_image_indices.append(np.full(valid_count, img_idx, dtype=np.int32))
        
        # 处理尺度掩码
        if 'scale_masks' in iou_info and iou_info['scale_masks'].size > 0:
            # 只取有效检测的尺度掩码
            scale_mask_valid = iou_info['scale_masks'][:, valid_mask]
            all_scale_masks.append(scale_mask_valid)
        elif num_scales is not None:
            # 如果当前元素没有尺度掩码但我们已经知道尺度数量，创建默认值
            scale_mask_valid = np.zeros((num_scales, valid_count), dtype=bool)
            all_scale_masks.append(scale_mask_valid)
    
    # 合并所有数据
    combined = {
        'ious': np.concatenate(all_ious) if all_ious else np.array([], dtype=np.float32),
        'scores': np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32),
        'areas': np.concatenate(all_areas) if all_areas else np.array([], dtype=np.float32),
        'indices': np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32),
        'image_indices': np.concatenate(all_image_indices) if all_image_indices else np.array([], dtype=np.int32),
        'num_images': len(iou_info_list),
        'total_count': sum([len(arr) for arr in all_ious]) if all_ious else 0
    }
    
    # 合并尺度掩码
    if all_scale_masks:
        combined['scale_masks'] = np.hstack(all_scale_masks)
        if combined['scale_masks'].size > 0:
            combined['num_scales'] = combined['scale_masks'].shape[0]
    else:
        combined['scale_masks'] = np.array([], dtype=bool)
    
    return combined

def analyze_iou_distribution_from_tpfp(tp_iou_info, fp_iou_info, iou_thr=0.5, scale_names=None):
    """分析从tpfp函数获取的IoU分布信息
    
    Args:
        tp_iou_info: TP的IoU信息字典
        fp_iou_info: FP的IoU信息字典
        iou_thr: IoU阈值
        scale_names: 尺度名称列表
        
    Returns:
        dict: 包含详细分析结果的字典
    """
    if len(tp_iou_info['ious']) == 0 and len(fp_iou_info['ious']) == 0:
        return {}
    
    # 基础统计
    analysis = {
        'basic_stats': {
            'tp_count': len(tp_iou_info['ious']),
            'fp_count': len(fp_iou_info['ious']),
            'total_detections': len(tp_iou_info['ious']) + len(fp_iou_info['ious']),
            'precision': len(tp_iou_info['ious']) / max(len(tp_iou_info['ious']) + len(fp_iou_info['ious']), 1)
        },
        
        'tp_stats': {},
        'fp_stats': {},
        
        'iou_threshold_analysis': {},
        'score_vs_iou_analysis': {},
        'area_vs_iou_analysis': {}
    }
    
    # TP统计
    if len(tp_iou_info['ious']) > 0:
        analysis['tp_stats'] = {
            'mean_iou': float(np.mean(tp_iou_info['ious'])),
            'std_iou': float(np.std(tp_iou_info['ious'])),
            'median_iou': float(np.median(tp_iou_info['ious'])),
            'min_iou': float(np.min(tp_iou_info['ious'])),
            'max_iou': float(np.max(tp_iou_info['ious'])),
            'q25_iou': float(np.percentile(tp_iou_info['ious'], 25)),
            'q75_iou': float(np.percentile(tp_iou_info['ious'], 75)),
            'mean_score': float(np.mean(tp_iou_info['scores'])),
            'mean_area': float(np.mean(tp_iou_info['areas']))
        }
    
    # FP统计
    if len(fp_iou_info['ious']) > 0:
        analysis['fp_stats'] = {
            'mean_iou': float(np.mean(fp_iou_info['ious'])),
            'std_iou': float(np.std(fp_iou_info['ious'])),
            'median_iou': float(np.median(fp_iou_info['ious'])),
            'min_iou': float(np.min(fp_iou_info['ious'])),
            'max_iou': float(np.max(fp_iou_info['ious'])),
            'q25_iou': float(np.percentile(fp_iou_info['ious'], 25)),
            'q75_iou': float(np.percentile(fp_iou_info['ious'], 75)),
            'mean_score': float(np.mean(fp_iou_info['scores'])),
            'mean_area': float(np.mean(fp_iou_info['areas']))
        }
    
    # IoU阈值分析
    iou_thresholds = np.arange(0.5, 0.95, 0.05)
    for thr in iou_thresholds:
        tp_at_thr = np.sum(tp_iou_info['ious'] >= thr)
        fp_at_thr = np.sum(fp_iou_info['ious'] >= thr)
        total_at_thr = tp_at_thr + fp_at_thr
        
        analysis['iou_threshold_analysis'][f'iou_{thr:.2f}'] = {
            'tp_count': int(tp_at_thr),
            'fp_count': int(fp_at_thr),
            'total_count': int(total_at_thr),
            'precision': tp_at_thr / max(total_at_thr, 1),
            'recall': tp_at_thr / max(len(tp_iou_info['ious']) + len(fp_iou_info['ious']), 1)
        }
    
    # 分数与IoU的相关性分析
    if len(tp_iou_info['ious']) > 0:
        analysis['score_vs_iou_analysis']['tp'] = {
            'correlation': float(np.corrcoef(tp_iou_info['scores'], tp_iou_info['ious'])[0, 1]),
            'score_bins': {},
            'iou_by_score_bin': {}
        }
        
        # 按分数分箱分析
        score_bins = np.linspace(0, 1, 11)
        for i in range(len(score_bins)-1):
            bin_mask = (tp_iou_info['scores'] >= score_bins[i]) & (tp_iou_info['scores'] < score_bins[i+1])
            if np.any(bin_mask):
                bin_name = f"{score_bins[i]:.1f}-{score_bins[i+1]:.1f}"
                analysis['score_vs_iou_analysis']['tp']['score_bins'][bin_name] = int(np.sum(bin_mask))
                analysis['score_vs_iou_analysis']['tp']['iou_by_score_bin'][bin_name] = float(np.mean(tp_iou_info['ious'][bin_mask]))
    
    # 尺度级别的分析（如果有尺度信息）
    if 'scale_masks' in tp_iou_info and 'scale_masks' in fp_iou_info:
        if tp_iou_info['scale_masks'].size > 0 and fp_iou_info['scale_masks'].size > 0:
            num_scales = tp_iou_info['scale_masks'].shape[0]
            analysis['scale_wise_analysis'] = {}
            
            for scale_idx in range(num_scales):
                scale_name = scale_names[scale_idx] if scale_names and scale_idx < len(scale_names) else f'scale_{scale_idx}'
                
                # TP在特定尺度
                if tp_iou_info['scale_masks'].size > 0:
                    tp_scale_mask = tp_iou_info['scale_masks'][scale_idx, :]
                    tp_ious_in_scale = tp_iou_info['ious'][tp_scale_mask] if tp_scale_mask.any() else np.array([])
                else:
                    tp_ious_in_scale = np.array([])
                
                # FP在特定尺度
                if fp_iou_info['scale_masks'].size > 0:
                    fp_scale_mask = fp_iou_info['scale_masks'][scale_idx, :]
                    fp_ious_in_scale = fp_iou_info['ious'][fp_scale_mask] if fp_scale_mask.any() else np.array([])
                else:
                    fp_ious_in_scale = np.array([])
                
                analysis['scale_wise_analysis'][scale_name] = {
                    'tp_count': len(tp_ious_in_scale),
                    'fp_count': len(fp_ious_in_scale),
                    'tp_mean_iou': float(np.mean(tp_ious_in_scale)) if len(tp_ious_in_scale) > 0 else 0.0,
                    'fp_mean_iou': float(np.mean(fp_ious_in_scale)) if len(fp_ious_in_scale) > 0 else 0.0,
                    'precision': len(tp_ious_in_scale) / max(len(tp_ious_in_scale) + len(fp_ious_in_scale), 1)
                }
    
    return analysis


def visualize_iou_distribution(tp_iou_info, fp_iou_info, save_path=None, title_prefix=""):
    """可视化IoU分布
    
    Args:
        tp_iou_info: TP的IoU信息
        fp_iou_info: FP的IoU信息
        save_path: 保存路径
        title_prefix: 标题前缀
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    

    
    plt.figure(figsize=(16, 10))
    
    # 1. IoU直方图
    plt.subplot(2, 3, 1)
    if len(tp_iou_info['ious']) > 0:
        plt.hist(tp_iou_info['ious'], bins=50, alpha=0.7, label='TP', density=True, color='green')
    if len(fp_iou_info['ious']) > 0:
        plt.hist(fp_iou_info['ious'], bins=50, alpha=0.7, label='FP', density=True, color='red')
    plt.yscale('log') # 取对数，避免极值点主导全局
    plt.xlabel('IoU')
    plt.ylabel('Density')
    plt.title(f'{title_prefix}IoU Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 置信度分数vs IoU
    plt.subplot(2, 3, 2)
    if len(tp_iou_info['ious']) > 0:
        plt.scatter(tp_iou_info['scores'], tp_iou_info['ious'], alpha=0.5, label='TP', s=10, color='green')
    if len(fp_iou_info['ious']) > 0:
        plt.scatter(fp_iou_info['scores'], fp_iou_info['ious'], alpha=0.5, label='FP', s=10, color='red')
    plt.xlabel('Confidence Score')
    plt.ylabel('IoU')
    plt.title('Score vs IoU')
    plt.legend()
    # 添加阈值参考线
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=1.0, label='IoU=0.5')
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=1.0, label='Score=0.5')
    # 添加四个象限的说明
    # plt.text(0.25, 0.75, 'Low Score\nHigh IoU', transform=plt.gca().transAxes, alpha=0.7, fontsize=8)
    # plt.text(0.75, 0.75, 'High Score\nHigh IoU', transform=plt.gca().transAxes, alpha=0.7, fontsize=8)
    # plt.text(0.25, 0.25, 'Low Score\nLow IoU', transform=plt.gca().transAxes, alpha=0.7, fontsize=8)
    # plt.text(0.75, 0.25, 'High Score\nLow IoU', transform=plt.gca().transAxes, alpha=0.7, fontsize=8)
    plt.grid(True, alpha=0.3)
    # 可选：添加回归线或密度热力图
    from scipy import stats
    if len(tp_iou_info['ious']) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            tp_iou_info['scores'], tp_iou_info['ious'])
        x_line = np.linspace(0, 1, 10)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, 'b--', alpha=1.0, label=f'TP fit (r={r_value:.2f})')
    # if len(fp_iou_info['ious']) > 0:
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(
    #         fp_iou_info['scores'], fp_iou_info['ious'])
    #     x_line = np.linspace(0, 1, 10)
    #     y_line = slope * x_line + intercept
    #     plt.plot(x_line, y_line, 'b--', alpha=1.0, label=f'FP fit (r={r_value:.2f})')
    
    # 3. 面积vs IoU
    plt.subplot(2, 3, 3)
    if len(tp_iou_info['ious']) > 0:
        plt.scatter(np.log1p(tp_iou_info['areas']), tp_iou_info['ious'], alpha=0.5, label='TP', s=10, color='green')
    if len(fp_iou_info['ious']) > 0:
        plt.scatter(np.log1p(fp_iou_info['areas']), fp_iou_info['ious'], alpha=0.5, label='FP', s=10, color='red')
    plt.xlabel('log(Area)')
    plt.ylabel('IoU')
    plt.title('Area vs IoU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 不同IoU阈值下的性能
    plt.subplot(2, 3, 4)
    iou_thresholds = np.arange(0.5, 0.95, 0.05)
    precisions = []
    recalls = []
    
    for thr in iou_thresholds:
        tp_at_thr = np.sum(tp_iou_info['ious'] >= thr) if len(tp_iou_info['ious']) > 0 else 0
        fp_at_thr = np.sum(fp_iou_info['ious'] >= thr) if len(fp_iou_info['ious']) > 0 else 0
        total_at_thr = tp_at_thr + fp_at_thr
        
        precisions.append(tp_at_thr / max(total_at_thr, 1))
        recalls.append(tp_at_thr / max(len(tp_iou_info['ious']) + len(fp_iou_info['ious']), 1))
    
    plt.plot(iou_thresholds, precisions, 'bo-', label='Precision', linewidth=2)
    plt.plot(iou_thresholds, recalls, 'ro-', label='Recall', linewidth=2)
    plt.xlabel('IoU Threshold')
    plt.ylabel('Value')
    plt.title('Performance at Different IoU Thresholds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. TP和FP的统计摘要
    plt.subplot(2, 3, 5)
    stats_labels = ['TP Count', 'FP Count', 'TP Mean IoU', 'FP Mean IoU']
    stats_values = [
        len(tp_iou_info['ious']),
        len(fp_iou_info['ious']),
        np.mean(tp_iou_info['ious']) if len(tp_iou_info['ious']) > 0 else 0,
        np.mean(fp_iou_info['ious']) if len(fp_iou_info['ious']) > 0 else 0
    ]
    
    colors = ['green', 'red', 'green', 'red']
    bars = plt.bar(range(len(stats_labels)), stats_values, color=colors, alpha=0.7)
    plt.xticks(range(len(stats_labels)), stats_labels, rotation=45, ha='right')
    plt.ylabel('Value')
    plt.title('Detection Statistics')
    
    # 在柱子上添加数值
    for bar, val in zip(bars, stats_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')
    
    # 6. IoU累积分布函数
    plt.subplot(2, 3, 6)
    if len(tp_iou_info['ious']) > 0:
        sorted_tp_ious = np.sort(tp_iou_info['ious'])
        tp_cdf = np.arange(1, len(sorted_tp_ious)+1) / len(sorted_tp_ious)
        plt.plot(sorted_tp_ious, tp_cdf, 'g-', label='TP CDF', linewidth=2)
    
    if len(fp_iou_info['ious']) > 0:
        sorted_fp_ious = np.sort(fp_iou_info['ious'])
        fp_cdf = np.arange(1, len(sorted_fp_ious)+1) / len(sorted_fp_ious)
        plt.plot(sorted_fp_ious, fp_cdf, 'r-', label='FP CDF', linewidth=2)
    
    plt.xlabel('IoU')
    plt.ylabel('Cumulative Probability')
    plt.title('IoU Cumulative Distribution Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def tpfp_default(det_bboxes,
                gt_bboxes,
                gt_bboxes_ignore=None,
                iou_thr=0.5,
                area_ranges=None,
                verbose=False):
    """Enhanced version with complete multi-scale support.
    
    Args:
        det_bboxes (ndarray): Detected bboxes, shape (m, 6)
        gt_bboxes (ndarray): GT bboxes, shape (n, 5)
        gt_bboxes_ignore (ndarray): Ignored GT bboxes, shape (k, 5)
        iou_thr (float): IoU threshold
        area_ranges (list[tuple]): Area ranges for multi-scale evaluation
        verbose (bool): Whether to print debug information
        
    Returns:
        tuple: (tp, fp) arrays of shape (num_scales, num_dets)
    """
    # 输入验证
    det_bboxes = np.array(det_bboxes)
    if len(det_bboxes) == 0:
        num_scales = len(area_ranges) if area_ranges is not None else 1
        return (np.zeros((num_scales, 0), dtype=np.float32),
                np.zeros((num_scales, 0), dtype=np.float32))
    
    # 处理忽略的GT框
    if gt_bboxes_ignore is None or gt_bboxes_ignore.shape[0] == 0:
        gt_bboxes_ignore = np.zeros((0, 5), dtype=np.float32)
        gt_ignore_inds = np.zeros(gt_bboxes.shape[0], dtype=np.bool)
    else:
        gt_ignore_inds = np.concatenate(
            (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
             np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    
    gt_all = np.vstack((gt_bboxes, gt_bboxes_ignore))
    
    # 设置尺度范围
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    num_dets = det_bboxes.shape[0]
    num_gts = gt_all.shape[0]
    
    # 初始化结果数组
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    
    if verbose:
        print(f"\n=== TP/FP计算开始 ===")
        print(f"检测框数量: {num_dets}")
        print(f"GT框数量: {num_gts} (含{gt_bboxes_ignore.shape[0]}个忽略框)")
        print(f"尺度数量: {num_scales}")
        print(f"尺度范围: {area_ranges}")
    
    # 情况1: 没有GT框
    if num_gts == 0:
        if verbose:
            print("没有GT框，所有在尺度范围内的检测都是FP")
        
        for k, (min_area, max_area) in enumerate(area_ranges):
            for i in range(num_dets):
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                
                # 检查检测框是否在当前尺度范围内
                in_range = False
                if min_area is None and max_area is None:
                    in_range = True
                elif min_area is None:
                    in_range = area < max_area
                elif max_area is None:
                    in_range = area >= min_area
                else:
                    in_range = min_area <= area < max_area
                
                if in_range:
                    fp[k, i] = 1
        
        if verbose:
            print(f"FP分布: {[np.sum(fp[k, :]) for k in range(num_scales)]}")
        return tp, fp
    
    # 情况2: 有GT框，计算IoU
    try:
        ious = box_iou_rotated(
            torch.from_numpy(det_bboxes).float(),
            torch.from_numpy(gt_all).float()).numpy()
    except Exception as e:
        if verbose:
            print(f"IoU计算错误: {e}")
        # 返回空结果
        return tp, fp
    
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-det_bboxes[:, -1])
    
    # 计算GT框面积（用于尺度过滤）
    gt_areas = gt_all[:, 2] * gt_all[:, 3]
    
    if verbose:
        print(f"GT框面积范围: [{gt_areas.min():.1f}, {gt_areas.max():.1f}]")
        print(f"检测框面积范围: [{(det_bboxes[:, 2] * det_bboxes[:, 3]).min():.1f}, "
              f"{(det_bboxes[:, 2] * det_bboxes[:, 3]).max():.1f}]")
    
    # 多尺度评估
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        
        # 1. 计算当前尺度下的gt_area_ignore
        if min_area is None and max_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
            
            if min_area is None:
                # (None, max): 忽略面积大于等于max_area的GT
                gt_area_ignore = gt_areas >= max_area
            elif max_area is None:
                # (min, None): 忽略面积小于min_area的GT
                gt_area_ignore = gt_areas < min_area
            else:
                # (min, max): 忽略不在范围内的GT
                gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        
        if verbose and np.any(gt_area_ignore):
            print(f"尺度{k}({min_area}, {max_area}): "
                  f"忽略{np.sum(gt_area_ignore)}个GT框")
        
        # 2. 处理每个检测框
        for i in sort_inds:
            bbox = det_bboxes[i, :5]
            area = bbox[2] * bbox[3]
            
            # 检查检测框是否在当前尺度范围内
            in_scale_range = True
            if min_area is not None or max_area is not None:
                in_scale_range = False
                
                if min_area is None:
                    if area < max_area:
                        in_scale_range = True
                elif max_area is None:
                    if area >= min_area:
                        in_scale_range = True
                else:
                    if min_area <= area < max_area:
                        in_scale_range = True
            
            # 如果不在当前尺度范围内，跳过
            if not in_scale_range:
                continue
            
            # 获取最佳匹配的GT框
            matched_gt = ious_argmax[i]
            max_iou = ious_max[i]
            
            # 判断是TP还是FP
            if max_iou >= iou_thr:
                # 潜在的真阳性
                if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                        if verbose:
                            print(f"  检测框{i}: TP (IoU={max_iou:.3f}, 匹配GT{matched_gt})")
                    else:
                        fp[k, i] = 1
                        if verbose:
                            print(f"  检测框{i}: FP (重复匹配GT{matched_gt}, IoU={max_iou:.3f})")
                else:
                    # 忽略的GT框，视为FP
                    fp[k, i] = 1
                    if verbose:
                        ignore_type = "忽略GT" if gt_ignore_inds[matched_gt] else "尺度外GT"
                        print(f"  检测框{i}: FP ({ignore_type}, IoU={max_iou:.3f})")
            else:
                # IoU低于阈值，FP
                fp[k, i] = 1
                if verbose:
                    print(f"  检测框{i}: FP (IoU={max_iou:.3f} < {iou_thr})")
    
    if verbose:
        print(f"\nTP统计: {[np.sum(tp[k, :]) for k in range(num_scales)]}")
        print(f"FP统计: {[np.sum(fp[k, :]) for k in range(num_scales)]}")
        print("=== TP/FP计算结束 ===\n")
    
    return tp, fp

def tpfp_default_with_iou(det_bboxes,
                         gt_bboxes,
                         gt_bboxes_ignore=None,
                         iou_thr=0.5,
                         area_ranges=None,
                         verbose=False):
    """Enhanced version with complete multi-scale support and IoU distribution analysis.
    
    Args:
        det_bboxes (ndarray): Detected bboxes, shape (m, 6) [x, y, w, h, angle, score]
        gt_bboxes (ndarray): GT bboxes, shape (n, 5) [x, y, w, h, angle]
        gt_bboxes_ignore (ndarray): Ignored GT bboxes, shape (k, 5)
        iou_thr (float): IoU threshold
        area_ranges (list[tuple]): Area ranges for multi-scale evaluation
        verbose (bool): Whether to print debug information
        
    Returns:
        tuple: (tp, fp, tp_iou_info, fp_iou_info)
            - tp: TP array of shape (num_scales, num_dets)
            - fp: FP array of shape (num_scales, num_dets)
            - tp_iou_info: dict containing TP IoU statistics
            - fp_iou_info: dict containing FP IoU statistics
    """
    import numpy as np
    import torch
    
    # 输入验证
    det_bboxes = np.array(det_bboxes)
    if len(det_bboxes) == 0:
        num_scales = len(area_ranges) if area_ranges is not None else 1
        empty_iou_info = {
            'ious': np.array([], dtype=np.float32),
            'scores': np.array([], dtype=np.float32),
            'areas': np.array([], dtype=np.float32),
            'indices': np.array([], dtype=np.int32)
        }
        return (np.zeros((num_scales, 0), dtype=np.float32),
                np.zeros((num_scales, 0), dtype=np.float32),
                empty_iou_info,
                empty_iou_info)
    
    # 处理忽略的GT框
    if gt_bboxes_ignore is None or gt_bboxes_ignore.shape[0] == 0:
        gt_bboxes_ignore = np.zeros((0, 5), dtype=np.float32)
        gt_ignore_inds = np.zeros(gt_bboxes.shape[0], dtype=np.bool)
    else:
        gt_ignore_inds = np.concatenate(
            (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
             np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    
    gt_all = np.vstack((gt_bboxes, gt_bboxes_ignore))
    
    # 设置尺度范围
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    num_dets = det_bboxes.shape[0]
    num_gts = gt_all.shape[0]
    
    # 初始化结果数组
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    
    # 初始化IoU信息存储
    tp_iou_data = {
        'ious': np.full(num_dets, -1.0, dtype=np.float32),      # IoU值
        'scores': det_bboxes[:, -1].copy(),                     # 置信度分数
        'areas': det_bboxes[:, 2] * det_bboxes[:, 3],           # 面积
        'indices': np.arange(num_dets, dtype=np.int32),         # 检测框索引
        'scale_masks': np.zeros((num_scales, num_dets), dtype=bool)  # 尺度掩码
    }
    
    fp_iou_data = {
        'ious': np.full(num_dets, -1.0, dtype=np.float32),
        'scores': det_bboxes[:, -1].copy(),
        'areas': det_bboxes[:, 2] * det_bboxes[:, 3],
        'indices': np.arange(num_dets, dtype=np.int32),
        'scale_masks': np.zeros((num_scales, num_dets), dtype=bool)
    }
    
    if verbose:
        print(f"\n=== TP/FP计算开始（含IoU分析） ===")
        print(f"检测框数量: {num_dets}")
        print(f"GT框数量: {num_gts} (含{gt_bboxes_ignore.shape[0]}个忽略框)")
        print(f"尺度数量: {num_scales}")
        print(f"尺度范围: {area_ranges}")
        print(f"检测框置信度范围: [{det_bboxes[:, -1].min():.3f}, {det_bboxes[:, -1].max():.3f}]")
    
    # 情况1: 没有GT框
    if num_gts == 0:
        if verbose:
            print("没有GT框，所有在尺度范围内的检测都是FP")
        
        for k, (min_area, max_area) in enumerate(area_ranges):
            for i in range(num_dets):
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                
                # 检查检测框是否在当前尺度范围内
                in_range = False
                if min_area is None and max_area is None:
                    in_range = True
                elif min_area is None:
                    in_range = area < max_area
                elif max_area is None:
                    in_range = area >= min_area
                else:
                    in_range = min_area <= area < max_area
                
                if in_range:
                    fp[k, i] = 1
                    fp_iou_data['ious'][i] = 0.0  # 没有GT，IoU设为0
                    fp_iou_data['scale_masks'][k, i] = True
        
        if verbose:
            print(f"FP分布: {[np.sum(fp[k, :]) for k in range(num_scales)]}")
        
        return tp, fp, tp_iou_data, fp_iou_data
    
    # 情况2: 有GT框，计算IoU
    try:
        ious = box_iou_rotated(
            torch.from_numpy(det_bboxes[:, :5]).float(),
            torch.from_numpy(gt_all[:, :5]).float()).numpy()
    except Exception as e:
        if verbose:
            print(f"IoU计算错误: {e}")
        # 返回空结果
        empty_iou_info = {
            'ious': np.array([], dtype=np.float32),
            'scores': np.array([], dtype=np.float32),
            'areas': np.array([], dtype=np.float32),
            'indices': np.array([], dtype=np.int32),
            'scale_masks': np.array([], dtype=bool)
        }
        return tp, fp, empty_iou_info, empty_iou_info
    
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-det_bboxes[:, -1])
    
    # 计算GT框面积（用于尺度过滤）
    gt_areas = gt_all[:, 2] * gt_all[:, 3]
    
    if verbose:
        print(f"GT框面积范围: [{gt_areas.min():.1f}, {gt_areas.max():.1f}]")
        print(f"检测框面积范围: [{(det_bboxes[:, 2] * det_bboxes[:, 3]).min():.1f}, "
              f"{(det_bboxes[:, 2] * det_bboxes[:, 3]).max():.1f}]")
        print(f"IoU矩阵形状: {ious.shape}")
        print(f"最大IoU范围: [{ious_max.min():.3f}, {ious_max.max():.3f}]")
    
    # 存储每对匹配的详细IoU信息（用于后续分析）
    detailed_iou_info = {
        'all_ious': ious.copy(),           # 完整的IoU矩阵
        'best_gt_indices': ious_argmax,    # 最佳匹配的GT索引
        'det_scores': det_bboxes[:, -1],   # 检测框分数
        'det_areas': det_bboxes[:, 2] * det_bboxes[:, 3]  # 检测框面积
    }
    
    # 多尺度评估
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        
        # 1. 计算当前尺度下的gt_area_ignore
        if min_area is None and max_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
            
            if min_area is None:
                # (None, max): 忽略面积大于等于max_area的GT
                gt_area_ignore = gt_areas >= max_area
            elif max_area is None:
                # (min, None): 忽略面积小于min_area的GT
                gt_area_ignore = gt_areas < min_area
            else:
                # (min, max): 忽略不在范围内的GT
                gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        
        if verbose and np.any(gt_area_ignore):
            print(f"尺度{k}({min_area}, {max_area}): "
                  f"忽略{np.sum(gt_area_ignore)}个GT框")
        
        # 2. 处理每个检测框
        for i in sort_inds:
            bbox = det_bboxes[i, :5]
            area = bbox[2] * bbox[3]
            
            # 检查检测框是否在当前尺度范围内
            in_scale_range = True
            if min_area is not None or max_area is not None:
                in_scale_range = False
                
                if min_area is None:
                    if area < max_area:
                        in_scale_range = True
                elif max_area is None:
                    if area >= min_area:
                        in_scale_range = True
                else:
                    if min_area <= area < max_area:
                        in_scale_range = True
            
            # 如果不在当前尺度范围内，跳过
            if not in_scale_range:
                continue
            
            # 记录尺度掩码
            tp_iou_data['scale_masks'][k, i] = True
            fp_iou_data['scale_masks'][k, i] = True
            
            # 获取最佳匹配的GT框
            matched_gt = ious_argmax[i]
            max_iou = ious_max[i]
            
            # 判断是TP还是FP
            if max_iou >= iou_thr:
                # 潜在的真阳性
                if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                        tp_iou_data['ious'][i] = max_iou
                        if verbose:
                            print(f"  检测框{i}: TP (IoU={max_iou:.3f}, 匹配GT{matched_gt})")
                    else:
                        fp[k, i] = 1
                        fp_iou_data['ious'][i] = max_iou
                        if verbose:
                            print(f"  检测框{i}: FP (重复匹配GT{matched_gt}, IoU={max_iou:.3f})")
                else:
                    # 忽略的GT框，视为FP
                    fp[k, i] = 1
                    fp_iou_data['ious'][i] = max_iou
                    if verbose:
                        ignore_type = "忽略GT" if gt_ignore_inds[matched_gt] else "尺度外GT"
                        print(f"  检测框{i}: FP ({ignore_type}, IoU={max_iou:.3f})")
            else:
                # IoU低于阈值，FP
                fp[k, i] = 1
                fp_iou_data['ious'][i] = max_iou
                if verbose:
                    print(f"  检测框{i}: FP (IoU={max_iou:.3f} < {iou_thr})")
    
    # 过滤掉无效的IoU值（-1表示该检测框在当前尺度未被考虑）
    tp_mask = tp_iou_data['ious'] >= 0
    fp_mask = fp_iou_data['ious'] >= 0
    
    # 整理最终的IoU信息
    final_tp_iou_info = {
        'ious': tp_iou_data['ious'][tp_mask],
        'scores': tp_iou_data['scores'][tp_mask],
        'areas': tp_iou_data['areas'][tp_mask],
        'indices': tp_iou_data['indices'][tp_mask],
        'scale_masks': tp_iou_data['scale_masks'][:, tp_mask] if tp_mask.any() else np.array([], dtype=bool),
        'detailed_info': detailed_iou_info
    }
    
    final_fp_iou_info = {
        'ious': fp_iou_data['ious'][fp_mask],
        'scores': fp_iou_data['scores'][fp_mask],
        'areas': fp_iou_data['areas'][fp_mask],
        'indices': fp_iou_data['indices'][fp_mask],
        'scale_masks': fp_iou_data['scale_masks'][:, fp_mask] if fp_mask.any() else np.array([], dtype=bool),
        'detailed_info': detailed_iou_info
    }
    
    if verbose:
        print(f"\nTP统计: {[np.sum(tp[k, :]) for k in range(num_scales)]}")
        print(f"FP统计: {[np.sum(fp[k, :]) for k in range(num_scales)]}")
        print(f"TP IoU数量: {len(final_tp_iou_info['ious'])}")
        print(f"FP IoU数量: {len(final_fp_iou_info['ious'])}")
        if len(final_tp_iou_info['ious']) > 0:
            print(f"TP IoU范围: [{final_tp_iou_info['ious'].min():.3f}, {final_tp_iou_info['ious'].max():.3f}]")
            print(f"TP平均IoU: {final_tp_iou_info['ious'].mean():.3f}")
        if len(final_fp_iou_info['ious']) > 0:
            print(f"FP IoU范围: [{final_fp_iou_info['ious'].min():.3f}, {final_fp_iou_info['ious'].max():.3f}]")
            print(f"FP平均IoU: {final_fp_iou_info['ious'].mean():.3f}")
        print("=== TP/FP计算结束（含IoU分析） ===\n")
    
    return tp, fp, final_tp_iou_info, final_fp_iou_info

# def tpfp_default(det_bboxes,
#                  gt_bboxes,
#                  gt_bboxes_ignore=None,
#                  iou_thr=0.5,
#                  area_ranges=None):
#     """Check if detected bboxes are true positive or false positive.

#     Args:
#         det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6).
#         gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5).
#         gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
#             of shape (k, 5). Default: None
#         iou_thr (float): IoU threshold to be considered as matched.
#             Default: 0.5.
#         area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
#             in the format [(min1, max1), (min2, max2), ...]. Default: None.

#     Returns:
#         tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
#             each array is (num_scales, m).
#     """
#     # an indicator of ignored gts
#     det_bboxes = np.array(det_bboxes)
#     gt_ignore_inds = np.concatenate(
#         (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
#          np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
#     # stack gt_bboxes and gt_bboxes_ignore for convenience
#     gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

#     num_dets = det_bboxes.shape[0]
#     num_gts = gt_bboxes.shape[0]
#     if area_ranges is None:
#         area_ranges = [(None, None)]
#     num_scales = len(area_ranges)
#     # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
#     # a certain scale
#     tp = np.zeros((num_scales, num_dets), dtype=np.float32)
#     fp = np.zeros((num_scales, num_dets), dtype=np.float32)

#     # if there is no gt bboxes in this image, then all det bboxes
#     # within area range are false positives
#     if gt_bboxes.shape[0] == 0:
#         if area_ranges == [(None, None)]:
#             fp[...] = 1
#         else:
#             raise NotImplementedError
#         return tp, fp
    
#     # print('gt_bboxes: ', gt_bboxes)
#     # print('det_bboxes: ', det_bboxes)

#     ious = box_iou_rotated(
#         torch.from_numpy(det_bboxes).float(),
#         torch.from_numpy(gt_bboxes).float()).numpy()
#     # print('ious: ', ious)
#     # for each det, the max iou with all gts
#     ious_max = ious.max(axis=1)
#     # for each det, which gt overlaps most with it
#     ious_argmax = ious.argmax(axis=1)
#     # sort all dets in descending order by scores
#     sort_inds = np.argsort(-det_bboxes[:, -1])
#     for k, (min_area, max_area) in enumerate(area_ranges):
#         gt_covered = np.zeros(num_gts, dtype=bool)
#         # if no area range is specified, gt_area_ignore is all False
#         if min_area is None:
#             gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
#         else:
#             raise NotImplementedError
#         for i in sort_inds:
#             if ious_max[i] >= iou_thr:
#                 matched_gt = ious_argmax[i]
#                 if not (gt_ignore_inds[matched_gt]
#                         or gt_area_ignore[matched_gt]):
#                     if not gt_covered[matched_gt]:
#                         gt_covered[matched_gt] = True
#                         tp[k, i] = 1
#                     else:
#                         fp[k, i] = 1
#                 # otherwise ignore this detected bbox, tp = 0, fp = 0
#             elif min_area is None:
#                 fp[k, i] = 1
#             else:
#                 bbox = det_bboxes[i, :5]
#                 area = bbox[2] * bbox[3]
#                 if area >= min_area and area < max_area:
#                     fp[k, i] = 1
#     return tp, fp


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]

    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])

        else:
            cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))

    return cls_dets, cls_gts, cls_gts_ignore


def eval_rbbox_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   save_path=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)
    
    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset
    
    if save_path is not None:
        save_dir = os.path.dirname(save_path)  # 获取目录部分
        original_name = os.path.basename(save_path)  # 获取文件名部分

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    tp_iou_info_all = []
    fp_iou_info_all = []
    # for i in range(num_classes):
    for i, label_name in enumerate(label_names):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes
        # tpfp = pool.starmap(
        #     tpfp_default,
        #     zip(cls_dets, cls_gts, cls_gts_ignore,
        #         [iou_thr for _ in range(num_imgs)],
        #         [area_ranges for _ in range(num_imgs)]))
        # tp, fp = tuple(zip(*tpfp))
        # _, _, tp_iou_info, fp_iou_info = tpfp_default_with_iou(cls_dets, cls_gts, cls_gts_ignore, iou_thr, area_ranges)
        tpfp = pool.starmap(
            tpfp_default_with_iou,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp, tp_iou_info, fp_iou_info = tuple(zip(*tpfp))
        
        if save_path is not None:
            tp_iou_info = combine_iou_info_list(tp_iou_info)
            fp_iou_info = combine_iou_info_list(fp_iou_info)
            # print('tp: ', tp)
            # print('tp_iou_info: ', len(tp_iou_info))
            # print('tp_iou_info: ', len(tp_iou_info['ious']))
            tp_iou_info_all.append(tp_iou_info)
            fp_iou_info_all.append(fp_iou_info)
            
            # 分析IoU分布
            if scale_ranges is not None:
                scale_names = ['small', 'medium', 'large']
            else:
                scale_names = None
            analysis = analyze_iou_distribution_from_tpfp(
                tp_iou_info, fp_iou_info, iou_thr=0.5, scale_names=scale_names
            )
            
            # 打印分析结果
            if analysis:
                print("\n" + "=" * 60)
                print("{} IoU分布分析结果".format(label_name))
                print("=" * 60)
                
                print(f"\n基础统计:")
                print(f"  TP数量: {analysis['basic_stats']['tp_count']}")
                print(f"  FP数量: {analysis['basic_stats']['fp_count']}")
                print(f"  总检测数: {analysis['basic_stats']['total_detections']}")
                print(f"  精确率: {analysis['basic_stats']['precision']:.3f}")
                
                if 'tp_stats' in analysis and analysis['tp_stats']:
                    print(f"\nTP统计:")
                    print(f"  平均IoU: {analysis['tp_stats']['mean_iou']:.3f}")
                    print(f"  IoU标准差: {analysis['tp_stats']['std_iou']:.3f}")
                    print(f"  IoU中位数: {analysis['tp_stats']['median_iou']:.3f}")
                    print(f"  IoU范围: [{analysis['tp_stats']['min_iou']:.3f}, {analysis['tp_stats']['max_iou']:.3f}]")
                
                if 'fp_stats' in analysis and analysis['fp_stats']:
                    print(f"\nFP统计:")
                    print(f"  平均IoU: {analysis['fp_stats']['mean_iou']:.3f}")
                    print(f"  IoU标准差: {analysis['fp_stats']['std_iou']:.3f}")
                    print(f"  IoU中位数: {analysis['fp_stats']['median_iou']:.3f}")
                    print(f"  IoU范围: [{analysis['fp_stats']['min_iou']:.3f}, {analysis['fp_stats']['max_iou']:.3f}]")
                
                # 可视化

                save_name = label_name + '_' + original_name  # 构建新文件名
                save_path = os.path.join(save_dir, save_name)  # 组合新路径
                visualize_iou_distribution(
                    tp_iou_info, fp_iou_info, 
                    save_path=save_path,
                    title_prefix="Test {} ".format(label_name)
                )
                
                print(f"\n可视化图表已保存为: {label_name}_iou_distribution.png")
        
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        #mode = '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    
    if save_path is not None:
        tp_iou_info_all = combine_iou_info_list(tp_iou_info_all)
        fp_iou_info_all = combine_iou_info_list(fp_iou_info_all)
        analysis = analyze_iou_distribution_from_tpfp(
            tp_iou_info_all, fp_iou_info_all, iou_thr=0.5, scale_names=scale_names
        )
        
        # 打印分析结果
        if analysis:
            print("\n" + "=" * 60)
            print("全部类 IoU分布分析结果")
            print("=" * 60)
            
            print(f"\n基础统计:")
            print(f"  TP数量: {analysis['basic_stats']['tp_count']}")
            print(f"  FP数量: {analysis['basic_stats']['fp_count']}")
            print(f"  总检测数: {analysis['basic_stats']['total_detections']}")
            print(f"  精确率: {analysis['basic_stats']['precision']:.3f}")
            
            if 'tp_stats' in analysis and analysis['tp_stats']:
                print(f"\nTP统计:")
                print(f"  平均IoU: {analysis['tp_stats']['mean_iou']:.3f}")
                print(f"  IoU标准差: {analysis['tp_stats']['std_iou']:.3f}")
                print(f"  IoU中位数: {analysis['tp_stats']['median_iou']:.3f}")
                print(f"  IoU范围: [{analysis['tp_stats']['min_iou']:.3f}, {analysis['tp_stats']['max_iou']:.3f}]")
            
            if 'fp_stats' in analysis and analysis['fp_stats']:
                print(f"\nFP统计:")
                print(f"  平均IoU: {analysis['fp_stats']['mean_iou']:.3f}")
                print(f"  IoU标准差: {analysis['fp_stats']['std_iou']:.3f}")
                print(f"  IoU中位数: {analysis['fp_stats']['median_iou']:.3f}")
                print(f"  IoU范围: [{analysis['fp_stats']['min_iou']:.3f}, {analysis['fp_stats']['max_iou']:.3f}]")
            
            # 可视化

            save_name = 'all_' + original_name  # 构建新文件名
            save_path = os.path.join(save_dir, save_name)  # 组合新路径
            visualize_iou_distribution(
                tp_iou_info, fp_iou_info, 
                save_path=save_path,
                title_prefix="Test all "
            )
            
            print(f"\n可视化图表已保存为: all_iou_distribution.png")

    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results


def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    precisions = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        if cls_result['precision'].size > 0:
            precisions[:, i] = np.array(cls_result['precision'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall','precision', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{precisions[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', f'{recalls[i,:].mean():.3f}',f'{precisions[i,:].mean():.3f}', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
