# Copyright (c) OpenMMLab. All rights reserved.
from .atss_kld_assigner import ATSSKldAssigner
from .atss_obb_assigner import ATSSObbAssigner
from .convex_assigner import ConvexAssigner
from .max_convex_iou_assigner import MaxConvexIoUAssigner
from .sas_assigner import SASAssigner
from .coarse2fine_assigner import C2FAssigner
from .ranking_assigner import RRankingAssigner

from .class_aware_assigner import (CAAssigner_v1, CAAssigner_v2, CAAssigner_v3, CAAssigner_v41, CAAssigner_v42,
    CAAssigner_v50)

from .rotated_atss_assigner import RotatedATSSAssigner

__all__ = [
    'ConvexAssigner', 'MaxConvexIoUAssigner', 'SASAssigner', 'ATSSKldAssigner',
    'ATSSObbAssigner','C2FAssigner','RRankingAssigner',
    
    'CAAssigner_v1', 'CAAssigner_v2', 'CAAssigner_v3', 'CAAssigner_v41', 'CAAssigner_v42', 
    'CAAssigner_v50',
    
    'RotatedATSSAssigner'
]
