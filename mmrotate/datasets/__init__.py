# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset  # noqa: F401, F403
from .dota import DOTADataset  # noqa: F401, F403
from .hrsc import HRSCDataset  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403
from .sar import SARDataset  # noqa: F401, F403
from .dotav2 import DOTAv2Dataset
from .dotav1_5 import DOTAv1_5Dataset
from .dior import DIORDataset
from .dior_txt import DIOR_txt_Dataset
from .hrsc2016_txt import HRSC2016_txt_Dataset

from .fair1m import FAIR1MDataset
from .dota_hm import DOTA_HM_Dataset
from .fddv1_0 import FDDv1_0Dataset
from .ucas_aod import UCAS_AODv10Dataset

from .dotav1_5_random import DOTAv1_5Dataset_RT02, DOTAv1_5Dataset_RT04, DOTAv1_5Dataset_RT06

from .sku110k_r import SKU110k_RDataset

__all__ = ['SARDataset', 'DOTADataset', 'build_dataset', 'HRSCDataset',
            'DOTAv2Dataset','DIORDataset','DOTAv1_5Dataset',
            'FAIR1MDataset', 'DIOR_txt_Dataset', 'HRSC2016_txt_Dataset',
            'DOTA_HM_Dataset', 'FDDv1_0Dataset', 'UCAS_AODv10Dataset',
            
            'DOTAv1_5Dataset_RT02', 'DOTAv1_5Dataset_RT04', 'DOTAv1_5Dataset_RT06',
            
            'SKU110k_RDataset'

]
