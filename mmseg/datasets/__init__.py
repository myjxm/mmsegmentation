# Copyright (c) OpenMMLab. All rights reserved.
from .ade import ADE20KDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
from .coco_stuff import COCOStuffDataset
from .custom import CustomDataset
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .drive import DRIVEDataset
from .hrf import HRFDataset
from .night_driving import NightDrivingDataset
from .pascal_context import PascalContextDataset, PascalContextDataset59
from .stare import STAREDataset
from .voc import PascalVOCDataset
from .mastr1325 import MaSTr1325
from .combine import Combine
from .zhuhai15708 import Zhuhai15708
from .zhuhai15708_3class import Zhuhai15708_3class
from .zhuhai12749_3class import Zhuhai12749_3class
__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
    'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
    'PascalContextDataset59', 'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset',
    'STAREDataset', 'DarkZurichDataset', 'NightDrivingDataset',
    'COCOStuffDataset','MaSTr1325','Combine','Zhuhai15708','Zhuhai15708_3class','Zhuhai12749_3class'
]
