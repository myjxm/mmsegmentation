import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class Zhuhai15708_3class(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = ('other','water','sky')

    PALETTE = [[0, 0, 0], [128, 0, 0],[128,128,0]]

    def __init__(self, split, **kwargs):
        super(Zhuhai15708_3class, self).__init__(
            split=split,
            reduce_zero_label=False,
            #att_metrics = ['PRE','REC','F-measure','F-max','FPR','FNR'],
            #att_metrics=['Grmse','Gmax'],  ##训练不能价att_metrics因为pre_eval_to_metrics(results, metric)
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
