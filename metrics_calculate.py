from mmseg.core import eval_metrics,eval_attach_metrics
from mmseg.apis import inference_segmentor, init_segmentor,inference_segmentor_concactdata
import mmcv
import os
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from skimage import io
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument("--seg-path", type=str,
                        help="Path to dataset files on which inference is performed.")
    parser.add_argument("--test-path", type=str,
                        help="Where to save predicted mask.")
    return parser.parse_args()
    
    
def evaluate(results,
             gt_seg_maps,
             num_classes=2,
             metric='mIoU',
             logger=None,
             ignore_index=255,
             label_map=None,
             reduce_zero_label=False,
             class_names=['other','water'],
             att_metrics = ['PRE','REC','F-measure','F-max','FPR','FNR'],
             efficient_test=False,
             **kwargs):
    """Evaluate the dataset.

    Args:
        results (list): Testing results of the dataset.
        metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
            'mDice' are supported.
        logger (logging.Logger | None | str): Logger used for printing
            related information during evaluation. Default: None.

    Returns:
        dict[str, float]: Default metrics.
    """

    if isinstance(metric, str):
        metric = [metric]
    allowed_metrics = ['mIoU', 'mDice']  # 'PRE','REC','F-measure','F-max','FPR','FNR'
    if not set(metric).issubset(set(allowed_metrics)):
        raise KeyError('metric {} is not supported'.format(metric))
    eval_results = {}
    ret_metrics = eval_metrics(
        results,
        gt_seg_maps,
        num_classes,
        ignore_index,
        metric,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
    ret_metrics_round = [
        np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
    ]
    for i in range(num_classes):
        class_table_data.append([class_names[i]] +
                                [m[i] for m in ret_metrics_round[2:]] +
                                [ret_metrics_round[1][i]])
    ret_metrics_mean = [
        np.round(np.nanmean(ret_metric) * 100, 2)
        for ret_metric in ret_metrics
    ]
    if att_metrics is not None:
        attach_metrics = eval_attach_metrics(
            results,
            gt_seg_maps,
            num_classes,
            ignore_index,
            att_metrics,
            label_map=label_map,
            reduce_zero_label=reduce_zero_label)
        summary_table_data = \
            [['Scope'] + ['m' + head for head in class_table_data[0][1:]] + [attach for attach in att_metrics] + [
                'aAcc']]
        summary_table_data.append(
            ['global'] + ret_metrics_mean[2:] + [ret_metrics_mean[1]] + [att_metrics for att_metrics in
                                                                         attach_metrics] + [ret_metrics_mean[0]])
    else:
        summary_table_data = [['Scope'] + ['m' + head for head in class_table_data[0][1:]] + ['aAcc']]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] + [ret_metrics_mean[1]] + [ret_metrics_mean[0]])

    print_log('per class results:', logger)
    table = AsciiTable(class_table_data)
    print_log('\n' + table.table, logger=logger)
    print_log('Summary:', logger)
    table = AsciiTable(summary_table_data)
    print_log('\n' + table.table, logger=logger)

    for i in range(1, len(summary_table_data[0])):
        eval_results[summary_table_data[0]
        [i]] = summary_table_data[1][i] / 100.0
    if mmcv.is_list_of(results, str):
        for file_name in results:
            os.remove(file_name)
    return eval_results



def main():
    args = parse_args()
    seg_list=[]
    test_list=[]
    for file_path  in os.listdir(args.seg_path):
        if os.path.isfile(os.path.join(args.seg_path, file_path)) == True:
            seg_file = os.path.join(args.seg_path, file_path)
            test_file = os.path.join(args.test_path, file_path)
            seg = io.imread(seg_file)
            test = io.imread(test_file)
            if seg.shape == test.shape:
               test_list.append(test)
               seg_list.append(seg)
    evaluate(test_list,seg_list)
if __name__ == '__main__':
    main()
