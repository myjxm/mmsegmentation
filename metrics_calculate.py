from mmseg.core import eval_metrics,eval_attach_metrics
import mmcv
import os
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from skimage import io
from prettytable import PrettyTable
from collections import OrderedDict
import argparse
import requests
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument("--seg-path", type=str,
                       help="Path to dataset files on which inference is performed.")
    parser.add_argument("--test-path", type=str,
                        help="Where to save predicted mask.")
    parser.add_argument("--modelname", type=str, default='',
                        help="modelname")
    parser.add_argument("--roc", type=float, default=-1,
                        help="roc threshold")
    parser.add_argument("--dataset", type=str, default='',
                        help="validation dataset")
    return parser.parse_args()
    
    
def evaluate(results,
             gt_seg_maps,
             num_classes=2,
             metric=['mIoU', 'mFscore','mFpr','mFnr','kappa','mcc','hloss'],
             logger=None,
             ignore_index=255,
             label_map=None,
             reduce_zero_label=False,
             class_names=['other','water'],
             #att_metrics = ['PRE','REC','F-measure','F-max','FPR','FNR'],
             att_metrics = ['Grmse','Gmax'],
             #att_metrics = None,
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
    allowed_metrics = ['mIoU', 'mDice','mFscore','mFpr','mFnr','kappa','mcc','hloss']  # 'PRE','REC','F-measure','F-max','FPR','FNR'
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
    # summary table
    ret_metrics_summary = OrderedDict({
        ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })

    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': class_names})
    ret_metrics_class.move_to_end('Class', last=False)

    # for logger
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            summary_table_data.add_column(key, [val])
        else:
            summary_table_data.add_column('m' + key, [val])
    if att_metrics is not None:
        attach_metrics = eval_attach_metrics(
            results,
            gt_seg_maps,
            num_classes,
            ignore_index,
            att_metrics,
            label_map=label_map,
            reduce_zero_label=reduce_zero_label)
        for key,value in attach_metrics.items():
            summary_table_data.add_column(key,[value])
        for key, value in attach_metrics.items():
            eval_results[key] = value
    print_log('per class results:', logger)
    print_log('\n' + class_table_data.get_string(), logger=logger)
    print_log('Summary:', logger)
    print_log('\n' + summary_table_data.get_string(), logger=logger)
    # each metric dict
    for key, value in ret_metrics_summary.items():
        if key == 'aAcc':
            eval_results[key] = value
        else:
            eval_results['m' + key] = value

    ret_metrics_class.pop('Class', None)
    for key, value in ret_metrics_class.items():
        eval_results.update({
            key + '.' + str(name): value[idx]
            for idx, name in enumerate(class_names)
        })

    # class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
    # ret_metrics_round = [
    #     np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
    # ]
    # for i in range(num_classes):
    #     class_table_data.append([class_names[i]] +
    #                             [m[i] for m in ret_metrics_round[2:]] +
    #                             [ret_metrics_round[1][i]])
    # ret_metrics_mean = [
    #     np.round(np.nanmean(ret_metric) * 100, 2)
    #     for ret_metric in ret_metrics
    # ]
    # if att_metrics is not None:
    #     attach_metrics = eval_attach_metrics(
    #         results,
    #         gt_seg_maps,
    #         num_classes,
    #         ignore_index,
    #         att_metrics,
    #         label_map=label_map,
    #         reduce_zero_label=reduce_zero_label)
    #     summary_table_data = \
    #         [['Scope'] + ['m' + head for head in class_table_data[0][1:]] + [attach for attach in att_metrics] + [
    #             'aAcc']]
    #     summary_table_data.append(
    #         ['global'] + ret_metrics_mean[2:] + [ret_metrics_mean[1]] + [att_metrics for att_metrics in
    #                                                                      attach_metrics] + [ret_metrics_mean[0]])
    # else:
    #     summary_table_data = [['Scope'] + ['m' + head for head in class_table_data[0][1:]] + ['aAcc']]
    #     summary_table_data.append(['global'] + ret_metrics_mean[2:] + [ret_metrics_mean[1]] + [ret_metrics_mean[0]])
    #
    # print_log('per class results:', logger)
    # table = AsciiTable(class_table_data)
    # print_log('\n' + table.table, logger=logger)
    # print_log('Summary:', logger)
    # table = AsciiTable(summary_table_data)
    # print_log('\n' + table.table, logger=logger)

    # for i in range(1, len(summary_table_data[0])):
    #     eval_results[summary_table_data[0][i]] = summary_table_data[1][i] / 100.0
    # if mmcv.is_list_of(results, str):
    #     for file_name in results:
    #         os.remove(file_name)
    return eval_results



def main():
    args = parse_args()
    url = r"http://localhost:8080/query/statistic_status/" + args.modelname + "/" + str(args.roc) + "/" + str(args.dataset)
    res = requests.get(url)
    res = json.loads(res.text)
    print(res)
    if res['code'] == 0 and len(res['data']) > 0 :
        if res['data'][0]['metric_status'] == 'Y':
           return
    print(res)
    url = r"http://localhost:8080/init_insert/statistic_class/" + args.modelname + "/" + str(args.roc) + "/" + str(args.dataset)
    res = requests.get(url)
    print(json.loads(res.text))
    seg_list=[]
    test_list=[]
    classes = 2;
    for file_path  in os.listdir(args.seg_path):
        if os.path.isfile(os.path.join(args.seg_path, file_path)) == True:
            seg_file = os.path.join(args.seg_path, file_path)
            test_file = os.path.join(args.test_path, file_path)
            seg = io.imread(seg_file)
            test = io.imread(test_file)
            if len(np.unique(seg))> classes:
                classes = len(np.unique(seg))
            if seg.shape == test.shape:
               test_list.append(test)
               seg_list.append(seg)
    if classes > 2:
        eval_results = evaluate(test_list, seg_list, num_classes=3, class_names=['other','water','sky'])
    else:
        eval_results = evaluate(test_list, seg_list, num_classes=2, class_names=['other', 'water'])
    eval_results['modelname'] = args.modelname
    eval_results['roc'] = args.roc
    eval_results['dataset'] = args.dataset
    if classes > 2:
        url = r"http://localhost:8080/update/statistic_three_class/"
    else:
        url = r"http://localhost:8080/update/statistic_two_class/"
    headers = {'content-type': 'application/json'}
    #requests.post(url, json=json.dumps(eval_results),headers=headers)
    res = requests.post(url, json=eval_results)
    #print(res)
    print(json.loads(res.text))
    url = r"http://localhost:8080/update/statistic_status/" + args.modelname + "/" + str(args.roc) + "/" + str(args.dataset)
    res = requests.get(url)
    print(json.loads(res.text))
if __name__ == '__main__':
    main()
