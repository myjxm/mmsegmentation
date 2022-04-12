# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import pandas as pd


def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]


    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)

        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result

def get_confusion_matrix(pred_label, label, num_classes, ignore_index,label_map,reduce_zero_label):
    """Intersection over Union
       Args:
           pred_label (np.ndarray): 2D predict map
           label (np.ndarray): label 2D label map
           num_classes (int): number of categories
           ignore_index (int): index ignore in evaluation
       """
    #print(pred_label[0][0])
    #print(type(pred_label[0][0]))
    if type(pred_label).__name__ == 'list':
       pred_label = [np.array(list(t)) for t in pred_label]
    if type(label).__name__ == 'generator':
       label = [np.array(t) for t in list(label)]
    total_mat = np.zeros((num_classes,num_classes), dtype=np.float)
    num_imgs = len(pred_label)
    assert len(label) == num_imgs
    for i in range(num_imgs):
        if label_map is not None:
            for old_id, new_id in label_map.items():
                label[i][label == old_id] = new_id
        #label[i][label[i] ==4 ] = 0
        if reduce_zero_label:
            # avoid using underflow conversion
            label[i][label[i] == 0] = 255
            label[i] = label[i] - 1
            label[i][label[i] == 254] = 255
        #mask = (label[i] != ignore_index)
        mask = (pred_label[i] <= 1)
        pred_label[i] = pred_label[i][mask]
        label[i] = label[i][mask]
        #print(pred_label[i].shape)
        #print(label[i].shape)
        n = num_classes
        inds = n * label[i] + pred_label[i]
        #print(np.bincount(inds, minlength=n**2).shape)
        mat = np.bincount(inds, minlength=n**2).reshape(n, n)
        #print(mat)
        total_mat += mat

    return total_mat

def eval_attach_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=None,
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):
    """Calculate user attach evaluation metrics
        Args:
            results (list[ndarray]): List of prediction segmentation maps.
            gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.
            metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be replaced
                by the numbers defined by the user. Default: None.
            label_map (dict): Mapping old labels to new labels. Default: dict().
            reduce_zero_label (bool): Wether ignore zero label. Default: False.
         Returns:
             float: Overall accuracy on all images.
             ndarray: Per category accuracy, shape (num_classes, ).
             ndarray: Per category evalution metrics, shape (num_classes, ).
           """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['PRE','REC','F-measure','F-max','FPR','FNR','Grmse','Gmax']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    att_metrics = OrderedDict()
    imagesnum=len(results)
    width=results[0].shape[1]
    high=results[0].shape[0]
    #print(imagesnum)
    #print(width)
    #print(high)
    #pd.DataFrame(np.argwhere(results[0] == 1), columns=['value', 'key']).sort_values(by='value')['key'].tolist()

    y = np.array([np.array(pd.DataFrame(np.argwhere(np.row_stack((result,np.ones(width,dtype='int'))) == 1), columns=['value', 'key']).groupby('key',as_index=False).min().sort_values(by='key')['value'].tolist()) for result in results])
    ygt = np.array([np.array(pd.DataFrame(np.argwhere(np.row_stack((gt_seg_map,np.ones(width,dtype='int'))) == 1), columns=['value', 'key']).groupby('key',as_index=False).min().sort_values(by='key')['value'].tolist()) for gt_seg_map in gt_seg_maps])
    for metric in metrics:
        if metric == 'Grmse':
            #print(type(y[0]))
            #print(y.shape)
            #print(type(ygt[0]))
            #print(ygt.shape)
            #print(np.square(y-ygt).sum(1))
            #print(np.sqrt(np.square(y-ygt).sum(1)))
            #aa = y-ygt
            att_metrics['Grmse'] = (np.sqrt(np.square(y-ygt).sum(1)/(width))).sum()/imagesnum
            #print(att_metrics)
            #print(type(att_metrics['Grmse']))
        if metric == 'Gmax':
            att_metrics['Gmax'] = (np.max(np.absolute(y-ygt), 1).sum())/imagesnum

    ret_metrics = {
        metric: value
        for metric, value in att_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics
    # mat = get_confusion_matrix(results,gt_seg_maps,num_classes,ignore_index,label_map,reduce_zero_label)
    # tp,tn,fp,fn = 1,1,1,1 ##初始值
    # if num_classes ==2 :
    #     tp = mat[1,1]
    #     tn = mat[0,0]
    #     fn = mat[1,0]
    #     fp = mat[0,1]
    # pre = tp / (tp + fp)
    # rec = tp / (tp + fn)
    # fmeasures = 2 * (pre * rec) / (pre + rec)
    # fmax = fmeasures
    # fpr = fp / (fp + tn)
    # fnr = fn / (tp + fn)
    # att_metrics = []
    # for metric in metrics:
    #     if metric == 'PRE':
    #         att_metrics.append(pre)
    #     elif metric == 'REC':
    #         att_metrics.append(rec)
    #     elif metric == 'F-measure':
    #         att_metrics.append(fmeasures)
    #     elif metric == 'F-max':
    #         att_metrics.append(fmax)
    #     elif metric == 'FPR':
    #         att_metrics.append(fpr)
    #     elif metric == 'FNR':
    #         att_metrics.append(fnr)
    # if nan_to_num is not None:
    #     att_metrics = [
    #         np.nan_to_num(metric, nan=nan_to_num) for metric in att_metrics
    #     ]
    # dict_metrics = dict(zip(metrics,att_metrics))
    # return dict_metrics



def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def pre_eval_to_metrics(pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    #print(pre_eval_results)
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4
    #print('pre_eval_results')
    #print(len(pre_eval_results[0])) 评估集的大小
    eval_img_nums = len(pre_eval_results[0])
    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta,eval_img_nums=eval_img_nums)

    return ret_metrics


def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1,eval_img_nums=3674):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore','mFpr','mFnr','kappa','mcc','hloss']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    all_precision = total_area_intersect.sum() / total_area_pred_label.sum()
    #print('all_precision')
    #print(all_precision)
    ret_metrics = OrderedDict({'aAcc': all_acc})
    #ret_metrics['aPre'] = all_precision
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall
        elif metric == 'mFpr':
            #print(total_area_pred_label)
            #print(total_area_intersect)
            #print(total_area_label)
            #print(512*512*eval_img_nums)
            #print(512*512*eval_img_nums - total_area_label)
            #print(total_area_pred_label.sum()/360)
            fpr = (total_area_pred_label-total_area_intersect)/(total_area_pred_label.sum() - total_area_label)
            ret_metrics['fpr'] = fpr
        elif metric == 'mFnr':
            fnr = (total_area_union - total_area_pred_label) / total_area_label
            ret_metrics['fnr'] = fnr
        elif metric == 'kappa':
            pe = (total_area_pred_label * total_area_label + (total_area_pred_label.sum()-total_area_union + total_area_label - total_area_intersect)*(total_area_pred_label.sum() - total_area_label))/(total_area_pred_label.sum()*total_area_pred_label.sum())
            allacc = (total_area_pred_label.sum()-total_area_union+total_area_intersect) / total_area_pred_label.sum()
            kappa = 1 - (1 - allacc) / (1 - pe)
            print('allacc')
            print(allacc)
            print('pe')
            print(pe)
            ret_metrics['kappa'] = kappa
        elif metric == 'mcc':
            mcc=((total_area_pred_label.sum() - total_area_union) * total_area_intersect - (
                        total_area_union - total_area_label)*(total_area_label - total_area_intersect)) / \
            np.sqrt(total_area_pred_label * total_area_label * (total_area_pred_label.sum() - total_area_label) * (
                        total_area_pred_label.sum() - total_area_union + total_area_label - total_area_intersect))
            ret_metrics['mcc'] = mcc
        elif  metric == 'hloss':
            hloss = (total_area_union - total_area_intersect) / total_area_pred_label.sum()
            ret_metrics['hloss'] = hloss

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics


if __name__ == '__main__':
    fusion = eval_attach_metrics(np.array([[1,1,1,0,0,1,0,0,0],[1,0,0,1,0,1,1,1,0]]),np.array([[1,0,1,1,0,1,0,0,1],[1,0,0,1,0,0,0,0,0]]),2,255,metrics=['PRE','REC','F-measure','F-max','FPR','FNR'])
    metris = eval_metrics(np.array([[1,1,1,0,0,1,0,0,0],[1,0,0,1,0,1,1,1,0]]),np.array([[1,0,1,1,0,1,0,0,1],[1,0,0,1,0,0,0,0,0]]),2,255,metrics=['mIoU'])
    print(fusion)
    print(metris)
