# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import onnxruntime as ort
import torch
from mmcv.ops import get_onnxruntime_op_path
from mmcv.tensorrt import (TRTWraper, is_tensorrt_plugin_loaded, onnx2trt,
                           save_trt_engine)

from mmseg.apis.inference import LoadImage
from mmseg.datasets import DATASETS
from mmseg.datasets.pipelines import Compose
import datetime

def get_GiB(x: int):
    """return x GiB."""
    return x * (1 << 30)


def _prepare_input_img(img_path: str,
                       test_pipeline: Iterable[dict],
                       shape: Optional[Iterable] = None,
                       rescale_shape: Optional[Iterable] = None) -> dict:
    # build the data pipeline
    if shape is not None:
        test_pipeline[1]['img_scale'] = (shape[1], shape[0])
    test_pipeline[1]['transforms'][0]['keep_ratio'] = False
    test_pipeline = [LoadImage()] + test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img_path)
    data = test_pipeline(data)
    imgs = data['img']
    img_metas = [i.data for i in data['img_metas']]

    if rescale_shape is not None:
        for img_meta in img_metas:
            img_meta['ori_shape'] = tuple(rescale_shape) + (3, )

    mm_inputs = {'imgs': imgs, 'img_metas': img_metas}

    return mm_inputs


def _update_input_img(img_list: Iterable, img_meta_list: Iterable):
    # update img and its meta list
    N = img_list[0].size(0)
    img_meta = img_meta_list[0][0]
    img_shape = img_meta['img_shape']
    ori_shape = img_meta['ori_shape']
    pad_shape = img_meta['pad_shape']
    new_img_meta_list = [[{
        'img_shape':
        img_shape,
        'ori_shape':
        ori_shape,
        'pad_shape':
        pad_shape,
        'filename':
        img_meta['filename'],
        'scale_factor':
        (img_shape[1] / ori_shape[1], img_shape[0] / ori_shape[0]) * 2,
        'flip':
        False,
    } for _ in range(N)]]

    return img_list, new_img_meta_list


def show_result_pyplot(img: Union[str, np.ndarray],
                       result: np.ndarray,
                       palette: Optional[Iterable] = None,
                       fig_size: Iterable[int] = (15, 10),
                       opacity: float = 0.5,
                       title: str = '',
                       block: bool = True):
    img = mmcv.imread(img)
    img = img.copy()
    seg = result[0]
    seg = mmcv.imresize(seg, img.shape[:2][::-1])
    palette = np.array(palette)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)

    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)


def test_rate(
                  trt_file: str,
                  config: dict,
                  input_config: dict,
                  ):
        min_shape = input_config['min_shape']
        max_shape = input_config['max_shape']
        start_time = datetime.datetime.now()
        count = 0
        for file_path in os.listdir(input_config['input_path']):
            if os.path.isfile(os.path.join(input_config['input_path'], file_path)) == True:
                img_test = os.path.join(input_config['input_path'], file_path)
                inputs = _prepare_input_img(
                    img_test,
                    config.data.test.pipeline,
                    shape=min_shape[2:])

                imgs = inputs['imgs']
                img_metas = inputs['img_metas']
                img_list = [img[None, :] for img in imgs]
                img_meta_list = [[img_meta] for img_meta in img_metas]
                # update img_meta
                img_list, img_meta_list = _update_input_img(img_list, img_meta_list)

                if max_shape[0] > 1:
                    # concate flip image for batch test
                    flip_img_list = [_.flip(-1) for _ in img_list]
                    img_list = [
                        torch.cat((ori_img, flip_img), 0)
                        for ori_img, flip_img in zip(img_list, flip_img_list)
                    ]

                # Get results from TensorRT
                trt_model = TRTWraper(trt_file, ['input'], ['output'])
                with torch.no_grad():
                    trt_outputs = trt_model({'input': img_list[0].contiguous().cuda()})
                count = count + 1
            end_time = datetime.datetime.now()
            print('endtime: ' + str(end_time))
            print(count)
            print('spendtime(ms) :' + str((end_time - start_time).seconds / count))



def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMSegmentation models from ONNX to TensorRT')
    parser.add_argument('config', help='Config file of the model')
    parser.add_argument('model', help='Path to the input ONNX model')
    parser.add_argument(
        '--trt-file', type=str, help='Path to the output TensorRT engine')
    parser.add_argument(
        '--max-shape',
        type=int,
        nargs=4,
        default=[1, 3, 400, 600],
        help='Maximum shape of model input.')
    parser.add_argument(
        '--min-shape',
        type=int,
        nargs=4,
        default=[1, 3, 400, 600],
        help='Minimum shape of model input.')
    parser.add_argument('--fp16', action='store_true', help='Enable fp16 mode')
    parser.add_argument(
        '--workspace-size',
        type=int,
        default=1,
        help='Max workspace size in GiB')
    parser.add_argument(
        '--input-img', type=str, default='', help='Image for test')
    parser.add_argument(
        '--show', action='store_true', help='Whether to show output results')
    parser.add_argument(
        '--dataset',
        type=str,
        default='CityscapesDataset',
        help='Dataset name')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the outputs of ONNXRuntime and TensorRT')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether to verbose logging messages while creating \
                TensorRT engine.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    assert is_tensorrt_plugin_loaded(), 'TensorRT plugin should be compiled.'
    args = parse_args()

    if not args.input_img:
        args.input_img = osp.join(osp.dirname(__file__), '../demo/demo.png')

    # check arguments
    assert osp.exists(args.config), 'Config {} not found.'.format(args.config)
    assert args.workspace_size >= 0, 'Workspace size less than 0.'
    for max_value, min_value in zip(args.max_shape, args.min_shape):
        assert max_value >= min_value, \
            'max_shape should be larger than min shape'

    input_config = {
        'min_shape': args.min_shape,
        'max_shape': args.max_shape,
        'input_path': args.input_img
    }

    cfg = mmcv.Config.fromfile(args.config)
    test_rate(
        args.trt_file,
        cfg,
        input_config
        )
