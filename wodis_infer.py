#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 21:49
# @File  : inference.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import os
import pandas as pd
import time
import numpy as np
import torch as t
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from WODIS import WODIS_model
import cv2
import cfg
import requests
import argparse
import json
#from ptflops import get_model_complexity_info
from thop import profile
import torch.nn.functional as F
import torch
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument("--image-path", type=str,
                        help="Path to dataset files on which inference is performed.")
    parser.add_argument("--output-path", type=str,
                        help="Where to save predicted mask.")
    parser.add_argument("--roc", type=float,default=-1,
                        help="roc threshold")
    parser.add_argument("--modelname", type=str, default='',
                        help="modelname")
    parser.add_argument("--dataset", type=str, default='',
                        help="validation dataset")
    return parser.parse_args()

def create_visual_anno(anno):
    label2color_dict = {
        0: [128, 0, 0],  # 0: the object
        1: [192, 128, 0],  # 1: water surface
        2: [128, 128, 128] , # 2: sky
        4: [128, 0, 0]  # void color
    }

    visual_anno = np.zeros((anno.shape[0], anno.shape[1]), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):
        for j in range(visual_anno.shape[1]):
            color = anno[i, j].tolist()
            if color[0] == 128 and  color[1] == 128 and color[2] == 128 :
                visual_anno[i, j] = 2
            elif color[0] == 192 and  color[1] == 128 and color[2] == 0 :
                visual_anno[i, j] = 1
            elif color[0] == 128 and  color[1] == 0 and color[2] == 0 :
                visual_anno[i, j] = 0
            else :
                visual_anno[i, j] = 3
    return visual_anno



def main():
    if t.cuda.is_available():
        device = t.device('cuda')
    else:
        device = t.device('cpu')

    '''
    create the model and start the inference...
    '''
    args = parse_args()
    url = r"http://localhost:8080/query/statistic_status/" + args.modelname + "/" + str(args.roc) + "/" + str(
        args.dataset)
    res = requests.get(url)
    print(res)
    res = json.loads(res.text)
    print(res)
    if res['code'] == 0 and len(res['data']) > 0:
        if res['data'][0]['metric_status'] == 'Y':
            return
    url = r"http://localhost:8080/init_insert/performance/" + args.modelname
    requests.get(url)

    path = args.image_path
    output_path = args.output_path
    # load model
    net = WODIS_model(is_training=False, num_classes=cfg.NUM_CLASSES).eval().to(device)
    checkpoint = t.load(cfg.MODEL_WEIGHTS, map_location='cpu')
    net.load_state_dict(checkpoint)
    pd_label_color = pd.read_csv(cfg.class_dict_path, sep=',')
    name_value = pd_label_color['name'].values
    num_class = len(name_value)
    colormap = []

    # read label color from the class dict
    for i in range(num_class):
        tmp = pd_label_color.iloc[i]
        color = [tmp['r'], tmp['g'], tmp['b']]
        colormap.append(color)

    cm = np.array(colormap).astype('uint8')

    # create output folder if it does not exist.

    input_shape = (3, 384, 512)

    rgb = t.randn(1, 3, 352, 480).cuda()
    flops, params = profile(net, (rgb,))
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
            split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
              'You may need to check if all ops are supported and verify that the '
              'flops computation is correct.')
    url = r"http://localhost:8080/update/performance/gflops/" + args.modelname + "/" + str(flops/1024/1024/1024) + "/" + str(params/1024/1024)
    requests.get(url)

    net = WODIS_model(is_training=False, num_classes=cfg.NUM_CLASSES).eval().to(device)
    checkpoint = t.load(cfg.MODEL_WEIGHTS, map_location='cpu')
    net.load_state_dict(checkpoint)


    counter = 0
    sum_times = 0
    for file_path in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_path)) == True:
            image_base_name = file_path.split('.')[0]
            img = os.path.join(path, file_path)
            img = cv2.imread(img)
            height = img.shape[0]
            weight = img.shape[1]
            #print(img.shape)
            if img is None:
                break

            img_reverse = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            transform = transforms.Compose([transforms.Resize(cfg.IMG_SIZE),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            # read one image for the inference process, it needs to expand the first dimension to match the tensor.
            img_out = transform(img_reverse).unsqueeze(0)

            # inference starting...
            start_time = time.time()
            valImg = img_out.to(device)
            #out, cx1 = net(valImg)
            out = net(valImg)
            elapsed_time = time.time() - start_time
            sum_times += elapsed_time

            out = F.log_softmax(out, dim=1)
            pre_label = out.max(dim=1)[1].squeeze().cpu().data.numpy()
            pre = cm[pre_label]
            pre_output = create_visual_anno(pre)
            #transform_output = transforms.Compose([transforms.Resize((height,weight))])
            #pre_output = torch.from_numpy(pre_output)
            #pre_output = F.interpolate(pre_output, (height, weight), mode='bilinear', align_corners=True)
            #pre_output = pre_output.resize(height,weight)
            #pre_output = pre_output.numpy()
            preds_out = cv2.resize(pre_output, (weight,height), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(output_path + image_base_name + '.png', preds_out.astype('uint8'))
            counter += 1
            #print('Elapsed time: %.04f for image num %03d' % (elapsed_time, counter))
    print('Average time per image: %.5f' % (sum_times / counter))
    print(output_path + ' fps :' + str(1 / (sum_times / counter)))
    url = r"http://localhost:8080/update/performance/fps/" + args.modelname + "/" + str(
        1 / (sum_times / counter))
    requests.get(url)
    url = r"http://localhost:8080/update/test_batch_status/" + args.modelname
    requests.get(url)




if __name__ == '__main__':
    main()