from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os
import argparse
import datetime
from mmcv.cnn import get_model_complexity_info
import cv2
import requests
import json
import time

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

def save(imgPath,savePath,subPath,model,roc):
    full_path=os.path.join(imgPath,subPath)
    for img in os.listdir(full_path):
        curPath = os.path.join(full_path, img)
        if os.path.isdir(curPath) and img != 'datasets' and img !='descriptors':
            save(imgPath,savePath,os.path.join(subPath,img),model,roc)
        elif os.path.isfile(curPath) and img != 'imgcon.py':
                imgfile = curPath
                result = inference_segmentor(model, imgfile, roc)
                outputPath = os.path.join(savePath, subPath)
                if not os.path.exists(outputPath):
                    os.makedirs(outputPath)
                model.save_other(imgfile, result, out_file=os.path.join(outputPath, img.replace('jpg', 'bmp')))



def main():
    args = parse_args()
    config_file = args.config
    checkpoint_file = args.load_from
    path = args.image_path
    output_path = args.output_path
    roc = args.roc
    model = init_segmentor(config_file, checkpoint_file, device='cuda:1')
    save(path, output_path, '', model, roc)

if __name__ == '__main__':
    main()
