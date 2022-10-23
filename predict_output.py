# _*_ coding: utf-8 _*_
"""
Time:     2020/11/30 17:02
Author:   Ding Cheng(Deeachain)
File:     predict.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
from argparse import ArgumentParser
from prettytable import PrettyTable
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from builders.loss_builder import build_loss
from builders.validation_builder import predict_multiscale_sliding
import numpy as np
import torch.nn.functional as F
from PIL import Image
from skimage import io
from torchvision import transforms
from utils import image_transform as tr
import datetime
import torchvision.models as models
from ptflops import get_model_complexity_info
import requests
import json
import time


def main(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    url = r"http://localhost:8080/query/statistic_status/" + args.modelname + "/" + str(args.roc)  + "/" + str(args.datasetname)
    res = requests.get(url)
    res = json.loads(res.text)
    print(res)
    if res['code'] == 0 and len(res['data']) > 0:
        if res['data'][0]['metric_status'] == 'Y':
            return
    url = r"http://localhost:8080/init_insert/performance/" + args.modelname
    requests.get(url)
    t = PrettyTable(['args_name', 'args_value'])
    for k in list(vars(args).keys()):
        t.add_row([k, vars(args)[k]])
    print(t.get_string(title="Predict Arguments"))

    # build the model
    model = build_model(args.model, args.classes, args.backbone, args.pretrained, args.out_stride, args.mult_grid)

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        model = model.cuda()
        cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)['model']
            check_list = [i for i in checkpoint.items()]
            # Read weights with multiple cards, and continue training with a single card this time
            if 'module.' in check_list[0][0]:  # 读取使用多卡训练权重,并且此次使用单卡预测
                new_stat_dict = {}
                for k, v in checkpoint.items():
                    new_stat_dict[k[7:]] = v
                model.load_state_dict(new_stat_dict, strict=True)
            # Read the training weight of a single card, and continue training with a single card this time
            else:
                model.load_state_dict(checkpoint)
        else:
            print("no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))
    model.eval()
    with torch.cuda.device(0):
        net = model
        macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('segnet flops MACs = 2 * FLOPs')
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    url = r"http://localhost:8080/update/performance/gflops/" + args.modelname +  "/" + macs + "(macs)/" + params
    requests.get(url)
    # if hasattr(model, 'forward_dummy'):
    #    model.forward = model.forward_dummy
    # else:
    #    raise NotImplementedError(
    #        'FLOPs counter is currently not currently supported with {}'.
    #        format(model.__class__.__name__))

    # flops, params = get_model_complexity_info(model, input_shape)
    # split_line = '=' * 30
    # print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
    #    split_line, input_shape, flops, params))
    # print('!!!Please be cautious if you use the results in papers. '
    #      'You may need to check if all ops are supported and verify that the '
    #    'flops computation is correct.')
    count = 0
    sum_times = 0
    #start_time = datetime.datetime.now()
    for file_path in os.listdir(args.img_path):
        if os.path.isfile(os.path.join(args.img_path, file_path)) == True:
            img = os.path.join(args.img_path, file_path)
            image = Image.open(img).convert('RGB')
            # print(np.asarray(image).shape)
            composed_transforms = transforms.Compose([
                # tr.FixedResize(size=(1024,512)),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])
            sample = {'image': image}
            sampled = composed_transforms(sample)
            image = sampled['image']

            image = np.expand_dims(image, axis=0)
            # print(image.shape)
            B, C, H, W = image.shape
            # image = np.array(image).astype(np.float32).transpose((0,3,1,2))
            image = torch.from_numpy(image).float().cuda()
            sh = int(H * 0.5)
            sw = int(W * 0.5)
            image = F.interpolate(image, (540, 960), mode='bilinear', align_corners=True)
            # image.resize([B,C,H,W])
            # image=torch.from_numpy(image)
            start_time = time.time()

            outputs = model(image)

            # print(outputs.cpu().detach().numpy().astype(np.uint8).shape)
            outputs = F.interpolate(outputs, (H, W), mode='bilinear', align_corners=True)
            # print(outputs)
            outputs = F.softmax(outputs, dim=1)
            # print(outputs)
            if args.roc > 0:
                outputs[0][1][outputs[0][1] >= args.roc] = 1
                outputs[0][1][outputs[0][1] < args.roc] = 0
            # print(outputs)
            # print('end')
            outputs = torch.argmax(outputs, 1).long()
            outputs = np.asarray(outputs.cpu(), dtype=np.uint8)
            elapsed_time = time.time() - start_time
            sum_times += elapsed_time
            res = Image.fromarray(outputs[0])
            # print(outputs)
            res.save(os.path.join(args.output_path, file_path.replace('jpg', 'png')))
            # io.imsave(os.path.join(args.output_path, file_path.replace('jpg','png')),
            # outputs.cpu().detach().numpy().astype(np.uint8)[0])
            count = count + 1
    #end_time = datetime.datetime.now()
    print('Average time per image: %.5f' % (sum_times / count))
    print(args.output_path + ' fps :' + str(1 / (sum_times / count)))
    url = r"http://localhost:8080/update/performance/fps/" + args.modelname + "/" + str(
        1 / (sum_times / count))
    requests.get(url)
    url = r"http://localhost:8080/update/test_batch_status/" + args.modelname
    requests.get(url)


    # print(args.output_path + ' fps :' + str(1 / ((end_time - start_time).seconds / count)))
    # url = r"http://localhost:8080/update/performance/fps/" + args.modelname + "/" + str(
    #     1 / ((end_time - start_time).seconds / count))
    # requests.get(url)
    # url = r"http://localhost:8080/update/test_batch_status/" + args.modelname
    # requests.get(url)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="UNet", help="model name")
    parser.add_argument('--backbone', type=str, default="resnet18", help="backbone name")
    parser.add_argument('--pretrained', action='store_true',
                        help="whether choice backbone pretrained on imagenet")
    parser.add_argument('--out_stride', type=int, default=32, help="output stride of backbone")
    parser.add_argument('--mult_grid', action='store_true',
                        help="whether choice mult_grid in backbone last layer")
    parser.add_argument('--root', type=str, default="", help="path of datasets")
    parser.add_argument('--predict_mode', default="sliding", choices=["sliding", "whole"],
                        help="Defalut use whole predict mode")
    parser.add_argument('--predict_type', default="validation", choices=["validation", "predict"],
                        help="Defalut use validation type")
    parser.add_argument('--flip_merge', action='store_true', help="Defalut use predict without flip_merge")
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0], help="predict with multi_scales")
    parser.add_argument('--overlap', type=float, default=0.0, help="sliding predict overlap rate")
    parser.add_argument('--dataset', default="paris", help="dataset: cityscapes")
    parser.add_argument('--num_workers', type=int, default=4, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing NOTES:image size should fixed!")
    parser.add_argument('--tile_hw_size', type=str, default='512, 512',
                        help=" the tile_size is when evaluating or testing")
    parser.add_argument('--crop_size', type=int, default=769, help="crop size of image")
    parser.add_argument('--input_size', type=str, default=(769, 769),
                        help=" the input_size is for build ProbOhemCrossEntropy2d loss")
    parser.add_argument('--checkpoint', type=str, default='',
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./outputs/",
                        help="saving path of prediction result")
    parser.add_argument('--loss', type=str, default="CrossEntropyLoss2d",
                        choices=['CrossEntropyLoss2d', 'ProbOhemCrossEntropy2d', 'CrossEntropyLoss2dLabelSmooth',
                                 'LovaszSoftmax', 'FocalLoss2d'], help="choice loss for train or val in list")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--roc', type=float, default=-1, help="roc threshold")
    parser.add_argument("--modelname", type=str, default='',
                        help="modelname")
    parser.add_argument("--datasetname", type=str, default='',
                        help="validation dataset")
    args = parser.parse_args()

    save_dirname = args.checkpoint.split('/')[-2] + '_' + args.checkpoint.split('/')[-1].split('.')[0]

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.predict_mode, save_dirname)

    if args.dataset == 'combine_zrdy':
        args.classes = 2
    elif args.dataset == 'zhuhai12749_3class_image' or args.dataset == 'zhuhai433_3class_image' or args.dataset == 'zhuhai480_3class_image' or args.dataset == 'zhuhai500_3class_image' or args.dataset == 'zhuhai15708_3class_image' or args.dataset == 'MaSTr1325_images_512x384' :
        args.classes = 3
    else:
        raise NotImplementedError(
            "This repository now supports datasets %s is not included" % args.dataset)

    main(args)