import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import os
import time

import wasr.models as models
import requests
import json
from ptflops import get_model_complexity_info
from thop import profile

NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])

# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

BATCH_SIZE = 12
ARCHITECTURE = 'wasr_resnet101_imu'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WaSR Network MaSTr1325 Inference")
    parser.add_argument("img_path", type=str,
                        help="Path to the image to run inference on.")
    parser.add_argument("output_dir", type=str,
                        help="Path to the file, where the output prediction will be saved.")
    parser.add_argument("--imu_mask", type=str, default=None,
                        help="Path to the corresponding IMU mask (if needed by the model).")
    parser.add_argument("--architecture", type=str, choices=models.model_list, default=ARCHITECTURE,
                        help="Model architecture.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights or a model checkpoint.")
    parser.add_argument("--roc", type=float, default=-1,
                        help="roc threshold")
    parser.add_argument("--modelname", type=str, default='',
                        help="modelname")
    parser.add_argument("--dataset", type=str, default='',
                        help="validation dataset")
    return parser.parse_args()


def predict_image(model, image, imu_mask=None):
    feat = {'image': image.cuda()}
    if imu_mask is not None:
        feat['imu_mask'] = imu_mask.cuda()

    res = model(feat)
    prediction = res['out'].detach().softmax(1).cpu()
    return prediction


def predict(args):
    # Load and prepare model
    model = models.get_model(args.architecture, pretrained=False)

    state_dict = torch.load(args.weights, map_location='cpu')
    if 'model' in state_dict:
        # Loading weights from checkpoint
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)

    # Enable eval mode and move to CUDA
    model = model.eval().cuda()

    # model = models.get_model(args.architecture, pretrained=false)
    #
    # state_dict = torch.load(args.weights, map_location='cpu')
    # if 'model' in state_dict:
    #     # loading weights from checkpoint
    #     state_dict = state_dict['model']
    # model.load_state_dict(state_dict)
    #
    # # enable eval mode and move to cuda
    # model = model.eval().cuda()
    sum_times = 0
    counter   = 0

    for file_path in os.listdir(args.img_path):
        if os.path.isfile(os.path.join(args.img_path, file_path)) == True:
            img = os.path.join(args.img_path, file_path)

            # Load and normalize image
            img = Image.open(img)
            print(img.size[0])
            print(img.size[1])
            img = np.array(img)
            H,W,_ = img.shape
            img = os.path.join(args.img_path, file_path)
            img = Image.open(img)
            img = img.resize((512,384),Image.ANTIALIAS)
            img = np.array(img)
            print(img.shape)

            #img = torch.from_numpy(img) / 255.0
            img = torch.from_numpy(img)
            print(img.shape)
            #img = torch.nn.functional.interpolate(img, (384,512,3), mode='bilinear')
            img = img / 255.0
            img = (img - NORM_MEAN) / NORM_STD
            img = img.permute(2,0,1).unsqueeze(0) # [1xCxHxW]
            img = img.float()

            # Load IMU mask if provided
            imu_mask = None
            if args.imu_mask is not None:
                imu_mask = np.array(Image.open(args.imu_mask))
                imu_mask = imu_mask.astype(np.bool)
                imu_mask = torch.from_numpy(imu_mask).unsqueeze(0) # [1xHxW]


            # Run inference
            start_time = time.time()
            probs = predict_image(model, img, imu_mask)
            elapsed_time = time.time() - start_time
            sum_times += elapsed_time
            probs = torch.nn.functional.interpolate(probs, (H,W), mode='bilinear')
            preds = probs.argmax(1)[0]
            #print(preds)
            #print(preds.shape)

            # Convert predictions to RGB class colors
            preds_rgb = SEGMENTATION_COLORS[preds]


            #preds_img = Image.fromarray(preds_rgb)
            preds_img = Image.fromarray(np.uint8(preds.numpy()))

            output_dir = Path(args.output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)

            preds_img.save(os.path.join(args.output_dir, file_path.replace('jpg', 'png')))
            counter += 1
    net = model

    rgb = torch.randn(1, 3, 7, 7).cuda()
    flops, params = profile(net, ({'image': rgb},))
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, (64, 3, 7, 7), flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
    url = r"http://localhost:8080/update/performance/gflops/" + args.modelname + "/" + str(
        flops / 1024 / 1024 / 1024) + "/" + str(params / 1024 / 1024)
    requests.get(url)
    print('Average time per image: %.5f' % (sum_times / counter))
    print(args.output_dir + ' fps :' + str(1 / (sum_times / counter)))
    url = r"http://localhost:8080/update/performance/fps/" + args.modelname + "/" + str(
            1 / (sum_times / counter))
    requests.get(url)
    url = r"http://localhost:8080/update/test_batch_status/" + args.modelname
    requests.get(url)

def main():
    args = get_arguments()
    print(args)
    '''
        create the model and start the inference...
        '''
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

    predict(args)


if __name__ == '__main__':
    main()