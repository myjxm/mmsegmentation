from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os
import argparse
import datetime
from mmcv.cnn import get_model_complexity_info
import cv2
import requests
import json

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
    
    
 
#config_file = '/home/zrd/mmsegmentation-master/configs/cpnet/cpnet_r50-d8.py'
#config_file = '/home/zrd/mmsegmentation-master/configs/cpnet/cpnetplus_r50-d8.py'
#config_file = '/home/zrd/mmsegmentation-master/configs/cpnet/cpnetplus_rs101-d8.py'
#config_file = '/home/zrd/mmsegmentation-master/configs/cpnet/cpnet_rs101-d8.py'
#config_file  = '/home/zrd/mmsegmentation-master/configs/cpnet/cpnetplus_inter_r50-d8-combine-zrdy.py/'
#config_file  = '/home/zrd/mmsegmentation-master/configs/cpnet/cpnetplus_intra_r50-d8-combine-zrdy.py/'
#config_file  = '/home/zrd/mmsegmentation-master/configs/mobilenet_v2/cpnetplus_m-v2-d8_512x512_160k_combine-zrdy.py/'
#config_file = '/home/zrd/mmsegmentation-master/configs/cpnet/cpnetplus_r50-d8-combine-zrdy.py'
#config_file = '/home/zrd/mmsegmentation-master/configs/cpnet/cpnetplus_rs101-d8-combine-zrdy.py'
#config_file = '/home/zrd/mmsegmentation-master/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_combine-zrdy.py'
#config_file = '/home/zrd/mmsegmentation-master/configs/pspnet/pspnet_r50-d8_512x512_160k_combine-zrdy.py'

#config_file = '/home/zrd/mmsegmentation-master/configs/cpnet/cpnet_r50-d8-classs3.py'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnet_r50-d8-combine/latest.pth'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnet_r50-d8-combine-box/latest.pth'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnet_r50-d8-master1325/latest.pth'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnetplus_r50-d8-combine-box/latest.pth'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnetplus_rs101-d8-combine-box/latest.pth'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnet-rs101-d8-combine-box/latest.pth'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnet_r50-d8-combine-zrdy/latest.pth'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnetplus_inter_r50-d8-combine-zrdy/latest.pth'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnetplus_intra_r50-d8-combine-zrdy/latest.pth'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnetplus_m-v2-d8_512x512_160k_combine-zrdy/latest.pth'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnetplus_r50-d8-combine-zrdy/latest.pth'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnetplus_rs101-d8-combine-zrdy/latest.pth'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/deeplabv3plus_r50-d8_512x512_80k_combine-zrdy/latest.pth'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/pspnet_r50-d8_combine-zrdy/latest.pth'






#path = '/home/zrd/segment-val'
#path = '/home/home2/zrd/data/datasets/slam/oka/N02_4_Sequence_155_370-Image/PIC_Left_Rectified'
#path = '/home/home2/zrd/data/datasets/slam/oka/N02_4_Sequence_155_370-Image/PIC_Right_Rectified'
#path = '/home/home2/zrd/data/datasets/slam/dumuchuan/527-8/rectified/cam0'
#path = '/home/home2/zrd/data/datasets/slam/dumuchuan/527-8/rectified/cam1'
#path = '/home/home2/zrd/data/val/validation-paper1/image_deal'
#output_path = '/home/zrd/segment-val/cpnet_r50-d8-combine-box/'
#output_path = '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-combine-box/'
#output_path = '/home/home2/zrd/data/val/validation-paper1/cpnet_r50-d8-combine-zrdy/'
#output_path = '/home/home2/zrd/data/val/validation-paper1/cpnetplus_inter_r50-d8-combine-zrdy/'
#output_path = '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-combine-zrdy/'
#output_path = '/home/home2/zrd/data/val/validation-paper1/cpnetplus_rs101-d8-combine-zrdy/'
#output_path = '/home/home2/zrd/data/val/validation-paper1/pspnet_r50-d8_combine-zrdy/'
#output_path = '/home/home2/zrd/data/val/validation-paper1/cpnetplus_intra_r50-d8-combine-zrdy/'
#output_path = '/home/zrd/segment-val/cpnetplus_rs101-d8-combine-box/'
#output_path = '/home/zrd/segment-val/cpnet-rs101-d8-combine-box/'
#output_path =  '/home/home2/zrd/data/datasets/slam/oka/N02_4_Sequence_155_370-Image/PIC_Left_Rectified_annotation'
#output_path =  '/home/home2/zrd/data/datasets/slam/oka/N02_4_Sequence_155_370-Image/PIC_Right_Rectified_annotation'
#output_path =  '/home/home2/zrd/data/datasets/slam/dumuchuan/527-8/rectified/cam0annotation'
#output_path =  '/home/home2/zrd/data/datasets/slam/dumuchuan/527-8/rectified/cam1annotation'

#output_path =  '/home/home2/zrd/data/datasets/slam/oka/N02_4_Sequence_155_370-Image/PIC_Left_Rectified_visionimg'
#output_path =  '/home/home2/zrd/data/datasets/slam/oka/N02_4_Sequence_155_370-Image/PIC_Right_Rectified_visionimg'
#output_path =  '/home/home2/zrd/data/datasets/slam/dumuchuan/527-8/rectified/cam0visionimg'
#output_path =  '/home/home2/zrd/data/datasets/slam/dumuchuan/527-8/rectified/cam1visionimg'

def main():
    args = parse_args()
    url = r"http://localhost:8080/query/statistic_status/" + args.modelname + "/" + str(args.roc) + "/" + str(args.dataset)
    res = requests.get(url)
    print(res)
    res = json.loads(res.text)
    print(res)
    if res['code'] == 0 and len(res['data']) > 0:
        if res['data'][0]['metric_status'] == 'Y':
            return
    url = r"http://localhost:8080/init_insert/performance/" + args.modelname
    requests.get(url)
    config_file = args.config
    checkpoint_file = args.load_from
    path = args.image_path
    output_path = args.output_path
    roc = args.roc
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:1')
    for file_path  in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_path)) == True:
           flopsimg = cv2.imread(os.path.join(path, file_path))
           break;
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
    #input_shape=(3,flopsimg.shape[0],flopsimg.shape[1])
    input_shape = (3, 512, 512)
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
    url = r"http://localhost:8080/update/performance/gflops/" + args.modelname +  "/" + flops + "/" + params
    requests.get(url)

    model = init_segmentor(config_file, checkpoint_file, device='cuda:1')
    start_time=datetime.datetime.now()
    print('starttime: '+ str(start_time))
    count=0 
    for file_path  in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_path)) == True:
           img = os.path.join(path, file_path)
           result = inference_segmentor(model, img, roc)
           model.show_result(img, result, palette=model.PALETTE,
                             out_file=os.path.join(output_path, file_path.replace('jpg', 'png')), save_annotation=True)
           #model.show_result(img, result, palette=[[0, 0, 0], [128, 0, 0], [128, 128, 0]],out_file=os.path.join(output_path, file_path.replace('jpg','png')),save_annotation=True)
           count = count + 1
    end_time=datetime.datetime.now()
    print('endtime: '+ str(end_time))
    print(count)
    print(output_path + ' fps :' + str(1/((end_time-start_time).seconds/count)))
    url = r"http://localhost:8080/update/performance/fps/" + args.modelname + "/" + str(1/((end_time-start_time).seconds/count))
    requests.get(url)
    url = r"http://localhost:8080/update/test_batch_status/" + args.modelname
    requests.get(url)

if __name__ == '__main__':
    main()
