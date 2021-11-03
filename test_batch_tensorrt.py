from mmseg.apis import inference_segmentor, init_segmentor,inference_segmentor_concactdata
import mmcv
import os
import argparse
import datetime
import torch
from torch2trt import torch2trt

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument("--image-path", type=str,
                        help="Path to dataset files on which inference is performed.")
    parser.add_argument("--output-path", type=str,
                        help="Where to save predicted mask.")
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
    config_file = args.config
    checkpoint_file = args.load_from
    path = args.image_path
    output_path = args.output_path
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:1')
    
    start_time=datetime.datetime.now()
    print('starttime: '+ str(start_time))
    count=0 
    for file_path  in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_path)) == True:
           img = os.path.join(path, file_path)
           result = inference_segmentor_concactdata(model, img)
           model.show_result(img, result, palette=[[0, 0, 0], [128, 0, 0]],out_file=os.path.join(output_path, file_path.replace('jpg','png')),save_annotation=True)
           count = count + 1
    end_time=datetime.datetime.now()
    print('endtime: '+ str(end_time))
    print(count)
    print(output_path + ' spendtime(ms) :' + str((end_time-start_time).seconds/count))

if __name__ == '__main__':
    main()
