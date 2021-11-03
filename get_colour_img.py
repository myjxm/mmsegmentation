from mmseg.apis import inference_segmentor, init_segmentor,inference_segmentor_concactdata
import mmcv
import os
from skimage import io,data,color,img_as_ubyte
import numpy as np
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument("--img-path", type=str,
                        help="Path to dataset files on which inference is performed.")
    parser.add_argument("--output-path", type=str,
                        help="Where to save predicted mask.")
    parser.add_argument("--save-path", type=str,
                        help="Where to save predicted color mask.")                    
    return parser.parse_args()
    
    
    


#save_path = '/home/home2/zrd/data/val/validation-paper1/wl_col/'
#save_path = '/home/home2/zrd/data/val/validation-paper1/wasr_col/'



def main():
    args = parse_args()
    config_file = '/home/zrd/mmsegmentation-master/configs/pspnet/pspnet_r50-d8_512x512_160k_combine-zrdy.py'
    checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/pspnet_r50-d8_combine-zrdy/latest.pth'
    model = init_segmentor(config_file, checkpoint_file, device='cuda:1')
    path = args.img_path
    output_path = args.output_path
    save_path = args.save_path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    for file_path  in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_path)) == True:
            image_file = os.path.join(path, file_path)
            seg_file = os.path.join(output_path, file_path.replace('jpg','png'))
            img = mmcv.imread(image_file)
            seg = io.imread(seg_file)
            seg = np.expand_dims(seg, axis=0)
            if img.shape[:2] == seg.shape[1:3]:
                model.show_result(img, seg, palette=[[0, 0, 0], [128, 0, 0]],out_file=os.path.join(save_path, file_path.replace('jpg','png')),save_annotation=False)

if __name__ == '__main__':
    main()



# test a single image and show the results
#img = '/home/zrd/segment-val/X31_1_0000052250.jpg-1.jpg'  # or img = mmcv.imread(img), which will only load it once
#result = inference_segmentor_concactdata(model, img)
#print(result[0].shape)
#model.show_result(img, result,palette = [[0, 0, 0], [128, 0, 0]],out_file='X31_1_0000052250.jpg-1.png')
#model.show_result(img, result,palette = [[0, 0, 0], [128, 0, 0],[0, 128, 0]],out_file='X31_1_0000052250.jpg-1.png')
