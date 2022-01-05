#!/bin/sh
imagepath=$1
#segpath=$2
source /home/zrd/anaconda3/etc/profile.d/conda.sh
conda init bash
conda deactivate
conda activate open-mmlab

python /home/home2/zrd/project/mmsegmentation/test_batch_mastr1325.py --config /home/home2/zrd/project/mmsegmentation/configs/mobilenet_v2/cpnetplus_m-v2-d8_mastr1325.py --load-from /home/home2/zrd/project/mmsegmentation/work-dirs/cpnetplus_m-v2-d8_mastr1325/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_m-v2-d8_mastr1325-'$imagepath'/'
python /home/home2/zrd/project/mmsegmentation/test_batch_mastr1325.py --config /home/home2/zrd/project/mmsegmentation/configs/bisenetv2/bisenetv2-mastr1325.py --load-from latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/bisenetv2-mastr1325c-'$imagepath'/'
python /home/home2/zrd/project/mmsegmentation/test_batch_mastr1325.py --config /home/home2/zrd/project/mmsegmentation/configs/cpnet/cpnetplus_r50-d8-master1325.py  --load-from /home/home2/zrd/project/mmsegmentation/work-dirs/cpnetplus_r50-d8-master1325/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-master1325-'$imagepath'/'

#echo 'metrics for cpnet_unet-combine-zrdy'
#python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/cpnet_unet-combine-zrdy-'$imagepath'/'
#echo 'metrics for pspnet_unet_s5-d16_256x256_40k_combine-zrdy'
#python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/pspnet_unet_s5-d16_256x256_40k_combine-zrdy-'$imagepath'/'
#echo 'metrics for deeplabv3_unet_s5-d16_256x256_40k_combine-zrdy'
#python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/deeplabv3_unet_s5-d16_256x256_40k_combine-zrdy-'$imagepath'/'
echo 'metrics for bisenetv2-combine-zrdy'
#python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath --test-path '/home/home2/zrd/data/val/validation-paper1/bisenetv2-combine-zrdy-'$imagepath'/'


#python /home/home2/zrd/project/mmsegmentation/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnet_unet-combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/cpnet_unet-combine-zrdy-'$imagepath'-col/'
python /home/home2/zrd/project/mmsegmentation/get_colour_img_mastr1325.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_m-v2-d8_mastr1325-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_m-v2-d8_mastr1325-'$imagepath'-col/'
python /home/home2/zrd/project/mmsegmentation/get_colour_img_mastr1325.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/bisenetv2-mastr1325-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/bisenetv2-mastr1325-'$imagepath'-col/'
python /home/home2/zrd/project/mmsegmentation/get_colour_img_mastr1325.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-master1325-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-master1325-'$imagepath'-col/'