#!/bin/sh
imagepath=$1
segpath=$2
source /home/zrd/anaconda3/etc/profile.d/conda.sh
conda init bash
conda deactivate
conda activate open-mmlab
#python /home/home2/zrd/project/mmsegmentation/test_batch.py --config /home/home2/zrd/project/mmsegmentation/configs/unet/cpnet_unet-combine-zrdy.py --load-from /home/home2/zrd/project/mmsegmentation/work_dirs/cpnet_unet-combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnet_unet-combine-zrdy-'$imagepath'/'
#python /home/home2/zrd/project/mmsegmentation/test_batch.py --config /home/home2/zrd/project/mmsegmentation/configs/unet/pspnet_unet_s5-d16_256x256_40k_combine-zrdy.py --load-from /home/home2/zrd/project/mmsegmentation/work_dirs/pspnet_unet_s5-d16_256x256_40k_combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/pspnet_unet_s5-d16_256x256_40k_combine-zrdy-'$imagepath'/'
#python /home/home2/zrd/project/mmsegmentation/test_batch.py --config /home/home2/zrd/project/mmsegmentation/configs/unet/deeplabv3_unet_s5-d16_256x256_40k_combine-zrdy.py --load-from /home/home2/zrd/project/mmsegmentation/work_dirs/deeplabv3_unet_s5-d16_256x256_40k_combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/deeplabv3_unet_s5-d16_256x256_40k_combine-zrdy-'$imagepath'/'
python /home/home2/zrd/project/mmsegmentation/test_batch.py --config /home/home2/zrd/project/mmsegmentation/configs/bisenetv2/bisenetv2-combine-zrdy.py  --load-from /home/home2/zrd/project/mmsegmentation/work_dirs/cbisenetv2-combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath --output-path '/home/home2/zrd/data/val/validation-paper1/bisenetv2-combine-zrdy-'$imagepath'/'
#echo 'metrics for cpnet_unet-combine-zrdy'
#python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/cpnet_unet-combine-zrdy-'$imagepath'/'
#echo 'metrics for pspnet_unet_s5-d16_256x256_40k_combine-zrdy'
#python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/pspnet_unet_s5-d16_256x256_40k_combine-zrdy-'$imagepath'/'
#echo 'metrics for deeplabv3_unet_s5-d16_256x256_40k_combine-zrdy'
#python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/deeplabv3_unet_s5-d16_256x256_40k_combine-zrdy-'$imagepath'/'
echo 'metrics for bisenetv2-combine-zrdy'
python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath --test-path '/home/home2/zrd/data/val/validation-paper1/bisenetv2-combine-zrdy-'$imagepath'/'


#python /home/home2/zrd/project/mmsegmentation/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnet_unet-combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/cpnet_unet-combine-zrdy-'$imagepath'-col/'
#python /home/home2/zrd/project/mmsegmentation/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/pspnet_unet_s5-d16_256x256_40k_combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/pspnet_unet_s5-d16_256x256_40k_combine-zrdy-'$imagepath'-col/'
#python /home/home2/zrd/project/mmsegmentation/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/deeplabv3_unet_s5-d16_256x256_40k_combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/deeplabv3_unet_s5-d16_256x256_40k_combine-zrdy-'$imagepath'-col/'
python /home/home2/zrd/project/mmsegmentation/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/bisenetv2-combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/bisenetv2-combine-zrdy-'$imagepath'-col/'