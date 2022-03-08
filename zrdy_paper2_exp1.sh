#!/bin/sh
imagepath=$1
segpath=$2
source /home/zrd/anaconda3/etc/profile.d/conda.sh
conda init bash
conda deactivate
conda activate open-mmlab

python /home/home2/zrd/project/mmsegmentation/test_batch.py --config configs/mobilenet_v2/cpnetplus_v2_m-v2-detail_loss_combine-zrdy.py --load-from /home/home2/zrd/project/mmsegmentation/work-dirs/cpnetplus_v2_m-v2-detail_loss_combine-zrdy_11/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_v2_m-v2-detail_loss_combine-zrdy_11-'$imagepath'/'



echo 'metrics for cpnetplus_v2_m-v2-detail_loss_combine-zrdy_11'
#python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath --test-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_v2_m-v2-detail_loss_combine-zrdy_01-'$imagepath'/'



python /home/home2/zrd/project/mmsegmentation/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_v2_m-v2-detail_loss_combine-zrdy_11-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_v2_m-v2-detail_loss_combine-zrdy_11-'$imagepath'-col/'
