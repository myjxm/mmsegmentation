#!/bin/sh
imagepath=$1
segpath=$2
source /home/zrd/anaconda3/etc/profile.d/conda.sh
conda init bash
conda deactivate
conda activate open-mmlab
python /home/zrd/mmsegmentation-master/test_batch.py --config /home/zrd/mmsegmentation-master/configs/fcn/fcn_plus_r50-d8.py --load-from /home/zrd/mmsegmentation-master/work_dirs/fcnplus_r50-d8-combine-zrdy/latest.pth --image-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath --output-path '/home/home2/zrd/data/val/validation-paper1/fcn_plus_r50-d8-combine-zrdy-'$imagepath'/'
echo 'metrics for fcn_plus_-combine-zrdy'
python /home/zrd/mmsegmentation-master/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/fcn_plus_r50-d8-combine-zrdy-'$imagepath'/'
python /home/zrd/mmsegmentation-master/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/fcn_plus_r50-d8-combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/fcn_plus_r50-d8-combine-zrdy-'$imagepath'-col/'
