#!/bin/sh
imagepath=$1
segpath=$2
source /home/zrd/anaconda3/etc/profile.d/conda.sh
conda init bash
conda deactivate
conda activate open-mmlab
python /home/home2/zrd/project/mmsegmentation/test_batch.py --config /home/home2/zrd/project/mmsegmentation/configs/cpnet/cpnetplus_r50-d8-combine-zrdy.py --load-from /home/home2/zrd/project/mmsegmentation/work_dirs/cpnetplus_r50-d8-combine-zrdy-trts/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-combine-zrdy-trts-'$imagepath'/'
python /home/home2/zrd/project/mmsegmentation/test_batch.py --config /home/home2/zrd/project/mmsegmentation/configs/cpnet/cpnetplus_r50-d8-combine-zrdy.py --load-from /home/home2/zrd/project/mmsegmentation/work_dirs/cpnetplus_r50-d8-combine-zrdy-tetp/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-combine-zrdy-tetp-'$imagepath'/'
python /home/home2/zrd/project/mmsegmentation/test_batch.py --config /home/home2/zrd/project/mmsegmentation/configs/cpnet/cpnetplus_r50-d8-combine-zrdy.py --load-from /home/home2/zrd/project/mmsegmentation/work_dirs/cpnetplus_r50-d8-combine-zrdy-tetptrts/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-combine-zrdy-tetptrts-'$imagepath'/'
echo 'metrics for cpnetplus_r50-d8-combine-zrdy'
python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-combine-zrdy-tetptrts-'$imagepath'/'
python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-combine-zrdy-tetp-'$imagepath'/'
python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-combine-zrdy-trts-'$imagepath'/'