#!/bin/sh
model=$1
config=$2
imagepath=$3
segpath=$4

#model=(deeplabv3plus_m-v2-d8_512x512_160k_zhuhai15708)
source /home/zrd/anaconda3/etc/profile.d/conda.sh
conda init bash
#roc=(-1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
#roc=(-1)
#roc=(0.0001 0.0005 0.001 0.004 0.007 0.01 0.03 0.05 0.07 0.09 0.91 0.93 0.95 0.97 0.99 0.993 0.996 0.999 0.9993 0.9996 0.9999)
#roc=(0.99999 0.999999 0.9999999 0.99999999 0.999999999)
roc=(-1)

for j in ${roc[*]}; do
echo 'inference '$model'roc: '$j
conda deactivate
conda activate open-mmlab
python /home/home2/zrd/project/mmsegmentation/test_batch_mysql.py --config $config --load-from '/home/home2/zrd/project/mmsegmentation/work-dirs/'$model'/latest.pth' --image-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath --output-path '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'/' --roc $j --modelname $model --dataset $imagepath
conda deactivate
conda activate open-mmlab
echo 'metrics for  '$model'roc: '$j
python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'/' --roc $j --modelname $model --dataset $imagepath  --roc $j --modelname $model --dataset $imagepath
echo 'get_colour_img for '$model'roc: '$j
python /home/home2/zrd/project/mmsegmentation/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'-col/'  --config-file $config --checkpoint-file  '/home/home2/zrd/project/mmsegmentation/work-dirs/'$model'/latest.pth'
done
