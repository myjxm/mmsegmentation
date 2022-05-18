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
if  [ "$model" == "segnet_paper2" ]; then
conda deactivate
conda activate segmen
python /home/zrd/Segmentation-Pytorch-master/predict_output.py  --model SegNet  --checkpoint /home/zrd/Segmentation-Pytorch-master/checkpoint/zhuhai12749_3class/SegNet/best_model.pth  --out_stride 8   --root '/data/open_data'   --dataset $imagepath   --predict_type validation  --predict_mode whole  --crop_size 768  --tile_hw_size '768,768'   --batch_size 2   --gpus 1 --img_path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output_path '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'/' --roc $j --modelname $model --datasetname $imagepath
elif [ "$model" == "wl-largeBN_paper2" ]; then
conda deactivate
conda activate keras
python /home/zrd/wl/wl6.py --dataset-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --save-dir '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'/' --model-weights /home/zrd/wl/models_saved/final_models/unet160--largeBN--dice_coef_loss --roc $j --modelname $model --dataset $imagepath
elif [ "$model" == "wl-largeBN_mobilenetv2_paper2" ]; then
conda deactivate
conda activate keras
python /home/zrd/wl/wl8.py --dataset-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --save-dir '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'/' --model-weights /home/zrd/wl/models_saved/final_models/unet160--largeBN_mobilenetv2--dice_coef_loss --roc $j --modelname $model --dataset $imagepath
elif [ "$model" == "wasr_paper2" ] || [ "$model" == "wasr_paper2_2800" ] || [ "$model" == "wasr_paper2_fine_3000" ] || [ "$model" == "wasr_paper2_fine_17900" ] || [ "$model" == "wasr_paper2_init_16000" ];  then
conda deactivate
conda activate wasr
#conda activate wasr_new
python /home/zrd/wasr_network-master/wasr_inference_noimu_batch.py --dataset-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath --save-dir  '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'/'
#python /home/zrd/wasr_network-master/wasr.py --dataset-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath --save-dir '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'/' --roc $j --modelname $model --dataset $imagepath
else
conda deactivate
conda activate open-mmlab
python /home/home2/zrd/project/mmsegmentation/test_batch_mysql.py --config $config --load-from '/home/home2/zrd/project/mmsegmentation/work-dirs/'$model'/latest.pth' --image-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath --output-path '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'/' --roc $j --modelname $model --dataset $imagepath
fi
conda deactivate
conda activate open-mmlab
echo 'metrics for  '$model'roc: '$j
python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'/' --roc $j --modelname $model --dataset $imagepath  --roc $j --modelname $model --dataset $imagepath
echo 'get_colour_img for '$model'roc: '$j
if [ "$model" == "wasr_paper2" ] || [ "$model" == "wl-largeBN_mobilenetv2_paper2" ] || [ "$model" == "wl-largeBN_paper2" ] || [ "$model" == "segnet_paper2" ] || [ "$model" == "wasr_paper2_2800" ] || [ "$model" == "wasr_paper2_fine_3000" ] || [ "$model" == "wasr_paper2_fine_17900" ] || [ "$model" == "wasr_paper2_init_16000" ]; then
python /home/home2/zrd/project/mmsegmentation/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'-col/'  --config-file $config --checkpoint-file  '/home/home2/zrd/project/mmsegmentation/work-dirs/deeplabv3plus_r50-d8_512x512_80k_zhuhai12749_3class/latest.pth'
else
python /home/home2/zrd/project/mmsegmentation/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/'$model'-'$imagepath'-col/'  --config-file $config --checkpoint-file  '/home/home2/zrd/project/mmsegmentation/work-dirs/'$model'/latest.pth'
fi
done
