#!/bin/sh
imagepath=$1
segpath=$2
model=(cpnetplus_r50-d8-combine-zrdy_2048*1024 cpnetplus_inter_r50-d8-combine-zrdy cpnetplus_intra_r50-d8-combine-zrdy cpnetplus_m-v2-d8_512x512_160k_combine-zrdy cpnetplus_r50-d8-combine-zrdy cpnetplus_rs101-d8-combine-zrdy deeplabv3plus_r50-d8_512x512_80k_combine-zrdy pspnet_r50-d8_512x512_160k_combine-zrdy cpnet_r50-d8-combine-zrdy fcn_plus_r50-d8-combine-zrdy segnet wl-largeBN wl-largeBN_mobilenetv2 wasr)
source /home/zrd/anaconda3/etc/profile.d/conda.sh
conda init bash
#roc=(-1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
#roc=(-1)
#roc=(0.0001 0.0005 0.001 0.004 0.007 0.01 0.03 0.05 0.07 0.09 0.91 0.93 0.95 0.97 0.99 0.993 0.996 0.999 0.9993 0.9996 0.9999)
#roc=(0.99999 0.999999 0.9999999 0.99999999 0.999999999)
roc=(-1)

for j in ${roc[*]}; do
for i in ${model[*]}; do
echo 'inference '$i'roc: '$j
if  [ "$i" == "segnet" ]; then
conda deactivate
conda activate segmen
python /home/zrd/Segmentation-Pytorch-master/predict_output.py  --model SegNet  --checkpoint /home/zrd/Segmentation-Pytorch-master/checkpoint/combine_zrdy/SegNet/best_model.pth  --out_stride 8   --root '/data/open_data'   --dataset 'combine_zrdy'   --predict_type validation  --predict_mode whole  --crop_size 768  --tile_hw_size '768,768'   --batch_size 2   --gpus 1 --img_path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output_path '/home/home2/zrd/data/val/validation-paper1/'$i'-'$imagepath'/' --roc $j --modelname $i --datasetname $imagepath
elif [ "$i" == "wl-largeBN" ]; then
conda deactivate
conda activate keras
python /home/zrd/wl/wl6.py --dataset-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --save-dir '/home/home2/zrd/data/val/validation-paper1/'$i'-'$imagepath'/' --model-weights /home/zrd/wl/models_saved/final_models/unet160--largeBN--dice_coef_loss.bak0610 --roc $j --modelname $i --dataset $imagepath
elif [ "$i" == "wl-largeBN_mobilenetv2" ]; then
conda deactivate
conda activate keras
python /home/zrd/wl/wl8.py --dataset-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --save-dir '/home/home2/zrd/data/val/validation-paper1/'$i'-'$imagepath'/' --model-weights /home/zrd/wl/models_saved/final_models/unet160--largeBN_mobilenetv2--dice_coef_loss.bak0610 --roc $j --modelname $i --dataset $imagepath
elif [ "$i" == "wasr" ]; then
conda deactivate
conda activate wasr_new
#python /home/zrd/wasr_network-master/wasr_inference_noimu_batch.py --dataset-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath --save-dir '/home/home2/zrd/data/val/validation-paper1/wasr-'$imagepath'/'
python /home/zrd/wasr_network-master/wasr.py --dataset-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath --save-dir '/home/home2/zrd/data/val/validation-paper1/'$i'-'$imagepath'/' --roc $j --modelname $i --dataset $imagepath
elif [ "$i" == "cpnetplus_m-v2-d8_512x512_160k_combine-zrdy" ]; then
conda deactivate
conda activate open-mmlab
python /home/home2/zrd/project/mmsegmentation/test_batch.py --config /home/home2/zrd/project/mmsegmentation/configs/mobilenet_v2/cpnetplus_m-v2-d8_512x512_160k_combine-zrdy.py --load-from /home/home2/zrd/project/mmsegmentation/work_dirs/cpnetplus_m-v2-d8_512x512_160k_combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/'$i'-'$imagepath'/' --roc $j --modelname $i --dataset $imagepath
elif [ "$i" == "deeplabv3plus_r50-d8_512x512_80k_combine-zrdy" ]; then
conda deactivate
conda activate open-mmlab
python /home/home2/zrd/project/mmsegmentation/test_batch.py --config /home/home2/zrd/project/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_combine-zrdy.py --load-from /home/home2/zrd/project/mmsegmentation/work_dirs/deeplabv3plus_r50-d8_512x512_80k_combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/'$i'-'$imagepath'/' --roc $j --modelname $i --dataset $imagepath
elif [ "$i" == "pspnet_r50-d8_512x512_160k_combine-zrdy" ]; then
conda deactivate
conda activate open-mmlab
python /home/home2/zrd/project/mmsegmentation/test_batch.py --config /home/home2/zrd/project/mmsegmentation/configs/pspnet/pspnet_r50-d8_512x512_160k_combine-zrdy.py --load-from /home/home2/zrd/project/mmsegmentation/work_dirs/pspnet_r50-d8_combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/'$i'-'$imagepath'/' --roc $j --modelname $i --dataset $imagepath
elif [ "$i" == "fcn_plus_r50-d8-combine-zrdy" ]; then
conda deactivate
conda activate open-mmlab
python /home/home2/zrd/project/mmsegmentation/test_batch.py --config /home/home2/zrd/project/mmsegmentation/configs/fcn/fcn_plus_r50-d8.py --load-from /home/home2/zrd/project/mmsegmentation/work_dirs/fcnplus_r50-d8-combine-zrdy/latest.pth --image-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath --output-path '/home/home2/zrd/data/val/validation-paper1/'$i'-'$imagepath'/' --roc $j --modelname $i --dataset $imagepath
else
conda deactivate
conda activate open-mmlab
python /home/home2/zrd/project/mmsegmentation/test_batch.py --config /home/home2/zrd/project/mmsegmentation/configs/cpnet/$i.py --load-from /home/home2/zrd/project/mmsegmentation/work_dirs/$i/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/'$i'-'$imagepath'/' --roc $j --modelname $i --dataset $imagepath
fi
conda deactivate
conda activate open-mmlab
echo 'metrics for  '$i'roc: '$j
python /home/home2/zrd/project/mmsegmentation/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/'$i'-'$imagepath'/' --roc $j --modelname $i --dataset $imagepath
#echo 'get_colour_img for '$i'roc: '$j
#python /home/home2/zrd/project/mmsegmentation/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/'$i'-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/'$i'-'$imagepath'-col/'
done
done
