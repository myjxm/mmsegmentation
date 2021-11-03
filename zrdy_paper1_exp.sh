#!/bin/sh
imagepath=$1
segpath=$2
source /home/zrd/anaconda3/etc/profile.d/conda.sh
conda init bash
conda deactivate
conda activate open-mmlab
python /home/zrd/mmsegmentation-master/test_batch.py --config /home/zrd/mmsegmentation-master/configs/cpnet/cpnetplus_inter_r50-d8-combine-zrdy.py --load-from /home/zrd/mmsegmentation-master/work_dirs/cpnetplus_inter_r50-d8-combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_inter_r50-d8-combine-zrdy-'$imagepath'/'
python /home/zrd/mmsegmentation-master/test_batch.py --config /home/zrd/mmsegmentation-master/configs/cpnet/cpnetplus_intra_r50-d8-combine-zrdy.py --load-from /home/zrd/mmsegmentation-master/work_dirs/cpnetplus_intra_r50-d8-combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_intra_r50-d8-combine-zrdy-'$imagepath'/'
python /home/zrd/mmsegmentation-master/test_batch.py --config /home/zrd/mmsegmentation-master/configs/mobilenet_v2/cpnetplus_m-v2-d8_512x512_160k_combine-zrdy.py --load-from /home/zrd/mmsegmentation-master/work_dirs/cpnetplus_m-v2-d8_512x512_160k_combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_m-v2-d8_512x512_160k_combine-zrdy-'$imagepath'/'
python /home/zrd/mmsegmentation-master/test_batch.py --config /home/zrd/mmsegmentation-master/configs/cpnet/cpnetplus_r50-d8-combine-zrdy.py --load-from /home/zrd/mmsegmentation-master/work_dirs/cpnetplus_r50-d8-combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-combine-zrdy-'$imagepath'/'
python /home/zrd/mmsegmentation-master/test_batch.py --config /home/zrd/mmsegmentation-master/configs/cpnet/cpnetplus_rs101-d8-combine-zrdy.py --load-from /home/zrd/mmsegmentation-master/work_dirs/cpnetplus_rs101-d8-combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_rs101-d8-combine-zrdy-'$imagepath'/'
python /home/zrd/mmsegmentation-master/test_batch.py --config /home/zrd/mmsegmentation-master/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_combine-zrdy.py --load-from /home/zrd/mmsegmentation-master/work_dirs/deeplabv3plus_r50-d8_512x512_80k_combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/deeplabv3plus_r50-d8_512x512_80k_combine-zrdy-'$imagepath'/'
python /home/zrd/mmsegmentation-master/test_batch.py --config /home/zrd/mmsegmentation-master/configs/pspnet/pspnet_r50-d8_512x512_160k_combine-zrdy.py --load-from /home/zrd/mmsegmentation-master/work_dirs/pspnet_r50-d8_combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/pspnet_r50-d8_combine-zrdy-'$imagepath'/'
python /home/zrd/mmsegmentation-master/test_batch.py --config /home/zrd/mmsegmentation-master/configs/cpnet/cpnet_r50-d8-combine-zrdy.py --load-from  /home/zrd/mmsegmentation-master/work_dirs/cpnet_r50-d8-combine-zrdy/latest.pth --image-path  '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnet_r50-d8-combine-zrdy-'$imagepath'/'
python /home/zrd/mmsegmentation-master/test_batch.py --config /home/zrd/mmsegmentation-master/configs/fcn/fcn_plus_r50-d8.py --load-from /home/zrd/mmsegmentation-master/work_dirs/fcnplus_r50-d8-combine-zrdy/latest.pth --image-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath --output-path '/home/home2/zrd/data/val/validation-paper1/fcn_plus_r50-d8-combine-zrdy-'$imagepath'/'
conda deactivate
conda activate segmen
python /home/zrd/Segmentation-Pytorch-master/predict_output.py  --model SegNet  --checkpoint /home/zrd/Segmentation-Pytorch-master/checkpoint/combine_zrdy/SegNet/best_model.pth  --out_stride 8   --root '/data/open_data'   --dataset 'combine_zrdy'   --predict_type validation  --predict_mode whole  --crop_size 768  --tile_hw_size '768,768'   --batch_size 2   --gpus 1 --img_path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output_path '/home/home2/zrd/data/val/validation-paper1/segnet-'$imagepath'/'
conda deactivate
conda activate keras
python /home/zrd/wl/inference_customer_model6.py --dataset-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --save-dir '/home/home2/zrd/data/val/validation-paper1/wl-largeBN-'$imagepath'/' --model-weights /home/zrd/wl/models_saved/final_models/unet160--largeBN--dice_coef_loss.bak0610
python /home/zrd/wl/inference_customer_model8.py --dataset-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --save-dir '/home/home2/zrd/data/val/validation-paper1/wl-largeBN_mobilenetv2-'$imagepath'/' --model-weights /home/zrd/wl/models_saved/final_models/unet160--largeBN_mobilenetv2--dice_coef_loss.bak0610
conda deactivate
conda activate wasr
python /home/zrd/wasr_network-master/wasr_inference_noimu_batch.py --dataset-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath --save-dir '/home/home2/zrd/data/val/validation-paper1/wasr-'$imagepath'/'
conda deactivate
conda activate open-mmlab
echo 'metrics for cpnetplus_inter_r50-d8-combine-zrdy' 
python /home/zrd/mmsegmentation-master/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_inter_r50-d8-combine-zrdy-'$imagepath'/'
echo 'metrics for cpnetplus_intra_r50-d8-combine-zrdy' 
python /home/zrd/mmsegmentation-master/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_intra_r50-d8-combine-zrdy-'$imagepath'/'
echo 'metrics for cpnetplus_m-v2-d8_512x512_160k_combine-zrdy' 
python /home/zrd/mmsegmentation-master/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_m-v2-d8_512x512_160k_combine-zrdy-'$imagepath'/'
echo 'metrics for cpnetplus_r50-d8-combine-zrdy' 
python /home/zrd/mmsegmentation-master/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-combine-zrdy-'$imagepath'/'
echo 'metrics for cpnetplus_rs101-d8-combine-zrdy' 
python /home/zrd/mmsegmentation-master/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_rs101-d8-combine-zrdy-'$imagepath'/'
echo 'metrics for deeplabv3plus_r50-d8_512x512_80k_combine-zrdy' 
python /home/zrd/mmsegmentation-master/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/deeplabv3plus_r50-d8_512x512_80k_combine-zrdy-'$imagepath'/'
echo 'metrics for pspnet_r50-d8_combine-zrdy' 
python /home/zrd/mmsegmentation-master/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/pspnet_r50-d8_combine-zrdy-'$imagepath'/'
echo 'metrics for cpnet_r50-d8-combine-zrdy' 
python /home/zrd/mmsegmentation-master/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/cpnet_r50-d8-combine-zrdy-'$imagepath'/'
echo 'metrics for segnet-combine-zrdy' 
python /home/zrd/mmsegmentation-master/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/segnet-'$imagepath'/'
echo 'metrics for wl-largeBN-combine-zrdy' 
python /home/zrd/mmsegmentation-master/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/wl-largeBN-'$imagepath'/'
echo 'metrics for wl-largeBN_mobilenetv2-combine-zrdy' 
python /home/zrd/mmsegmentation-master/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/wl-largeBN_mobilenetv2-'$imagepath'/'
echo 'metrics for wasr-combine-zrdy' 
python /home/zrd/mmsegmentation-master/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/wasr-'$imagepath'/'
echo 'metrics for fcn_plus_-combine-zrdy'
python /home/zrd/mmsegmentation-master/metrics_calculate.py --seg-path  '/home/home2/zrd/data/val/validation-paper1/'$segpath  --test-path '/home/home2/zrd/data/val/validation-paper1/fcn_plus_r50-d8-combine-zrdy-'$imagepath'/'
python /home/zrd/mmsegmentation-master/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_inter_r50-d8-combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_inter_r50-d8-combine-zrdy-'$imagepath'-col/'
python /home/zrd/mmsegmentation-master/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_intra_r50-d8-combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_intra_r50-d8-combine-zrdy-'$imagepath'-col/'
python /home/zrd/mmsegmentation-master/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_m-v2-d8_512x512_160k_combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_m-v2-d8_512x512_160k_combine-zrdy-'$imagepath'-col/'
python /home/zrd/mmsegmentation-master/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_r50-d8-combine-zrdy-'$imagepath'-col/'
python /home/zrd/mmsegmentation-master/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_rs101-d8-combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/cpnetplus_rs101-d8-combine-zrdy-'$imagepath'-col/'
python /home/zrd/mmsegmentation-master/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/deeplabv3plus_r50-d8_512x512_80k_combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/deeplabv3plus_r50-d8_512x512_80k_combine-zrdy-'$imagepath'-col/'
python /home/zrd/mmsegmentation-master/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/pspnet_r50-d8_combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/pspnet_r50-d8_combine-zrdy-'$imagepath'-col/'
python /home/zrd/mmsegmentation-master/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/cpnet_r50-d8-combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/cpnet_r50-d8-combine-zrdy-'$imagepath'-col/'
python /home/zrd/mmsegmentation-master/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/segnet-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/segnet-combine-zrdy-'$imagepath'-col/'
python /home/zrd/mmsegmentation-master/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/wl-largeBN-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/wl-largeBN-combine-zrdy-'$imagepath'-col/'
python /home/zrd/mmsegmentation-master/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/wl-largeBN_mobilenetv2-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/wl-largeBN_mobilenetv2-combine-zrdy-'$imagepath'-col/'
python /home/zrd/mmsegmentation-master/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/wasr-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/wasr-combine-zrdy-'$imagepath'-col/'
python /home/zrd/mmsegmentation-master/get_colour_img.py --img-path '/home/home2/zrd/data/val/validation-paper1/'$imagepath  --output-path '/home/home2/zrd/data/val/validation-paper1/fcn_plus_r50-d8-combine-zrdy-'$imagepath'/'  --save-path '/home/home2/zrd/data/val/validation-paper1/fcn_plus_r50-d8-combine-zrdy-'$imagepath'-col/'
