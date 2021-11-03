from mmseg.apis import inference_segmentor, init_segmentor,inference_segmentor_concactdata
import mmcv
import datetime

config_file = '/home/zrd/mmsegmentation-master/configs/cpnet/cpnet_r50-d8.py'
#config_file = '/home/zrd/mmsegmentation-master/configs/cpnet/cpnet_r50-d8-classs3.py'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnet_r50-d8-combine/latest.pth'
checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnet_r50-d8-combine-box/latest.pth'
#checkpoint_file = '/home/zrd/mmsegmentation-master/work_dirs/cpnet_r50-d8-master1325/latest.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/home/zrd/segment-val/X31_1_0000052250.jpg-1.jpg'  # or img = mmcv.imread(img), which will only load it once
start_time=datetime.datetime.now()
print('starttime: '+ str(start_time))
result = inference_segmentor_concactdata(model, img)
print(result[0].shape)
end_time=datetime.datetime.now()
print('endtime: '+ str(end_time))
print('spendtime(s) :' + str((end_time-start_time).microseconds))


model.show_result(img, result,palette = [[0, 0, 0], [128, 0, 0]],out_file='X31_1_0000052250.jpg-1.png',save_annotation=True)
#model.show_result(img, result,palette = [[0, 0, 0], [128, 0, 0],[0, 128, 0]],out_file='X31_1_0000052250.jpg-1.png')
