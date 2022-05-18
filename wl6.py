from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import load_model
from dataset import Dataset
from losses import *
import cv2
import numpy as np
import os
from models import Models
import argparse
import datetime
import tensorflow as tf
import requests
import json

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--dataset-path", type=str,
                        help="Path to dataset files on which inference is performed.")
    parser.add_argument("--model-weights", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--save-dir", type=str,
                        help="Where to save predicted mask.")
    parser.add_argument("--roc", type=float, default=-1,
						help="roc thresold")
    parser.add_argument("--modelname", type=str, default='',
						help="modelname")
    parser.add_argument("--dataset", type=str, default='',
						help="validation dataset")
    return parser.parse_args()



def main():
		args = get_arguments()
		url = r"http://localhost:8080/query/statistic_status/" + args.modelname + "/" + str(args.roc) + "/" + str(args.dataset)
		res = requests.get(url)
		res = json.loads(res.text)
		print(res)
		if res['code'] == 0 and len(res['data']) > 0:
			if res['data'][0]['metric_status'] == 'Y':
				return
		url = r"http://localhost:8080/init_insert/performance/" + args.modelname
		requests.get(url)
		#model = load_model(args.dataset_path,custom_objects={'dice_coef': dice_coef})
		model_class = Models()
		input_shape = (160, 160, 3)
		if args.dataset == 'zhuhai12749_3class_image' or args.dataset == 'zhuhai433_3class_image' or args.dataset == 'zhuhai480_3class_image' or args.dataset == 'zhuhai500_3class_image' or args.dataset == 'zhuhai15708_3class_image' or args.dataset == 'MaSTr1325_images_512x384':
		   nb_classes = 3
		elif args.dataset == 'combine_zrdy' :
		   nb_classes = 2
		else:
			raise NotImplementedError(
				"This repository now supports datasets %s is not included" % args.dataset)
		metrics = ['binary_crossentropy', 'mse', 'mae', dice_coef]
		#model = model_class.get_unet_model_8(input_shape, nb_classes, dice_coef_loss, metrics)
		print("nb_classes")
		print(nb_classes)
		model = model_class.get_unet_model_6(input_shape, nb_classes, dice_coef_loss, metrics)
		model.load_weights(args.model_weights)
		print('wl6 flops MACs = 2 * FLOPs')
		with tf.Session() as sess:
			# The session is binding to the default global graph
			flops = tf.profiler.profile(
				sess.graph,
				options=tf.profiler.ProfileOptionBuilder.float_operation())
			params = tf.profiler.profile(sess.graph,
										 options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())

		print('GFLOPs: {};    Trainable params: {}'.format(flops.total_float_ops / 1000000000.0,
														   params.total_parameters))
		url = r"http://localhost:8080/update/performance/gflops/" + args.modelname + "/" + str(
			flops.total_float_ops / 1000000000.0) + "/" + str(params.total_parameters / 1000000.0)
		requests.get(url)
		# create output folder/s if they dont exist yet
		if not os.path.exists(args.save_dir):
			os.makedirs(args.save_dir)
		count=0
		start_time=datetime.datetime.now()
		for file_path  in os.listdir(args.dataset_path):
			if os.path.isfile(os.path.join(args.dataset_path, file_path)) == True:
				bgr=cv2.imread(os.path.join(args.dataset_path, file_path))
				rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
				rgb_convert = cv2.resize(rgb,(160,160))
				rgb_convert = rgb_convert/255.0
				x = np.expand_dims(rgb_convert, axis=0)
				#x = rgb_convert
				print('x')
				print(x.shape)
				y=model.predict(x)
				print('y')
				print(y.shape)
				print(y)
				y_out = y.squeeze()
				print('y_out')
				print(y_out.shape)
				print(y_out)
				y_output = cv2.resize(y_out,(rgb.shape[1],rgb.shape[0]))
				print('y_output')
				print(y_output.shape)
				print(y_output)
				if args.roc >= 0:
					y_output[y_output >= args.roc] = 1
					y_output[y_output < args.roc] = 0
				print(y_output)
				cv2.imwrite(args.save_dir + file_path.replace('jpg','png'), y_output)  #imwrite写入的时候按照会有一个四舍五入的过程， 并且有saturation的过程。小于0置为0，大于255置于255。
				count = count + 1
		end_time=datetime.datetime.now()
		#print(args.save_dir + ' spendtime(ms) :' + str((end_time-start_time).seconds/count))
		print(args.save_dir + ' fps :' + str(1/((end_time-start_time).seconds/count)))
		url = r"http://localhost:8080/update/performance/fps/" + args.modelname + "/" + str(
			1 / ((end_time - start_time).seconds / count))
		requests.get(url)
		url = r"http://localhost:8080/update/test_batch_status/" + args.modelname
		requests.get(url)


if __name__ == '__main__':
    main()