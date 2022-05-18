"""
Run WASR NOIMU on any image...
Specify the image with a full name as an input argument to: --img-path example_1.jpg

This script segments provided image into three semantic regions: sky, water and obstacles.
"""

from __future__ import print_function

import argparse
import datetime
import os
import sys
import time
import cv2
import scipy.io

from PIL import Image
from os import listdir
from os.path import isfile, join

import tensorflow as tf
import numpy as np

from wasr_models import wasr_NOIMU2, ImageReader, decode_labels, prepare_label
import requests
import json

# COLOR MEANS OF IMAGES FROM MODDv1 DATASET
#IMG_MEAN = np.array((148.8430, 171.0260, 162.4082), dtype=np.float32)
IMG_MEAN = np.array((123.675, 116.28, 103.53), dtype=np.float32)
# Number of classes
NUM_CLASSES = 3

# Output dir, where segemntation mask is saved
SAVE_DIR = 'output/'  # save directory

# Full path to the folder where images are stored
DATASET_PATH = 'test_images/'

# Path to trained weights
MODEL_WEIGHTS = '/home/zrd/wasr_network-master/weights_models/snapshots_wasr_noimu_zhuhai12749_3class_oldwasr_newmean/arm8imu3_noimu.ckpt-600'
#MODEL_WEIGHTS = '/home/zrd/wasr_network-master/example_weights/arm8imu3_noimu.ckpt-80000'
# Input image size. Our network expects images of resolution 512x384
#IMG_SIZE = [384, 512]
IMG_SIZE = [1080,1920]


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH,
                        help="Path to dataset files on which inference is performed.")
    parser.add_argument("--model-weights", type=str, default=MODEL_WEIGHTS,
                        help="Path to the file with model weights.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    parser.add_argument("--img-path", type=str, required=False,
                        help="Path to the image on which we want to run inference.")
    parser.add_argument("--roc", type=float, default=-1, required=False,
                        help="roc  threshold.")
    parser.add_argument("--modelname", type=str, default='',
                        help="modelname")
    parser.add_argument("--dataset", type=str, default='',
                        help="validation dataset")
    return parser.parse_args()


def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def count_flops(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))

def counta():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("parameters: {}".format(total_parameters))



def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    """Create the model and start the evaluation process."""
    args = get_arguments()
    sess = None
    flops =None
    params = None
    # create output folder/s if they dont exist yet
    url = r"http://localhost:8080/query/statistic_status/" + args.modelname + "/" + str(args.roc)  + "/" + str(args.dataset)
    res = requests.get(url)
    res = json.loads(res.text)
    print(res)
    if res['code'] == 0 and len(res['data']) > 0:
        if res['data'][0]['metric_status'] == 'Y':
            return
    url = r"http://localhost:8080/init_insert/performance/" + args.modelname
    requests.get(url)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    count = 0
    start_time = datetime.datetime.now()
    for file_path in os.listdir(args.dataset_path):
        if os.path.isfile(os.path.join(args.dataset_path, file_path)) == True:
            tf.reset_default_graph()
            # Read image
            img_org = cv2.imread(os.path.join(args.dataset_path, file_path))
            img_in = cv2.resize(img_org, (512, 384), interpolation=cv2.INTER_LINEAR)
            # Create network
            img_input = tf.placeholder(dtype=tf.uint8, shape=(img_in.shape[0], img_in.shape[1], 3))

            # Convert from opencv BGR to tensorflow's RGB format
            img_b, img_g, img_r = tf.split(axis=2, num_or_size_splits=3, value=img_input)

            # Join and subtract means
            img = tf.cast(tf.concat(axis=2, values=[img_r, img_g, img_b]), dtype=tf.float32)

            img -= IMG_MEAN

            # Expand first dimension
            img = tf.expand_dims(img, dim=0)
            with tf.variable_scope('', reuse=False):
                net = wasr_NOIMU2({'data': img}, is_training=False, num_classes=args.num_classes)

            # Which variables to load...
            restore_var = tf.global_variables()

            # Predictions (last layer of the decoder)
            raw_output = net.layers['fc1_voc12']
            #print('raw_output')
            #print(raw_output)
            # Upsample image to the original resolution
            raw_output = tf.image.resize_bilinear(raw_output, tf.shape(img)[1:3, ])
            raw_output = tf.nn.softmax(raw_output, dim=-1)
            #print(raw_output)
            # raw_output[:][:][:][1][raw_output[:][:][:][1] >= 0.8] = 1
            # raw_output[:][:][:][1][raw_output[:][:][:][1] < 0.8] = 0
            #print(type(raw_output))
            #raw_output = tf.argmax(raw_output, dimension=3)
            #print('raw_output after argmax')
            #print(raw_output)
            #pred = tf.expand_dims(raw_output, dim=3)
            pred=raw_output
            # Set up TF session and initialize variables.
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            init = tf.global_variables_initializer()

            sess.run(init)

            # Load weights
            loader = tf.train.Saver(var_list=restore_var)
            load(loader, sess, args.model_weights)
            break;
    for file_path in os.listdir(args.dataset_path):
        if os.path.isfile(os.path.join(args.dataset_path, file_path)) == True:
            img_org = cv2.imread(os.path.join(args.dataset_path, file_path))
            img_in = cv2.resize(img_org, (512, 384), interpolation=cv2.INTER_LINEAR)
            # Run inference
            preds = sess.run(pred, feed_dict={img_input: img_in})
            print('preds after run')
            print(preds.shape)
            print(preds)
            #print(type(preds))
            if args.roc >0:
               preds[:,:,:,1][preds[:,:,:,1] >= args.roc] = 1
               preds[:,:,:,1][preds[:,:,:,1] < args.roc] = 0
            #print('preds after roc')
            #print((preds))
            preds = preds.argmax(axis=-1)
            print('preds after argmax')
            print(preds.shape)
            print(preds)
            preds_squeeze = preds[0].squeeze().astype(np.uint8)
            print('preds after squeeze')
            print(preds_squeeze.shape)
            print(preds[0].shape)
            preds_out = cv2.resize(preds_squeeze, (img_org.shape[1], img_org.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Decode segmentation mask
            #msk = decode_labels(preds, num_classes=args.num_classes)

            # Save mask
            cv2.imwrite(args.save_dir + file_path.replace('.png', '').replace('.jpg', '') + '.png', preds_out)

            #flops = tf.profiler.profile(tf.get_default_graph(), options=tf.profiler.ProfileOptionBuilder.float_operation())
            #print('FLOPs: {}'.format(flops.total_float_ops))
            #count_flops(tf.get_default_graph())
            flops = 'no cal'
            if flops is None:
                print('wasr flops MACs = 2 * FLOPs')
                with tf.Session() as sesses:
                    # The session is binding to the default global graph
                    flops=tf.profiler.profile(
                        sesses.graph,
                        options=tf.profiler.ProfileOptionBuilder.float_operation())
                    params = tf.profiler.profile(sesses.graph,
                                                 options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())

                print('GFLOPs: {};    Trainable params: {}'.format(flops.total_float_ops / 1000000000.0,
                                                                   params.total_parameters))
                url = r"http://localhost:8080/update/performance/gflops/" + args.modelname + "/" + str(flops.total_float_ops / 1000000000.0) + "/" + str(params.total_parameters/1000000.0)
                requests.get(url)

            #counta()
            #sess.close()
            count = count + 1

    end_time = datetime.datetime.now()
    print(args.save_dir + ' fps :' + str(1 / ((end_time - start_time).seconds / count)))
    print('DONE!')
    url = r"http://localhost:8080/update/performance/fps/" + args.modelname + "/" + str(
        1 / ((end_time - start_time).seconds / count))
    requests.get(url)
    url = r"http://localhost:8080/update/test_batch_status/" + args.modelname
    requests.get(url)


if __name__ == '__main__':
    main()