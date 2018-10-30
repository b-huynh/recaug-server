from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf

from ssd_keras.models.keras_ssd300 import ssd_300
from ssd_keras.keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_keras.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd_keras.keras_layers.keras_layer_DecodeDetections import DecodeDetections
from ssd_keras.keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from ssd_keras.keras_layers.keras_layer_L2Normalization import L2Normalization
from ssd_keras.data_generator.object_detection_2d_data_generator import DataGenerator
from ssd_keras.eval_utils.coco_utils import get_coco_category_maps, predict_all_to_json

from predict_utils.coco_predictions import CocoPredictions

import glob
import os
import itertools
import sys

_dir_path, _ = os.path.split(__file__) 

class Coco300(object):
    IMG_HEIGHT = 300
    IMG_WIDTH = 300
    PRETRAINED_WEIGHTS_PATH = os.path.join(_dir_path, 'trained_weights', 'VGG_coco_SSD_300x300_iter_400000.h5')

    """A 300x300 SSD Model with pre-trained weights for inference"""
    def __init__(self):
        self._build_model()

    def _build_model(self):
        # 1: Build the Keras model
        K.clear_session() # Clear previous models from memory.

        # Config custom session params
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.7
        # K.set_session(tf.Session(config=config))

        self._model = ssd_300(image_size=(self.IMG_HEIGHT, self.IMG_WIDTH, 3),
                        n_classes=80,
                        mode='inference',
                        l2_regularization=0.0005,
                        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], # The scales for Pascal VOC are [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
                        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                [1.0, 2.0, 0.5],
                                                [1.0, 2.0, 0.5]],
                        two_boxes_for_ar1=True,
                        steps=[8, 16, 32, 64, 100, 300],
                        offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        clip_boxes=False,
                        variances=[0.1, 0.1, 0.2, 0.2],
                        normalize_coords=True,
                        subtract_mean=[123, 117, 104],
                        swap_channels=[2, 1, 0],
                        confidence_thresh=0.01,
                        iou_threshold=0.45,
                        top_k=200,
                        nms_max_output_size=400)

        # 2: Load the trained weights into the model.
        weights_path = self.PRETRAINED_WEIGHTS_PATH
        self._model.load_weights(weights_path, by_name=True)

        # 3: Compile the model so that Keras won't complain the next time you load it.
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self._model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
        
    def _remove_low_confidence(self, pred):
        confidence_threshold = 0.5
        pred = [pred[k][pred[k,:,1] > confidence_threshold] for k in range(pred.shape[0])]
        return pred

    def predict(self, img):
        resized_img = cv2.resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH))
        input_img = np.array([resized_img])
        pred = self._model.predict(input_img)
        pred = self._remove_low_confidence(pred)
        return CocoPredictions(pred, img.shape, self.IMG_HEIGHT, self.IMG_WIDTH)