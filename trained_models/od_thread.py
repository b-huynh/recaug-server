import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict, deque
from io import StringIO
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# sys.path.append("C:\\Users\\yukib\\Projects\\recaug-server")
from predict_utils.coco_predictions import CocoPredictions

import threading
import queue
import cv2
import time

from trained_models.od_utils import visualize_labels

def inference_loop(graph, input_deque, output_deque, stop_event):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}

            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            while not stop_event.is_set():
                try: 
                    image = input_deque.pop()
                    # image = in_buf

                    # Handle for input image
                    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                    # Run inference
                    output_dict = sess.run(tensor_dict,
                                           feed_dict={image_tensor: np.expand_dims(image, 0)})

                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]

                    output_deque.append((image, output_dict))
                except IndexError:
                    continue
                except Exception as e:
                    raise e

def object_detection_worker(input_deque, output_deque, stop_event):
    MODEL_DIR = "C:\\Users\\yukib\\Projects\\recaug-server\\trained_models\\trained_weights\\ssd_mobilenet_v1_coco_2018_01_28"
    PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_DIR, 'frozen_inference_graph.pb')

    try:
        # Load model
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        inference_loop(detection_graph, input_deque, output_deque, stop_event)
    except Exception as e:
        raise e

class ObjectDetector(object):

    def __init__(self, threshold = 0.5, single_instance = False):
        # self.in_q = queue.Queue(maxsize = 1)
        # self.out_q = queue.Queue(maxsize = 1)

        self.in_q = deque(maxlen = 1)
        self.out_q = deque(maxlen = 1)

        self.stop_event = threading.Event()
        self.t = threading.Thread(target=object_detection_worker, args=(self.in_q, self.out_q, self.stop_event))
        self.t.start()

        self._latest_result = None
        self._threshold = threshold
        self._single_instance = single_instance

    @property
    def result_ready(self):
        return self.latest_result != None

    @property
    def latest_result(self):
        try:
            out_img, pred_dict = self.out_q.pop()
            coco_pred = CocoPredictions(threshold=self._threshold,
                single_instance=self._single_instance)
            coco_pred.init_from_tf_od(out_img, pred_dict)
            self._latest_result = (out_img, coco_pred)
        except IndexError:
            pass
        return self._latest_result

    def enqueue(self, img):
        self.in_q.append(img)

    def close(self):
        self.stop_event.set()
        self.t.join()



if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 60)

    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_buf = np.zeros((height, width, 3), dtype=np.uint8)

    detector = ObjectDetector()

    try:
        start_time = time.time()
        x = 1 # displays the frame rate every 1 second
        counter = 0
        fps = 0

        first_frame_displayed = False
        while True:
            ret, img = cam.read() # OpenCV natively uses numpy array
            
            if not first_frame_displayed:
                cv2.imshow('Predictions', img)
                first_frame_displayed = True

            if img is None:
                print("Not receiving frames...")
                continue

            img = cv2.flip(img, 1)

            detector.enqueue(img)

            if detector.result_ready:
                # out_img, output_dict = detector.latest_result
                out_img, predictions = detector.latest_result
                
                # visualize_labels(out_img, output_dict)
                predictions.visualize(out_img)
                
                # Write FPS
                font = cv2.FONT_HERSHEY_SIMPLEX
                display_str = 'FPS: {:.2f}'.format(fps)
                cv2.putText(out_img, display_str, (10, 470), font, 1, (255,255,255), 2, cv2.LINE_AA)

                cv2.imshow('Predictions', out_img)
                counter += 1
                if (time.time() - start_time) > x :
                    fps = counter / (time.time() - start_time)
                    counter = 0
                    start_time = time.time()
            
            if cv2.waitKey(1) == ord('q'):
                detector.close()
                break

        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass