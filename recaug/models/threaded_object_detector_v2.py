import collections
import os
import queue
import threading
import time

import cv2
import numpy as np
import tensorflow as tf

from ..predictions.coco_predictions import CocoPredictions

def inference_loop(graph, in_q, out_q, stop_event):
    sess = tf.Session(graph=graph)

    tensor_dict = {
        'num_detections'   : graph.get_tensor_by_name('num_detections:0'),
        'detection_boxes'  : graph.get_tensor_by_name('detection_boxes:0'),
        'detection_scores' : graph.get_tensor_by_name('detection_scores:0'),
        'detection_classes': graph.get_tensor_by_name('detection_classes:0')
        # 'detection_masks': graph.get_tensor_by_name('detection_masks:0')
    }

    while not stop_event.is_set():
        if in_q:
            # Get handle for input image
            image, client, client_callback = in_q.pop()
            image_tensor = graph.get_tensor_by_name('image_tensor:0')

            # Run inference
            output = sess.run(tensor_dict,
                feed_dict={image_tensor: np.expand_dims(image, 0)})

            # Outputs are float32 numpy arrays, so convert to appropriate types
            output['num_detections']    = int(output['num_detections'][0])
            output['detection_classes'] = output['detection_classes'][0].astype(np.uint8)
            output['detection_boxes']   = output['detection_boxes'][0]
            output['detection_scores']  = output['detection_scores'][0]

            out_q.append((image, output, client))
            if client_callback:
                client_callback(image, output, client)
        # except IndexError:
        #     continue
        # except Exception as e:
        #     raise e

class ThreadedObjectDetector:
    model_dir = os.path.join(os.path.dirname(__file__), 'trained_weights')

    def __init__(self, model_name):
        self.in_q = collections.deque(maxlen = 10)
        self.out_q = collections.deque(maxlen = 10)

        self._load_model(model_name)

        self._latest_result = None

    def _load_model(self, model_name):
        # TODO: If model does not exist in trained_weights, download it
        model_path = os.path.join(
            type(self).model_dir, model_name, 'frozen_inference_graph.pb')

        graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            graph_def.ParseFromString(fid.read())

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.stop_event = threading.Event()
        self.t = threading.Thread(
            target=inference_loop,
            args=(self.graph, self.in_q, self.out_q, self.stop_event)
        )
        self.t.start()

    def enqueue(self, message, client, client_callback = None):
        self.in_q.append((message, client, client_callback))

    def stop(self):
        self.stop_event.set()
        self.t.join()


# if __name__ == '__main__':
#     cam = cv2.VideoCapture(0)
#     cam.set(cv2.CAP_PROP_FPS, 60)

#     height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
#     in_buf = np.zeros((height, width, 3), dtype=np.uint8)

#     detector = ObjectDetector('ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03')

#     try:
#         start_time = time.time()
#         x = 1 # displays the frame rate every 1 second
#         counter = 0
#         fps = 0

#         first_frame_displayed = False
#         while True:
#             ret, img = cam.read() # OpenCV natively uses numpy array
            
#             if not first_frame_displayed:
#                 cv2.imshow('Predictions', img)
#                 first_frame_displayed = True

#             if img is None:
#                 print("Not receiving frames...")
#                 continue

#             img = cv2.flip(img, 1)

#             detector.enqueue(img)

#             if detector.result_ready:
#                 # out_img, output_dict = detector.latest_result
#                 out_img, predictions = detector.latest_result
                
#                 # visualize_labels(out_img, output_dict)
#                 predictions.visualize(out_img)
                
#                 # Write FPS
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 display_str = 'FPS: {:.2f}'.format(fps)
#                 cv2.putText(out_img, display_str, (10, 470), font, 1, (255,255,255), 2, cv2.LINE_AA)

#                 cv2.imshow('Predictions', out_img)
#                 counter += 1
#                 if (time.time() - start_time) > x :
#                     fps = counter / (time.time() - start_time)
#                     counter = 0
#                     start_time = time.time()
            
#             if cv2.waitKey(1) == ord('q'):
#                 detector.close()
#                 break

#         cv2.destroyAllWindows()
#     except KeyboardInterrupt:
#         pass
