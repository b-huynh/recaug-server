import os
from collections import namedtuple

import cv2

from object_detection.utils import label_map_util

dirname = os.path.dirname(__file__)
labels_path = os.path.join(dirname, 'mscoco_label_map.pbtxt')
# labels_path = os.path.join(config_server.MODEL_DATA_DIR, 'mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)

# Wrapper for predicted box values
PredictedBox = namedtuple('PredictedBox', ['class_id', 'class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])

class CocoPredictions(object):
    """Wrapper around predicted output from coco SSD models"""
    def __init__(self, threshold=0.5, single_instance=False):
        self._predictions = []
        self._threshold = threshold
        self._single_instance = single_instance

    def init_from_tf_od(self, img, pred):
        if self._single_instance:
            self._filter_confidence_single(img, pred)
        else:
            self._filter_confidence(img, pred)

    def _filter_confidence(self, img, pred):
        for i in range(0, len(pred['detection_classes'])):
            confidence = pred['detection_scores'][i]
            if confidence < self._threshold:
                continue

            class_id = pred['detection_classes'][i]
            if class_id in category_index.keys():
                class_name = category_index[class_id]['name']

            box = pred['detection_boxes'][i]
            ymin, xmin, ymax, xmax = box
            height, width, _ = img.shape
            ymin, xmin, ymax, xmax = (ymin * height, xmin * width,
                                      ymax * height, xmax * width)

            p = PredictedBox(
                class_id, class_name, confidence, xmin, ymin, xmax, ymax)
            self._predictions.append(p)

    def _filter_confidence_single(self, img, pred):
        # Dictionary to ensure keeping only highest confidence prediction
        confidence_map = {}  # key: string, value: (float, PredictedBox)

        for i in range(0, len(pred['detection_classes'])):
            confidence = pred['detection_scores'][i]
            if confidence < self._threshold:
                continue

            class_id = pred['detection_classes'][i]
            if class_id in category_index.keys():
                class_name = category_index[class_id]['name']

            if (class_name in confidence_map.keys() and 
                confidence < confidence_map[class_name][0]):
                continue

            box = pred['detection_boxes'][i]
            ymin, xmin, ymax, xmax = box
            height, width, _ = img.shape
            ymin, xmin, ymax, xmax = (ymin * height, xmin * width,
                                      ymax * height, xmax * width)

            p = PredictedBox(
                class_id, class_name, confidence,xmin, ymin, xmax,ymax)
            confidence_map[class_name] = (confidence, p)

        # Convert confidence map to list of predictions
        self._predictions.extend([x[1] for x in confidence_map.values()])

    def visualize(self, canvas):
        for pbox in self._predictions:
            xmin, ymin, xmax, ymax = [int(x) for x in [pbox.xmin, pbox.ymin, pbox.xmax, pbox.ymax]]
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
            display_str = '{}: {:.2f}'.format(pbox.class_name, pbox.confidence)
            cv2.putText(canvas, display_str, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255), 1)

    def get_predicted_points(self):
        labels = []
        for pbox in self._predictions:
            xcen = int(pbox.xmin + ((pbox.xmax - pbox.xmin) / 2))
            ycen = int(pbox.ymin + ((pbox.ymax - pbox.ymin) / 2))
            label_dict = {
                "className": pbox.class_name,
                "xmin": pbox.xmin,
                "xmax": pbox.xmax,
                "ymin": pbox.ymin,
                "ymax": pbox.ymax,
                "xcen": xcen,
                "ycen": ycen
            }
            labels.append(label_dict)
        return {"labels": labels}

    def print(self):
        pass