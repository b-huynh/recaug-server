import os
from collections import namedtuple

import cv2

from ssd_keras.eval_utils.coco_utils import get_coco_category_maps

dir_path, _ = os.path.split(__file__)
ann_path = os.path.join(dir_path, "annotations", "instances_val2017.json")
cats_to_classes, classes_to_cats, cats_to_names, classes_to_names = get_coco_category_maps(ann_path)

def get_classname(class_id):
    return classes_to_names[class_id]

# Wrapper for predicted box values
PredictedBox = namedtuple('PredictedBox', ['class_id', 'class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])

class CocoPredictions(object):
    """Wrapper around predicted output from coco SSD models"""
    def __init__(self, pred, orig_shape, ssd_height, ssd_width):
        self._predictions = []
        for box in pred[0]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = box[-4] * orig_shape[1] / ssd_width
            ymin = box[-3] * orig_shape[0] / ssd_height
            xmax = box[-2] * orig_shape[1] / ssd_width
            ymax = box[-1] * orig_shape[0] / ssd_height
            
            # Predicted class and confidence
            class_id = int(box[0])
            class_name = get_classname(class_id)
            confidence = box[1]
            
            self._predictions.append(PredictedBox(class_id, class_name, confidence, xmin, ymin, xmax, ymax))

    def visualize(self, canvas):
        for pbox in self._predictions:
            xmin, ymin, xmax, ymax = [int(x) for x in [pbox.xmin, pbox.ymin, pbox.xmax, pbox.ymax]]
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            display_str = '{}: {:.2f}'.format(pbox.class_name, pbox.confidence)
            cv2.putText(canvas, display_str, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

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