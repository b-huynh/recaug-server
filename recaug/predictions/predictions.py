import collections
import json
import os

import cv2

PredictedBox = collections.namedtuple(
    'PredictedBox', 
    ['class_id', 'class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])

def create_category_index(label_map_path):
    with open(label_map_path, 'r') as f:
        category_index = json.load(f)
    static_keys = list(category_index.keys())
    for k in static_keys:
        category_index[int(k)] = category_index.pop(k)
    return category_index

class PredictionsUtil:
    """
    Factory utility object for formatting Predictions out of nn outputs.
    
    Must choose a label_map to use for conversion. Available choices are:
        * TensorFlow2COCO
    """
    
    label_map_dir = os.path.join(os.path.dirname(__file__), 'label_maps')
    
    def __init__(self, label_map, filters=[]):
        # Look for relevant namemap in the namemap directory
        label_map_path = os.path.join(
            type(self).label_map_dir, label_map + '.json')
        self.category_index = create_category_index(label_map_path)
        self.filters = filters

    # Ingest nn detections and returns a list of formatted predictions.
    def format(self, image, nn_output):
        predictions = []
        for i in range(0, len(nn_output['detection_classes'])):
            # Convert network specific class ids to category names
            class_id = nn_output['detection_classes'][i]
            if class_id in self.category_index.keys():
                class_name = self.category_index[class_id]['name']

            confidence = nn_output['detection_scores'][i]

            # Convert normalized coords to pixel coords
            ymin, xmin, ymax, xmax = nn_output['detection_boxes'][i]
            height, width, _ = image.shape
            ymin, xmin, ymax, xmax = (
                ymin * height, xmin * width, ymax * height, xmax * width)
            predictions.append(PredictedBox(
                class_id, class_name, confidence, xmin, ymin, xmax, ymax))

        # Apply filters
        for filt in self.filters:
            predictions = filt.filter(predictions)
        
        result = Predictions()
        result.predictions = predictions
        return result

def draw(predictions, canvas):
    for pbox in predictions:
        xmin, ymin, xmax, ymax = [int(x) for x in [pbox.xmin, pbox.ymin, pbox.xmax, pbox.ymax]]
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        display_str = '{}: {:.2f}'.format(pbox.class_name, pbox.confidence)
        cv2.putText(canvas, display_str, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255), 1)  

class Predictions:
    """Don't instantiate yourself, use PredictionsUtil"""
    def __init__(self):
        self.predictions = []
    
    def visualize(self, canvas):
        for pbox in self.predictions:
            xmin, ymin, xmax, ymax = [int(x) for x in [pbox.xmin, pbox.ymin, pbox.xmax, pbox.ymax]]
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
            display_str = '{}: {:.2f}'.format(pbox.class_name, pbox.confidence)
            cv2.putText(canvas, display_str, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255), 1)

    def get_predicted_points(self):
        labels = []
        for pbox in self.predictions:
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
