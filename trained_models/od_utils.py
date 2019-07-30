import os

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import config_server

labels_path = os.path.join(config_server.MODEL_DATA_DIR, 'mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)

def visualize_labels(out_img, output_dict):
    vis_util.visualize_boxes_and_labels_on_image_array(
        out_img,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)