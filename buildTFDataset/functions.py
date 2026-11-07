
import cv2
import sys
sys.path.insert(0,'./')
from utils.utils import get_roi_from_region, polygon_area
from utils.features import _bytes_feature, _int64_feature, _float_feature
import numpy as np
import sys
import tensorflow as tf

# 2D symbologies, as named in the per-source BarBeR JSONs and in the unified annotation
TYPES_2D = {'QR': 'QR', 'DATAMATRIX': 'DATAMATRIX', 'AZTEC': 'AZTEC',
            'QR Code': 'QR', 'Data Matrix': 'DATAMATRIX', 'Aztec': 'AZTEC'}

def to_quadrilateral(polygon):
    # The TFRecord readers expect exactly 4 corners; curved codes may be annotated
    # with more points, so they are replaced by their minimum-area rectangle.
    if len(polygon) == 4:
        return polygon
    rect = cv2.minAreaRect(np.asarray(polygon, dtype=np.float32))
    return np.round(cv2.boxPoints(rect)).astype(np.int64)

def convert_to_defect_dictionary(json_data, dictionary=None):
    if dictionary is None:
        dictionary = {}
    keys = list(json_data['_via_img_metadata'].keys())
    for key in keys:
        regions = []
        file_name = json_data['_via_img_metadata'][key]['filename']
        for region in json_data['_via_img_metadata'][key]['regions']:
            polygon = get_roi_from_region(region)
            if 'PPE_x' in region['region_attributes']:
                rows = region['region_attributes']['Rows']
                cols = region['region_attributes']['Cols']
                area = polygon_area(polygon)
                ppe = np.sqrt(area/((rows+2)*(cols+2)))
            else:
                if region['region_attributes']['Type'] not in TYPES_2D:
                    continue
                ppe = float(region['region_attributes']['PPE'])
                if ppe < 0:  # -1 (string or number) means PPE not annotated
                    continue
            
            regions.append({'polygon': to_quadrilateral(polygon), 'ppe': ppe,
                            'type': TYPES_2D.get(region['region_attributes']['Type'],
                                                 region['region_attributes']['Type'])})
        dictionary[file_name] = regions
    return dictionary

def image_example(img, polygon, file_name, ppe):
    shape = img.shape
    _,img = cv2.imencode('.png',img)
    feature = {
        'img': _bytes_feature(img.tobytes()),
        'shape': _int64_feature(shape),
        'polygon': _int64_feature(polygon.ravel()),
        'file_name': _bytes_feature(str(file_name).encode('ascii') ),
        'ppe': _float_feature(ppe)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))