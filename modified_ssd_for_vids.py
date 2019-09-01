
from __future__ import print_function, division
from builtins import range, input

import os, sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import imageio

if tf.__version__ < '1.14.0':
  raise ImportError(
    'Please upgrade your tensorflow installation to v1.4.* or later!'
  )

RESEARCH_PATH = '/home/bharat/Downloads/models-master/research'
MODELS_PATH = '/home/bharat/Downloads/models-master/research/object_detection'
sys.path.append(RESEARCH_PATH)
sys.path.append(MODELS_PATH)

import object_detection
from utils import label_map_util
from utils import visualization_utils as vis_util


MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = '%s/%s/frozen_inference_graph.pb' % (MODELS_PATH, MODEL_NAME)
PATH_TO_LABELS = '%s/data/mscoco_label_map.pbtxt' % MODELS_PATH
NUM_CLASSES = 90


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print("categories:")
print(categories)


# convert image -> numpy array
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    input_video = 'safari'
    video_reader = imageio.get_reader('%s.mp4' % input_video)
    video_writer = imageio.get_writer('%s_annotated.mp4' % input_video, fps=10)
    t0 = datetime.now()
    n_frames = 0
    for frame in video_reader:
      image_np = frame
      n_frames += 1
      image_np_expanded = np.expand_dims(image_np, axis=0)

      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      
      video_writer.append_data(image_np)

    fps = n_frames / (datetime.now() - t0).total_seconds()
    print("Frames processed: %s, Speed: %s fps" % (n_frames, fps))

    video_writer.close()
