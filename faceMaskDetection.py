import numpy as np
import os, sys, cv2, tarfile, zipfile
from itertools import combinations 
import six.moves.urllib as urllib
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.14.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.14.*.')

from utils import label_map_util

from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'training/labelmap.pbtxt'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

'''
OUTPUT = '<path/to/output.avi>'

# Checks and deletes the output file.
# You cant have a existing file or it will through an error.
if os.path.isfile(OUTPUT):
    os.remove(OUTPUT)

# Playing video from file
cap = cv2.VideoCapture("<path/to/video>")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# Convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
out = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                      10, (frame_width, frame_height))

sys.path.append("..")
'''

cap = cv2.VideoCapture(0) #Use only if not started capturing video before.

with detection_graph.as_default():
    with tf.Session() as sess:
        while True:
            ret, image_np = cap.read()
            
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=6)
            '''
            #use this only if saving the output video.
            if ret == True:
                # Saves for video
                out.write(image_np)

                # Display the resulting frame
                cv2.imshow('Mask Detection', image_np)
            '''
            cv2.imshow('Mask Detection System', cv2.resize(image_np, (800, 600)))
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        cap.release()
        #out.release()