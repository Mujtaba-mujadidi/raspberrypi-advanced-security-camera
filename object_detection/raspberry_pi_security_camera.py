#import required libraries and packages
import os
import cv2
import tensorflow as tf
import numpy as np

import RPi.GPIO as GPIO
import time
from time import sleep

from picamera import PiCamera
from picamera.array import PiRGBArray
import argparse
import sys

#To start from exisitng folder 
sys.path.append('..')

from utils import label_map_util
from utils import visualization_utils as vis_util

print("Packages import success!")


#Name of the object detection pre trained model used
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

IM_WIDTH = 540    #Use smaller resolution for
IM_HEIGHT = 380   #slightly faster framerate


#Current working director path
CWD_PATH = os.getcwd()

#Frozen detection graph .pb file path. it has the model that will be used for object detection
#frozenDetectionGraphPath = os.path.join(currentWorkingDirectoryPath, ObjectDetectionModelName, 'frozen_inference_graph.pb') 
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

#90 classes of object can be detected by the model.
NUM_CLASSES = 90

# Labels map file path
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

print("Paths initialisation success!")

# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine


#Loads labels map
#Converst label map into list of categories
#converts list of categories into a dictionary
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print("Labels Mapping success!")

# Load the Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

print("Loading detection model success!")

#Input data for the model is image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

#Output data from the model is list of detectio boxes, classs and scores

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

print("Input output type intialisation sucess!")

# Number of objects detected
#detectionCount = detection_graph.get_tensor_by_name('num_detections:0')

#Frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

print("Frame rate assignment")

#Camera initialisation



#Sensor initialisation
GPIO.setmode(GPIO.BCM)
PIR_PIN = 7
GPIO.setup(PIR_PIN, GPIO.IN)

print ("PIR module test (CTR+C to exit)")

def detect():
    print("Detecting")
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        
        
        #print(object_id_to_class_mapper[classes[0]])
        print(classes)
        print("--------------------------------")
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)
        
      
        
        

        cv2.putText(frame,"FPS: {0:.2f}".format(1),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

    cv2.destroyAllWindows()

    
'''
def detectObjects():
    print("Object detecion called")
    rawCapture = PiRGBArray(camera, size=(frameWidth,frameheight))
    rawCapture.truncate(0)
    for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        t1 = cv2.getTickCount()
    
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame.array)
        frame.setflags(write=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)
        
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        print(classes)
        rawCapture.truncate(0)
    
    print ("list of objetcs")

'''


def startRecording():
    print("Recording")
    
def stopRecording():
    print ("Not Recording")


GPIO.setmode(GPIO.BCM)

PIR_PIN = 7

GPIO.setup(PIR_PIN, GPIO.IN)

print ("PIR module test (CTR+C to exit)")

#time.sleep(2)

detection = False
recording = False

detect()

'''
while True:
    print("---------------------------------------------------")
    if GPIO.input(PIR_PIN):
        if recording == False:
            startRecording()
            recording = True
            
        if detection == False:
            detectObjects()
            detection = True
            
    else:
        stopRecording()
        recording = False
        detection = False
        
        
    time.sleep(5)

#camera.stop_preview();
'''
print("Execution success")


