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

print("Imprt success!")


#Name of the object detection pre trained model used
ObjectDetectionModelName = 'ssdlite_mobilenet_v2_coco_2018_05_09'

#Current working director path
currentWorkingDirectoryPath = os.getcwd()

#Frozen detection graph .pb file path. it has the model that will be used for object detection
frozenDetectionGraphPath = os.path.join(currentWorkingDirectoryPath, ObjectDetectionModelName, 'frozen_inference_graph.pb') 

#90 classes of object can be detected by the model.
numberOfClasses = 90;

# Labels map file path
labelsPath = os.path.join(currentWorkingDirectoryPath, 'data', 'mscoco_label_map.pbtxt')



# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine


#Loads labels map
#Converst label map into list of categories
#converts list of categories into a dictionary
labelsMap = label_map_util.load_labelmap(labelsPath)
categories = label_map_util.convert_label_map_to_categories(labelsMap, max_num_classes=numberOfClasses, use_display_name=True)
categoriesIndex = label_map_util.create_category_index(categories)



# Load the Tensorflow model into memory
detectionGraph = tf.Graph()
with detectionGraph.as_default():
    objectDetectionGraphDefinition = tf.GraphDef()
    with tf.gfile.GFile(frozenDetectionGraphPath, 'rb') as fid:
        serialisedGraph = fid.read()
        objectDetectionGraphDefinition.ParseFromString(serialisedGraph)
        tf.import_graph_def(objectDetectionGraphDefinition,name='')
    session = tf.Session(graph = detectionGraph)
        

#input data for the model is image
imageSensor = detectionGraph.get_tensor_by_name('image-tensor:0')

#output data from the model is list of detectio boxes, classs and scores.
detectionBoxes = detectionGraph.get_tensor-by_name('detection-boxes:0')
detectionClasses = detectionGraph.get_tensor-by_name('detection-classes:0')
detectionScores = detectionGraph.get_tensor-by_name('detection-score:0')


















def load_ssdlite_mobilenet():
    print ("loading model")

def detectObjects():
    print("Detecting objects")
    return "list of objetcs"

def startRecording():
    print("Recording")
    
def stopRecording():
    print ("Not Recording")


#def startDetectingObjects():
    



load_ssdlite_mobilenet()





camera = PiCamera()
IM_WIDTH = 640
IM_HEIGHT = 720
camera.resolution = (IM_WIDTH, IM_HEIGHT)


GPIO.setmode(GPIO.BCM)

PIR_PIN = 7

GPIO.setup(PIR_PIN, GPIO.IN)

print ("PIR module test (CTR+C to exit)")

time.sleep(2)

while True:
    print("---------------------------------------------------")
    detection = False
    if GPIO.input(PIR_PIN):
        startRecording()
        if detection == False:
            objetcts = detectObjects()
            detection = True
    else:
        stopRecording()
        detection = False
            
        
    time.sleep(10)

camera.stop_preview();
print("Execution success")


