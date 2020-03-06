#import required libraries and packages
import os
import cv2
import tensorflow as tf
import numpy as np

import RPi.GPIO as GPIO
import time
from datetime import datetime
from time import sleep

from picamera import PiCamera
from picamera.array import PiRGBArray
import argparse
import sys

import pyrebase

#To start from exisitng folder 
sys.path.append('..')

from utils import label_map_util
from utils import visualization_utils as vis_util


FIREBASE_CONFIG = {
    "apiKey": "AIzaSyCviGU6TFW6cwMoZN15L_pFmZEwPzLwfNk",
    "authDomain": "raspberry-pi-security-camera.firebaseapp.com",
    "databaseURL": "https://raspberry-pi-security-camera.firebaseio.com",
    "projectId": "raspberry-pi-security-camera",
    "storageBucket": "raspberry-pi-security-camera.appspot.com",
    "messagingSenderId": "20264300276",
    "appId": "1:20264300276:web:76054dc8a2882ba8cef8a1"
  };

firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
database = firebase.database()
fireStore = firebase.storage();

print("Packages import success!")


#Name of the object detection pre trained model used
mdelName = 'ssdlite_mobilenet_v2_coco_2018_05_09'

cameraFrameWidth = 540    #Use smaller resolution for
cameraFrameHeight = 380   #slightly faster framerate


#Current working director path
currentWorkingDirectoryPath = os.getcwd()

#Frozen detection graph .pb file path. it has the model that will be used for object detection
#frozenDetectionGraphPath = os.path.join(currentWorkingDirectoryPath, ObjectDetectionmdelName, 'frozen_inference_graph.pb') 
detectionGraphPath = os.path.join(currentWorkingDirectoryPath,mdelName,'frozen_inference_graph.pb')

#90 classes of object can be detected by the model.
numberOfClasses = 90

# Labels map file path
labelsPath = os.path.join(currentWorkingDirectoryPath,'data','mscoco_label_map.pbtxt')

print("Paths initialisation success!")

# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine


#Loads labels map
#Converst label map into list of categories
#converts list of categories into a dictionary
labelMap= label_map_util.load_labelmap(labelsPath)
categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=numberOfClasses, use_display_name=True)
categoryIndex = label_map_util.create_category_index(categories)

print("Labels Mapping success!")

# Load the Tensorflow model into memory
detectionGraph = tf.Graph()
with detectionGraph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(detectionGraphPath, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    session = tf.Session(graph=detectionGraph)

print("Loading detection model success!")

#Input data for the model is image
image_tensor = detectionGraph.get_tensor_by_name('image_tensor:0')

#Output data from the model is list of detectio boxes, classs and scores

detectionBoxes = detectionGraph.get_tensor_by_name('detection_boxes:0')
detectionScores = detectionGraph.get_tensor_by_name('detection_scores:0')
detectionClasses = detectionGraph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
numberOfDetections = detectionGraph.get_tensor_by_name('num_detections:0')

print("Input output type intialisation sucess!")

# Number of objects detected
#detectionCount = detectionGraph.get_tensor_by_name('num_detections:0')

#Frame rate calculation
frameRateCalculation = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

print("Frame rate assignment")

#Camera initialisation



#Sensor initialisation
GPIO.setmode(GPIO.BCM)
PIR_PIN = 7
GPIO.setup(PIR_PIN, GPIO.IN)

print ("PIR module test (CTR+C to exit)")

isFirstDetectionInThisSession = True

camera = PiCamera()
camera.resolution = (cameraFrameWidth,cameraFrameHeight)
camera.framerate = 10

def detect():
    global isFirstDetectionInThisSession
    global categoryIndex
    global camera
    listOfDectedobjects = []
    
    print("Detection method called")
   
    rawCapture = PiRGBArray(camera, size=(cameraFrameWidth,cameraFrameHeight))
    rawCapture.truncate(0)
    
    print("Checking the if condition inside detectio method!")
    
    timeInMilliSeconds = 0

    if isFirstDetectionInThisSession == True:
        timeInMilliSeconds = int(round(time.time() * 1000)) + 25000 #20 seconds from now. longer time as first intialistion takes  time
    else:
        timeInMilliSeconds = int(round(time.time() * 1000)) + 10000 #10 secinds from now. shorter time as subsequent calls are faster
        
    print("Start Detecting")
    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = session.run(
            [detectionBoxes, detectionScores, detectionClasses, numberOfDetections],
            feed_dict={image_tensor: frame_expanded})
        
        #global categoryIndex
        #print(object_id_to_class_mapper[classes[0]])
        listOfDectedobjects.append(categoryIndex[1]["name"])
        #print(boxes)
        #print("--------------------------------")
        # Draw the results of the detection (aka 'visulaize the results')
        '''
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            categoryIndex,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)
        
      
        
        

        cv2.putText(frame,"FPS: {0:.2f}".format(1),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frameRateCalculation = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
        '''
        if (int(round(time.time()*1000))) >= timeInMilliSeconds :
            isFirstDetectionInThisSession = False
            print ("Exiting detection")
            break;
        rawCapture.truncate(0)
        
    #camera.close()

    cv2.destroyAllWindows()
    return listOfDectedobjects
    




def startRecording():
    camera.start_recording('/home/pi/Desktop/prj2019/videos/video.h264')
    
def stopRecording():
    camera.stop_recording()


GPIO.setmode(GPIO.BCM)

PIR_PIN = 7

GPIO.setup(PIR_PIN, GPIO.IN)

print ("PIR module test (CTR+C to exit)")

#time.sleep(2)

detection = False
recording = False

listOfDectedobjects = [];
currentDateTimeString = ""
while True:
    if GPIO.input(PIR_PIN):
        print("movement detected!")
        if recording == False:
            print("start recording")
            currentDateTime = datetime.now()
            currentDateTimeString = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            camera.start_recording('/home/pi/Desktop/prj2019/videos/'+currentDateTimeString+".h264")
            
            recording = True
            
        if detection == False:
            listOfDectedobjects = detect();
            print(listOfDectedobjects)
            listOfDectedobjects = list(set(listOfDectedobjects))
            print(listOfDectedobjects)
            detection = True
          #  time.sleep(10)
        
        time.sleep(20)
            
    else:
        print("Stop recording")
        if recording == True:
            camera.stop_recording()
            recording = False
        
        if detection == True:
            user = firebase.auth().sign_in_with_email_and_password("r@g.com","123456")
            database.child(user['localId']).push({"incedentDateAndtime":currentDateTimeString, "detectedObjects":listOfDectedobjects})
            detection = False
        
        
    #time.sleep(5)

#camera.stop_preview();
print("Execution success")



