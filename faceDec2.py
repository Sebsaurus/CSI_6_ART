
from turtle import position
import cv2 as cv
from cv2 import INTER_AREA
from cv2 import INTER_LINEAR
from cv2 import INTER_CUBIC
import cv2
from matplotlib import image
from matplotlib.pyplot import sca
import numpy as np
import dlib
from matplotlib import pyplot as plt
import face_utils






videoType = input("Please type 1 for livefeed or 2 for video file: ")
classifier = input("Type one digit for the following; 1 for Haarcascade; 2 for Histogram of Oriented Gradient: ")
scale = input("Please enter a scale factor 1-100: ")
interpolation = input("Type one digit for the following; 1 = INTER_AREA; 2 = LINEAR; 3 = CUBIC: ")
videoType = int(videoType)
classifier = int(classifier)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def __init__(cap):
    cap.detector = dlib.get_frontal_face_detector()
    cap.predictor = dlib.shape_predictor("shape_predicotr_68_face_landmarks.dat")
    cap.fa = face_utils.FaceAligner(cap.predictor, desiredFaceWidth=256)


    


def scale_frame(img, scale):
    # read the actual height and weight, scale it and store
    height = ((int(img.shape[0])*int(scale))/100)
    width = ((int(img.shape[1])*int(scale))/100)
    

    #return the scaled frame
    
    if interpolation == '1':
        return cv.resize(img, (int(width), int(height)), (255,255),  fx = 250, fy=250, interpolation=cv.INTER_AREA)
    elif interpolation == '2':
        return cv.resize(img, (int(width), int(height)), (255,255),  fx = 250, fy=250, interpolation=cv.INTER_LINEAR)
    elif interpolation == '3':
        return cv.resize(img, (int(width), int(height)), (255,255),  fx = 250, fy=250, interpolation=cv.INTER_CUBIC)

def face_detection(frame):
    hog_face_detector = dlib.get_frontal_face_detector

    faces_detected = hog_face_detector(frame)

    faces_count = len(faces_detected)

   
    cv.putText(img=frame,
               text=f'Number of Faces = {faces_count}',
               color=(255,0,255),
               org=(50,50),fontFace=cv.FONT_HERSHEY_PLAIN,
               thickness=1, 
               fontScale=1
               )        

if ((classifier == 1) & (videoType == 1)):
    cap = cv2.VideoCapture(0)
    
    while True: 
        ret, img = cap.read()
        scaled_frame = scale_frame(img, scale)
        gray = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(scaled_frame, (x,y), (x+w, y+h), (255,0,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = scaled_frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes: 
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        cv2.imshow('img', scaled_frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
elif ((classifier == 1) & (videoType == 2)):
    video_location = input("Please type the location of your video file: ")
    cap = cv2.VideoCapture(video_location)
    
    while True: 
        ret, img = cap.read()
        scaled_frame = scale_frame(img, scale)
        gray = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(scaled_frame, (x,y), (x+w, y+h), (255,0,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = scaled_frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes: 
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
        cv2.imshow('img', scaled_frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

elif ((classifier == 2) & (videoType == 1)):
   
 cap = cv2.VideoCapture(0)
 cap.open(0)
 face_detector = dlib.get_frontal_face_detector()
 while (cap.isOpened()):
     isTrue, frame = cap.read()
     scaled_frame = scale_frame(cap, scale)
     scaled_frame = cv2.cvrColor(scaled_frame, cv2.COLOR_BGR2RGB)
 
     frame = scaled_frame.copy()
     frame_marked = face_detection(frame, face_detector)
     cv.imshow(winname='Image', mat=frame)
     cv.imshow(winname='Face Detection', mat=frame_marked)
     if cv.waitKey(20) & 0xFF == ord('f'):  
         break
 cap.release()  
 cv.destroyAllWindows()



elif ((classifier == 2) & (videoType == 2)):
 video_location = input("Please type the location of your file: ")   
 cap = cv2.VideoCapture(video_location)
 cap.open(video_location)
 face_detector = dlib.get_frontal_face_detector()
 while (cap.isOpened()):
     isTrue, frame = cap.read()
     scaled_frame = scale_frame(cap, scale)
     scaled_frame = cv2.cvrColor(scaled_frame, cv2.COLOR_BGR2RGB)
 
     frame = scaled_frame.copy()
     frame_marked = face_detection(frame, face_detector)
     cv.imshow(winname='Image', mat=frame)
     cv.imshow(winname='Face Detection', mat=frame_marked)
     if cv.waitKey(20) & 0xFF == ord('f'):  
         break
 cap.release()  
 cv.destroyAllWindows()

   
   
    

    

    
    
    





        








    
    
    
    
    
    
    
    
    







