#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 15:36:41 2018

@author: jiajiaping
"""

# Conclusion
"""
1. Viola-Jones algorithm
2. detect func(gray,frame) => gray detect feature, frame present rectangle
3. detectMultiScale(gray,reduce_rate,zone_number) return (x,y,w,h)
4. reduce_rate,zone_number larger, detect will more rigious
5. show the box - video_capture.release(),cv2.destroyAllWindows
"""


import cv2

face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_classifier=cv2.CascadeClassifier("haarcascade_smile.xml")
eyes_classifier=cv2.CascadeClassifier("haarcascade_eye.xml")

def detect(gray,frame):
    
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        #print("f",x,y,w,h)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_frame=frame[y:y+h,x:x+w]
        roi_gray=gray[y:y+h,x:x+w]
        
        eyes=eyes_classifier.detectMultiScale(roi_gray,1.3,22)
        for (ex,ey,ew,eh) in eyes:
            #print("e",ex,ey,ew,eh)
            
            cv2.rectangle(roi_frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) # show the frame in specific rectangle
        
        smiles=smile_classifier.detectMultiScale(roi_gray,1.7,22)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_frame,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    
    return frame

video_captrure=cv2.VideoCapture(0)
while True:
    _,frame=video_captrure.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=detect(gray,frame)
    cv2.imshow("Video",canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break;

video_captrure.release()
cv2.destroyAllWindows()
        
