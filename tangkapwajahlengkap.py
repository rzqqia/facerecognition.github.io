import cv2,os, time
import numpy as np
from PIL import Image
import pickle
import sqlite3
camera=0
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
video=cv2.VideoCapture(camera,cv2.CAP_DSHOW)
a=0
recognizer=cv2.face.LBPHFaceRecognizer_create();
recognizer.read("C://Users/user/Documents/TUGAS SEMESTER 7/AI/FACERECOGNITION/training/training.xml")
id=0
fontface=cv2.FONT_HERSHEY_SIMPLEX
fontscale=1
fontcolor=(0,0,255)
path='DataSet'
def getProfile(id):
 conn=sqlite3.connect("datawajah.db")
 cmd="SELECT * FROM orang WHERE id="+str(id)
 cursor=conn.execute(cmd)
 profile=None
 for row in cursor:
  profile=row
 conn.close()
 return profile
while(True):
 check, frame=video.read();
 gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 faces=faceDetect.detectMultiScale(gray,1.3,5);
 for(x,y,w,h) in faces:
  a=a+1
  cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
  id,conf=recognizer.predict(gray[y:y+h,x:x+w])
  profile=getProfile(id)
  if(profile!=None):
   cv2.putText(frame,str(profile[1]),(x,y+h+30),fontface,fontscale,fontcolor)
   cv2.putText(frame,str(profile[2]),(x,y+h+60),fontface,fontscale,fontcolor)
   cv2.putText(frame,str(profile[3]),(x,y+h+90),fontface,fontscale,fontcolor)
   #cv2.putText(frame,str(profile[4]),(x,y+h+100),fontface,fontscale,fontcolor);
 cv2.imshow("wajah",frame);
 if(cv2.waitKey(1)==ord('q')):
  break
 print(a)
cam.release()
cv2.destroyAllWindows()