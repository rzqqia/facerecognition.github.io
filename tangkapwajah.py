import cv2, time

camera = 0
video = cv2.VideoCapture(camera,cv2.CAP_DSHOW)
a = 0
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
id = input('masukkan id user ')
while True:
  a = a + 1  
  check, frame = video.read()  
  print(check)
  print(frame)  
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
  faces=faceDetect.detectMultiScale(gray,1.3,5);
  for (x,y,w,h) in faces:
    cv2.imwrite("DataSet/User."+str(id)+"."+str(a)+".jpg",
    gray[y:y+h,x:x+w])
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
  cv2.imshow("wajah",frame)
  key = cv2.waitKey(1)  
  if (a>24):
       break
  print(a)
video.release()
cv2.destroyAllWindows()