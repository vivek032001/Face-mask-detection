import numpy as np
import cv2
import tensorflow as tf
import keras
json_file = open("modelMask.json", "r")
model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(model_json)
model.load_weights("modelMask.h5")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
source = cv2.VideoCapture(0)
labels = {0:'without_mask_not_safe', 1:'with_mask_safe'}
colour = {0: (0, 0, 255), 1: (0, 255, 0)}
while(True):

    _,img=source.read()
    img = cv2.flip(img,1,1)
    face_re = cv2.resize(img,(img.shape[1], img.shape[0]))
    faces=face_cascade.detectMultiScale(face_re)  
    for x,y,w,h in faces:
        face_img=img[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(224, 224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1, 224, 224, 3))
        result=model.predict(reshaped)
        if(result[0][0] > result[0][1]):
            color, text = colour[1], "Mask"
        else:
            color, text = colour[0], "No_mask"
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        #cv2.rectangle(img,(x,y-40),(x+w,y),color,-1)
        cv2.putText(img, text, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('DETECTOR',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()
