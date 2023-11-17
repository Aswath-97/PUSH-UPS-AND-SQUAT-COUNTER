#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp
import time 


# In[2]:


mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils


# In[3]:


cap = cv2.VideoCapture('UMA MAHESHWARAN PUSH UPS.mp4')


# In[4]:


def rescale(img,scale = 0.1) :
    width = int(img.shape[1]*scale)
    height = int(img.shape[0]*scale)
    dimension = (width,height)
    return cv2.resize(img,dimension,interpolation = cv2.INTER_AREA)  


# In[5]:


right_shoulder = []
right_elbow = []

right_shoulder_y = []
right_elbow_y = []

left_shoulder = []
left_elbow = []

left_shoulder_y = []
left_elbow_y = []


# In[6]:


while (cap.isOpened()) :
    success,frame = cap.read()
    if success == True :
        img = rescale(frame,0.8)
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        points = {}
        if results.pose_landmarks :
            mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
            for i_d,land_mark in enumerate(results.pose_landmarks.landmark) :
                h,w,c = img.shape
                # print(i_d,land_mark)
                cx,cy = int(land_mark.x*w),int(land_mark.y*h)
                cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
                points[i_d] = (cx,cy)
                
            print(points[12])
            print(points[14])
            print(points[11])
            print(points[13])
            print("------------")
            
            right_shoulder.append(points[12])
            right_elbow.append(points[14])
            left_shoulder.append(points[11]) 
            left_elbow.append(points[13])
            
            right_shoulder_y.append(points[12][1])
            right_elbow_y.append(points[14][1])
            left_shoulder_y.append(points[11][1])
            left_elbow_y.append(points[13][1])
            
        cv2.imshow('PUSH UP VIDEO',img)  
    if success == False :
        break
        
    if cv2.waitKey(10) & 0xFF == ord('q') :
        break
        
cap.release()
cv2.destroyAllWindows()


# In[44]:


import matplotlib.pyplot as plt
import numpy as np


# In[45]:


x = np.arange(1,len(right_elbow_y)+1,1)
x = x.tolist()


# In[46]:


plt.figure(figsize = (9,9))

plt.plot(x,right_shoulder_y,label = 'SHOULDER',color = 'green')
plt.plot(x,right_elbow_y,label = 'ELBOW',color = 'blue')

plt.title("FLUCTUATION OF Y COORDINATE OF SHOLDER AND ELBOW DURING PUSH-UPS")
plt.legend()

plt.show()


# In[ ]:




