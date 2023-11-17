#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import mediapipe as mp
import time 


# In[9]:


mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils


# In[10]:


cap = cv2.VideoCapture('squat workout.mp4')


# In[11]:


def rescale(img,hscale = 0.1,vscale = 0.1) :
    width = int(img.shape[1]*hscale)
    height = int(img.shape[0]*vscale)
    dimension = (width,height)
    return cv2.resize(img,dimension,interpolation = cv2.INTER_AREA)  


# In[12]:


left_knee = []
left_butt = []
right_knee = []
right_butt = []


# In[13]:


left_knee_y = []
left_butt_y = []
right_knee_y = []
right_butt_y = []


# In[14]:


while (cap.isOpened()) :
    success,frame = cap.read()
    if success == True :
        img = rescale(frame,0.5,0.3)
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
            
            print(points[24])
            print(points[26])
            print(points[23])
            print(points[25])
            print('----------')
            
            left_butt.append(points[24])
            left_knee.append(points[26])
            right_knee.append(points[23])
            right_butt.append(points[25])
            
            left_butt_y.append(points[24][1])
            left_knee_y.append(points[26][1])
            right_knee_y.append(points[23][1])
            right_butt_y.append(points[25][1])
            
            
            
        cv2.imshow('squat',img)  
    if success == False :
        break
        
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
        
cap.release()
cv2.destroyAllWindows()


# In[15]:


left_butt_y


# In[16]:


left_knee_y


# In[17]:


right_butt_y


# In[18]:


right_knee_y


# In[19]:


import matplotlib.pyplot as plt
import numpy as np


# In[20]:


x = np.arange(1,len(left_butt_y)+1,1)
x = x.tolist()


# In[21]:


plt.figure(figsize = (9,9))

plt.plot(x,right_butt_y,label = 'SHOULDER',color = 'purple')
plt.plot(x,right_knee_y,label = 'ELBOW',color = 'orange')

plt.title("FLUCTUATION OF Y COORDINATE OF BUTT AND KNEE DURING PUSH-UPS")
plt.legend()

plt.show()


# In[ ]:




