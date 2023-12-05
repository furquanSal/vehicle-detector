import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils
import uuid
from LD import Lane_Detection

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5\\runs\\train\\exp15\\weights\\best.pt', force_reload=True)

cap = cv2.VideoCapture('Drive_01.mp4')
while cap.isOpened():
    ret, img = cap.read()
    results = model(img)
    
    lt_ln_fitted, rt_ln_fitted = Lane_Detection.LD(picture = img)

    if lt_ln_fitted.shape == (2,2):
        if (lt_ln_fitted[1][1]-lt_ln_fitted[0][1]) < (-0.65)*(lt_ln_fitted[1][0]-lt_ln_fitted[0][0])-0.01:
            cv2.line(img, tuple(lt_ln_fitted[0]), tuple(lt_ln_fitted[1]), color = (0,0,255),
                     thickness=5)                                                           

    if rt_ln_fitted.shape == (2,2):
        if (rt_ln_fitted[1][1]-rt_ln_fitted[0][1]) > 0.65*(rt_ln_fitted[1][0]-rt_ln_fitted[0][0])+0.01:
            cv2.line(img, tuple(rt_ln_fitted[0]), tuple(rt_ln_fitted[1]), color = (0,0,255), thickness=5)    
            
    cv2.imshow('frame', imutils.resize(np.squeeze(results.render()), width=800))
    
    if cv2.waitKey(10)==27:
        break