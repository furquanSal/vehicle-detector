import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils
import uuid

frame_count = 0
cap = cv2.VideoCapture('Drive_01.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    
    if frame_count ==25:
        cv2.imwrite('data\\images\img_' + str(uuid.uuid1()) + '.jpeg', frame)
        frame_count = 0
    else:
        frame_count = frame_count + 1