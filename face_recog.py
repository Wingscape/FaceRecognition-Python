import numpy as np
import cv2

# initialize Video Capture
cap = cv2.VideoCapture(0)

while(True):
    # reading frame by frame, cause the loop as well
    ret, frame = cap.read()

    # display the video frame
    cv2.imshow('frame', frame)
    