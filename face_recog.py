import numpy as np
import cv2

# initialize video capture
cap = cv2.VideoCapture(0)

while(True):
    # reading frame by frame, cause of the loop as well
    ret, frame = cap.read()

    # display the video frame
    cv2.imshow('frame', frame)

    # stop the program when 'q' is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    