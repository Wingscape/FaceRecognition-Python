import numpy as np
import cv2

std_dimensions = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080)
}

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# initialize video capture
cap = cv2.VideoCapture(0)
change_res(cap, std_dimensions["480p"])

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
    