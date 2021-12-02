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

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt2.xml')

# initialize video capture
cap = cv2.VideoCapture(0)

# change the resolution
width, height = std_dimensions["480p"]
change_res(cap, width, height)

while(True):
    # reading frame by frame, cause of the loop as well
    ret, frame = cap.read()

    # change it into gray picture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # finding face in any picture or video
    # scaleFactor is for how much accuracy to find the face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        print(x,y,w,h)

    # display the video frame
    cv2.imshow('frame', frame)

    # stop the program when 'q' is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    