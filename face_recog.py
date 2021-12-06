import cv2
import pickle
import numpy as np

STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080)
}

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

def rescale_frame(frame, percent=100):
    scale_percent = percent
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt2.xml')
# eye_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels = {}

# get value from label_ids and reverse it
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

# initialize video capture
cap = cv2.VideoCapture(0)

# change the resolution
width, height = STD_DIMENSIONS["720p"]
change_res(cap, width, height)

while(True):
    # reading frame by frame, cause of the loop as well
    ret, frame = cap.read()

    # rescale frame
    frame = rescale_frame(frame)

    # change it into gray picture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # finding face by camera
    # scaleFactor will detecting the face in what scale image factor is
    # minNeighbor, higher value results in less detection but with higher quality
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)

    for (x, y, w, h) in faces:
        # print coordinat
        # print(x, y, w, h)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)

        if conf >= 97 and conf <= 100:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            colors = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 0.5, colors, stroke, cv2.LINE_AA)

        # save gray picture with roi(region of interest)
        img_item = "image_identify.png"
        cv2.imwrite(img_item, roi_color)

        # creating ractangle around the face of camera
        color = (0, 255, 0) # BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # eyes = eye_cascade.detectMultiScale(roi_gray)

        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,0,255), 2)

    # display the video frame
    cv2.imshow('frame', frame)

    # stop the program when 'q' is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()