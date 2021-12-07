import os
from typing import final
import cv2
import shutil
import pickle
import numpy as np
from PIL import Image

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

def rescale_frame(frame, percent=100):
    scale_percent = percent
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

# get the current directory of the file
# os.path.dirname is used to get the directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# get directory of Images folder
image_dir = os.path.join(BASE_DIR, "Images")

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt2.xml')

inp = input("Do you want to get pictures of yourself? (y/n): ")

if inp == 'y' or inp == 'Y':
    STD_DIMENSIONS = {
        "480p": (640, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080)
    }

    amount_pic = 0

    inp_name = input("Please insert your name: ")
    path_name = os.path.join(image_dir, inp_name.replace(" ", "-").lower())

    # BUG: if there is symlink in directory or file
    for dir in next(os.walk(image_dir))[1]:
        name = os.path.basename(path_name)

        if dir == name:
            for file in next(os.walk(path_name))[2]:
                file_path = os.path.join(path_name, file)

                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            os.rmdir(path_name)

    os.mkdir(path_name)

    cap = cv2.VideoCapture(0)
    width, height = STD_DIMENSIONS["720p"]
    change_res(cap, width, height)

    while(True):     
        ret, frame = cap.read()
        frame = rescale_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get laggy when scaleFactor is decrease
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=[300, 300])

        for (x, y, w, h) in faces:
            amount_pic += 1

            roi_color = frame[y:y+h, x:x+w]
            img_item = str(amount_pic) + ".png"
            cv2.imwrite(os.path.join(path_name, img_item), roi_color)

            # creating ractangle around the face of camera
            color = (0, 255, 0) # BGR
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # display the video frame
        cv2.imshow('Analyzing face', frame)

        if amount_pic == 70:
            break

        # stop the program when 'q' is pressed
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
elif inp == 'n' or inp == 'N':
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    # separate file so that only picture that get taken
    # root is top-most directory in a hierarchy
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg") or file.endswith("jfif"):
                # join the root directory of this file with the file itself
                path = os.path.join(root, file)

                # make a label from the foldername of the file
                # basename is to get the folder of the path
                label = os.path.basename(root).replace(" ", "-").lower()
                # print(label, path)

                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                id_ = label_ids[label]
                print(label_ids)

                # pillow is python image library
                # open image of the directory and convert it into grayscale
                pil_image = Image.open(path).convert("L")

                # resize the image
                # size = (550, 550)
                # final_image = pil_image.resize(size, Image.ANTIALIAS)

                # change image into NUMPY array
                image_array = np.array(pil_image, "uint8")
                # print(image_array)

                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.05, minNeighbors=4, minSize=[300, 300])

                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]

                    x_train.append(roi)
                    y_labels.append(id_)

    # create lables.pickle file and send label_ids into a lables.pickle file
    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)

    # training the recognizer
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainner.yml")
else:
    print("The input is incorrect! Try again later")