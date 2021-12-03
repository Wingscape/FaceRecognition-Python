import os
from typing import final
import cv2
import pickle
import numpy as np
from PIL import Image

# get the current directory of the file
# os.path.dirname is used to get the directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# get directory of Images folder
image_dir = os.path.join(BASE_DIR, "Images")

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt2.xml')
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
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)

            # change image into NUMPY array
            image_array = np.array(final_image, "uint8")
            # print(image_array)

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=3)

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