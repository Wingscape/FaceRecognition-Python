import os

# get the current directory of the file
# os.path.dirname is used to get the directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# get directory of Images folder
image_dir = os.path.join(BASE_DIR, "Images")

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

            print(path)

