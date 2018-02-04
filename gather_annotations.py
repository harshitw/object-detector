# this code will detect the object in the image using HOG AND SVM Classifier
# TASK : to build and end to end custom object detector
# say we want to detect the clocks in the image
# So the steps involved in building the custom object detector are :
# 1.collecting the training images
# 2.Attonate the object location in the images
# 3.Train the object detector in the object regions
# 4.Save and test the trained detector

# PROJECT STRUCTURE
# detector.py -> gather_attonations -> selectors -> train.py -> test.py

# selector contains the boxselector class that helps to attonate(select) the object regions
# gather allows us to select the image using a selector
# detector contains the class ObjectDetector which is used for training and detecting the objects
# train.py used for training and testing the  object detector
# test.py the actual driver script to detect regions in the image

# So let's fucking start!! and it's super interesting and I'm fucking excited

# gather_annotations.py
import numpy as np
import cv2
import argparse
from imutils.paths import list_images
from selectors import BoxSelector

ap = argparse(ArgumentParser())
ap.add_argument("-d", "--dataset", required = True, help = "add path to the images")
ap.add_argument("-i", "--images", required = True, help = "path to save the images")
# now next line is there to ensure consistent annotations
ap.add_argument("-a", "--annotations", required = True, help = "path to save the annotations")
args = vars(ap.parse_args())
annotations = []
imPaths = []

# loop through each image and collect the annotations
for imagePath in list_images(args["dataset"]):

    #load the image and create a box selector instance
    image = cv2.imread(imagePath)
    bs = BoxSelector(image, "Image")
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    # order the points suitable for the object detector
    (pt1, pt2) = bs.roiPts
    (x, y, xb, yb) = [pt1[0], pt1[1], pt2[0], pt2[1]]
    annotations.append([int(x), int(y), int(xb), int(yb)])
    imPaths.append(imagePath)
    # we need to save the image paths as annotations for an image can be retrived by index
    # i.e no mistake of retrieving incorrect annotaions for the image
    # we loop over each image and create a box selector instance to help us select the regions from the mouse
    # we then collect object location using selection and append the annotations and image paths to their respective vectors

    # save the annotations and image paths to the disk
    # we convert them into numpy arrays and save them to the disk
    annotations = np.array(annotations)
    imPaths = np.array(imPaths, dtype = "unicode")
    np.save(args["annotations"], annotations)
    np.save(args["images"], imPaths)




















    1
