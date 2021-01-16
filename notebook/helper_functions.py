import cv2 as cv
import os


# Takes the image directory path and output a list of image matrices
def read_imgs(path):
    return [cv.imread(os.path.join(path, img)) for img in os.listdir(path)]
