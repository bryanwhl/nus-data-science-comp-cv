import cv2 as cv
import os


# Takes the image directory path and output a list of image matrices
def read_imgs(path):
    return {int(img.split(".")[0]): cv.imread(os.path.join(path, img)) for img in os.listdir(path)}


# Returns the coordinates of a bounding rectangle around all the input contours
def get_bounding_rect_coor(contours):
    min_x, min_y, max_x, max_y = 396, 549, 0, 0
    for cnt in contours:
        for pt in cnt:
            x, y = pt[0]
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
    return min_x, min_y, max_x, max_y