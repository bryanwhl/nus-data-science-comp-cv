import cv2 as cv
import imutils
import os
import pandas as pd
import numpy as np
from tensorflow import keras
import csv

# Takes the image directory path and output a list of image matrices
def read_imgs(path):
    return {int(img.split(".")[0]): cv.imread(os.path.join(path, img)) for img in os.listdir(path) if img.split(".")[0] != ""}

def test_results(csv_path, pred):
    df = pd.read_csv(csv_path)
    error = 0
    correct_image_count = 0
    for i in range(len(df)):
        k = df.iloc[i]["img_id"]
        error += abs(pred[k] - df.iloc[i]["object_count_gt"])
        print(str(i) + " " + str(pred[k]) + " " + str(df.iloc[i]["object_count_gt"]) + " " + str(
            (pred[k] - df.iloc[i]["object_count_gt"])))
        if (pred[k] == df.iloc[i]["object_count_gt"]):
            correct_image_count += 1
    print(error)
    print(correct_image_count)

def getRefContour(file_path):
    src = cv.imread(file_path)
    img = cv.resize(src, (396, 549))
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    ret, th = cv.threshold(b, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(th, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    ref_cnt = contours[0]
    return ref_cnt

def contourCounting(img, ref_cnt):
    img = cv.resize(img.copy(), (396, 549))
    imgShape = img.shape
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    # ret, th = cv.threshold(b, 142, 255, cv.THRESH_BINARY);
    ret, th = cv.threshold(b, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);

    contours, hierarchy = cv.findContours(th, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    mask = np.zeros_like(img)

    temp_lst = []
    for cnt in contours:
        temp_lst.append(cv.contourArea(cnt))

    temp_lst.sort()
    curr_ref_cnt = temp_lst[len(temp_lst) // 2]
    contourArea = curr_ref_cnt
    # Make sure the median is reasonable, otherwise take the largest contour that is reasonable, otherwise default
    median_area = contourArea if 150 < contourArea < 270 else next((x for x in temp_lst if 150 < x < 270), 180)
    area_threshold_high = median_area * 1.485 if 150 < contourArea < 270 else 220
    area_threshold_low = median_area * 0.46 if 150 < contourArea < 270 else 80
    # print("Median: ", median_area)
    # print("Low: ", area_threshold_low)
    # print("High: ", area_threshold_high)

    contourImg = np.copy(img)
    contourCounts = []
    for cnt in contours:
        count = 0
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
        area = cv.contourArea(approx)
        #         if (cv.matchShapes(curr_ref_cnt, cnt, 1, 0.0) < 0.076):
        #             count += 1

        # Ignore edge contours
        x, y, w, h = cv.boundingRect(cnt)
        if (0 == x or 0 == y or x + w == imgShape[1] or y + h == imgShape[0]) and cv.matchShapes(ref_cnt, cnt, 1, 0.0) > 0.075:
            cv.drawContours(contourImg, [cnt], -1, (0, 0, 0), 2)
            continue

        if (cv.matchShapes(ref_cnt, cnt, 1, 0.0) < 0.075):
            cv.drawContours(contourImg, [cnt], -1, (0, 0, 255), 2)
            count += 1
        elif area_threshold_high < area:
            cv.drawContours(contourImg, [approx], -1, (255, 255, 255), 2)
            count += round(area / median_area)
        elif area_threshold_low < area < area_threshold_high:
            cv.drawContours(contourImg, [approx], -1, (0, 255, 0), 2)
            count += 1
        else:
            cv.drawContours(contourImg, [approx], -1, (255, 0, 0), 2)
        contourCounts.append((cnt, count))
    # cv.imshow("Contours", contourImg)
    return contourCounts

def circleTest(img):
    # print(img.shape)
    img = imutils.resize(img, width=500)
    # cv.imshow("original", img)

    shape = img.shape
    src = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cimg = np.copy(img)

    src = cv.medianBlur(src, 5)
    # ret, th = cv.threshold(src, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    ret, th = cv.threshold(src, 70, 255, cv.THRESH_TRUNC)
    dilationKernal = np.ones((3, 3), np.uint8)
    dilated = cv.dilate(th, dilationKernal)
    canny = cv.Canny(dilated, 80, 125)
    # cv.imshow("Canny", canny)
    # cv.waitKey()

    circles = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1.5, 50,
                              param1=50, param2=30, minRadius=20, maxRadius=70)

    if circles is None:
        return []

    circles = np.uint16(np.around(circles))[0, :]

    validCircles = []
    for i in circles:
        if (i[0] > shape[1] / 2 and i[1] > shape[0] / 2) or (i[0] < shape[1] / 2 and i[1] < shape[0] / 2):
            validCircles.append(i)

    if len(validCircles) == 0:
        return []

    return validCircles

def laplaceTest(img):
    # img = cv.imread(file_path)
    # cv.imshow("Original", img)
    img = imutils.resize(img, 500)

    #Convert source image to LAB space
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    # cv.imshow("B", b)

    #Threshold the B channel to get binary image
    ret, th = cv.threshold(b, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # cv.imshow("Thresh", th)

    #Dilate slightly to pad out the mask
    dilateKernel = np.ones((3, 3), np.uint8)
    dilated = cv.dilate(th, dilateKernel, iterations=1)
    # cv.imshow("Dilated", dilated)

    #Smooth original image to reduce noise, in preparation for laplace
    smooth = cv.bilateralFilter(img, 9, 100, 100)
    # cv.imshow("Smoothed", smooth)

    #Equalize (not being used)
    greyImg = cv.cvtColor(smooth, cv.COLOR_BGR2GRAY)
    # equalized = cv.equalizeHist(greyImg)
    # cv.imshow("Equalized", equalized)
    equalized = greyImg

    #Mask Equalized image for laplace
    masked = cv.bitwise_and(equalized, equalized, mask=dilated)
    # cv.imshow("Masked", masked)

    #Generate laplace using masked image
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    imgLaplacian = cv.filter2D(masked, cv.CV_32F, kernel)

    #Use laplace to sharpen edges on b channel image
    sharp = np.float32(b)
    imgResult = sharp - imgLaplacian

    #Data conversion back to showable format
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)
    # cv.imshow('Laplace Filtered Image', imgLaplacian)
    # cv.imshow('New Sharped Image', imgResult)

    # Mask sharpened image before thresholding
    maskedLaplace = cv.bitwise_and(imgResult, imgResult, mask=dilated)
    # cv.imshow("Masked Laplace", maskedLaplace)

    #Threshold the sharpened image
    ret, thLap = cv.threshold(maskedLaplace, 40, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # cv.imshow("Threshed Laplace", thLap)

    #Use distance transform to split overlapping chips
    dist = cv.distanceTransform(thLap, cv.DIST_L2, 3)
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    # cv.imshow('Distance Transform Image', dist)

    # noise removal
    openKernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thLap, cv.MORPH_OPEN, openKernel, iterations=2)

    # noise removal
    closeKernel = np.ones((3, 3), np.uint8)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, closeKernel, iterations=2)

    # sure background area
    dilateKernel = np.ones((3, 3), np.uint8)
    sure_bg = cv.dilate(closing, dilateKernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(closing, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    # cv.imshow("Fg", sure_fg)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    numLabels, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    # cv.imshow("Result", img)
    # cv.waitKey()
    return numLabels - 1 #ignore background label

def handwritingExtract(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    inverted = s

    ret, th = cv.threshold(inverted, 150, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    dilateKernel = np.ones((1, 30), np.uint8)
    dilated = cv.dilate(th, dilateKernel)
    contours, hierarchy = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    imgShape = img.shape
    validContours = []
    for contour in contours:
        area = cv.contourArea(contour)
        x, y, w, h = cv.boundingRect(contour)
        if area > 100 and (y > 0.80 * imgShape[0] or y + h < 0.20 * imgShape[0]):
            validContours.append(contour)

    contourImg = np.copy(img)
    cv.drawContours(contourImg, validContours, -1, (0, 255, 0), 3)
    cv.line(contourImg, (0, int(0.8 * imgShape[0])), (imgShape[1], int(0.8 * imgShape[0])), (255, 0, 0))
    cv.line(contourImg, (0, int(0.2 * imgShape[0])), (imgShape[1], int(0.2 * imgShape[0])), (255, 0, 0))

    candidates = []
    for contour in validContours:
        x, y, w, h = cv.boundingRect(contour)
        candidate = img[y:y + h, x:x + w]
        candidates.append(candidate)

    return candidates

def handwritingTest(img, handwritingModel):
    # pass in path of image to be predicted
    candidates = handwritingExtract(img)

    handwritingResults = []

    for candidate in candidates:
        # cv.imshow("candidate", candidate)
        # cv.waitKey()

        digit_img = cv.resize(cv.cvtColor(candidate, cv.COLOR_BGR2GRAY), (84, 28)).reshape((28, 84, 1))

        # predict and output predicted value
        result_vector = np.where(handwritingModel.predict(np.array([digit_img]))[0] == 1)[0]
        if len(result_vector) > 0:
            handwritingResults.append(result_vector[0])

    handwritingResults.append(-1)
    return handwritingResults

def runModel(dir_path, ref_path):
    # src = cv.imread(file_path)
    # src = imutils.resize(src, 860)
    # roi, medianArea = findROI(src)
    ref_cnt = getRefContour(ref_path)
    img_dict = read_imgs(dir_path)

    # load trained handwritingModel from file
    handwritingModel = keras.models.load_model("model/final_model.h5")

    pred = {}
    for k in img_dict.keys():
        totalCount = 0
        img = img_dict[k]
        imgShape = img.shape
        contourCounts = contourCounting(img, ref_cnt)
        # contourCounts = findROI(img)

        anomalyImg = np.copy(img)
        anomalies = []
        for contour, count in contourCounts:
            # Add the humber of acumens in the countour
            totalCount += int(count)

            x, y, w, h = cv.boundingRect(contour)
            # (396, 549)
            x = round(x * (imgShape[1] / 396))
            y = round(y * (imgShape[0] / 549))
            w = round(w * (imgShape[1] / 396))
            h = round(h * (imgShape[0] / 549))

            acumenImg = img[y:y+h, x:x+w]

            # Black dot
            if count == 1:
                circles = circleTest(acumenImg)

                if len(circles) > 0:
                    cv.rectangle(anomalyImg, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    anomalies.append(('black_spot', [x, y, w, h], 1.0))

            if count > 1:
                numContours = laplaceTest(acumenImg)
                if count == 2 and numContours > count:
                    # count += 1
                    # Draw bbox for overlap
                    cv.rectangle(anomalyImg, (x, y), (x+w, y+h), (255, 0, 0), 3)
                    anomalies.append(('double_stack', [x, y, w, h], 1.0))

        # cv.imshow("Anomaly", anomalyImg)

        handwritingResults = handwritingTest(img, handwritingModel)

        print("Testing image: ", k)
        print("Acumen count: ", totalCount)
        print("Handwriting count: ", handwritingResults[0])
        print("Anomalities: ", anomalies)

        pred[int(k)] = [totalCount, handwritingResults[0], anomalies if len(anomalies) > 0 else "[]"]
        # cv.waitKey()
    return pred

def convert_to_csv(predictions):
    df = pd.DataFrame(columns=['acumen_count_pred', 'handwritten_count_pred', 'anomalies_bbox_pred'])
    for id, pred in sorted(predictions.items()):
        df.loc[id] = pred

    print(df)
    df.to_csv("submissions.csv", index_label='img_id', quoting=csv.QUOTE_NONNUMERIC)

if __name__ == '__main__':
    predictions = runModel('./data/',
                           './model/chip_benchmark.jpg')

    convert_to_csv(predictions)
