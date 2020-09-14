import numpy as np
import pandas as pd
import cv2

filename = "truth.csv"
truth = pd.read_csv(filename)

input = "clusterized_image/"
output = "no_background_mask/"

for index, row in truth.iterrows():
    image = row['image']
    filepath = input + image + ".pgm"
    img = cv2.imread(filepath, 0)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(hierarchy)

    if len(contours) != 0:
        # the contours are drawn here
        #img = cv2.drawContours(img, contours, -1, 255, -1)

        #find the biggest area of the contour
        c = max(contours, key = cv2.contourArea)

        #x, y, w, h = cv2.boundingRect(c)

        img[:] = 0
        #print(x, y, w, h)
        # draw the 'human' contour (in green)
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img = cv2.drawContours(img, c, -1, 255, thickness = -1)


    output_path = output + image + ".pgm"
    cv2.imwrite(output_path, img)


ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
