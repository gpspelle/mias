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

    if len(contours) != 0:
        img[:] = 0
        # the contours are drawn here
        #img = cv2.drawContours(img, contours, -1, 255, -1)

        #find the biggest area of the contour
        c = max(contours, key = cv2.contourArea)
        hull = cv2.convexHull(c, False)
        #for c in contours:
        #    hull = cv2.convexHull(c, False)
            #print(x, y, w, h)
            # draw the 'human' contour (in green)
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #img = cv2.drawContours(img, hull, -1, 255, thickness = -1)

        #    for point1, point2 in zip(hull, hull[1:]): 
        #        p1 = tuple(point1[0])
        #        p2 = tuple(point2[0])
        #        cv2.line(img, p1, p2, 255, 2) 

        for point1, point2 in zip(hull, hull[1:]): 
            p1 = tuple(point1[0])
            p2 = tuple(point2[0])
            cv2.line(img, p1, p2, 255, 2) 
        
        p1 = tuple(hull[0][0])
        p2 = tuple(hull[-1][0])

        cv2.line(img, p1, p2, 255, 2)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            print("Found!", image)
            img = cv2.drawContours(img, contours, -1, 255, -1)


    output_path = output + image + ".pgm"
    cv2.imwrite(output_path, img)
