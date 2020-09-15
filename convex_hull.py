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

            
        img_hull = cv2.drawContours(img, hull, -1, 255, 5)
        cv2.imshow(image, img_hull)
        cv2.waitKey(0)

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
