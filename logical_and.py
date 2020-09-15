import numpy as np
import pandas as pd
import cv2

filename = "truth.csv"
truth = pd.read_csv(filename)

input_image = ""
input_mask = "no_background_mask/"
output = "clean_image/"

for index, row in truth.iterrows():
    image = row['image']
    filepath = input_image + image + ".pgm"
    img = cv2.imread(filepath, 0)
    
    filepath = input_mask + image + ".pgm"
    mask = cv2.imread(filepath, 0)
    mask_inv = cv2.bitwise_not(mask)

    # Take only region of logo from logo image.
    cv2.bitwise_and(img, 0, img, mask = mask_inv)
    output_path = output + image + ".pgm"
    cv2.imwrite(output_path, img)


