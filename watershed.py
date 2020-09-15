from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np

from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte

import cv2 as cv
import pandas as pd

filename = "truth.csv"
truth = pd.read_csv(filename)

mias_input = ""
input = "clusterized_image/"
output = "watershed/"

for index, row in truth.iterrows():
    filename = row['image']

    filepath = input + filename + ".pgm"

    image = cv.imread(filepath, 0)

    filepath = mias_input + filename + ".pgm" 
    original_image = cv.imread(filepath, 0)

    # denoise image
    denoised = rank.median(image, disk(2))

    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]

    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(2))

    # process the watershed
    labels = watershed(gradient, markers)

    unique_entries = np.unique(labels)

    contours = []

    for u in unique_entries:
        index = np.where(labels == u)
        contours.append(np.array([[j, i] for i, j in zip(index[0], index[1])]).reshape((-1,1,2)).astype(np.int32))

    contours_areas = []
    contours_colors = []
    for c in contours:
        area = len(c)
        mask = np.zeros(image.shape, np.uint8)

        cv.drawContours(mask, c, -1, 255, -1)

        mean = cv.mean(original_image, mask=mask)[0]

        if mean > 80:
            contours_colors.append(mean)
            contours_areas.append(area)
        else:
            contours_colors.append(0)
            contours_areas.append(0)


    biggest = np.argmax(contours_areas)
    mask = np.zeros(image.shape, np.uint8)
    mask = cv.drawContours(mask, contours[biggest], -1, 255, -1)
    mask_inv = cv.bitwise_not(mask)

    img = np.zeros(image.shape, np.uint8)
    # Take only region of logo from logo image.
    cv.bitwise_and(original_image, original_image, img, mask = mask)

    output_path = output + filename + ".pgm"
    cv.imwrite(output_path, img)
    
    output_path = output + filename + "_mask.pgm"
    cv.imwrite(output_path, mask_inv)
