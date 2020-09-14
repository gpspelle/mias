import numpy as np
from sklearn.cluster import KMeans
import cv2
import datetime
from matplotlib import pyplot as plt
import time

image1=cv2.imread("mdb001.pgm")
image2=cv2.imread("mdb002.pgm")
image3=cv2.imread("mdb003.pgm")

image=[image1,image2,image3]
reshaped=[0,0,0]
for i in range(0,3):
    reshaped[i] = image[i].reshape(image[i].shape[0] * image[i].shape[1], image[i].shape[2])

num_cluster = 2
clustering=[0,0,0]
for i in range(0,3):
    kmeans = KMeans(n_clusters=num_cluster, n_init=40, max_iter=500).fit(reshaped[i])
    clustering[i] = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
    (image[i].shape[0], image[i].shape[1]))

sortedLabels=[[],[],[]]
for i in range(0,3):
    sortedLabels[i] = sorted([n for n in range(num_cluster)],
        key=lambda x: -np.sum(clustering[i] == x))


kmeansImage=[0,0,0]
concatImage=[[],[],[]]
for j in range(0,3):
    kmeansImage[j] = np.zeros(image[j].shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels[j]):
        kmeansImage[j][ clustering[j] == label ] = int((255) / (num_cluster - 1)) * i
    
    concatImage[j] = image[j] 
    #concatImage[j] = np.concatenate((image[j],193 * np.ones((image[j].shape[0], int(0.0625 * image[j].shape[1]), 3), dtype=np.uint8),cv2.cvtColor(kmeansImage[j], cv2.COLOR_GRAY2BGR)), axis=1)

for i in range(0,3):
    cv2.imwrite("image_" + str(i) + ".png", concatImage[i])
