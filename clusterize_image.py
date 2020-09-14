import numpy as np
from sklearn.cluster import KMeans
import cv2
import datetime
from matplotlib import pyplot as plt
import time
import pandas as pd

filename = "truth.csv"
truth = pd.read_csv(filename)

input = ""
output = "clusterized_image/"

num_clusters = 2
for index, row in truth.iterrows():
    image  = row["image"]
    filepath = input + image + ".pgm"
    img = cv2.imread(filepath)

    reshaped = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    kmeans = KMeans(n_clusters=num_clusters, n_init=40, max_iter=500).fit(reshaped)

    clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8), (img.shape[0], img.shape[1]))

    kmeans_image = np.zeros(img.shape[:2], dtype=np.uint8)

    sorted_labels = sorted([n for n in range(num_clusters)], key=lambda x: -np.sum(clustering == x))

    for i, label in enumerate(sorted_labels):
        kmeans_image[clustering == label] = int((255) / (num_clusters - 1)) * i

    output_path = output + image + ".pgm"
    cv2.imwrite(output_path, kmeans_image)
