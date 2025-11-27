# python_server/clustering.py

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_object_centroids(image):
    h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_area = 50
    centroids = []
    object_contours = []
    for c in contours:
        if cv2.contourArea(c) > min_contour_area:
            M = cv2.moments(c)
            if M["m00"] != 0:
                object_contours.append(c)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append([cx, cy])
                
    return np.array(centroids), object_contours

def find_optimal_k(points, max_k=10):
    if len(points) <= max_k:
        max_k = len(points) - 1
    if max_k < 2: return max_k

    inertias = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(points)
        inertias.append(kmeans.inertia_)
        
    p1 = np.array([k_range[0], inertias[0]])
    p2 = np.array([k_range[-1], inertias[-1]])
    
    distances = []
    for i in range(len(inertias)):
        p3 = np.array([k_range[i], inertias[i]])
        distance = np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
        distances.append(distance)
        
    optimal_k = k_range[np.argmax(distances)]
    
    # plt.figure(figsize=(7, 5))
    # plt.plot(k_range, inertias, 'bo-')
    # plt.title('Elbow Method')
    # plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles='--', colors='r')
    # plt.show()
    
    return optimal_k