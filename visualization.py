import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_result(image, initial_centroids, clustered_centroids, spiral_params):
    final_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for (cx, cy) in initial_centroids:
        cv2.circle(final_image, (cx, cy), 2, (150, 150, 150), -1)
    for (cx, cy) in clustered_centroids:
        cv2.circle(final_image, (int(cx), int(cy)), 8, (0, 0, 255), -1)
    cx, cy, a, b = spiral_params['cx'], spiral_params['cy'], spiral_params['a'], spiral_params['b']
    theta_fit = np.linspace(-np.pi * 5, np.pi * 5, 500)
    r_fit = a * np.exp(b * theta_fit)
    x_fit = cx + r_fit * np.cos(theta_fit)
    y_fit = cy + r_fit * np.sin(theta_fit)
    fit_points = np.vstack((x_fit, y_fit)).T.astype(np.int32)
    cv2.polylines(final_image, [fit_points], isClosed=False, color=(255, 0, 0), thickness=3)
    plt.figure(figsize=(10, 10))
    plt.imshow(final_image)
    plt.title('Fibonacci Spiral Detection')
    plt.show()
