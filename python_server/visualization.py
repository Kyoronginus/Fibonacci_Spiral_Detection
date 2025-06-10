import cv2
import numpy as np

def draw_result(image, initial_centroids, clustered_centroids, spiral_params):
    final_image = image.copy()
    
    if initial_centroids is not None and len(initial_centroids) > 0:
        for (cx, cy) in initial_centroids:
            cv2.circle(final_image, (int(cx), int(cy)), 2, (150, 150, 150), -1)

    if clustered_centroids is not None and len(clustered_centroids) > 0:
        for (cx, cy) in clustered_centroids:
            cv2.circle(final_image, (int(cx), int(cy)), 8, (0, 0, 255), -1)

    # ★★★ spiral_paramsがNoneでない場合のみ、螺旋を描画 ★★★
    if spiral_params:
        cx, cy, a, b = spiral_params['cx'], spiral_params['cy'], spiral_params['a'], spiral_params['b']
        theta_fit = np.linspace(-np.pi * 5, np.pi * 5, 500)
        r_fit = a * np.exp(b * theta_fit)
        x_fit = cx + r_fit * np.cos(theta_fit)
        y_fit = cy + r_fit * np.sin(theta_fit)
        fit_points = np.vstack((x_fit, y_fit)).T.astype(np.int32)
        cv2.polylines(final_image, [fit_points], isClosed=False, color=(255, 0, 0), thickness=3)

    return final_image