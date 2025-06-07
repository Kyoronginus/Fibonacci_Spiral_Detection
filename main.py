import cv2
from clustering import extract_object_centroids, find_optimal_k
from spiral_fit import optimize_spiral,optimize_spiral_with_golden_ratio
from visualization import draw_result
from sklearn.cluster import KMeans
import time

def main(image_path):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"エラー: 画像ファイル '{image_path}' が見つかりません。")
        return
    initial_centroids, object_contours = extract_object_centroids(original_image)
    if len(initial_centroids) < 3:
        print("分析対象となるオブジェクトが3つ未満です。処理を中断します。")
        return
    print(f"初期重心が {len(initial_centroids)}個 検出されました。クラスタリングで主要な点にまとめます...")
    optimal_k = find_optimal_k(initial_centroids)
    print(f"エルボー法により、最適なクラスタ数 k = {optimal_k} と判断しました。")
    kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=0).fit(initial_centroids)
    clustered_centroids = kmeans.cluster_centers_
    print(f"{len(clustered_centroids)}個のクラスタ中心に対してフィッティングを開始します...")
    start_time = time.time()
    h, w, _ = original_image.shape
    initial_search_ranges = {'cx': [0, w], 'cy': [0, h], 'a': [10.0, 400.0], 'b': [0.1, 0.5]}
    # best_params, best_score = optimize_spiral(clustered_centroids, (h, w), initial_search_ranges)
    best_params = optimize_spiral_with_golden_ratio(clustered_centroids, (h, w))
    print("\nフィッティング完了！")
    print(f"処理時間: {time.time() - start_time:.2f} 秒")
    draw_result(original_image, initial_centroids, clustered_centroids, best_params)

if __name__ == '__main__':
    # ここで画像パスを指定
    image_path = 'images\Illustration137.cliptry.clipbu.clip.bu.png.bestcomposition.png'
    main(image_path)
