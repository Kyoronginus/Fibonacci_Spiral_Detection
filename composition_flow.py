import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
# scikit-learnライブラリが必要です
# pip install scikit-learn
from sklearn.cluster import KMeans

# --- アルゴリズム用の設定項目 ---
NUM_GENERATIONS = 100  # 探索が複雑なため、世代数を少し増やす
NUM_CANDIDATES = 300   # 候補も増やす
N_ELITES = 15          # 次世代に引き継ぐエリート（優秀な候補）の数
MUTATION_RATE = 0.25   # 全く新しい候補を混ぜる「突然変異」の割合
MIN_A_THRESHOLD = 15.0 # これよりパラメータ'a'が小さい螺旋は無効とする

def calculate_composition_score(candidate_params, points, image_shape):
    """
    生成した螺旋と、指定された点群との「近さ」を計算するスコア関数。
    """
    h, w = image_shape
    cx, cy, a, b = candidate_params['cx'], candidate_params['cy'], candidate_params['a'], candidate_params['b']
    
    # 評価用の螺旋の点を生成
    theta = np.linspace(-np.pi * 4, np.pi * 4, 200)
    r = a * np.exp(b * theta)
    x_fit = cx + r * np.cos(theta)
    y_fit = cy + r * np.sin(theta)
    
    # 画面外の点は除外
    valid_mask = (x_fit >= 0) & (x_fit < w) & (y_fit >= 0) & (y_fit < h)
    spiral_points = np.vstack((x_fit[valid_mask], y_fit[valid_mask])).T
    
    # 螺旋がほとんど画面外なら悪いスコア（無限大）を返す
    if len(spiral_points) < 10: 
        return float('inf')

    # 各点から、最も近い螺旋上の点までの距離を計算
    total_distance = 0
    for p in points:
        # NumPyのブロードキャスト機能を使って、ある1つの点と全ての螺旋点との距離を一括計算
        distances = np.sqrt(np.sum((spiral_points - p)**2, axis=1))
        # 最も近い距離を足し合わせる
        total_distance += np.min(distances)
        
    # 点あたりの平均距離を最終的なスコアとする（小さいほど良い）
    return total_distance / len(points)

def find_optimal_k(points, max_k=10):
    """エルボー法で最適なクラスタ数kを見つける"""
    # 点の数よりkが大きくならないように調整
    if len(points) <= max_k:
        max_k = len(points) -1
    
    # kを2からmax_kまで変化させて、それぞれのクラスタ内距離の総和（inertia）を計算
    inertias = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(points)
        inertias.append(kmeans.inertia_)
    
    # エルボー（肘）を検出
    # グラフの最初の点と最後の点を結ぶ直線から、最も距離が遠い点を探す
    p1 = np.array([k_range[0], inertias[0]])
    p2 = np.array([k_range[-1], inertias[-1]])
    
    distances = []
    for i in range(len(inertias)):
        p3 = np.array([k_range[i], inertias[i]])
        # 点と直線の距離の公式を利用
        distance = np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
        distances.append(distance)
        
    # 最も距離が遠かった点のkを最適値とする
    optimal_k = k_range[np.argmax(distances)]
    
    # エルボーグラフをプロットして可視化
    plt.figure(figsize=(7, 5))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    # 最適と判断したkの位置に赤い破線を引く
    plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles='--', colors='r', label=f'Optimal k = {optimal_k}')
    plt.legend()
    plt.show()
    
    return optimal_k


# --- メインの実行部分 ---
if __name__ == '__main__':
    image_path = 'images\Illustration119.png' # ここに分析したいイラストのパスを指定
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"エラー: 画像ファイル '{image_path}' が見つかりません。")
    else:
        # --- 1. 全オブジェクトの重心を抽出 ---
        h, w, _ = original_image.shape
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 50 
        initial_centroids = []
        object_contours = []
        for c in contours:
            if cv2.contourArea(c) > min_contour_area:
                object_contours.append(c)
                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                initial_centroids.append([cx, cy])

        if len(initial_centroids) < 3:
            print("分析対象となるオブジェクトが3つ未満です。処理を中断します。")
        else:
            initial_centroids = np.array(initial_centroids)
            print(f"初期重心が {len(initial_centroids)}個 検出されました。クラスタリングで主要な点にまとめます...")
            
            # --- 2. クラスタリングで主要な重心点を抽出 ---
            optimal_k = find_optimal_k(initial_centroids)
            print(f"エルボー法により、最適なクラスタ数 k = {optimal_k} と判断しました。")
            kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=0).fit(initial_centroids)
            clustered_centroids = kmeans.cluster_centers_

            # --- 3. クラスタリング後の重心点群に対してフィッティングを実行 ---
            print(f"{len(clustered_centroids)}個のクラスタ中心に対してフィッティングを開始します...")
            start_time = time.time()
            
            initial_search_ranges = {'cx': [0, w], 'cy': [0, h], 'a': [10.0, 400.0], 'b': [0.1, 0.5]}
            candidates = []
            for _ in range(NUM_CANDIDATES):
                candidates.append({
                    'cx': np.random.uniform(*initial_search_ranges['cx']), 
                    'cy': np.random.uniform(*initial_search_ranges['cy']), 
                    'a': np.random.uniform(*initial_search_ranges['a']), 
                    'b': np.random.uniform(*initial_search_ranges['b'])
                })

            best_overall_params = {}
            best_overall_score = float('inf')

            for generation in range(NUM_GENERATIONS):
                scores = []
                for params in candidates:
                    if params['a'] < MIN_A_THRESHOLD:
                        scores.append(float('inf'))
                        continue
                    
                    score = calculate_composition_score(params, clustered_centroids, (h, w))
                    scores.append(score)
                
                sorted_candidates = sorted(zip(scores, candidates), key=lambda x: x[0])
                
                if sorted_candidates[0][0] < best_overall_score:
                    best_overall_score = sorted_candidates[0][0]
                    best_overall_params = sorted_candidates[0][1]
                
                elites = [c for s, c in sorted_candidates[:N_ELITES]]
                
                next_generation = elites[:]
                
                num_mutations = int(NUM_CANDIDATES * MUTATION_RATE)
                for _ in range(num_mutations):
                    next_generation.append({
                        'cx': np.random.uniform(*initial_search_ranges['cx']), 
                        'cy': np.random.uniform(*initial_search_ranges['cy']),
                        'a': np.random.uniform(*initial_search_ranges['a']), 
                        'b': np.random.uniform(*initial_search_ranges['b'])
                    })
                
                num_offspring = NUM_CANDIDATES - len(next_generation)
                for i in range(num_offspring):
                    parent = elites[i % len(elites)]
                    next_generation.append({
                        'cx': np.random.normal(parent['cx'], w * 0.1), 
                        'cy': np.random.normal(parent['cy'], h * 0.1), 
                        'a': np.random.normal(parent['a'], 20.0), 
                        'b': np.random.normal(parent['b'], 0.05)
                    })
                
                candidates = next_generation

                if (generation + 1) % 10 == 0:
                    print(f"世代 {generation+1}/{NUM_GENERATIONS}: ベストスコア = {best_overall_score:.4f}")
            
            # --- 4. 最終結果の描画 ---
            print("\nフィッティング完了！")
            print(f"処理時間: {time.time() - start_time:.2f} 秒")
            
            final_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # クラスタリング前の重心を小さな点で描画
            for (cx, cy) in initial_centroids:
                cv2.circle(final_image, (cx, cy), 2, (150, 150, 150), -1)
            # クラスタリング後の重心を大きな点で描画
            for (cx, cy) in clustered_centroids:
                cv2.circle(final_image, (int(cx), int(cy)), 8, (0, 0, 255), -1)
            # 最適な螺旋を描画
            cx, cy, a, b = best_overall_params['cx'], best_overall_params['cy'], best_overall_params['a'], best_overall_params['b']
            theta_fit = np.linspace(-np.pi * 5, np.pi * 5, 500)
            r_fit = a * np.exp(b * theta_fit)
            x_fit = cx + r_fit * np.cos(theta_fit)
            y_fit = cy + r_fit * np.sin(theta_fit)
            fit_points = np.vstack((x_fit, y_fit)).T.astype(np.int32)
            cv2.polylines(final_image, [fit_points], isClosed=False, color=(255, 0, 0), thickness=3)

            plt.figure(figsize=(10, 10))
            plt.imshow(final_image)
            plt.title('Compositional Flow with Clustering')
            plt.show()
            # plt.savefig('composition_result.png')