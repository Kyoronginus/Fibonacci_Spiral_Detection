import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from interactive_contour import select_contour_interactively  # 追加

# --- アルゴリズム用の設定項目 ---
NUM_GENERATIONS = 50
NUM_CANDIDATES = 200
N_ELITES = 10
MUTATION_RATE = 0.2
# --- ★新しい設定項目：小さすぎる螺旋を除外するための閾値 ---
MIN_A_THRESHOLD = 500.0  # aの値がこれより小さい候補は無効とする

# --- 1-8. これまでのコード (変更なし) ---
# (前処理、輪郭抽出、距離変換画像の作成コードは全く同じなので、ここでは省略します)
# (もし完全なファイルが必要な場合は、前回のコードをベースにしてください)
image_path = 'Illustration137.cliptry.clipbu.clip.bu.png.bestcomposition.png'
original_image = cv2.imread(image_path)
if original_image is None:
    print(f"エラー: 画像ファイル '{image_path}' が見つからないか、読み込めません。")
else:
    h, w, _ = original_image.shape
    # --- ここでインタラクティブに輪郭を選択 ---
    largest_contour = select_contour_interactively(image_path)
    if largest_contour is None:
        print("輪郭が見つかりませんでした。")
    else:
        print("妥当性チェック付き・改良版アルゴリズムを開始します...")
        start_time = time.time()
        dist_transform_img = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(dist_transform_img, [largest_contour], -1, 255, 1)
        dist_transform_img = cv2.distanceTransform(255 - dist_transform_img, cv2.DIST_L2, 3)

        initial_search_ranges = {
            'cx': [0, w], 'cy': [0, h], 'a': [1.0, 10000.0], 'b': [0.1, 0.5]
        }
        
        candidates = []
        for _ in range(NUM_CANDIDATES):
            candidates.append({
                'cx': np.random.uniform(*initial_search_ranges['cx']),
                'cy': np.random.uniform(*initial_search_ranges['cy']),
                'a': np.random.uniform(*initial_search_ranges['a']),
                'b': np.random.uniform(*initial_search_ranges['b']),
            })

        best_overall_params = {}
        best_overall_score = float('inf')

        for generation in range(NUM_GENERATIONS):
            scores = []
            for params in candidates:
                # --- ★ここに追加した妥当性チェック ---
                # パラメータ'a'が閾値より小さい場合、無効な候補として最低スコアを与える
                if params['a'] < MIN_A_THRESHOLD:
                    scores.append(float('inf')) # infは「無限大」。つまり最低のスコア。
                    continue # この候補の評価を中断し、次の候補へ

                # --- スコア計算部分は同じ ---
                theta = np.linspace(-np.pi * 4, np.pi * 4, 200)
                r = params['a'] * np.exp(params['b'] * theta)
                x_fit = params['cx'] + r * np.cos(theta)
                y_fit = params['cy'] + r * np.sin(theta)
                valid_mask = (x_fit >= 0) & (x_fit < w) & (y_fit >= 0) & (y_fit < h)
                x_fit, y_fit = x_fit[valid_mask], y_fit[valid_mask]
                if len(x_fit) == 0:
                    scores.append(float('inf'))
                    continue
                score = np.mean(dist_transform_img[y_fit.astype(int), x_fit.astype(int)])
                scores.append(score)

            # --- 以降のコード（エリート選択、次世代生成、描画）は前回と全く同じ ---
            # ... (省略) ...
            sorted_candidates = sorted(zip(scores, candidates), key=lambda x: x[0])
            if sorted_candidates[0][0] < best_overall_score:
                best_overall_score = sorted_candidates[0][0]
                best_overall_params = sorted_candidates[0][1]
            elites = [c for s, c in sorted_candidates[:N_ELITES]]
            next_generation = []
            next_generation.extend(elites)
            num_mutations = int(NUM_CANDIDATES * MUTATION_RATE)
            for _ in range(num_mutations):
                next_generation.append({
                    'cx': np.random.uniform(*initial_search_ranges['cx']),
                    'cy': np.random.uniform(*initial_search_ranges['cy']),
                    'a': np.random.uniform(*initial_search_ranges['a']),
                    'b': np.random.uniform(*initial_search_ranges['b']),
                })
            num_offspring = NUM_CANDIDATES - len(next_generation)
            for i in range(num_offspring):
                parent = elites[i % len(elites)]
                next_generation.append({
                    'cx': np.random.normal(parent['cx'], w * 0.05),
                    'cy': np.random.normal(parent['cy'], h * 0.05),
                    'a': np.random.normal(parent['a'], 20.0),
                    'b': np.random.normal(parent['b'], 0.05),
                })
            candidates = next_generation
            if (generation + 1) % 5 == 0:
                print(f"世代 {generation+1}/{NUM_GENERATIONS}: ベストスコア = {best_overall_score:.4f}")

        # ... (最終描画コードも省略) ...
        print("\nフィッティング完了！")
        print(f"処理時間: {time.time() - start_time:.2f} 秒")
        print("最終パラメータ:")
        print(f"  cx={best_overall_params['cx']:.2f}, cy={best_overall_params['cy']:.2f}, a={best_overall_params['a']:.4f}, b={best_overall_params['b']:.4f}")
        
        # (描画コード)
        cx, cy, a, b = best_overall_params['cx'], best_overall_params['cy'], best_overall_params['a'], best_overall_params['b']
        theta_fit = np.linspace(-np.pi * 5, np.pi * 5, 500)
        r_fit = a * np.exp(b * theta_fit)
        x_fit = cx + r_fit * np.cos(theta_fit)
        y_fit = cy + r_fit * np.sin(theta_fit)
        fit_points = np.vstack((x_fit, y_fit)).T.astype(np.int32)
        
        final_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        cv2.drawContours(final_image, [largest_contour], -1, (0, 255, 0), 2)
        cv2.circle(final_image, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.polylines(final_image, [fit_points], isClosed=False, color=(255, 0, 0), thickness=2)

        plt.figure(figsize=(8, 8))
        plt.imshow(final_image)
        plt.title('Constraint-based Fitting Result')
        plt.xlabel(f'a = {a:.3f}, b = {b:.3f}, score = {best_overall_score:.3f}')
        plt.show()