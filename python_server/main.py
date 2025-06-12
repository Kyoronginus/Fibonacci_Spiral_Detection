# python_server/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
import base64
from sklearn.cluster import KMeans

# 他のファイルから関数をインポート
from clustering import extract_object_centroids, find_optimal_k
from spiral_fit import optimize_spiral_with_golden_ratio,calculate_composition_score
from visualization import draw_result
from preprocessing import smart_resize
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "null", # ローカルのHTMLファイルを直接開いた場合のオリジン
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ★★★ run_analysis_pipelineの引数に`b_weight`を追加 ★★★
def run_analysis_pipeline(image_bytes: bytes, k: int, b_weight: float):
    # ... (重心抽出、クラスタリング部分は変更なし) ...
    # ... (中略) ...
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    resized_image = smart_resize(original_image)
    initial_centroids, _ = extract_object_centroids(resized_image)
    if len(initial_centroids) < 3: raise HTTPException(status_code=400, detail="分析対象オブジェクトが3つ未満です。")
    if k > 0 and k > len(initial_centroids): k = len(initial_centroids)
    if k == 0: optimal_k = find_optimal_k(initial_centroids)
    else: optimal_k = k
    if optimal_k < 2 and len(initial_centroids) >= 2: optimal_k = 2
    elif len(initial_centroids) < 2: raise HTTPException(status_code=400, detail="分析対象オブジェクトが2つ未満です。")
    kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=0).fit(initial_centroids)
    clustered_centroids = kmeans.cluster_centers_

    # 3. 螺旋フィッティング
    image_shape = resized_image.shape[:2]
    # ★★★最適化関数に`b_weight`を渡す ★★★
    best_params = optimize_spiral_with_golden_ratio(clustered_centroids, image_shape, b_weight)

    # 4. スコアリングと描画
    # ... (スコアリングと描画、レスポンス作成部分は変更なし) ...
    distance_score = calculate_composition_score(best_params, clustered_centroids, image_shape)
    score_fit = np.exp(-0.05 * distance_score)
    b_value = best_params.get('b', 0)
    GOLDEN_B = 0.30635
    score_golden = np.exp(-50 * abs(b_value - GOLDEN_B))
    final_score = (0.6 * score_fit + 0.4 * score_golden) * 100
    result_image = draw_result(resized_image, initial_centroids, clustered_centroids, best_params)
    _, buffer = cv2.imencode(".png", result_image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    return { "score": round(final_score, 1), "b_value": round(b_value, 4), "golden_b": GOLDEN_B, "image_base64": "data:image/png;base64," + image_base64 }


# ★★★ APIエンドポイントの引数に`b_weight`を追加 ★★★
@app.post("/analyze/")
async def analyze_image(
    file: UploadFile = File(...),
    k: int = Form(0),
    b_weight: float = Form(100.0) # デフォルト値を設定
):
    image_bytes = await file.read()
    try:
        # パイプラインにkとb_weightの値を渡す
        analysis_result = run_analysis_pipeline(image_bytes, k, b_weight)
        return JSONResponse(content=analysis_result)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"予期せぬエラーが発生: {e}")
        raise HTTPException(status_code=500, detail="分析中に内部エラーが発生しました。")


# ★★★ バグを修正したプレビュー用APIエンドポイント ★★★
@app.post("/preview_clusters/")
async def preview_clusters(
    file: UploadFile = File(...),
    k: int = Form(0)
):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    resized_image = smart_resize(original_image)
    initial_centroids, _ = extract_object_centroids(resized_image)
    
    clustered_centroids = None # 結果を格納する変数を初期化

    if len(initial_centroids) < 2:
        # プレビューではエラーにせず、そのまま画像を返す
        clustered_centroids = initial_centroids
    elif k == 1:
        # k=1の場合は、全重心の平均点を計算（これが唯一のクラスタ中心）
        print("k=1, calculating the mean of all centroids.")
        clustered_centroids = np.array([np.mean(initial_centroids, axis=0)])
    elif k >= 2:
        # kが重心の数より多い場合は、重心の数に丸める
        num_points = len(initial_centroids)
        k_to_use = min(k, num_points)
        print(f"k={k}, running KMeans with {k_to_use} clusters.")
        kmeans = KMeans(n_clusters=k_to_use, n_init='auto', random_state=0).fit(initial_centroids)
        clustered_centroids = kmeans.cluster_centers_
    # k=0 (Auto) の場合は、clustered_centroidsがNoneのままになり、青い点は描画されない
    
    # 螺旋なしで、重心の位置だけを描画
    preview_image = draw_result(resized_image, initial_centroids, clustered_centroids, None)

    _, buffer = cv2.imencode(".png", preview_image)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")