# python_server/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
from sklearn.cluster import KMeans

# 他のファイルから関数をインポート
from clustering import extract_object_centroids, find_optimal_k
from spiral_fit import optimize_spiral_with_golden_ratio
from visualization import draw_result
from preprocessing import smart_resize
app = FastAPI()

def run_analysis_pipeline(image_bytes: bytes, k: int):
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    resized_image = smart_resize(original_image)
    
    # 1. 重心抽出
    initial_centroids, _ = extract_object_centroids(resized_image)
    if len(initial_centroids) < 3:
        raise HTTPException(status_code=400, detail="分析対象オブジェクトが3つ未満です。")
    if k > len(initial_centroids):
        print(f"Warning: k({k}) is larger than the number of centroids({len(initial_centroids)}). Clamping k.")
        k = len(initial_centroids)

    # 2. クラスタリング
    # ★★★ kの値に応じて処理を分岐 ★★★
    if k == 0:
        # kが0の場合はエルボー法で自動計算
        print("k=0, running elbow method to find optimal k...")
        optimal_k = find_optimal_k(initial_centroids)
        print(f"Optimal k found: {optimal_k}")
    else:
        # kが指定されている場合はその値を使う
        print(f"Using user-specified k: {k}")
        optimal_k = k

    kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=0).fit(initial_centroids)
    clustered_centroids = kmeans.cluster_centers_

    # 3. 螺旋フィッティング
    image_shape = resized_image.shape[:2]
    best_params = optimize_spiral_with_golden_ratio(clustered_centroids, image_shape)

    # 4. 結果の描画
    result_image = draw_result(resized_image, initial_centroids, clustered_centroids, best_params)

    # 5. 画像をエンコードしてレスポンスとして返す
    _, buffer = cv2.imencode(".png", result_image)
    return io.BytesIO(buffer.tobytes())


# ★★★ APIエンドポイントのシグネチャを修正 ★★★
@app.post("/analyze/")
async def analyze_image(
    file: UploadFile = File(...),
    k: int = Form(0)  # kという名前のフォームデータを受け取る。デフォルトは0
):
    image_bytes = await file.read()
    
    try:
        # パイプラインにkの値を渡す
        result_image_io = run_analysis_pipeline(image_bytes, k)
        return StreamingResponse(result_image_io, media_type="image/png")
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"予期せぬエラーが発生: {e}")
        raise HTTPException(status_code=500, detail="分析中に内部エラーが発生しました。")
