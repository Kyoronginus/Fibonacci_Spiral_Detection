# python_server/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
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

def run_analysis_pipeline(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ★★★ 分析パイプラインの最初にリサイズ処理を挟む ★★★
    print(f"Original image size: {original_image.shape[1]}x{original_image.shape[0]}")
    resized_image = smart_resize(original_image)
    print(f"Resized image size: {resized_image.shape[1]}x{resized_image.shape[0]}")

    try:
        # 1. 重心抽出 (リサイズ後の画像から抽出)
        initial_centroids, _ = extract_object_centroids(resized_image)
        if len(initial_centroids) < 3:
            raise HTTPException(status_code=400, detail="分析対象オブジェクトが3つ未満です。")

        # 2. クラスタリング
        optimal_k = find_optimal_k(initial_centroids)
        kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=0).fit(initial_centroids)
        clustered_centroids = kmeans.cluster_centers_

        # 3. 螺旋フィッティング
        image_shape = resized_image.shape[:2]
        best_params = optimize_spiral_with_golden_ratio(clustered_centroids, image_shape)

        # 4. 結果の描画 (リサイズ後の画像に描画)
        result_image = draw_result(resized_image, initial_centroids, clustered_centroids, best_params)

        # 5. 画像をエンコードしてレスポンスとして返す
        _, buffer = cv2.imencode(".png", result_image)
        return io.BytesIO(buffer.tobytes())

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"予期せぬエラーが発生: {e}")
        raise HTTPException(status_code=500, detail="分析中に内部エラーが発生しました。")


@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    result_image_io = run_analysis_pipeline(image_bytes)

    return StreamingResponse(result_image_io, media_type="image/png")