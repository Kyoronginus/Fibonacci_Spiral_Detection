import cv2
import numpy as np
import matplotlib.pyplot as plt # 最終結果の表示にのみ使用

# --- ★ ユーザー設定項目 ★ ---
# 最初に輪郭を探して番号付き画像を保存する場合は 'FIND' に設定
# 番号を確認した後、特定の輪郭でフィッティングする場合は 'FIT' に設定
ANALYSIS_MODE = 'FIND' 
# ANALYSIS_MODE = 'FIT' # こちらのコメントを外して実行

# 'FIT'モードの時に、分析したい輪郭の番号を指定
SELECTED_CONTOUR_INDEX = 0 


def run_fitting_process(contour_to_fit, original_image_rgb):
    """
    指定された輪郭に対してフィッティング処理を行い、結果を表示する関数
    （この中身はステップ3のフィッティングコードとほぼ同じ）
    """
    print(f"\n輪郭 {SELECTED_CONTOUR_INDEX} のフィッティングを開始します...")
    
    # 重心計算による中心点推定
    M = cv2.moments(contour_to_fit)
    if M["m00"] == 0:
        print("エラー: 輪郭の面積が0です。")
        return
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # 座標変換とフィッティング
    points = contour_to_fit.squeeze()
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.unwrap(np.arctan2(dy, dx))
    
    valid_indices = r > 0
    if not np.any(valid_indices):
        print("エラー: 有効な点が見つかりません。")
        return
        
    coeffs = np.polyfit(theta[valid_indices], np.log(r[valid_indices]), 1)
    b, a = coeffs[0], np.exp(coeffs[1])
    
    print("フィッティング結果:")
    print(f"  a = {a:.4f}, b = {b:.4f}")

    # 結果の描画
    theta_fit = np.linspace(theta.min(), theta.max(), 500)
    r_fit = a * np.exp(b * theta_fit)
    x_fit = cx + r_fit * np.cos(theta_fit)
    y_fit = cy + r_fit * np.sin(theta_fit)
    fit_points = np.vstack((x_fit, y_fit)).T.astype(np.int32)
    
    final_image = original_image_rgb.copy()
    cv2.drawContours(final_image, [contour_to_fit], -1, (0, 255, 0), 2)
    cv2.circle(final_image, (cx, cy), 5, (0, 0, 255), -1)
    cv2.polylines(final_image, [fit_points], isClosed=False, color=(255, 0, 0), thickness=2)

    # MatplotlibはGUIがなくてもファイル保存が可能なので、結果表示に使う
    plt.figure(figsize=(8, 8))
    plt.imshow(final_image)
    plt.title(f'Fitting Result for Contour #{SELECTED_CONTOUR_INDEX}')
    plt.xlabel(f'a = {a:.3f}, b = {b:.3f}')
    plt.axis('off')
    # plt.show() # GUIがない環境ではこれもエラーになることがある
    output_filename = f'result_contour_{SELECTED_CONTOUR_INDEX}.png'
    plt.savefig(output_filename)
    print(f"結果を '{output_filename}' に保存しました。")
    plt.close() # メモリを解放


# --- メインの実行部分 ---
if __name__ == '__main__':
    image_path = 'spiral.jpg'
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"エラー: 画像ファイル '{image_path}' が見つかりません。")
    else:
        # 輪郭抽出はどちらのモードでも共通
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        # RETR_CCOMPで階層構造を保持して抽出
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # 面積でソートし、ノイズの可能性が高い小さい輪郭を除外
        min_area = 100
        valid_contours = []
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > min_area:
                valid_contours.append(contour)
        
        # 面積の大きい順にソート
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        print(f"有効な輪郭が {len(valid_contours)} 個見つかりました。")

        if ANALYSIS_MODE == 'FIND':
            print("モード: FIND - 輪郭に番号を付けて画像を保存します。")
            
            image_with_numbers = original_image.copy()
            for i, contour in enumerate(valid_contours):
                # 輪郭を描画
                color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                cv2.drawContours(image_with_numbers, [contour], -1, color, 2)
                
                # 番号を描画
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(image_with_numbers, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(image_with_numbers, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            output_filename = 'contours_with_ids.jpg'
            cv2.imwrite(output_filename, image_with_numbers)
            print(f"番号付きの輪郭画像を '{output_filename}' に保存しました。")
            print("画像を開いて分析したい輪郭の番号を確認し、SELECTED_CONTOUR_INDEXを書き換えて、モードを'FIT'にして再度実行してください。")

        elif ANALYSIS_MODE == 'FIT':
            print(f"モード: FIT - 輪郭番号 {SELECTED_CONTOUR_INDEX} を選択して分析します。")
            if 0 <= SELECTED_CONTOUR_INDEX < len(valid_contours):
                selected_contour = valid_contours[SELECTED_CONTOUR_INDEX]
                
                # フィッティング処理を実行する関数を呼び出し
                original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                run_fitting_process(selected_contour, original_image_rgb)
            else:
                print(f"エラー: 指定された番号 {SELECTED_CONTOUR_INDEX} は無効です。0から{len(valid_contours)-1}の範囲で指定してください。")