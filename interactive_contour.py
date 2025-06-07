import cv2
import numpy as np

# グローバル変数でクリック位置を保持
click_point = None

def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)
        print(f"クリックを検出: ({x}, {y})")

def select_contour_interactively(image_path):
    global click_point
    
    original_image = cv2.imread(image_path)
    if original_image is None: return None

    # 輪郭抽出までを行う
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("輪郭がありません。")
        return None

    # ウィンドウを作成し、マウスコールバックを設定
    window_name = 'Select a Contour by Clicking'
    try:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
    except cv2.error as e:
        print("OpenCVのGUI機能がサポートされていません。画像ウィンドウを開けません。")
        print("エラー詳細:", e)
        return None
    
    print("分析したい螺旋の近くをクリックして、何かキーを押してください...")
    
    selected_contour = None
    
    while True:
        display_image = original_image.copy()
        
        if click_point:
            # 最もクリック位置に近い輪郭を探す
            min_dist = float('inf')
            for contour in contours:
                # cv2.pointPolygonTestは点と輪郭の距離を返す
                dist = cv2.pointPolygonTest(contour, click_point, True)
                # 最も「内側」にあるか、最も近くにある輪郭を選ぶ
                if abs(dist) < min_dist:
                    min_dist = abs(dist)
                    selected_contour = contour
        
        # 見つかった輪郭をハイライト表示
        if selected_contour is not None:
            cv2.drawContours(display_image, [selected_contour], -1, (0, 255, 0), 3)

        try:
            cv2.imshow(window_name, display_image)
        except cv2.error as e:
            print("OpenCVのGUI機能がサポートされていません。画像ウィンドウを開けません。")
            print("エラー詳細:", e)
            break

        # キーが押されたらループを抜ける
        key = cv2.waitKey(1) & 0xFF
        if key != 255: # 何かキーが押された
            break
            
    cv2.destroyAllWindows()
    return selected_contour

# --- 実行部分 ---
if __name__ == '__main__':
    largest_contour = select_contour_interactively('spiral.jpg')
    
    if largest_contour is not None:
        print("輪郭が選択されました。この輪郭を使ってフィッティング処理に進みます。")
        # ここに、ステップ3以降のフィッティングコードを続ける
    else:
        print("輪郭は選択されませんでした。")