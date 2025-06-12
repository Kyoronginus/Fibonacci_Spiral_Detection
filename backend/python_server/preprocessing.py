import cv2

# ★★★ スマートリサイズ関数を新しく追加 ★★★
def smart_resize(image, max_dim=2048):
    """
    画像の縦横比を維持したまま、長辺がmax_dimになるようにリサイズする
    """
    h, w, _ = image.shape
    # すでに十分に小さい場合は何もしない
    if h <= max_dim and w <= max_dim:
        return image

    # 長辺がどちらかを判断
    if w > h:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    else:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    
    # cv2.INTER_AREAは、画像を縮小する際に最も品質が良いとされる補間方法
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)