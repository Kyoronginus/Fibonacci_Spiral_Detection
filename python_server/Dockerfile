# ベースとなるPythonの公式イメージを選択
FROM python:3.11-slim

# ★★★ ここに不足しているOSライブラリをインストールする命令を追加 ★★★
# apt-get(OSのパッケージマネージャ)を更新し、libgl1-mesa-glxをインストールする
# これにより、libGL.so.1がコンテナ内に提供される
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# 依存ライブラリのリストをコピー
COPY requirements.txt .

# 依存ライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのコードを全てコピー
COPY . .

# コンテナ起動時に実行するコマンド
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT