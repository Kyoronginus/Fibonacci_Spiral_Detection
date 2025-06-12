# --- ステージ1: ビルド環境 ---
# (FROM句は変更なし)
FROM rust:1.87.0-slim-bullseye as builder

# ビルドに必要な基本ツールをインストール
RUN apt-get update && apt-get install -y build-essential pkg-config musl-tools

# muslビルドターゲットを追加
RUN rustup target add x86_64-unknown-linux-musl

WORKDIR /usr/src/app

# ★★★ パスを修正 ★★★
# プロジェクトルートからの相対パスで指定
COPY backend/rust_axum_server/Cargo.toml backend/rust_axum_server/Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release --target x86_64-unknown-linux-musl

# ★★★ パスを修正 ★★★
COPY ./rust_axum_server/src ./src
COPY ./frontend ./frontend 
RUN cargo build --release --target x86_64-unknown-linux-musl


# --- ステージ2: 最終的な実行環境 ---
FROM gcr.io/distroless/cc-debian12
WORKDIR /app

# ★★★ パスを修正 ★★★
COPY --from=builder /usr/src/app/target/x86_64-unknown-linux-musl/release/rust_axum_server /app/rust_axum_server
COPY --from=builder /usr/src/app/frontend /app/frontend
EXPOSE 8080
CMD ["/app/rust_axum_server"]