# --- Stage 1: Build Environment ---
FROM rust:1.87.0-slim-bullseye as builder
RUN apt-get update && apt-get install -y build-essential pkg-config musl-tools
RUN rustup target add x86_64-unknown-linux-musl
WORKDIR /usr/src/app

# Copy manifests first for caching
COPY backend/rust_axum_server/Cargo.toml backend/rust_axum_server/Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release --target x86_64-unknown-linux-musl

# Copy source code
COPY backend/rust_axum_server/.cargo ./.cargo
COPY backend/rust_axum_server/src ./src

# Touch main.rs to force rebuild
RUN touch src/main.rs
RUN cargo build --release --target x86_64-unknown-linux-musl

# --- Stage 2: Runtime Environment ---
FROM gcr.io/distroless/cc-debian12
WORKDIR /app
COPY --from=builder /usr/src/app/target/x86_64-unknown-linux-musl/release/rust_axum_server /app/rust_axum_server
EXPOSE 8080
CMD ["/app/rust_axum_server"]