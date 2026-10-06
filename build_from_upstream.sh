#!/usr/bin/env bash
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${1:-$KIT_DIR/.build}"
OUT_DIR="${2:-$KIT_DIR/dist}"
SRC_DIR="$WORK_ROOT/pearl-hashrate-miner"
PKG_DIR="$WORK_ROOT/prl0"
VERSION="0.1.2"

for x in git cargo nvcc; do
  command -v "$x" >/dev/null 2>&1 || { echo "Falta dependência de build: $x" >&2; exit 1; }
done

mkdir -p "$WORK_ROOT" "$OUT_DIR"
if [[ ! -d "$SRC_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/puneet-mehta/pearl-hashrate-miner.git "$SRC_DIR"
else
  git -C "$SRC_DIR" fetch --depth 1 origin
  git -C "$SRC_DIR" reset --hard origin/HEAD
fi

cp "$KIT_DIR/src/kryptex_miner.rs" "$SRC_DIR/src/bin/kryptex_miner.rs"

if ! grep -q '^name = "prl0-kryptex"$' "$SRC_DIR/Cargo.toml"; then
  cat >> "$SRC_DIR/Cargo.toml" <<'CARGO'

[[bin]]
name = "prl0-kryptex"
path = "src/bin/kryptex_miner.rs"
required-features = ["cuda"]
CARGO
fi

pushd "$SRC_DIR" >/dev/null
./csrc/build_fatbin.sh
cargo build --release --bin prl0-kryptex --features cuda
popd >/dev/null

FATBIN="/tmp/pearl_gemm.fatbin"
[[ -f "$FATBIN" ]] || FATBIN="$SRC_DIR/pearl_gemm.fatbin"
[[ -f "$FATBIN" ]] || { echo "pearl_gemm.fatbin não foi encontrado" >&2; exit 1; }

rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR"
cp "$SRC_DIR/target/release/prl0-kryptex" "$PKG_DIR/prl0-kryptex"
cp "$FATBIN" "$PKG_DIR/pearl_gemm.fatbin"
cp "$KIT_DIR/hiveos/h-manifest.conf" "$PKG_DIR/h-manifest.conf"
cp "$KIT_DIR/hiveos/h-config.sh" "$PKG_DIR/h-config.sh"
cp "$KIT_DIR/hiveos/h-run.sh" "$PKG_DIR/h-run.sh"
cp "$KIT_DIR/hiveos/h-stats.sh" "$PKG_DIR/h-stats.sh"
cp "$KIT_DIR/NOTICE.md" "$PKG_DIR/NOTICE.md"
cp "$SRC_DIR/LICENSE-MIT" "$PKG_DIR/LICENSE-MIT"
cp "$SRC_DIR/LICENSE-APACHE" "$PKG_DIR/LICENSE-APACHE"
chmod +x "$PKG_DIR/prl0-kryptex" "$PKG_DIR"/*.sh

tar -C "$WORK_ROOT" -czf "$OUT_DIR/prl0-$VERSION.tar.gz" prl0
sha256sum "$OUT_DIR/prl0-$VERSION.tar.gz" > "$OUT_DIR/prl0-$VERSION.tar.gz.sha256"
echo "Criado: $OUT_DIR/prl0-$VERSION.tar.gz"
