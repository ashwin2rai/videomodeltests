#!/usr/bin/env bash
# Downloads the fixed "stock" supporting models (Qwen encoder + video/audio
# VAEs) from the official Comfy-Org/MiniMax-H3 repo, and/or a user-supplied
# DiT checkpoint from an arbitrary URL. Never bulk-clones the repo (it's
# ~480GB; V1 only needs 3 specific files from it — see objective/status.md).
#
# Hugging Face-hosted sources are fetched with the official `hf` CLI (via
# `uvx`, no project dependency needed) for resumable, integrity-checked
# transfers. Non-HF sources (e.g. CivitAI) fall back to a plain curl
# download with a manual size check.
set -euo pipefail

STOCK_REPO="Comfy-Org/MiniMax-H3"
HF_REPO_URL="https://huggingface.co/${STOCK_REPO}/resolve/main"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Auto-load .env (e.g. HF_TOKEN=...) if present, without overriding a value
# already set explicitly in the environment (e.g. `HF_TOKEN=x make ...`).
if [ -f "$REPO_ROOT/.env" ]; then
  existing_hf_token="${HF_TOKEN:-}"
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
  [ -n "$existing_hf_token" ] && HF_TOKEN="$existing_hf_token"
fi

py() {
  ( cd "$REPO_ROOT" && uv run python3 -c "$1" )
}

hf() {
  # HF_TOKEN, if set, is picked up automatically by huggingface_hub from the
  # environment — deliberately not passed as `--token <value>` here, since
  # the CLI itself warns that leaks into shell history and process listings.
  uvx --from huggingface_hub hf "$@"
}

# Sets AUTH_HEADER to an Authorization header, but only when the target is
# huggingface.co — never send the token anywhere else (e.g. a CivitAI URL in
# the curl fallback path).
auth_header_for() {
  AUTH_HEADER=""
  if [ -n "${HF_TOKEN:-}" ]; then
    case "$1" in
      https://huggingface.co/*) AUTH_HEADER="Authorization: Bearer ${HF_TOKEN}" ;;
    esac
  fi
}

# curl wrapper that attaches AUTH_HEADER (if any) via `-K -` (config on
# stdin) rather than `-H` on the command line, so the token never appears in
# `ps` output. The URL must be the last argument.
curl_auth() {
  local url="${!#}"
  auth_header_for "$url"
  if [ -n "$AUTH_HEADER" ]; then
    printf 'header = "%s"\n' "$AUTH_HEADER" | curl -K - "$@"
  else
    curl "$@"
  fi
}

resolve_models_root() {
  MODELS_ROOT="${MODELS_ROOT:-$(py 'import h3; print(h3.MODELS_ROOT)')}"
  echo "INFO: using MODELS_ROOT=$MODELS_ROOT"
  [ -n "${HF_TOKEN:-}" ] || echo "WARNING: no HF_TOKEN set — requests to huggingface.co will be unauthenticated (lower rate limits)."
}

usage() {
  cat <<EOF
Usage:
  $0 stock                 Download the fixed Qwen encoder + video/audio VAEs
  $0 dit <url> [filename]  Download a DiT checkpoint from <url> into diffusion_models/
                            (uses the official 'hf' CLI for huggingface.co URLs,
                            falls back to curl for anything else, e.g. CivitAI)
  $0 all <url> [filename]  Both of the above

Environment:
  MODELS_ROOT   Root of the ComfyUI-style models folder (default: value of h3.MODELS_ROOT)
  HF_TOKEN      Optional Hugging Face access token, for authenticated requests
                (higher rate limits, access to gated repos). Only ever sent to
                huggingface.co — never forwarded to a non-HF URL. Can also be
                set via a .env file in the repo root (auto-loaded if present).
EOF
}

# Prints "repo_id<TAB>revision<TAB>path" if $1 is a huggingface.co blob/resolve
# URL, else prints nothing.
parse_hf_url() {
  py "
import re
m = re.match(r'https?://huggingface\.co/([^/]+/[^/]+)/(?:blob|resolve)/([^/]+)/(.+?)(?:\?.*)?\$', '$1')
if m:
    print('\t'.join(m.groups()))
"
}

remote_size_bytes() {
  curl_auth -sIL "$1" | tr -d '\r' | awk 'tolower($0) ~ /^content-length:/ {v=$2} END {print v}'
}

check_disk_space() {
  local dest_dir="$1" required_bytes="$2" avail_bytes
  mkdir -p "$dest_dir"
  avail_bytes=$(df --output=avail -B1 "$dest_dir" | tail -1 | tr -d ' ')
  if [ -n "$required_bytes" ] && [ "$avail_bytes" -lt "$required_bytes" ]; then
    echo "Error: only $((avail_bytes / 1024**3))GB free in $dest_dir, need $((required_bytes / 1024**3))GB" >&2
    exit 1
  fi
}

# Generic curl-based fallback for non-HF sources (e.g. CivitAI).
curl_download() {
  local url="$1" dest="$2" size gb_msg
  size=$(remote_size_bytes "$url" || true)
  check_disk_space "$(dirname "$dest")" "${size:-}"
  gb_msg="unknown size"
  [ -n "${size:-}" ] && gb_msg="$((size / 1024**3))GB"
  echo "Downloading $(basename "$dest") ($gb_msg) via curl..."
  curl_auth -L -C - --fail --retry 3 -o "$dest" "$url"

  if [ -n "${size:-}" ]; then
    local actual
    actual=$(stat -c%s "$dest")
    if [ "$actual" -ne "$size" ]; then
      echo "Error: downloaded size ($actual bytes) does not match expected size ($size bytes) for $dest" >&2
      exit 1
    fi
  fi
}

fetch_stock() {
  local rel_paths total_bytes=0 size rel
  rel_paths=$(py '
import h3
root = h3.MODELS_ROOT + "/"
for p in (h3.QWEN_ENCODER_PATH, h3.VIDEO_VAE_PATH, h3.AUDIO_VAE_PATH):
    print(p[len(root):])
')

  while IFS= read -r rel; do
    size=$(remote_size_bytes "$HF_REPO_URL/$rel" || true)
    total_bytes=$((total_bytes + ${size:-0}))
  done <<< "$rel_paths"
  check_disk_space "$MODELS_ROOT" "$total_bytes"

  echo "Downloading 3 fixed files (~$((total_bytes / 1024**3))GB total) into $MODELS_ROOT..."
  echo "(hf CLI shows little/no live progress when not attached to an interactive terminal —"
  echo " this is normal, not a hang. Check with: du -sh $MODELS_ROOT in another shell.)"
  # shellcheck disable=SC2086
  hf download "$STOCK_REPO" $rel_paths --local-dir "$MODELS_ROOT" --format human
  echo "Done."
}

fetch_dit() {
  local url="$1" filename="${2:-}" hf_match dest

  hf_match=$(parse_hf_url "$url")

  if [ -n "$hf_match" ]; then
    local repo_id revision path scratch_dir downloaded_path
    IFS=$'\t' read -r repo_id revision path <<< "$hf_match"
    filename="${filename:-$(basename "$path")}"
    dest="$MODELS_ROOT/diffusion_models/$filename"

    # check_disk_space creates the destination directory as a side effect.
    check_disk_space "$MODELS_ROOT/diffusion_models" "$(remote_size_bytes "https://huggingface.co/$repo_id/resolve/$revision/$path" || true)"

    scratch_dir=$(mktemp -d)
    echo "Downloading $filename via hf ($repo_id, revision $revision)..."
    downloaded_path=$(hf download "$repo_id" "$path" --revision "$revision" --local-dir "$scratch_dir" --quiet | tail -1)
    mv "$downloaded_path" "$dest"
    rm -rf "$scratch_dir"
  else
    filename="${filename:-$(basename "$url" | cut -d'?' -f1)}"
    dest="$MODELS_ROOT/diffusion_models/$filename"
    curl_download "$url" "$dest"
  fi

  echo "Validating $filename against h3.validate_checkpoint()..."
  py "
import h3
info = h3.validate_checkpoint('$dest')
print('OK:', info)
"
}

case "${1:-}" in
  stock)
    resolve_models_root
    fetch_stock
    ;;
  dit)
    [ -n "${2:-}" ] || { usage; exit 1; }
    resolve_models_root
    fetch_dit "$2" "${3:-}"
    ;;
  all)
    [ -n "${2:-}" ] || { usage; exit 1; }
    resolve_models_root
    fetch_stock
    fetch_dit "$2" "${3:-}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
