#!/usr/bin/env bash
# Downloads the fixed "stock" supporting models (Qwen encoder + video/audio
# VAEs) from the official Comfy-Org/MiniMax-H3 repo, and/or a user-supplied
# DiT checkpoint from an arbitrary URL. Never bulk-clones the repo (it's
# ~480GB; V1 only needs 3 specific files from it — see objective/status.md).
#
# Hugging Face-hosted sources are fetched with the official `hf` CLI (via
# `uvx`, no project dependency needed) for resumable, integrity-checked
# transfers, using the Rust-based hf-xet backend (huggingface_hub's default
# transfer path for Xet-enabled repos, which Comfy-Org/MiniMax-H3 is) with
# high-performance mode auto-enabled on boxes with enough RAM to back it —
# see the `hf()`/`HF_XET_HIGH_PERFORMANCE` block below. Non-HF sources (e.g.
# CivitAI) fall back to a plain curl download with a manual size check.
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

# hf-xet's "high performance" mode saturates network bandwidth and all CPU
# cores for parallel transfer — a real win for the tens-of-GB files this
# script moves, but the docs call for ~64GB+ RAM to safely buffer at that
# rate, so only auto-enable it on a box that has that much (the RunPod
# target has ~92GB; a small dev machine wouldn't, but this script also isn't
# meant to run there — see objective/objective.md). An explicit shell-level
# or .env value always wins over this auto-detection.
if [ -z "${HF_XET_HIGH_PERFORMANCE:-}" ]; then
  mem_kb=$(awk '/MemTotal:/ {print $2; exit}' /proc/meminfo 2>/dev/null || echo 0)
  if [ "${mem_kb:-0}" -ge $((64 * 1024 * 1024)) ]; then
    HF_XET_HIGH_PERFORMANCE=1
  fi
fi
export HF_XET_HIGH_PERFORMANCE

py() {
  ( cd "$REPO_ROOT" && uv run python3 -c "$1" )
}

hf() {
  # HF_TOKEN, if set, is picked up automatically by huggingface_hub from the
  # environment — deliberately not passed as `--token <value>` here, since
  # the CLI itself warns that leaks into shell history and process listings.
  # `[hf_xet]` pulls in the Rust-based Xet transfer backend huggingface_hub
  # uses by default for Xet-enabled repos — without it, `hf` falls back to
  # plain HTTP and prints a "package not installed" warning on every call.
  uvx --from 'huggingface_hub[hf_xet]' hf "$@"
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
  if [ "${HF_XET_HIGH_PERFORMANCE:-0}" = "1" ]; then
    echo "INFO: HF_XET_HIGH_PERFORMANCE=1 — hf-xet will use all CPU cores and try to saturate network bandwidth."
  else
    echo "INFO: HF_XET_HIGH_PERFORMANCE not set (this box has <64GB RAM, or you set it to a non-1 value) — using hf-xet's normal auto-tuned transfer speed."
  fi
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
  HF_XET_HIGH_PERFORMANCE
                Optional. 1 saturates network+CPU for faster hf-xet transfers
                (needs ~64GB+ RAM); auto-set to 1 when this box has that much
                RAM, otherwise left to hf-xet's own auto-tuned default. Set to
                0 to force it off regardless of detected RAM.
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
  local t0=$SECONDS
  # shellcheck disable=SC2086
  hf download "$STOCK_REPO" $rel_paths --local-dir "$MODELS_ROOT" --format human
  echo "Done. (took $((SECONDS - t0))s)"
}

fetch_dit() {
  local url="$1" filename="${2:-}" hf_match dest t0=$SECONDS

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
  echo "Downloaded $filename in $((SECONDS - t0))s."

  echo "Validating $filename against h3.validate_checkpoint()..."
  # Prints a one-line summary, not the raw dict -- for a real DiT checkpoint,
  # `metadata` can be a multi-KB-per-tensor quantization blob (hundreds of layers),
  # which previously made this the single least-readable line in the whole log
  # (see objective/status.md's Docker debugging notes). h3.py's own logger already
  # logs the same tensor_count/dtypes concisely; this mirrors that instead of
  # dumping the full return value.
  py "
import h3
info = h3.validate_checkpoint('$dest')
meta = info['metadata']
meta_summary = 'present (%d top-level keys)' % len(meta) if meta else 'none'
print('OK: tensor_count=%s dtypes=%s metadata=%s' % (info['tensor_count'], info['dtypes'], meta_summary))
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
