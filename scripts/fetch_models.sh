#!/usr/bin/env bash
# Downloads the fixed "stock" models (Qwen encoder + video/audio VAEs) from the official
# Comfy-Org/MiniMax-H3 repo, and/or a user-supplied DiT checkpoint from an arbitrary URL.
# Never bulk-clones the repo (~480GB; only 3 files are needed — see objective/status.md).
# HF-hosted sources go through the official `hf` CLI (via `uvx`, no project dependency)
# for resumable, integrity-checked transfers over the hf-xet backend. Non-HF sources
# (e.g. CivitAI) fall back to plain curl with a manual size check.
set -euo pipefail

STOCK_REPO="Comfy-Org/MiniMax-H3"
HF_REPO_URL="https://huggingface.co/${STOCK_REPO}/resolve/main"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Auto-load .env (e.g. HF_TOKEN=...), without overriding any value already exported into
# this process's environment (e.g. by RunPod's pod env vars) -- a .env entry should only
# fill in what isn't already set, never clobber it.
if [ -f "$REPO_ROOT/.env" ]; then
  existing_hf_token="${HF_TOKEN:-}"
  existing_xet_perf="${HF_XET_HIGH_PERFORMANCE:-}"
  existing_progress_interval="${PROGRESS_INTERVAL_SECONDS:-}"
  existing_models_root="${MODELS_ROOT:-}"
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
  [ -n "$existing_hf_token" ] && HF_TOKEN="$existing_hf_token"
  [ -n "$existing_xet_perf" ] && HF_XET_HIGH_PERFORMANCE="$existing_xet_perf"
  [ -n "$existing_progress_interval" ] && PROGRESS_INTERVAL_SECONDS="$existing_progress_interval"
  [ -n "$existing_models_root" ] && MODELS_ROOT="$existing_models_root"
fi

# hf-xet's "high performance" mode saturates network+CPU for much faster transfer of
# the tens-of-GB files this script moves, but needs ~64GB+ RAM to safely buffer at that
# rate, so it's only auto-enabled above that (an explicit env/.env value always wins).
# Note: /proc/meminfo inside a container may not match the host's if RunPod applies a
# lower memory cgroup limit than the pod's advertised total.
mem_kb=$(awk '/MemTotal:/ {print $2; exit}' /proc/meminfo 2>/dev/null || echo 0)
if [ -z "${HF_XET_HIGH_PERFORMANCE:-}" ] && [ "$mem_kb" -ge $((64 * 1024 * 1024)) ]; then
  HF_XET_HIGH_PERFORMANCE=1
fi
export HF_XET_HIGH_PERFORMANCE

py() {
  ( cd "$REPO_ROOT" && uv run python3 -c "$1" )
}

# `[hf_xet]` pulls in the Rust-based Xet transfer backend; without it `hf` silently falls
# back to plain HTTP. HF_TOKEN, if set, is picked up automatically from the environment.
hf() {
  uvx --from 'huggingface_hub[hf_xet]' hf "$@"
}

# Sets AUTH_HEADER, but only for huggingface.co targets -- never leaked to e.g. CivitAI.
auth_header_for() {
  AUTH_HEADER=""
  if [ -n "${HF_TOKEN:-}" ]; then
    case "$1" in
      https://huggingface.co/*) AUTH_HEADER="Authorization: Bearer ${HF_TOKEN}" ;;
    esac
  fi
}

# curl wrapper that sends AUTH_HEADER (if any) via `-K -` so the token never appears in
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
  if [ -n "${HF_TOKEN:-}" ]; then
    echo "INFO: HF_TOKEN is set — requests to huggingface.co will be authenticated (higher rate limits)."
  else
    echo "WARNING: no HF_TOKEN set — requests to huggingface.co will be unauthenticated (lower rate limits)."
  fi
  if [ "${HF_XET_HIGH_PERFORMANCE:-0}" = "1" ]; then
    echo "INFO: HF_XET_HIGH_PERFORMANCE=1 (detected $((mem_kb / 1024 / 1024))GB RAM) — hf-xet will saturate CPU/bandwidth."
  else
    echo "INFO: HF_XET_HIGH_PERFORMANCE not enabled (detected $((mem_kb / 1024 / 1024))GB RAM, need >=64GB, or an explicit override) — using hf-xet's normal transfer speed."
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
                huggingface.co. Can also be set via a .env file in the repo root.
  HF_XET_HIGH_PERFORMANCE
                Optional. 1 saturates network+CPU for faster hf-xet transfers (needs
                ~64GB+ RAM); auto-set to 1 when this box has that much RAM. Set to 0
                to force it off regardless of detected RAM.
  PROGRESS_INTERVAL_SECONDS
                Optional. How often (in seconds) to log download progress (default: 15).
EOF
}

# Prints "repo_id<TAB>revision<TAB>path" if $1 is a huggingface.co blob/resolve URL,
# else prints nothing.
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

# Logs periodic, newline-terminated progress by polling bytes-on-disk, rather than
# relying on hf/curl's own \r-animated bars -- those are invisible in anything that
# doesn't emulate a terminal (RunPod's log panel, a piped/redirected log file, etc).
log_progress_until_done() {
  local target="$1" total_bytes="${2:-0}" pid="$3" interval="${PROGRESS_INTERVAL_SECONDS:-15}"
  local done_bytes elapsed pct start=$SECONDS
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$interval"
    kill -0 "$pid" 2>/dev/null || break
    if [ -d "$target" ]; then
      done_bytes=$(du -sb "$target" 2>/dev/null | awk '{print $1}') || true
    else
      done_bytes=$(stat -c%s "$target" 2>/dev/null) || true
    fi
    [ -n "${done_bytes:-}" ] || continue
    elapsed=$((SECONDS - start))
    if [ "${total_bytes:-0}" -gt 0 ]; then
      pct=$((done_bytes * 100 / total_bytes))
      [ "$pct" -gt 100 ] && pct=100  # du's block-rounding can overshoot right at the end
      echo "  ...${elapsed}s: ${pct}% ($((done_bytes / 1024**2))MB / $((total_bytes / 1024**2))MB)"
    else
      echo "  ...${elapsed}s: $((done_bytes / 1024**2))MB downloaded so far"
    fi
  done
}

# Runs "$@" in the background with a progress monitor against $target ($total_bytes,
# 0 if unknown), then waits for it and propagates its exit status.
run_with_progress() {
  local target="$1" total_bytes="$2" pid
  shift 2
  "$@" &
  pid=$!
  log_progress_until_done "$target" "$total_bytes" "$pid"
  wait "$pid"
}

# Generic curl-based fallback for non-HF sources (e.g. CivitAI).
curl_download() {
  local url="$1" dest="$2" size gb_msg
  size=$(remote_size_bytes "$url" || true)
  check_disk_space "$(dirname "$dest")" "${size:-}"
  gb_msg="unknown size"
  [ -n "${size:-}" ] && gb_msg="$((size / 1024**3))GB"
  echo "Downloading $(basename "$dest") ($gb_msg) via curl..."
  run_with_progress "$dest" "${size:-0}" curl_auth -L -C - --fail --retry 3 -o "$dest" "$url"

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
  local t0=$SECONDS
  # shellcheck disable=SC2086
  run_with_progress "$MODELS_ROOT" "$total_bytes" \
    hf download "$STOCK_REPO" $rel_paths --local-dir "$MODELS_ROOT" --format human
  echo "Done. (took $((SECONDS - t0))s)"
}

fetch_dit() {
  local url="$1" filename="${2:-}" hf_match dest t0=$SECONDS

  hf_match=$(parse_hf_url "$url")

  if [ -n "$hf_match" ]; then
    local repo_id revision path scratch_dir downloaded_path expected_bytes
    IFS=$'\t' read -r repo_id revision path <<< "$hf_match"
    filename="${filename:-$(basename "$path")}"
    dest="$MODELS_ROOT/diffusion_models/$filename"
    expected_bytes=$(remote_size_bytes "https://huggingface.co/$repo_id/resolve/$revision/$path" || true)
    check_disk_space "$MODELS_ROOT/diffusion_models" "${expected_bytes:-}"

    scratch_dir=$(mktemp -d)
    echo "Downloading $filename via hf ($repo_id, revision $revision)..."
    run_with_progress "$scratch_dir" "${expected_bytes:-0}" \
      hf download "$repo_id" "$path" --revision "$revision" --local-dir "$scratch_dir"
    # `--local-dir` also writes a tiny sidecar at .cache/huggingface/download/<path>.metadata
    # (etag/commit hash bookkeeping -- see huggingface_hub's _local_folder.py) alongside the
    # real payload, so `find "$scratch_dir" -type f | head -1` could pick either one depending
    # on directory-entry order -- it was consistently grabbing the metadata sidecar, which
    # then failed h3.validate_checkpoint() as "Not a valid safetensors file". The real payload
    # is always at "$scratch_dir/$path" since we asked `hf download` for exactly that path.
    downloaded_path="$scratch_dir/$path"
    [ -f "$downloaded_path" ] || { echo "Error: hf download reported success but $downloaded_path is missing" >&2; exit 1; }
    mv "$downloaded_path" "$dest"
    rm -rf "$scratch_dir"
  else
    filename="${filename:-$(basename "$url" | cut -d'?' -f1)}"
    dest="$MODELS_ROOT/diffusion_models/$filename"
    curl_download "$url" "$dest"
  fi
  echo "Downloaded $filename in $((SECONDS - t0))s."

  echo "Validating $filename against h3.validate_checkpoint()..."
  # One-line summary rather than the raw dict: `metadata` can be a multi-KB quantization
  # blob per tensor (hundreds of layers) that would otherwise dominate the log.
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
