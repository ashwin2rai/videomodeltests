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

# RAM actually available to *this process*. /proc/meminfo reports the host's total, which
# on RunPod can be an order of magnitude above the pod's cgroup limit — and hf-xet's
# high-performance buffering is bounded by the limit, not the host total. Overestimating
# here gets the download OOM-killed partway through, which presents as a container restart
# (and a fresh multi-GB re-download), not as an error message.
detect_mem_kb() {
  local total limit
  total=$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo 2>/dev/null || echo 0)
  [ -n "$total" ] || total=0
  # cgroup v2 first, then v1. v2's "max" and v1's near-2^63 sentinel both mean unlimited.
  limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null \
    || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null \
    || echo max)
  case "$limit" in
    '' | max | *[!0-9]*) echo "$total"; return ;;
  esac
  limit=$((limit / 1024))
  if [ "$limit" -gt 0 ] && [ "$limit" -lt "$total" ]; then echo "$limit"; else echo "$total"; fi
}
mem_kb=$(detect_mem_kb)

# hf-xet's "high performance" mode saturates network+CPU for much faster transfer of the
# tens-of-GB files this script moves, but needs ~64GB+ RAM to safely buffer at that rate,
# so it's only auto-enabled above that. An explicit value always wins.
#
# huggingface_hub evaluates this as `value.upper() in {"1","ON","YES","TRUE"}` with NO
# whitespace trimming (constants.py::_is_true), so a value pasted into a RunPod env-var
# field with a trailing space or newline reads as *set* here (suppressing auto-detection)
# while reading as *false* inside the library — the mode then silently never turns on.
# Normalising to a bare 1/0 makes that impossible and keeps the log line below honest.
xet_perf_raw="$(printf '%s' "${HF_XET_HIGH_PERFORMANCE-}" | tr -d '[:space:]')"
if [ -n "$xet_perf_raw" ]; then
  case "$(printf '%s' "$xet_perf_raw" | tr '[:lower:]' '[:upper:]')" in
    1 | ON | YES | TRUE) HF_XET_HIGH_PERFORMANCE=1 ;;
    *) HF_XET_HIGH_PERFORMANCE=0 ;;
  esac
elif [ "$mem_kb" -ge $((64 * 1024 * 1024)) ]; then
  HF_XET_HIGH_PERFORMANCE=1
else
  HF_XET_HIGH_PERFORMANCE=0
fi
export HF_XET_HIGH_PERFORMANCE

py() {
  ( cd "$REPO_ROOT" && uv run python3 -c "$1" )
}

# Prefer the `hf` CLI from the image's own venv: huggingface_hub 1.x requires hf-xet
# outright on x86_64/aarch64 (the `[hf_xet]` extra is now a no-op alias), so the venv copy
# already has the Rust Xet backend, and uv.lock pins its exact version. `uvx` instead
# resolves and downloads an *unpinned* huggingface_hub from PyPI on every container start:
# a needless network dependency in the critical startup path (a PyPI hiccup = the container
# can't start) and a silent upgrade surface (this script depends on `--local-dir`'s exact
# on-disk layout). uvx stays as the fallback for a plain checkout with no synced venv.
# HF_TOKEN, if set, is picked up automatically from the environment either way.
if command -v hf >/dev/null 2>&1; then
  hf() { command hf "$@"; }
else
  hf() { uvx --from 'huggingface_hub[hf_xet]' hf "$@"; }
fi

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
  local why="auto-detected from $((mem_kb / 1024 / 1024))GB usable RAM"
  [ -n "$xet_perf_raw" ] && why="explicitly set to '$xet_perf_raw' (usable RAM: $((mem_kb / 1024 / 1024))GB)"
  if [ "$HF_XET_HIGH_PERFORMANCE" = "1" ]; then
    echo "INFO: HF_XET_HIGH_PERFORMANCE=1 — $why — hf-xet will saturate CPU/bandwidth."
  else
    echo "INFO: HF_XET_HIGH_PERFORMANCE=0 — $why; needs >=64GB usable RAM or an explicit 1/on/yes/true — using hf-xet's normal transfer speed."
  fi
}

# Serialises fetches so the entrypoint's start-up fetch and a manual
# `./scripts/fetch_models.sh dit ...` from a shell in the same container can't interleave
# writes into the same incoming dir. Degrades to a no-op if flock is unavailable.
acquire_fetch_lock() {
  command -v flock >/dev/null 2>&1 || return 0
  mkdir -p "$MODELS_ROOT"
  exec 9>"$MODELS_ROOT/.fetch.lock"
  if ! flock -w "${FETCH_LOCK_TIMEOUT:-3600}" 9; then
    echo "Error: another fetch still holds $MODELS_ROOT/.fetch.lock after ${FETCH_LOCK_TIMEOUT:-3600}s" >&2
    exit 1
  fi
}

# Prints a one-line summary of the checkpoint; non-zero exit means it isn't usable.
# The path goes through the environment rather than into the Python source so a filename
# containing a quote can't break (or inject into) the snippet.
validate_checkpoint_file() {
  CHECKPOINT_PATH="$1" py '
import os
import h3
info = h3.validate_checkpoint(os.environ["CHECKPOINT_PATH"])
meta = info["metadata"]
# One-line summary rather than the raw dict: `metadata` can be a multi-KB quantization
# blob per tensor (hundreds of layers) that would otherwise dominate the log.
meta_summary = "present (%d top-level keys)" % len(meta) if meta else "none"
print("OK: tensor_count=%s dtypes=%s metadata=%s" % (info["tensor_count"], info["dtypes"], meta_summary))
'
}

# Records which checkpoint the server should load. h3.discover_dit_checkpoint() otherwise
# globs diffusion_models/*.safetensors and refuses to guess when there's more than one
# (e.g. after a second DIT_URL is fetched by hand into a running container). Dot-prefixed
# so it can never be mistaken for a checkpoint by that glob.
record_active_dit() {
  printf '%s\n' "$1" > "$MODELS_ROOT/diffusion_models/.active-dit"
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
  FORCE_REFETCH Optional. 1 re-downloads the DiT checkpoint even if a valid one is
                already in place (the default is to reuse it, so a container restart
                doesn't re-transfer tens of GB).
  FETCH_RETRIES Optional. Attempts per DiT download before giving up (default: 3).
                Retries resume the partial transfer rather than restarting it.
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

# X-Linked-Size is the authoritative payload size for LFS/Xet-backed files on
# huggingface.co; Content-Length on the same response can describe the pointer or a
# redirect hop instead. huggingface_hub itself prefers it (file_download.py), so this
# matches what the download will actually write. Non-HF hosts don't send it — hence the
# Content-Length fallback.
remote_size_bytes() {
  curl_auth -sIL "$1" | tr -d '\r' | awk '
    tolower($1) == "x-linked-size:" { linked = $2 }
    tolower($1) == "content-length:" { len = $2 }
    END { print (linked != "" ? linked : len) }'
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
  # Returns rather than exits on failure so the caller's retry loop can resume (-C -)
  # instead of the whole script dying and the container restarting from zero.
  run_with_progress "$dest" "${size:-0}" curl_auth -L -C - --fail --retry 3 -o "$dest" "$url" || return 1

  if [ -n "${size:-}" ] && [ "${size:-0}" -gt 0 ]; then
    local actual
    actual=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    if [ "$actual" -ne "$size" ]; then
      echo "Error: downloaded size ($actual bytes) does not match expected size ($size bytes) for $dest" >&2
      return 1
    fi
  fi
}

fetch_stock() {
  local total_bytes=0 size rel i rels=() sizes=()
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    rels+=("$rel")
    size=$(remote_size_bytes "$HF_REPO_URL/$rel" || true)
    sizes+=("${size:-0}")
    total_bytes=$((total_bytes + ${size:-0}))
  done <<< "$(py '
import h3
root = h3.MODELS_ROOT + "/"
for p in (h3.QWEN_ENCODER_PATH, h3.VIDEO_VAE_PATH, h3.AUDIO_VAE_PATH):
    print(p[len(root):])
')"
  check_disk_space "$MODELS_ROOT" "$total_bytes"

  echo "Downloading ${#rels[@]} fixed files (~$((total_bytes / 1024**3))GB total) into $MODELS_ROOT..."
  local t0=$SECONDS
  # `hf` skips files already present with matching metadata, so a container restart
  # re-runs this for free rather than re-downloading ~20GB.
  run_with_progress "$MODELS_ROOT" "$total_bytes" \
    hf download "$STOCK_REPO" "${rels[@]}" --local-dir "$MODELS_ROOT" --format human
  echo "Done. (took $((SECONDS - t0))s)"

  # Verify rather than assume. A file truncated by a container killed mid-write would
  # otherwise stay silently broken until it surfaced as a confusing load error partway
  # into the first real generation.
  local bad=0 actual
  for i in "${!rels[@]}"; do
    actual=$(stat -c%s "$MODELS_ROOT/${rels[$i]}" 2>/dev/null || echo 0)
    if [ "$actual" -eq 0 ]; then
      echo "ERROR: stock model missing after fetch: $MODELS_ROOT/${rels[$i]}" >&2
      bad=1
    elif [ "${sizes[$i]}" -gt 0 ] && [ "$actual" -ne "${sizes[$i]}" ]; then
      echo "ERROR: stock model is $actual bytes, expected ${sizes[$i]}: $MODELS_ROOT/${rels[$i]}" >&2
      echo "ERROR: delete it and re-run to fetch it again." >&2
      bad=1
    fi
  done
  [ "$bad" -eq 0 ] || return 1
  echo "Verified ${#rels[@]}/${#rels[@]} stock files against their published sizes."
}

# Runs one download attempt into $incoming, setting DIT_DOWNLOADED_PATH to the payload.
# Kept separate from fetch_dit so a transient network failure can be retried against the
# same incoming dir, where both `hf` (via its .metadata bookkeeping) and `curl -C -`
# resume rather than start the multi-GB transfer over. The result comes back in a global
# rather than on stdout so that progress logging stays on stdout where it belongs.
# Every failure path returns explicitly: `set -e` is suspended inside a function called
# from an `if`, so a bare failing command here would otherwise fall through.
download_dit_attempt() {
  local url="$1" incoming="$2" expected_bytes="$3" downloaded_path
  DIT_DOWNLOADED_PATH=""

  if [ -n "$DIT_HF_REPO" ]; then
    echo "Downloading $DIT_FILENAME via hf ($DIT_HF_REPO, revision $DIT_HF_REVISION)..."
    run_with_progress "$incoming" "${expected_bytes:-0}" \
      hf download "$DIT_HF_REPO" "$DIT_HF_PATH" --revision "$DIT_HF_REVISION" --local-dir "$incoming" || return 1
    # `--local-dir` also writes a tiny sidecar at .cache/huggingface/download/<path>.metadata
    # (etag/commit-hash bookkeeping — see huggingface_hub's _local_folder.py) next to the real
    # payload, so the previous `find "$incoming" -type f | head -1` could return either one
    # depending on directory-entry order. It consistently picked the ~16-byte sidecar, which
    # then got renamed over the destination and failed validation as "Not a valid safetensors
    # file" — with the actual 20GB download discarded. The payload is always at the repo path
    # we asked for; the fallback below still excludes the bookkeeping dir and tiny files.
    downloaded_path="$incoming/$DIT_HF_PATH"
    if [ ! -f "$downloaded_path" ]; then
      downloaded_path=$(find "$incoming" -type f -not -path '*/.cache/*' -size +1M | head -1)
    fi
  else
    downloaded_path="$incoming/$DIT_FILENAME"
    curl_download "$url" "$downloaded_path" || return 1
  fi

  if [ -z "$downloaded_path" ] || [ ! -f "$downloaded_path" ]; then
    echo "Error: the download reported success but no payload was found under $incoming" >&2
    return 1
  fi
  DIT_DOWNLOADED_PATH="$downloaded_path"
}

fetch_dit() {
  local url="$1" filename="${2:-}" hf_match dest incoming downloaded_path t0=$SECONDS
  local expected_bytes attempt=1 max_attempts="${FETCH_RETRIES:-3}" actual_bytes

  # Resolve names without touching the network, so the reuse check below still works when
  # huggingface.co is unreachable -- an outage must not stop a container that already has
  # a valid checkpoint on disk from starting.
  DIT_HF_REPO="" DIT_HF_REVISION="" DIT_HF_PATH=""
  hf_match=$(parse_hf_url "$url")
  if [ -n "$hf_match" ]; then
    IFS=$'\t' read -r DIT_HF_REPO DIT_HF_REVISION DIT_HF_PATH <<< "$hf_match"
    filename="${filename:-$(basename "$DIT_HF_PATH")}"
  else
    filename="${filename:-$(basename "$url" | cut -d'?' -f1)}"
  fi
  DIT_FILENAME="$filename"
  dest="$MODELS_ROOT/diffusion_models/$filename"
  mkdir -p "$MODELS_ROOT/diffusion_models"

  # A container restart must not re-download tens of GB it already has: RunPod restarts
  # the container on any non-zero exit, so a failure downstream of this point would
  # otherwise mean a full re-download on every single retry, forever. `dest` only ever
  # holds a checkpoint that already passed validation (the download lands in `incoming`
  # and is only renamed into place afterwards), so its presence is a trustworthy skip
  # signal — but re-validating costs one header read, so do it anyway.
  if [ "${FORCE_REFETCH:-0}" != "1" ] && [ -f "$dest" ]; then
    echo "Found an existing $filename ($(du -h "$dest" | cut -f1)) — validating before reuse..."
    if validate_checkpoint_file "$dest"; then
      echo "Reusing it; no download needed (set FORCE_REFETCH=1 to fetch it again)."
      record_active_dit "$dest"
      return 0
    fi
    echo "WARNING: the existing $filename failed validation — discarding and re-downloading." >&2
    rm -f "$dest"
  fi

  # Stage inside diffusion_models rather than `mktemp -d` under /tmp. Three reasons: the
  # disk-space check above then covers the filesystem actually being written to; the final
  # `mv` is a same-filesystem rename (atomic and instant) instead of a 20GB cross-device
  # copy that leaves a truncated destination if the container dies mid-copy; and a fixed
  # path lets a retry resume, where a fresh mktemp dir guaranteed starting over. Hidden so
  # h3.discover_dit_checkpoint()'s non-recursive *.safetensors glob can't see it.
  incoming="$MODELS_ROOT/diffusion_models/.incoming"
  if [ -n "$DIT_HF_REPO" ]; then
    expected_bytes=$(remote_size_bytes "https://huggingface.co/$DIT_HF_REPO/resolve/$DIT_HF_REVISION/$DIT_HF_PATH" || true)
  else
    expected_bytes=$(remote_size_bytes "$url" || true)
  fi
  check_disk_space "$MODELS_ROOT/diffusion_models" "${expected_bytes:-}"
  mkdir -p "$incoming"

  while :; do
    if download_dit_attempt "$url" "$incoming" "${expected_bytes:-}"; then
      downloaded_path="$DIT_DOWNLOADED_PATH"
      break
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "ERROR: giving up on $filename after $attempt attempt(s)." >&2
      echo "ERROR: the partial transfer is kept in $incoming so a re-run resumes it." >&2
      return 1
    fi
    echo "WARNING: download attempt $attempt/$max_attempts failed — retrying in $((attempt * 10))s (it resumes, it doesn't restart)." >&2
    sleep $((attempt * 10))
    attempt=$((attempt + 1))
  done
  echo "Downloaded $filename in $((SECONDS - t0))s."

  echo "Validating $filename against h3.validate_checkpoint()..."
  if ! validate_checkpoint_file "$downloaded_path"; then
    actual_bytes=$(stat -c%s "$downloaded_path" 2>/dev/null || echo 0)
    if [ -n "${expected_bytes:-}" ] && [ "${expected_bytes:-0}" -gt 0 ] && [ "$actual_bytes" -ne "$expected_bytes" ]; then
      echo "ERROR: $filename is $actual_bytes bytes but the server advertised $expected_bytes — the transfer is incomplete." >&2
      echo "ERROR: keeping it in $incoming so the next run resumes instead of starting over." >&2
    else
      # The bytes are all there and they still aren't an H3 checkpoint: re-downloading
      # cannot change that, so delete the staged copy instead of leaving something that
      # would make every future run retry a transfer that can never succeed.
      echo "ERROR: $filename transferred completely ($actual_bytes bytes) but is not a usable H3 FL2VA checkpoint." >&2
      echo "ERROR: re-downloading will not fix this — check that DIT_URL points at the right file." >&2
      rm -f "$downloaded_path"
    fi
    return 1
  fi

  mv "$downloaded_path" "$dest"
  rm -rf "$incoming"
  record_active_dit "$dest"
  echo "Installed validated checkpoint at $dest"
}

case "${1:-}" in
  stock)
    resolve_models_root
    acquire_fetch_lock
    fetch_stock
    ;;
  dit)
    [ -n "${2:-}" ] || { usage; exit 1; }
    resolve_models_root
    acquire_fetch_lock
    fetch_dit "$2" "${3:-}"
    ;;
  all)
    [ -n "${2:-}" ] || { usage; exit 1; }
    resolve_models_root
    acquire_fetch_lock
    fetch_stock
    fetch_dit "$2" "${3:-}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
