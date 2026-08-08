#!/bin/zsh
# Self-healing download + incremental repack, detached from any Claude session.
# Logs: ~/work/research/dsv4-streaming/logs/{download,repack}.log
# Done when repack.log ends with REPACK COMPLETE.

ROOT=~/work/research/dsv4-streaming
SNAP_GLOB=~/.cache/huggingface/hub/models--pipenetwork--DeepSeek-V4-Flash-MLX-mixed-4_8bit/snapshots
mkdir -p $ROOT/logs

download_loop() {
  export HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; print(get_token())")
  export HF_HUB_DISABLE_XET=1
  local n=0
  until hf download pipenetwork/DeepSeek-V4-Flash-MLX-mixed-4_8bit --max-workers 1; do
    n=$((n+1)); echo "== crashed (attempt $n) $(date +%H:%M), retry in 60s =="
    [ $n -gt 200 ] && return 1
    sleep 60
  done
  echo "DOWNLOAD COMPLETE $(date +%H:%M)"
}

repack_loop() {
  while true; do
    local snap=$(ls -d $SNAP_GLOB/*/ 2>/dev/null | head -1)
    local n=$(ls $snap/model-*.safetensors 2>/dev/null | wc -l | tr -d ' ')
    echo "$(date +%H:%M) shards complete: $n/33"
    if [ "$n" -gt 0 ]; then
      python3 $ROOT/streaming/repack.py --snapshot "$snap" --out "$ROOT/repacked" \
        --skip-resident 2>&1 | grep -E "written|not ready" | head -50
    fi
    if [ "$n" -eq 33 ]; then
      echo "final pass with resident tensors"
      python3 $ROOT/streaming/repack.py --snapshot "$snap" --out "$ROOT/repacked" 2>&1 | tail -20
      echo "REPACK COMPLETE $(date +%H:%M)"
      return 0
    fi
    sleep 900
  done
}

case "$1" in
  download) download_loop ;;
  repack)   repack_loop ;;
  *) echo "usage: overnight.sh download|repack"; exit 1 ;;
esac
