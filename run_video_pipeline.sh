#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$#" -eq 0 ]; then
    set -- bedroom_01
fi
for video in "$@"; do
    python -m preprocessing.prepare_video "$video"
    python process_full_video.py "$video"
    python reason.py "$video"
done
