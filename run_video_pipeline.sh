#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

jobs=${MAX_PARALLEL_JOBS:-4}
if ! [[ $jobs =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_PARALLEL_JOBS must be a positive integer." >&2
    exit 2
fi

video_ids() {
    python - "$@" <<'PY'
import json
import sys

with open("data/robot.json", encoding="utf-8") as source:
    benchmark = json.load(source)
videos = dict.fromkeys(sys.argv[1:] or benchmark)
unknown = videos.keys() - benchmark.keys()
if unknown:
    sys.exit("Unknown Robot video(s): " + ", ".join(sorted(unknown)))
if not videos:
    sys.exit("No videos in data/robot.json")
for video in videos:
    print(video, end="\0")
PY
}

video_ids "$@" | xargs -0 -r -n 1 -P "$jobs" sh -c '
    video=$1
    for stage in preprocessing/prepare_video.py process_full_video.py reason.py; do
        if ! python "$stage" "$video"; then
            printf "%s failed at %s\n" "$video" "$stage" >&2
            exit 1
        fi
    done
' cam-video
