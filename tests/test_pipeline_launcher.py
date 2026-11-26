import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("preprocessing/prepare_video.py", "process_full_video.py", "reason.py")
FAKE_STAGE = """\
import os
import sys
import time
from pathlib import Path

video, stage = sys.argv[1], Path(__file__).name
calls = Path("calls")
with calls.open("a") as log:
    print(video, stage, file=log)
if stage == "prepare_video.py":
    if video == os.environ.get("FAIL_VIDEO"):
        sys.exit(1)
    # Hold the first workers until all expected slots are occupied.
    deadline = time.monotonic() + 5
    while calls.read_text().count("prepare_video.py") < int(os.environ["WAIT_FOR_WORKERS"]):
        if time.monotonic() >= deadline:
            sys.exit("Timed out waiting for parallel workers")
        time.sleep(0.02)
"""


class PipelineLauncherTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        shutil.copy(ROOT / "run_video_pipeline.sh", self.root)
        (self.root / "python").symlink_to(sys.executable)
        (self.root / "data").mkdir()
        self.videos = ["a", "b", "c", "d", "e"]
        (self.root / "data/robot.json").write_text(json.dumps(dict.fromkeys(self.videos)))
        for stage in STAGES:
            path = self.root / stage
            path.parent.mkdir(exist_ok=True)
            path.write_text(FAKE_STAGE)

    def launch(self, *videos, jobs=None, wait_for=0, fail=""):
        env = dict(
            os.environ,
            PATH=str(self.root) + os.pathsep + os.environ["PATH"],
            WAIT_FOR_WORKERS=str(wait_for),
            FAIL_VIDEO=fail,
        )
        env.pop("MAX_PARALLEL_JOBS", None)
        if jobs is not None:
            env["MAX_PARALLEL_JOBS"] = jobs
        return subprocess.run(
            ["bash", self.root / "run_video_pipeline.sh", *videos],
            env=env, capture_output=True, text=True, timeout=30,
        )

    def assert_parallel_run(self, limit, jobs=None):
        result = self.launch(jobs=jobs, wait_for=limit)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [line.split() for line in (self.root / "calls").read_text().splitlines()]
        active = set()
        peak = 0
        for video, stage in calls:
            if stage == "prepare_video.py":
                active.add(video)
                peak = max(peak, len(active))
            elif stage == "reason.py":
                active.remove(video)
        self.assertEqual(peak, limit)
        for video in self.videos:
            self.assertEqual([stage for name, stage in calls if name == video],
                             [Path(stage).name for stage in STAGES])

    def test_full_benchmark_defaults_to_four_parallel_videos(self):
        self.assert_parallel_run(4)

    def test_parallel_limit_can_be_changed(self):
        self.assert_parallel_run(2, jobs="2")

    def test_serial_subset_deduplicates_and_reports_a_failed_video(self):
        result = self.launch("b", "a", "a", jobs="1", fail="b")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("b failed at preprocessing/prepare_video.py", result.stderr)
        self.assertEqual((self.root / "calls").read_text().splitlines(),
                         ["b prepare_video.py", "a prepare_video.py", "a process_full_video.py", "a reason.py"])

    def test_invalid_selection_starts_no_workers(self):
        for videos, jobs in [(("a", "unknown"), "2"), (("a",), "0")]:
            with self.subTest(videos=videos, jobs=jobs):
                self.assertNotEqual(self.launch(*videos, jobs=jobs).returncode, 0)
                self.assertFalse((self.root / "calls").exists())
