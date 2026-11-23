import tempfile
import unittest
from pathlib import Path

from utils.general import find_pkl_files


class UtilsStructureTest(unittest.TestCase):
    def test_graph_discovery_ignores_preabstraction_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_dir = Path(directory)
            (graph_dir / "bedroom_01.pkl").touch()
            (graph_dir / "bedroom_01_preabstraction.pkl").touch()
            (graph_dir / "gym_02.pkl").touch()

            self.assertEqual(
                find_pkl_files(graph_dir),
                ["bedroom_01", "gym_02"],
            )


if __name__ == "__main__":
    unittest.main()
