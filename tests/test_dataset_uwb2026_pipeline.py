"""Regressão para o log PCNT+UWB 2026 de uma tag."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.ui.algo_modes.dataset_uwb2026_pipeline import (
    load_robot2026_single_tag_uwb, read_pcnt_anchor_slot_mapping,
)


class TestSingleTagUwb2026(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.anchors = self.root / "anchors.json"
        self.anchors.write_text(json.dumps({
            "anchors_xy": [[0, 0, 1], [2, 0, 1], [0, 2, 1], [2, 2, 1], [1, 1, 1]],
            "pcnt_slot_to_anchor_index": [0, 1, 2, 3, 4],
        }))
        self.mapping = read_pcnt_anchor_slot_mapping(self.anchors, 5)

    def write(self, rows):
        path = self.root / "pcntLog.csv"
        path.write_text("\n".join(rows) + "\n")
        return path

    def line(self, t, left, right, base=100.0, invalid_slot=None):
        values = [str(t), str(left), str(right), "0", "0"]
        for slot in range(5):
            avg = -255 if invalid_slot == slot else base + slot * 10
            values.extend([str(avg), "4.0", "2.0"])
        return ",".join(values)

    def test_converts_units_and_discards_repeated_snapshots(self):
        path = self.write([
            self.line(100, 0, 0),
            self.line(350, 1000, -1000),
            self.line(600, 2000, -2000, base=110),
        ])
        data = load_robot2026_single_tag_uwb(
            path, session_id=1, anchor_count=5, slot_to_anchor_index=self.mapping
        )
        self.assertEqual(data.odometry.sample_count, 3)
        np.testing.assert_array_equal(data.sample_indices, [0, 2])
        np.testing.assert_allclose(data.timestamps_s, [0, .5])
        self.assertAlmostEqual(data.distances[0, 0], 1)
        self.assertAlmostEqual(data.deviations[0, 0], .02)
        self.assertEqual(data.repeated_readings, 5)

    def test_rejects_old_five_column_log(self):
        path = self.write(["100,0,0,0,0", "350,10,-10,0,0"])
        with self.assertRaisesRegex(ValueError, "5 colunas"):
            load_robot2026_single_tag_uwb(
                path, session_id=1, anchor_count=5, slot_to_anchor_index=self.mapping
            )

    def test_requires_explicit_slot_mapping(self):
        self.anchors.write_text(json.dumps({"anchors_xy": [[0,0,1]] * 5}))
        with self.assertRaisesRegex(ValueError, "pcnt_slot_to_anchor_index"):
            read_pcnt_anchor_slot_mapping(self.anchors, 5)
        self.anchors.write_text(json.dumps({
            "pcnt_slot_to_anchor_index": [0, 0, 2, 3, 4]
        }))
        with self.assertRaisesRegex(ValueError, "mais de um slot"):
            read_pcnt_anchor_slot_mapping(self.anchors, 5)

    def test_slot_mapping_and_invalid_sentinels(self):
        path = self.write([
            self.line(100, 0, 0, base=200, invalid_slot=4),
            self.line(350, 100, -100, base=205, invalid_slot=4),
        ])
        data = load_robot2026_single_tag_uwb(
            path, session_id=1, anchor_count=5,
            slot_to_anchor_index=(3, 2, 1, 0, 4),
        )
        self.assertAlmostEqual(data.distances[0, 3], 2)
        self.assertAlmostEqual(data.distances[0, 0], 2.3)
        self.assertTrue(np.isnan(data.distances[0, 4]))

    def test_do_not_forge_synchronized_measurements(self):
        path = self.write([
            self.line(100, 0, 0),
            self.line(350, 100, -100),
        ])
        data = load_robot2026_single_tag_uwb(
            path, session_id=1, anchor_count=5, slot_to_anchor_index=self.mapping
        )
        self.assertEqual(data.sample_count, 1)
        self.assertEqual(data.odometry.sample_count, 2)

    def test_session_reset_separates_uwb_and_encoder(self):
        path = self.write([
            self.line(100, 0, 0), self.line(350, 20, -20, base=110),
            self.line(100, 0, 0, base=120), self.line(350, 20, -20, base=130),
        ])
        data = load_robot2026_single_tag_uwb(
            path, session_id=2, anchor_count=5, slot_to_anchor_index=self.mapping
        )
        self.assertEqual(data.odometry.session_id, 2)
        self.assertEqual(data.odometry.sample_count, 2)
        self.assertEqual(data.sample_count, 2)


if __name__ == "__main__":
    unittest.main()
