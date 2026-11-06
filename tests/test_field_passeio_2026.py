"""Integração de regressão do ensaio de campo com 4 âncoras.

Os valores brutos do laboratório NÃO são armazenados nos testes.
O CSV sintético preserva o layout real: WR/WL, slot 0 ausente,
quatro âncoras nos slots 1..4, médias em cm.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.odometry.pcntlog_adapter import RawPcntRow, Robot2026Profile, convert_session
from src.ui.algo_modes.dataset_lap_view import split_time_windows
from src.ui.algo_modes.dataset_uwb2026_pipeline import (
    inspect_robot2026_capture,
    load_robot2026_single_tag_uwb,
    read_pcnt_anchor_slot_mapping,
    read_robot2026_profile_from_anchor_json,
    read_robot2026_view_settings,
    read_uwb2026_processing_settings,
)


class TestFieldFourAnchors2026(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.layout = root / "anchor_sets" / "passeio_01_4_ancoras.json"
        cls.route = root / "routes" / "passeio_01_rota_nominal.json"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.csv = Path(self.temp.name) / "passeio.csv"

    @staticmethod
    def row(t, wr, wl, k, offset=0, missing=None):
        values = [t, wr, wl, k, 0, 0, 0, -255]
        for j in range(4):
            m = -255 if missing == j else 210 + offset + 15*j
            values.extend([m, 4, 2])
        return ",".join(map(str, values))

    def save(self, rows):
        self.csv.write_text("\n".join(rows)+"\n",encoding="utf-8")
        return self.csv

    def test_anchor_and_matlab_route_coordinates(self):
        a = json.loads(self.layout.read_text())
        p = json.loads(self.route.read_text())
        self.assertEqual(a["pcnt_slot_to_anchor_index"], [None,0,1,2,3])
        np.testing.assert_allclose(a["anchors_xy"], [
            [.77,1.77,1.85],[0,0,1.63],[-2.66,.13,1.28],[-3.04,2.1,.86]
        ])
        self.assertEqual(len(p["waypoints"]),9)
        self.assertEqual(p["waypoints"][0],p["waypoints"][-1])
        self.assertNotIn("ground_truth_synced",p)

    def test_wheel_swap_and_sign_is_explicit_and_reversible(self):
        p = read_robot2026_profile_from_anchor_json(self.layout)
        self.assertTrue(p.swap_encoder_columns)
        self.assertTrue(p.invert_left)
        self.assertFalse(p.invert_right)
        row0 = RawPcntRow(100, 0, 0, 0, 0)
        row1 = RawPcntRow(350, 4560, -2280, 0, 0)
        new = convert_session([row0, row1], p)
        self.assertEqual(new[-1]["left_ticks"], 2280)
        self.assertEqual(new[-1]["right_ticks"], 4560)
        self.assertGreater(new[-1]["dtheta_rad"],0)
        old = convert_session([row0,row1],Robot2026Profile(ticks_per_wheel_rev=4560))
        self.assertEqual(old[-1]["left_ticks"],4560)
        self.assertEqual(old[-1]["right_ticks"],2280)
        self.assertLess(old[-1]["dtheta_rad"],0)

    def test_snapshot_mapping_and_four_fresh_ranges(self):
        rows = [
            self.row(100,0,0,0),
            self.row(350,300,-290,0,20),
            self.row(600,600,-580,0,20),
            self.row(850,900,-870,0,35),
            self.row(1100,1200,-1160,0,45,missing=1),
        ]
        self.save(rows)
        slots = read_pcnt_anchor_slot_mapping(self.layout,4)
        profile = read_robot2026_profile_from_anchor_json(self.layout)
        settings = read_uwb2026_processing_settings(self.layout)
        d = load_robot2026_single_tag_uwb(
            self.csv,session_id=1,anchor_count=4,
            slot_to_anchor_index=slots,profile=profile,**settings
        )
        self.assertEqual(d.odometry.sample_count,5)
        self.assertEqual(d.sample_count,3)  # índices 0,1,3
        np.testing.assert_array_equal(d.sample_indices,[0,1,3])
        self.assertEqual(d.distances.shape,(3,4))
        self.assertTrue(np.all(np.isfinite(d.distances)))
        np.testing.assert_allclose(d.distances[0],[2.10,2.25,2.40,2.55])
        self.assertEqual(d.repeated_readings,4)
        self.assertEqual(d.invalid_readings,1)

    def test_preflight_and_per_slot_record_periods(self):
        self.save([
            self.row(100,0,0,0),
            self.row(350,300,-290,0,20),
            self.row(600,600,-580,0,20),
            self.row(850,900,-870,0,35)
        ])
        report=inspect_robot2026_capture(
            self.csv,session_id=1,anchor_path=self.layout
        )
        self.assertEqual(report["status"],"ready_for_static_batch")
        self.assertEqual(report["new_ranges_required"],4)
        self.assertEqual(report["uwb_snapshots"],3)
        stats=report["uwb_slot_record_update_stats"]
        self.assertEqual(stats[0]["valid_new_readings"],0)
        self.assertEqual(stats[1]["anchor_index"],0)
        self.assertEqual(stats[1]["median_log_update_ms"],375.)

    def test_time_windows_not_fake_physical_laps(self):
        v=read_robot2026_view_settings(self.layout)
        self.assertEqual(v["segmentation"],"time_windows")
        t=np.linspace(0,772.45,3091)
        sections=split_time_windows(t,v["window_duration_s"])
        self.assertEqual(len(sections),10)
        self.assertEqual(sections[0].start,0)
        self.assertEqual(sections[-1].stop,len(t))
        self.assertEqual(sections[-1].firmware_count,-1)


if __name__ == "__main__":
    unittest.main()
