"""Regressão: visualização de voltas não altera dados nem depende de Pygame."""
import unittest

import numpy as np

from src.ui.algo_modes.dataset_lap_view import (
    split_laps,
    select_laps,
    selection_label,
)


class TestDatasetLapView(unittest.TestCase):
    def test_segmenta_incrementos_do_contador(self):
        laps = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        original = laps.copy()
        parts = split_laps(laps)
        self.assertEqual([p.number for p in parts], [1, 2, 3])
        self.assertEqual((parts[0].start, parts[0].stop), (0, 3))
        self.assertEqual((parts[1].start, parts[1].stop), (2, 6))
        self.assertEqual([p.complete for p in parts], [True, True, False])
        np.testing.assert_array_equal(laps, original)

    def test_selecao_individual_intervalo_e_todas(self):
        parts = split_laps(np.repeat(np.arange(10), 5))
        self.assertEqual(len(select_laps(parts, "all")), 10)
        self.assertEqual(select_laps(parts, "single", 4)[0].number, 4)
        self.assertEqual(
            [x.number for x in select_laps(parts, "range", 8, 5)],
            [8, 9, 10],
        )
        self.assertEqual(
            selection_label("single", select_laps(parts, "single", 1), len(parts)),
            "Volta 1/10",
        )

    def test_sem_marcador_nao_inventa_voltas(self):
        parts = split_laps(np.zeros(5))
        self.assertEqual(len(parts), 1)
        self.assertEqual(selection_label("single", parts, len(parts)),
                         "Trecho sem volta marcada")

    def test_amostras_invalidas_e_intervalos_vazios(self):
        self.assertEqual(split_laps(np.array([1])), ())
        self.assertEqual(split_laps(np.array([])), ())
        with self.assertRaises(ValueError):
            split_laps(np.array([[0, 1]]))
        with self.assertRaises(ValueError):
            select_laps((), "invalid")

    def test_contagem_realista_de_voltas(self):
        # 103 cruzamentos do marcador + trecho final ainda sem conclusão.
        laps = np.repeat(np.arange(104), 10)
        parts = split_laps(laps)
        self.assertEqual(len(parts), 104)
        self.assertEqual(sum(p.complete for p in parts), 103)
        self.assertEqual(parts[-1].number, 104)
        self.assertFalse(parts[-1].complete)
        self.assertIn("parcial", selection_label(
            "single", select_laps(parts, "single", 104), len(parts)))
        self.assertIn("parcial", selection_label(
            "range", select_laps(parts, "range", 101, 5), len(parts)))
        self.assertEqual(len(select_laps(parts, "range", 101, 5)), 4)


if __name__ == "__main__":
    unittest.main()
