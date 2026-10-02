"""#9：只检查掩码、非有限值和不可判定分支；合成数值不作项目指标。"""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1] / 'scripts/validate_wt_candidates.py'
spec = importlib.util.spec_from_file_location('wt_validation', path)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def window(values):
    return {'oe': np.array(values, dtype=float), 'mask': np.ones((3, 3), dtype=bool),
            'bin_ids': np.arange(3), 'window_edges': np.arange(4) * 100}


class PairedStatisticsTests(unittest.TestCase):
    def test_strict_upper_triangle_and_shared_mask(self):
        left = window([[999, 0, 1], [-9, 999, 2], [-9, -9, 999]])
        right = window([[-999, 1, 3], [9999, -999, 5], [9999, 9999, -999]])
        result, _ = validation.paired_statistics(left, right)
        self.assertEqual(result['valid_upper_pixels'], 3)
        self.assertAlmostEqual(result['replicate_correlation'], 1)
        right['mask'][0, 2] = False
        result, _ = validation.paired_statistics(left, right)
        self.assertEqual(result['valid_upper_pixels'], 2)
        self.assertEqual(result['excluded_upper_pixels'], 1)
        self.assertEqual(result['reproducibility_status'], 'unassessable')
        self.assertEqual(result['replicate_correlation'], '')

    def test_nonfinite_is_excluded_in_either_replicate(self):
        left = window([[0, np.nan, 1], [0, 0, 2], [0, 0, 0]])
        right = window([[0, 1, 3], [0, 0, np.inf], [0, 0, 0]])
        result, common = validation.paired_statistics(left, right)
        self.assertEqual(result['valid_upper_pixels'], 1)
        self.assertFalse(common[0, 1])
        self.assertFalse(common[1, 2])

    def test_constant_oe_is_not_called_reproduced(self):
        left = window(np.zeros((3, 3)))
        right = window([[0, 1, 2], [0, 0, 3], [0, 0, 0]])
        result, _ = validation.paired_statistics(left, right)
        self.assertEqual(result['valid_upper_pixels'], 3)
        self.assertEqual(result['reproducibility_status'], 'unassessable')
        self.assertIn('常量', result['reason'])

    def test_coordinate_mismatch_stops_comparison(self):
        left = window(np.zeros((3, 3)))
        right = window(np.zeros((3, 3)))
        right['bin_ids'][0] = 10
        with self.assertRaisesRegex(ValueError, '坐标'):
            validation.paired_statistics(left, right)


if __name__ == '__main__':
    unittest.main()
