"""条件比较的计数口径、共同掩码和描述性阈值检查。"""

import unittest

import numpy as np

from src.condition_differences import compare_structure


def example_inputs():
    structure = {
        "ID": "example", "type": "CHIN", "chrom": "NC_000913.3",
        "start": 50, "end": 250, "center": 150, "length_bp": 200,
    }
    samples = [
        {"sample_id": f"{condition}_rep{replicate}", "condition": condition, "replicate": replicate}
        for condition in ("WT", "DstpA", "DhnsDstpA") for replicate in (1, 2)
    ]
    windows = {
        sample["sample_id"]: {
            "bin_ids": np.arange(4), "intensity": np.full((4, 4), 2.0),
            "intensity_mask": np.ones((4, 4), dtype=bool),
            "oe": np.ones((4, 4)), "mask": ~np.eye(4, dtype=bool),
        }
        for sample in samples
    }
    return structure, windows, samples


class ConditionDifferencesTests(unittest.TestCase):
    def test_identical_and_doubled_samples(self):
        structure, windows, samples = example_inputs()
        for replicate in (1, 2):
            windows[f"DstpA_rep{replicate}"]["intensity"] *= 2
            windows[f"DstpA_rep{replicate}"]["oe"] *= 2
        doubled, same = compare_structure(structure, windows, samples)
        self.assertEqual(doubled["log2_fc"], 1)
        self.assertEqual(doubled["effect_label"], "increase_consistent")
        self.assertTrue(doubled["replicate_direction_agreement"])
        self.assertAlmostEqual(doubled["oe_mean_abs_difference"], np.log(3) - np.log(2))
        self.assertEqual(same["log2_fc"], 0)
        self.assertEqual(same["effect_label"], "within_reference")
        self.assertTrue(same["replicate_direction_agreement"])
        self.assertEqual(same["oe_mean_abs_difference"], 0)

    def test_opposite_replicates_and_wt_reference_band(self):
        structure, windows, samples = example_inputs()
        windows["DstpA_rep1"]["intensity"] *= 2
        windows["DstpA_rep2"]["intensity"] *= 0.5
        row = compare_structure(structure, windows, samples)[0]
        self.assertEqual(row["rep1_log2_fc"], 1)
        self.assertEqual(row["rep2_log2_fc"], -1)
        self.assertFalse(row["replicate_direction_agreement"])
        self.assertEqual(row["effect_label"], "inconsistent_or_borderline")
        self.assertAlmostEqual(row["log2_fc"], np.log2(1.25))
        windows["WT_rep2"]["intensity"] *= 2
        row = compare_structure(structure, windows, samples)[0]
        self.assertEqual(row["wt_rep_log2_ratio"], 1)
        self.assertEqual(row["reference_band"], 1)

    def test_shared_mask_and_fractional_boundary_weights(self):
        structure, windows, samples = example_inputs()
        # ROI 两端覆盖半个 bin，三对像素的原权重为 0.5、0.25、0.5。
        values = np.array([[999, 2, 10, 999], [2, 999, 6, 999],
                           [10, 6, 999, 999], [999, 999, 999, 999]], dtype=float)
        for window in windows.values():
            window["intensity"] = values.copy()
        rows = compare_structure(structure, windows, samples)
        self.assertAlmostEqual(rows[0]["wt_mean"], (2 * 0.5 + 10 * 0.25 + 6 * 0.5) / 1.25)
        self.assertEqual(rows[0]["roi_weight_sum"], 1.25)
        self.assertEqual(rows[0]["valid_pair_count"], 3)
        # 只有第三条件 rep2 无效，也必须从 WT 和两个比较条件统一排除。
        mask = windows["DhnsDstpA_rep2"]["intensity_mask"]
        mask[0, 2] = mask[2, 0] = False
        rows = compare_structure(structure, windows, samples)
        for row in rows:
            self.assertEqual(row["wt_mean"], 4)
            self.assertEqual(row["mutant_mean"], 4)
            self.assertEqual(row["valid_pair_count"], 2)
            self.assertEqual(row["valid_weight_fraction"], 0.8)

    def test_no_pairs_or_nonpositive_mean_are_not_called_changes(self):
        structure, windows, samples = example_inputs()
        windows["DstpA_rep1"]["intensity"][:] = 0
        rows = compare_structure(structure, windows, samples)
        self.assertEqual(rows[0]["status"], "nonpositive_intensity")
        self.assertEqual(rows[0]["effect_label"], "not_evaluable")
        self.assertTrue(np.isnan(rows[0]["log2_fc"]))
        self.assertEqual(rows[1]["status"], "ok")
        windows["WT_rep1"]["intensity_mask"][:] = False
        for row in compare_structure(structure, windows, samples):
            self.assertEqual(row["status"], "no_common_valid_pairs")
            self.assertEqual(row["valid_pair_count"], 0)
            self.assertEqual(row["valid_weight_fraction"], 0)
            self.assertTrue(np.isnan(row["wt_mean"]))
            self.assertTrue(np.isnan(row["oe_mean_abs_difference"]))

    def test_circular_bin_order_preserves_roi_and_oe_mask(self):
        structure, windows, samples = example_inputs()
        for window in windows.values():
            # 环绕窗口的局部顺序先出现染色体末端，再出现本 ROI 的 0/1/2 号 bin。
            window["bin_ids"] = np.array([46416, 0, 1, 2])
        for replicate in (1, 2):
            windows[f"DstpA_rep{replicate}"]["intensity"] *= 0.5
        rows = compare_structure(structure, windows, samples)
        self.assertEqual(rows[0]["log2_fc"], -1)
        self.assertEqual(rows[0]["effect_label"], "decrease_consistent")
        self.assertEqual(rows[0]["roi_weight_sum"], 1.25)
        self.assertEqual(rows[0]["valid_pair_count"], 3)
        windows["DhnsDstpA_rep2"]["mask"][:] = False
        for row in compare_structure(structure, windows, samples):
            self.assertEqual(row["status"], "ok")
            self.assertTrue(np.isnan(row["oe_mean_abs_difference"]))


if __name__ == "__main__":
    unittest.main()
