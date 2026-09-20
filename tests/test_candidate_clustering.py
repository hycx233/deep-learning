import unittest

import pandas as pd

from src.candidate_clustering import assign_independent_loci, summarize_clusters


class CandidateClusteringTests(unittest.TestCase):
    def test_overlapping_windows_are_one_independent_locus(self) -> None:
        candidates = pd.DataFrame(
            {
                "chrom": ["chr", "chr", "chr"],
                "start": [0, 100, 1000],
                "end": [500, 600, 1200],
            }
        )
        loci = assign_independent_loci(candidates)
        self.assertEqual(loci.nunique(), 2)
        self.assertEqual(loci.iloc[0], loci.iloc[1])

    def test_noise_is_not_summarized_as_a_cluster(self) -> None:
        members = pd.DataFrame(
            {
                "cluster": [-1, 0, 0],
                "is_candidate": [True, True, True],
                "is_known_reference": [False, False, False],
                "known_overlap": [False, False, False],
                "independent_locus_id": ["a", "b", "c"],
            }
        )
        summary = summarize_clusters(members)
        self.assertEqual(summary["cluster"].tolist(), [0])

        all_noise = members.assign(cluster=-1)
        self.assertTrue(summarize_clusters(all_noise).empty)

    def test_new_cluster_rule_requires_five_clean_independent_loci(self) -> None:
        members = pd.DataFrame(
            {
                "cluster": [2] * 5,
                "is_candidate": [True] * 5,
                "is_known_reference": [False] * 5,
                "known_overlap": [False] * 5,
                "independent_locus_id": list("abcde"),
            }
        )
        summary = summarize_clusters(members)
        self.assertTrue(bool(summary.loc[0, "suspected_new_cluster"]))

        members.loc[0, "known_overlap"] = True
        summary = summarize_clusters(members)
        self.assertFalse(bool(summary.loc[0, "suspected_new_cluster"]))


if __name__ == "__main__":
    unittest.main()
