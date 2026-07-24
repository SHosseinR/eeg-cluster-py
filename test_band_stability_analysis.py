"""Focused tests for within-band and cross-band result analysis."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from band_stability_analysis import (
    build_analysis_tables,
    choose_conservative_winner,
    compute_band_summary,
    compute_bootstrap_rank_probabilities,
    compute_cross_band_tests,
    run_band_stability_analysis,
)
from optimization_config import OPTIMIZATION_OUTPUT_DIR
from plot_band_stability_analysis import align_subject_cohort, parse_args


def _synthetic_results(scales=None, n_subjects=8):
    scales = scales or {"delta": 0.65, "alpha": 0.35, "beta": 0.55}
    bands = list(scales)
    healthy_values = np.array([1.0, 2.0, 4.0])
    healthy = {"m1": 1.0, "m2": 2.0, "m3": 4.0}
    initial = np.array([1.5, 2.7, 5.2])
    results = {}
    for band_index, band in enumerate(bands):
        results[band] = {}
        for subject_index in range(n_subjects):
            scale = scales[band] + subject_index * 0.002
            final = healthy_values + (initial - healthy_values) * scale
            amplitude = (-1 if (subject_index + band_index) % 3 == 0 else 1) * (0.2 + subject_index * 0.03)
            solution = {
                "node": subject_index % 4,
                "band": band_index,
                "band_name": band,
                "stimulation_duration": 2.0 + subject_index * 0.1,
                "stimulation_amplitude": amplitude,
                "leak": 1.0,
                "objectives": np.full(3, scale),
                "measure_values": final.tolist(),
                "constraint_values": np.array([-0.2, -1.0]),
                "feasible": True,
                "raw_activation_ratio_min": 0.4,
                "raw_activation_ratio_max": 2.0,
            }
            infeasible = dict(solution)
            infeasible["constraint_values"] = np.array([0.5, -1.0])
            infeasible["feasible"] = False
            results[band][f"sub-{subject_index:02d}"] = {
                "best_solution": solution,
                "all_solutions": [solution, infeasible],
                "initial_metrics": initial.tolist(),
                "final_metrics": final.tolist(),
                "optimization_measures": ["m1", "m2", "m3"],
                "healthy_measure_baselines": healthy,
                "channel_display_names": ["F3", "F4", "C3", "C4"],
                "n_nodes": 4,
            }
    return results


class BandStabilityAnalysisTests(unittest.TestCase):
    def test_cli_uses_configured_results_dir_when_path_is_omitted(self):
        self.assertEqual(parse_args([]).results_dir, OPTIMIZATION_OUTPUT_DIR)
        self.assertEqual(parse_args(["explicit-results"]).results_dir, "explicit-results")

    def test_clear_winner_and_rank_probabilities(self):
        subject_df, _ = build_analysis_tables(_synthetic_results())
        summary = compute_band_summary(subject_df, n_resamples=200, random_seed=1)
        pairwise, omnibus = compute_cross_band_tests(subject_df)
        winner, _ = choose_conservative_winner(summary, pairwise, omnibus)
        self.assertEqual(winner, "alpha")

        ranks = compute_bootstrap_rank_probabilities(subject_df, n_resamples=200, random_seed=1)
        alpha = ranks[ranks["band"] == "alpha"].iloc[0]
        self.assertGreater(alpha["rank_1_probability"], 0.95)
        probability_columns = [column for column in ranks if column.startswith("rank_")]
        np.testing.assert_allclose(ranks[probability_columns].sum(axis=1), 1.0)

    def test_tied_bands_are_inconclusive(self):
        tied = _synthetic_results({"delta": 0.5, "alpha": 0.5, "beta": 0.5})
        subject_df, _ = build_analysis_tables(tied)
        summary = compute_band_summary(subject_df, n_resamples=100, random_seed=2)
        pairwise, omnibus = compute_cross_band_tests(subject_df)
        winner, _ = choose_conservative_winner(summary, pairwise, omnibus)
        self.assertEqual(winner, "inconclusive")

    def test_mismatched_cohorts_are_rejected(self):
        results = _synthetic_results()
        results["alpha"].pop("sub-00")
        with self.assertRaisesRegex(ValueError, "identical subject IDs"):
            build_analysis_tables(results, require_matched_subjects=True)

    def test_cli_intersection_policy_aligns_existing_band_cohorts(self):
        results = _synthetic_results()
        results["alpha"].pop("sub-00")
        results["beta"].pop("sub-01")
        aligned, summary = align_subject_cohort(results, policy="intersection")
        self.assertEqual({len(values) for values in aligned.values()}, {6})
        self.assertEqual(set.intersection(*(set(values) for values in aligned.values())), set(aligned["delta"]))
        self.assertEqual(set(summary["common_subject_count"]), {6})

    def test_cli_strict_policy_rejects_mismatch(self):
        results = _synthetic_results()
        results["alpha"].pop("sub-00")
        with self.assertRaisesRegex(ValueError, "Strict cohort policy"):
            align_subject_cohort(results, policy="strict")

    def test_zero_initial_distance_is_finite(self):
        results = _synthetic_results(n_subjects=2)
        for band_results in results.values():
            result = band_results["sub-00"]
            result["initial_metrics"] = [1.0, 2.0, 4.0]
            result["final_metrics"] = [1.0, 2.0, 4.0]
            result["best_solution"]["measure_values"] = [1.0, 2.0, 4.0]
        subject_df, _ = build_analysis_tables(results)
        self.assertTrue(np.isfinite(subject_df["relative_improvement"]).all())

    def test_complete_analysis_writes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            figures = os.path.join(directory, "figures")
            report = os.path.join(directory, "report.txt")
            output = run_band_stability_analysis(
                _synthetic_results(), directory, figures, report,
                n_resamples=50, random_seed=3,
            )
            self.assertEqual(output["decision"]["winner"], "alpha")
            for path in output["paths"].values():
                self.assertTrue(os.path.exists(path), path)
            expected_figures = {
                "delta_outcome_validity_dashboard.png",
                "delta_stimulation_profile_dashboard.png",
                "alpha_outcome_validity_dashboard.png",
                "alpha_stimulation_profile_dashboard.png",
                "beta_outcome_validity_dashboard.png",
                "beta_stimulation_profile_dashboard.png",
                "cross_band_efficacy_stability_dashboard.png",
                "cross_band_rank_probabilities.png",
            }
            self.assertEqual(set(os.listdir(figures)), expected_figures)

    def test_static_model_dashboard_uses_node_and_total_change(self):
        results = _synthetic_results(n_subjects=4)
        result_index = 0
        for band_results in results.values():
            for result in band_results.values():
                result["stimulation_model"] = "static_adjacency"
                solution = result["best_solution"]
                solution["stimulation_model"] = "static_adjacency"
                # Exercise bound-saturated optimizer output whose values differ
                # only by floating-point noise. Twenty ordinary bins cannot be
                # represented across this range.
                near_bound = (
                    3.0
                    if result_index % 2 == 0
                    else np.nextafter(3.0, 0.0)
                )
                result_index += 1
                solution["stimulation_amplitude"] = near_bound
                solution["stimulation_total_change"] = near_bound
                solution["stimulation_duration"] = None
                solution["leak"] = None
        with tempfile.TemporaryDirectory() as directory:
            figures = os.path.join(directory, "figures")
            output = run_band_stability_analysis(
                results,
                directory,
                figures,
                os.path.join(directory, "report.txt"),
                n_resamples=20,
                random_seed=4,
            )
            self.assertIn(
                "delta_stimulation_profile_dashboard.png",
                os.listdir(figures),
            )
            subject_table = pd.read_csv(output["paths"]["subject_table"])
            self.assertTrue(
                np.isfinite(subject_table["stimulation_total_change"]).all()
            )


if __name__ == "__main__":
    unittest.main()
