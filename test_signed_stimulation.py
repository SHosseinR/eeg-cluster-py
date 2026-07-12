"""Focused tests for configurable suppressive stimulation and safety constraints."""

import unittest

import numpy as np

from nsga_optimizer import EEGOptimizationProblem, NSGAIIOptimizer, ZeroAmplitudeAnchorSampling
from state_space_simulation import create_stimulation_signal, run_full_simulation


class SignedStimulationTests(unittest.TestCase):
    def test_negative_signal_is_preserved(self):
        signal = create_stimulation_signal(3, 1, duration=0.1, dt=0.01, amplitude=-0.4)
        np.testing.assert_allclose(signal[1], -0.4)
        np.testing.assert_allclose(signal[[0, 2]], 0.0)

    def test_raw_ratios_expose_saturation_before_clipping(self):
        adjacency = np.array([[0.0, 0.3, 0.1], [0.3, 0.0, 0.2], [0.1, 0.2, 0.0]])
        baseline = np.array([0.6, 0.7, 0.8])

        moderate = run_full_simulation(
            adjacency, baseline, 0, stimulation_duration=0.1,
            stimulation_amplitude=-0.1, dt=0.01, leak=1.0,
        )
        self.assertGreaterEqual(np.min(moderate["raw_activation_ratios"]), 0.1)
        self.assertLess(np.min(moderate["raw_activation_ratios"]), 1.0)

        extreme = run_full_simulation(
            adjacency, baseline, 0, stimulation_duration=10.0,
            stimulation_amplitude=-3.0, dt=0.01, leak=1.0,
        )
        self.assertLess(np.min(extreme["raw_activation_ratios"]), 0.1)
        self.assertEqual(float(np.min(extreme["activation_ratios"])), 0.1)

    def test_positive_negative_mixed_and_zero_width_bounds(self):
        def evaluate(node, band, duration, amplitude, leak):
            return np.array([amplitude ** 2])

        for bounds in [(0.1, 3.0), (-3.0, 0.0), (-3.0, 3.0), (0.0, 0.0)]:
            problem = EEGOptimizationProblem(
                3, 1, evaluate, duration_bounds=(1.0, 2.0),
                amplitude_bounds=bounds, leak_bounds=(0.0, 1.0),
                n_objectives=1, fixed_band_index=0,
            )
            self.assertEqual(float(problem.xl[2]), bounds[0])
            self.assertEqual(float(problem.xu[2]), bounds[1])
            midpoint = (bounds[0] + bounds[1]) / 2.0
            output = {}
            problem._evaluate(np.array([[1.0, 1.5, midpoint, 0.5]]), output)
            self.assertEqual(output["F"].shape, (1, 1))

        with self.assertRaisesRegex(ValueError, "Invalid amplitude bounds"):
            EEGOptimizationProblem(
                3, 1, evaluate, amplitude_bounds=(1.0, -1.0), fixed_band_index=0
            )

    def test_zero_anchor_is_inserted_only_when_zero_is_allowed(self):
        def evaluate(node, band, duration, amplitude, leak):
            return np.array([amplitude ** 2])

        mixed = EEGOptimizationProblem(
            3, 1, evaluate, amplitude_bounds=(-3.0, 3.0), fixed_band_index=0
        )
        samples = ZeroAmplitudeAnchorSampling(2)._do(
            mixed, 5, random_state=np.random.default_rng(7)
        )
        self.assertEqual(float(samples[0, 2]), 0.0)

        positive = EEGOptimizationProblem(
            3, 1, evaluate, amplitude_bounds=(0.1, 3.0), fixed_band_index=0
        )
        samples = ZeroAmplitudeAnchorSampling(2)._do(
            positive, 5, random_state=np.random.default_rng(7)
        )
        self.assertTrue(np.all(samples[:, 2] >= 0.1))

    def test_fixed_zero_range_runs_through_nsga(self):
        def evaluate(node, band, duration, amplitude, leak):
            return np.array([(duration - 1.5) ** 2 + amplitude ** 2])

        optimizer = NSGAIIOptimizer(
            n_nodes=3,
            n_bands=1,
            band_names=["alpha"],
            evaluate_func=evaluate,
            duration_bounds=(1.0, 2.0),
            amplitude_bounds=(0.0, 0.0),
            leak_bounds=(0.0, 1.0),
            n_objectives=1,
            population_size=10,
            n_generations=3,
            seed=4,
            fixed_band_index=0,
        )
        best_front, _ = optimizer.optimize(verbose=False)
        self.assertTrue(best_front)
        self.assertTrue(
            all(solution["stimulation_amplitude"] == 0.0 for solution in best_front)
        )

    def test_constrained_nsga_selects_only_safe_suppression(self):
        def evaluate(node, band, duration, amplitude, leak):
            raw_ratio = 1.0 + amplitude
            objectives = np.array([(amplitude + 2.0) ** 2])
            constraints = np.array([0.1 - raw_ratio, raw_ratio - 10.0])
            return objectives, constraints

        optimizer = NSGAIIOptimizer(
            n_nodes=3,
            n_bands=1,
            band_names=["alpha"],
            evaluate_func=evaluate,
            duration_bounds=(1.0, 2.0),
            amplitude_bounds=(-3.0, 0.0),
            leak_bounds=(0.0, 1.0),
            n_objectives=1,
            n_constraints=2,
            activation_ratio_bounds=(0.1, 10.0),
            population_size=20,
            n_generations=8,
            seed=5,
            fixed_band_index=0,
        )
        best_front, _ = optimizer.optimize(verbose=False)
        self.assertTrue(best_front)
        self.assertTrue(all(solution["feasible"] for solution in best_front))
        self.assertTrue(all(-0.9 - 1e-6 <= solution["stimulation_amplitude"] <= 0.0 for solution in best_front))
        self.assertTrue(any(not solution["feasible"] for solution in optimizer.all_solutions))


if __name__ == "__main__":
    unittest.main()
