import unittest

import mne
import numpy as np

from plot_weighted_selection_target_3d import (
    _channel_positions,
    _channel_positions_2d,
    _display_labels,
    _project_topdown,
)


class ChannelPositionTests(unittest.TestCase):
    def test_exact_labels_are_unchanged(self):
        expected = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
        actual = _channel_positions(["Fp1", "T7"], {})
        np.testing.assert_allclose(actual[0], expected["Fp1"])
        np.testing.assert_allclose(actual[1], expected["T7"])

    def test_legacy_linked_ear_suffix_is_removed_as_fallback(self):
        expected = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
        actual = _channel_positions(["Fp1-LE", "T3-LE"], {})
        np.testing.assert_allclose(actual[0], expected["Fp1"])
        np.testing.assert_allclose(actual[1], expected["T3"])

    def test_bipolar_reference_uses_electrode_midpoint(self):
        expected = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
        actual = _channel_positions(["A2-A1"], {})[0]
        np.testing.assert_allclose(actual, (expected["A2"] + expected["A1"]) / 2.0)

    def test_topdown_projection_preserves_eeg_orientation(self):
        positions = _channel_positions(["Fp1", "Fp2", "O1", "Cz"], {})
        projected = _project_topdown(positions)
        self.assertLess(projected[0, 0], 0.0)
        self.assertGreater(projected[1, 0], 0.0)
        self.assertGreater(projected[0, 1], 0.0)
        self.assertLess(projected[2, 1], 0.0)
        self.assertLess(np.linalg.norm(projected[3]), 0.1)

    def test_legacy_suffixes_are_removed_only_for_display(self):
        self.assertEqual(
            _display_labels(["Fp1-LE", "Cz-REF", "Pz-AVG", "A2-A1"]),
            ["Fp1", "Cz", "Pz", "A2-A1"],
        )

    def test_bipolar_reference_is_drawn_at_active_electrode_on_2d_map(self):
        bipolar = _channel_positions_2d(["A2-A1"], {})[0]
        active = _project_topdown(_channel_positions(["A2"], {}))[0]
        np.testing.assert_allclose(bipolar, active)


if __name__ == "__main__":
    unittest.main()
