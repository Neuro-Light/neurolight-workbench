"""Tests for settings_dialog module."""

import matplotlib

matplotlib.use("Agg")

from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from ui.settings_dialog import ROI_LABELS, THEME_VALUES, SettingsDialog


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


class TestSettingsDialogConstants:
    """Tests for module constants."""

    def test_theme_values_contains_dark(self):
        theme_keys = [t[0] for t in THEME_VALUES]
        assert "dark" in theme_keys

    def test_theme_values_contains_light(self):
        theme_keys = [t[0] for t in THEME_VALUES]
        assert "light" in theme_keys

    def test_theme_values_contains_high_contrast_variants(self):
        theme_keys = [t[0] for t in THEME_VALUES]
        assert "dark_high_contrast" in theme_keys
        assert "light_high_contrast" in theme_keys

    def test_theme_values_has_labels(self):
        for value, label in THEME_VALUES:
            assert isinstance(value, str)
            assert isinstance(label, str)
            assert len(label) > 0

    def test_roi_labels_contains_roi_1(self):
        assert "roi_1" in ROI_LABELS
        assert ROI_LABELS["roi_1"] == "ROI 1"

    def test_roi_labels_contains_roi_2(self):
        assert "roi_2" in ROI_LABELS
        assert ROI_LABELS["roi_2"] == "ROI 2"


class TestSettingsDialogInit:
    """Tests for dialog initialization."""

    def test_dialog_has_title(self, app):
        dialog = SettingsDialog()
        assert dialog.windowTitle() == "Preferences"

    def test_dialog_is_modal(self, app):
        dialog = SettingsDialog()
        assert dialog.isModal() is True

    def test_dialog_has_minimum_width(self, app):
        dialog = SettingsDialog()
        assert dialog.minimumWidth() >= 400

    def test_dialog_has_theme_radios(self, app):
        dialog = SettingsDialog()
        assert len(dialog.theme_radios) == len(THEME_VALUES)

    def test_dialog_has_current_theme_selected(self, app):
        with patch("ui.settings_dialog.get_theme", return_value="dark"):
            dialog = SettingsDialog()
            assert dialog.theme_radios["dark"].isChecked() is True

    def test_dialog_roi_swatches_exist(self, app):
        dialog = SettingsDialog()
        assert "roi_1" in dialog._roi_swatches
        assert "roi_2" in dialog._roi_swatches

    def test_dialog_roi_colors_loaded(self, app):
        dialog = SettingsDialog()
        assert "roi_1" in dialog._roi_colors
        assert "roi_2" in dialog._roi_colors


class TestSettingsDialogSetSwatchColor:
    """Tests for the _set_swatch_color static method."""

    def test_sets_pixmap_on_swatch(self, app):
        from PySide6.QtWidgets import QLabel

        swatch = QLabel()
        SettingsDialog._set_swatch_color(swatch, "#ff0000")
        pixmap = swatch.pixmap()
        assert pixmap is not None
        assert pixmap.width() == 24
        assert pixmap.height() == 24

    def test_accepts_various_color_formats(self, app):
        from PySide6.QtWidgets import QLabel

        swatch = QLabel()
        # Should not raise for valid colors
        SettingsDialog._set_swatch_color(swatch, "#ff0000")
        SettingsDialog._set_swatch_color(swatch, "#00ff00")
        SettingsDialog._set_swatch_color(swatch, "#0000ff")
        SettingsDialog._set_swatch_color(swatch, "#ffffff")
        SettingsDialog._set_swatch_color(swatch, "#000000")


class TestSettingsDialogThemeSelection:
    """Tests for theme radio button behavior."""

    def test_only_one_theme_selected(self, app):
        dialog = SettingsDialog()
        checked_count = sum(1 for radio in dialog.theme_radios.values() if radio.isChecked())
        assert checked_count == 1

    def test_selecting_theme_deselects_others(self, app):
        dialog = SettingsDialog()
        dialog.theme_radios["light"].setChecked(True)
        assert dialog.theme_radios["light"].isChecked() is True
        assert dialog.theme_radios["dark"].isChecked() is False


class TestSettingsDialogApply:
    """Tests for apply and accept behavior."""

    def test_apply_saves_theme(self, app):
        with patch("ui.settings_dialog.set_theme") as mock_set_theme:
            with patch("ui.settings_dialog.set_roi_color"):
                with patch("ui.settings_dialog.set_avg_trajectory_color"):
                    with patch("ui.settings_dialog.set_avg_trajectory_roi_color"):
                        with patch("ui.settings_dialog.set_peak_marker_color"):
                            with patch("ui.settings_dialog.set_trough_marker_color"):
                                dialog = SettingsDialog()
                                dialog.theme_radios["light"].setChecked(True)
                                dialog._apply_and_accept()
                                mock_set_theme.assert_called_once_with("light")

    def test_apply_saves_roi_colors(self, app):
        with patch("ui.settings_dialog.set_theme"):
            with patch("ui.settings_dialog.set_roi_color") as mock_set_roi:
                with patch("ui.settings_dialog.set_avg_trajectory_color"):
                    with patch("ui.settings_dialog.set_avg_trajectory_roi_color"):
                        with patch("ui.settings_dialog.set_peak_marker_color"):
                            with patch("ui.settings_dialog.set_trough_marker_color"):
                                dialog = SettingsDialog()
                                dialog._roi_colors["roi_1"] = "#aabbcc"
                                dialog._apply_and_accept()
                                mock_set_roi.assert_any_call("roi_1", "#aabbcc")

    def test_apply_saves_marker_colors(self, app):
        with patch("ui.settings_dialog.set_theme"):
            with patch("ui.settings_dialog.set_roi_color"):
                with patch("ui.settings_dialog.set_avg_trajectory_color"):
                    with patch("ui.settings_dialog.set_avg_trajectory_roi_color"):
                        with patch("ui.settings_dialog.set_peak_marker_color") as mock_peak:
                            with patch("ui.settings_dialog.set_trough_marker_color") as mock_trough:
                                dialog = SettingsDialog()
                                dialog._peak_marker_color = "#112233"
                                dialog._trough_marker_color = "#445566"
                                dialog._apply_and_accept()
                                mock_peak.assert_called_once_with("#112233")
                                mock_trough.assert_called_once_with("#445566")
