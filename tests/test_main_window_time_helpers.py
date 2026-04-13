"""Tests for time parsing helpers in main_window."""

from ui.main_window import _parse_time_string, _time_string_to_minutes


class TestMainWindowTimeHelpers:
    """Coverage for start-time helper parsing used by multiple widgets."""

    def test_parse_time_string_accepts_hh_mm_ss(self):
        t = _parse_time_string("13:47:09")
        assert t is not None
        assert t.hour() == 13
        assert t.minute() == 47
        assert t.second() == 9

    def test_parse_time_string_accepts_hh_mm_and_defaults_seconds(self):
        t = _parse_time_string("08:15")
        assert t is not None
        assert t.hour() == 8
        assert t.minute() == 15
        assert t.second() == 0

    def test_parse_time_string_rejects_invalid(self):
        assert _parse_time_string("not-a-time") is None

    def test_time_string_to_minutes_uses_hour_and_minute(self):
        assert _time_string_to_minutes("03:30:59") == 210

    def test_time_string_to_minutes_returns_none_for_invalid(self):
        assert _time_string_to_minutes(None) is None
