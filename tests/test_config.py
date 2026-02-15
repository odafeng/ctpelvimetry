"""Tests for ctpelvimetry.config."""

from ctpelvimetry.config import PelvicConfig, DEFAULT_PELVIC_CONFIG, SUBPROCESS_TIMEOUT_SECONDS


class TestPelvicConfigDefaults:
    """PelvicConfig should initialise with documented defaults."""

    def test_default_isd_range(self):
        cfg = PelvicConfig()
        assert cfg.min_isd_mm == 70.0
        assert cfg.max_isd_mm == 130.0

    def test_default_itd_range(self):
        cfg = PelvicConfig()
        assert cfg.min_itd_mm == 60.0
        assert cfg.max_itd_mm == 200.0

    def test_default_search_margins(self):
        cfg = PelvicConfig()
        assert cfg.search_margin_superior_mm == 20.0
        assert cfg.search_margin_inferior_mm == 100.0

    def test_default_quality_gates(self):
        cfg = PelvicConfig()
        assert cfg.rotation_warn_deg == 5.0
        assert cfg.rotation_fail_deg == 10.0
        assert cfg.tilt_warn_deg == 5.0
        assert cfg.tilt_fail_deg == 10.0

    def test_default_sacrum_offset(self):
        cfg = PelvicConfig()
        assert cfg.sacrum_offset_warn_mm == 5.0
        assert cfg.sacrum_offset_fail_mm == 10.0


class TestPelvicConfigCustom:
    """PelvicConfig should accept custom overrides."""

    def test_custom_isd(self):
        cfg = PelvicConfig(min_isd_mm=50.0, max_isd_mm=150.0)
        assert cfg.min_isd_mm == 50.0
        assert cfg.max_isd_mm == 150.0

    def test_custom_smoothing(self):
        cfg = PelvicConfig(smoothing_window=5, plateau_min_len=4)
        assert cfg.smoothing_window == 5
        assert cfg.plateau_min_len == 4


class TestModuleConstants:
    """Module-level constants should be set."""

    def test_subprocess_timeout(self):
        assert SUBPROCESS_TIMEOUT_SECONDS == 30 * 60

    def test_default_config_is_instance(self):
        assert isinstance(DEFAULT_PELVIC_CONFIG, PelvicConfig)
