"""
Configuration constants and parameter classes for the pelvimetry pipeline.
"""

# Subprocess timeout for dcm2niix / TotalSegmentator
SUBPROCESS_TIMEOUT_SECONDS = 30 * 60  # 30 minutes


class PelvicConfig:
    """ISD / ITD detection parameters and quality-gate thresholds.

    Attributes
    ----------
    min_isd_mm, max_isd_mm : float
        Acceptable range for Inter-Spinous Distance (mm).
    min_itd_mm, max_itd_mm : float
        Acceptable range for Inter-Tuberous Distance (mm).
    search_margin_superior_mm, search_margin_inferior_mm : float
        Z-axis search margins relative to femur head (mm).
    sacrum_buffer_mm : float
        Buffer around sacrum for sacrum-presence check (mm).
    edge_margin_slices : int
        Number of edge slices excluded from valley/plateau selection.
    prominence_mm : float
        Minimum prominence for valley detection via ``find_peaks``.
    smoothing_window : int
        Kernel size for moving-average smoothing of the distance curve.
    plateau_slope_thresh : float
        Maximum absolute slope to qualify as a plateau region.
    plateau_min_len : int
        Minimum contiguous length of a plateau (slices).
    rotation_warn_deg, rotation_fail_deg : float
        Thresholds for axial rotation quality gate (degrees).
    tilt_warn_deg, tilt_fail_deg : float
        Thresholds for coronal tilt quality gate (degrees).
    sacrum_offset_warn_mm, sacrum_offset_fail_mm : float
        Thresholds for sacrum–symphysis midline offset (mm).
    """

    def __init__(
        self,
        min_isd_mm=70.0,
        max_isd_mm=130.0,
        min_itd_mm=60.0,
        max_itd_mm=200.0,
        search_margin_superior_mm=20.0,
        search_margin_inferior_mm=100.0,
        sacrum_buffer_mm=15.0,
        edge_margin_slices=2,
        prominence_mm=1.0,
        smoothing_window=3,
        plateau_slope_thresh=1.0,
        plateau_min_len=2,
        rotation_warn_deg=5.0,
        rotation_fail_deg=10.0,
        tilt_warn_deg=5.0,
        tilt_fail_deg=10.0,
        sacrum_offset_warn_mm=5.0,
        sacrum_offset_fail_mm=10.0,
    ):
        """Initialise detection parameters with sensible defaults.

        Parameters
        ----------
        min_isd_mm : float, optional
            Minimum acceptable ISD in mm (default 70.0).
        max_isd_mm : float, optional
            Maximum acceptable ISD in mm (default 130.0).
        min_itd_mm : float, optional
            Minimum acceptable ITD in mm (default 60.0).
        max_itd_mm : float, optional
            Maximum acceptable ITD in mm (default 200.0).
        search_margin_superior_mm : float, optional
            Superior search margin above femur head in mm (default 20.0).
        search_margin_inferior_mm : float, optional
            Inferior search margin below femur head in mm (default 100.0).
        sacrum_buffer_mm : float, optional
            Buffer for sacrum presence check in mm (default 15.0).
        edge_margin_slices : int, optional
            Edge slices excluded from selection (default 2).
        prominence_mm : float, optional
            Minimum prominence for valley detection (default 1.0).
        smoothing_window : int, optional
            Moving-average kernel size (default 3).
        plateau_slope_thresh : float, optional
            Max absolute slope for plateau detection (default 1.0).
        plateau_min_len : int, optional
            Minimum plateau length in slices (default 2).
        rotation_warn_deg : float, optional
            Warning threshold for axial rotation (default 5.0).
        rotation_fail_deg : float, optional
            Failure threshold for axial rotation (default 10.0).
        tilt_warn_deg : float, optional
            Warning threshold for coronal tilt (default 5.0).
        tilt_fail_deg : float, optional
            Failure threshold for coronal tilt (default 10.0).
        sacrum_offset_warn_mm : float, optional
            Warning threshold for sacrum–symphysis offset (default 5.0).
        sacrum_offset_fail_mm : float, optional
            Failure threshold for sacrum–symphysis offset (default 10.0).
        """
        self.min_isd_mm = min_isd_mm
        self.max_isd_mm = max_isd_mm
        self.min_itd_mm = min_itd_mm
        self.max_itd_mm = max_itd_mm
        self.search_margin_superior_mm = search_margin_superior_mm
        self.search_margin_inferior_mm = search_margin_inferior_mm
        self.sacrum_buffer_mm = sacrum_buffer_mm
        self.edge_margin_slices = edge_margin_slices
        self.prominence_mm = prominence_mm
        self.smoothing_window = smoothing_window
        self.plateau_slope_thresh = plateau_slope_thresh
        self.plateau_min_len = plateau_min_len
        self.rotation_warn_deg = rotation_warn_deg
        self.rotation_fail_deg = rotation_fail_deg
        self.tilt_warn_deg = tilt_warn_deg
        self.tilt_fail_deg = tilt_fail_deg
        self.sacrum_offset_warn_mm = sacrum_offset_warn_mm
        self.sacrum_offset_fail_mm = sacrum_offset_fail_mm


DEFAULT_PELVIC_CONFIG = PelvicConfig()
