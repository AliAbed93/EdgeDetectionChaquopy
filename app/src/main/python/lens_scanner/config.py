"""
Configuration parameters for the lens scanning pipeline.

All parameters are tuned for edge-lit lens detection on matte black background.
These values represent reasonable defaults for MVP prototype testing.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class ScannerConfig:
    """
    Configuration for the lens scanner pipeline.
    
    Attributes:
        # Reference marker detection
        marker_type: Type of reference markers ('circle', 'aruco', 'checkerboard')
        marker_known_distance_mm: Known distance between reference markers in mm
        min_markers_required: Minimum markers needed for calibration
        
        # Image preprocessing
        gaussian_blur_kernel: Kernel size for Gaussian blur (must be odd)
        
        # Edge detection (Canny)
        canny_low_threshold: Lower threshold for Canny edge detection
        canny_high_threshold: Upper threshold for Canny edge detection
        
        # Morphological operations
        morph_kernel_size: Kernel size for morphological closing
        morph_iterations: Number of morphological iterations
        
        # Contour filtering
        min_contour_area_ratio: Minimum contour area as ratio of image area
        max_contour_area_ratio: Maximum contour area as ratio of image area
        min_contour_circularity: Minimum circularity (0-1) for lens-like shapes
        
        # Subpixel refinement
        subpixel_window_size: Window size for subpixel corner refinement
        subpixel_iterations: Max iterations for subpixel refinement
        subpixel_epsilon: Convergence epsilon for subpixel refinement
        
        # Spline smoothing
        spline_smoothing_factor: Smoothing factor for B-spline fitting
        spline_num_points: Number of points in final smoothed contour
        
        # SVG output
        svg_stroke_width_mm: Stroke width in SVG output (mm)
        svg_decimal_precision: Decimal places for SVG coordinates
    """
    
    # Reference marker detection
    marker_type: str = "circle"
    marker_known_distance_mm: float = 100.0  # Distance between markers
    min_markers_required: int = 4
    marker_min_area: int = 500  # Minimum pixel area for marker detection
    marker_max_area: int = 50000  # Maximum pixel area for marker detection
    marker_circularity_threshold: float = 0.7  # Minimum circularity for circle markers
    
    # Image preprocessing
    gaussian_blur_kernel: int = 5
    
    # Edge detection (Canny) - tuned for edge-lit bright edges on dark background
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    
    # Morphological operations
    morph_kernel_size: int = 3
    morph_iterations: int = 2
    
    # Contour filtering
    min_contour_area_ratio: float = 0.01  # At least 1% of image
    max_contour_area_ratio: float = 0.5   # At most 50% of image
    min_contour_circularity: float = 0.3  # Lens shapes are roughly elliptical
    contour_approx_epsilon_ratio: float = 0.001  # For polygon approximation
    
    # Subpixel refinement
    subpixel_window_size: int = 5
    subpixel_iterations: int = 100
    subpixel_epsilon: float = 0.001
    
    # Spline smoothing
    spline_smoothing_factor: float = 0.0  # 0 = interpolating spline
    spline_num_points: int = 500  # Points in final contour
    
    # SVG output
    svg_stroke_width_mm: float = 0.1
    svg_decimal_precision: int = 3
    
    # Debug options
    save_debug_images: bool = False
    debug_output_dir: Optional[str] = None


@dataclass
class CalibrationResult:
    """
    Result of reference marker calibration.
    
    Attributes:
        scale_mm_per_pixel: Conversion factor from pixels to millimeters
        perspective_matrix: 3x3 homography matrix for perspective correction (or None)
        markers_detected: Number of markers successfully detected
        calibration_error_mm: Estimated calibration error in mm
        is_valid: Whether calibration was successful
    """
    scale_mm_per_pixel: float = 0.0
    perspective_matrix: Optional[any] = None  # numpy array
    markers_detected: int = 0
    calibration_error_mm: float = float('inf')
    is_valid: bool = False


@dataclass 
class ContourResult:
    """
    Result of contour extraction.
    
    Attributes:
        contour_pixels: Raw contour points in pixel coordinates
        contour_mm: Contour points in millimeter coordinates
        perimeter_mm: Total perimeter length in mm
        area_mm2: Enclosed area in mm²
        bounding_box_mm: (x, y, width, height) in mm
        centroid_mm: (x, y) center point in mm
        is_closed: Whether contour forms a closed loop
    """
    contour_pixels: any = None  # numpy array Nx2
    contour_mm: any = None  # numpy array Nx2
    perimeter_mm: float = 0.0
    area_mm2: float = 0.0
    bounding_box_mm: Tuple[float, float, float, float] = (0, 0, 0, 0)
    centroid_mm: Tuple[float, float] = (0, 0)
    is_closed: bool = False
