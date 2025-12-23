"""
Reference Marker Detection and Calibration Module.

Handles detection of reference markers placed around the lens to:
1. Determine pixel-to-millimeter scale
2. Correct perspective distortion if camera is not perfectly orthogonal

Supported marker types:
- Circle markers (default): Simple white/bright circles on dark background
- ArUco markers: For more robust detection in challenging conditions

Design assumptions:
- Markers are placed OUTSIDE the lens area (never under the glass)
- Markers form a known geometric pattern (e.g., rectangle corners)
- At least 4 markers for perspective correction, 2 for scale only
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from .config import ScannerConfig, CalibrationResult


class ReferenceMarkerDetector:
    """
    Detects reference markers and computes calibration parameters.
    """
    
    def __init__(self, config: ScannerConfig):
        self.config = config
        
    def detect_and_calibrate(
        self, 
        image: np.ndarray,
        expected_marker_positions_mm: Optional[np.ndarray] = None
    ) -> CalibrationResult:
        """
        Detect reference markers and compute calibration.
        
        Args:
            image: Input BGR image
            expected_marker_positions_mm: Known marker positions in mm (Nx2 array)
                If None, assumes a standard rectangular pattern based on config
                
        Returns:
            CalibrationResult with scale and perspective correction
        """
        if self.config.marker_type == "circle":
            markers_px = self._detect_circle_markers(image)
        elif self.config.marker_type == "aruco":
            markers_px = self._detect_aruco_markers(image)
        else:
            raise ValueError(f"Unknown marker type: {self.config.marker_type}")
        
        if len(markers_px) < 2:
            return CalibrationResult(
                is_valid=False,
                markers_detected=len(markers_px)
            )
        
        # Sort markers to establish correspondence with expected positions
        markers_px = self._sort_markers_clockwise(markers_px)
        
        # Generate expected positions if not provided
        if expected_marker_positions_mm is None:
            expected_marker_positions_mm = self._generate_default_marker_positions()
        
        # Compute calibration based on number of markers
        if len(markers_px) >= 4:
            return self._calibrate_with_perspective(
                markers_px[:4], 
                expected_marker_positions_mm[:4]
            )
        else:
            return self._calibrate_scale_only(
                markers_px[:2],
                expected_marker_positions_mm[:2]
            )
    
    def _detect_circle_markers(self, image: np.ndarray) -> np.ndarray:
        """
        Detect circular reference markers using blob detection.
        
        Assumes bright circles on dark background (edge-lit setup).
        
        Returns:
            Nx2 array of marker center coordinates in pixels
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to isolate bright markers
        # Using Otsu's method for automatic threshold selection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        markers = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.config.marker_min_area or area > self.config.marker_max_area:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < self.config.marker_circularity_threshold:
                continue
            
            # Compute centroid using moments
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            
            # Refine center using minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Use moment-based center (more accurate for partial occlusions)
            markers.append([cx, cy])
        
        return np.array(markers, dtype=np.float32) if markers else np.array([])
    
    def _detect_aruco_markers(self, image: np.ndarray) -> np.ndarray:
        """
        Detect ArUco markers for more robust calibration.
        
        Returns:
            Nx2 array of marker center coordinates in pixels
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Use ArUco dictionary (4x4 markers, 50 IDs)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is None or len(ids) == 0:
            return np.array([])
        
        # Extract center of each marker
        markers = []
        for corner in corners:
            # corner shape is (1, 4, 2) - 4 corners of the marker
            center = corner[0].mean(axis=0)
            markers.append(center)
        
        return np.array(markers, dtype=np.float32)
    
    def _sort_markers_clockwise(self, markers: np.ndarray) -> np.ndarray:
        """
        Sort markers in clockwise order starting from top-left.
        
        This establishes consistent correspondence with expected positions.
        """
        if len(markers) < 2:
            return markers
        
        # Compute centroid
        centroid = markers.mean(axis=0)
        
        # Compute angles from centroid
        angles = np.arctan2(
            markers[:, 1] - centroid[1],
            markers[:, 0] - centroid[0]
        )
        
        # Sort by angle (clockwise from top-left means starting from -135 degrees)
        # Adjust angles so top-left is first
        adjusted_angles = (angles + np.pi * 0.75) % (2 * np.pi)
        sorted_indices = np.argsort(adjusted_angles)
        
        return markers[sorted_indices]
    
    def _generate_default_marker_positions(self) -> np.ndarray:
        """
        Generate default marker positions assuming a rectangular pattern.
        
        Markers are assumed to be at corners of a rectangle with
        known_distance_mm as the diagonal or side length.
        """
        d = self.config.marker_known_distance_mm
        
        # Assume square pattern with side length = known_distance
        # Positions: top-left, top-right, bottom-right, bottom-left
        return np.array([
            [0, 0],
            [d, 0],
            [d, d],
            [0, d]
        ], dtype=np.float32)
    
    def _calibrate_with_perspective(
        self,
        markers_px: np.ndarray,
        markers_mm: np.ndarray
    ) -> CalibrationResult:
        """
        Compute full calibration with perspective correction using 4+ markers.
        
        Uses homography to map from pixel coordinates to mm coordinates.
        """
        # Compute homography matrix
        H, mask = cv2.findHomography(markers_px, markers_mm, cv2.RANSAC, 5.0)
        
        if H is None:
            return CalibrationResult(
                is_valid=False,
                markers_detected=len(markers_px)
            )
        
        # Compute scale as average of x and y scaling factors
        # Extract from homography matrix
        scale_x = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
        scale_y = np.sqrt(H[0, 1]**2 + H[1, 1]**2)
        scale = (scale_x + scale_y) / 2
        
        # Compute reprojection error
        markers_px_h = np.hstack([markers_px, np.ones((len(markers_px), 1))])
        projected = (H @ markers_px_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        error = np.sqrt(((projected - markers_mm)**2).sum(axis=1)).mean()
        
        return CalibrationResult(
            scale_mm_per_pixel=scale,
            perspective_matrix=H,
            markers_detected=len(markers_px),
            calibration_error_mm=error,
            is_valid=True
        )
    
    def _calibrate_scale_only(
        self,
        markers_px: np.ndarray,
        markers_mm: np.ndarray
    ) -> CalibrationResult:
        """
        Compute scale-only calibration using 2 markers.
        
        No perspective correction - assumes camera is orthogonal to surface.
        """
        # Compute distance between markers in pixels and mm
        dist_px = np.linalg.norm(markers_px[1] - markers_px[0])
        dist_mm = np.linalg.norm(markers_mm[1] - markers_mm[0])
        
        if dist_px < 1e-6:
            return CalibrationResult(
                is_valid=False,
                markers_detected=len(markers_px)
            )
        
        scale = dist_mm / dist_px
        
        return CalibrationResult(
            scale_mm_per_pixel=scale,
            perspective_matrix=None,
            markers_detected=len(markers_px),
            calibration_error_mm=0.0,  # No error estimate for 2-point calibration
            is_valid=True
        )
    
    def apply_perspective_correction(
        self,
        image: np.ndarray,
        calibration: CalibrationResult,
        output_size_mm: Tuple[float, float] = (150, 150),
        pixels_per_mm: float = 10.0
    ) -> Tuple[np.ndarray, float]:
        """
        Apply perspective correction to image.
        
        Args:
            image: Input image
            calibration: Calibration result with perspective matrix
            output_size_mm: Output image size in mm (width, height)
            pixels_per_mm: Resolution of output image
            
        Returns:
            Tuple of (corrected_image, new_scale_mm_per_pixel)
        """
        if calibration.perspective_matrix is None:
            # No perspective correction needed
            return image, calibration.scale_mm_per_pixel
        
        # Compute output size in pixels
        out_w = int(output_size_mm[0] * pixels_per_mm)
        out_h = int(output_size_mm[1] * pixels_per_mm)
        
        # Scale the homography to output pixel coordinates
        # H maps pixels -> mm, we need pixels -> output_pixels
        scale_matrix = np.array([
            [pixels_per_mm, 0, 0],
            [0, pixels_per_mm, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        H_scaled = scale_matrix @ calibration.perspective_matrix
        
        # Apply warp
        corrected = cv2.warpPerspective(image, H_scaled, (out_w, out_h))
        
        return corrected, 1.0 / pixels_per_mm
