"""
Contour Processing Module.

Handles refinement and smoothing of detected lens contours:
1. Subpixel edge refinement for improved accuracy
2. B-spline fitting for smooth curves
3. Coordinate transformation to real-world units

Target accuracy: ~0.03-0.05mm with subpixel refinement
"""

import cv2
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, Optional
from .config import ScannerConfig, CalibrationResult, ContourResult


class ContourProcessor:
    """
    Processes raw contours into refined, smooth curves.
    """
    
    def __init__(self, config: ScannerConfig):
        self.config = config
    
    def process_contour(
        self,
        contour: np.ndarray,
        gray_image: np.ndarray,
        calibration: CalibrationResult
    ) -> ContourResult:
        """
        Process raw contour into refined result.
        
        Args:
            contour: Raw contour from edge detection (Nx1x2)
            gray_image: Grayscale image for subpixel refinement
            calibration: Calibration result for coordinate transformation
            
        Returns:
            ContourResult with refined contour and measurements
        """
        # Reshape contour to Nx2
        points = contour.reshape(-1, 2).astype(np.float32)
        
        # Step 1: Subpixel refinement
        refined_points = self._subpixel_refine(points, gray_image)
        
        # Step 2: Remove duplicate/close points
        cleaned_points = self._remove_duplicates(refined_points)
        
        # Step 3: Smooth with B-spline
        smoothed_points = self._spline_smooth(cleaned_points)
        
        # Step 4: Transform to mm coordinates
        points_mm = self._transform_to_mm(smoothed_points, calibration)
        
        # Step 5: Compute measurements
        result = self._compute_measurements(smoothed_points, points_mm)
        
        return result
    
    def _subpixel_refine(
        self,
        points: np.ndarray,
        gray_image: np.ndarray
    ) -> np.ndarray:
        """
        Refine contour points to subpixel accuracy.
        
        Uses OpenCV's cornerSubPix which works well for edge points too.
        The algorithm fits a gradient-based model around each point.
        """
        # cornerSubPix requires specific format
        points_for_refine = points.reshape(-1, 1, 2).astype(np.float32)
        
        # Define refinement criteria
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            self.config.subpixel_iterations,
            self.config.subpixel_epsilon
        )
        
        # Window size for refinement
        win_size = (self.config.subpixel_window_size, self.config.subpixel_window_size)
        zero_zone = (-1, -1)  # No dead zone
        
        try:
            refined = cv2.cornerSubPix(
                gray_image,
                points_for_refine,
                win_size,
                zero_zone,
                criteria
            )
            return refined.reshape(-1, 2)
        except cv2.error:
            # Fall back to original points if refinement fails
            return points
    
    def _remove_duplicates(
        self,
        points: np.ndarray,
        min_distance: float = 0.5
    ) -> np.ndarray:
        """
        Remove duplicate or very close points.
        
        This prevents issues with spline fitting.
        """
        if len(points) < 3:
            return points
        
        # Compute distances between consecutive points
        diffs = np.diff(points, axis=0)
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        
        # Keep points that are far enough from previous point
        keep_mask = np.concatenate([[True], distances >= min_distance])
        
        return points[keep_mask]
    
    def _spline_smooth(self, points: np.ndarray) -> np.ndarray:
        """
        Fit a smooth B-spline to the contour points.
        
        Uses periodic spline for closed contours.
        """
        if len(points) < 4:
            return points
        
        # Ensure contour is closed by checking first/last point distance
        dist_to_close = np.linalg.norm(points[0] - points[-1])
        is_closed = dist_to_close < 10  # pixels
        
        # Parameterize by arc length
        diffs = np.diff(points, axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        
        # Normalize parameter to [0, 1]
        if cumulative_length[-1] > 0:
            t = cumulative_length / cumulative_length[-1]
        else:
            t = np.linspace(0, 1, len(points))
        
        # Fit spline to x and y separately
        try:
            if is_closed:
                # For closed contours, use periodic spline
                # Extend points to ensure periodicity
                extended_points = np.vstack([points, points[0:1]])
                extended_t = np.concatenate([t, [1.0]])
                
                # Fit with smoothing
                tck_x, _ = interpolate.splprep(
                    [extended_points[:, 0]], 
                    u=extended_t,
                    s=self.config.spline_smoothing_factor,
                    per=True,
                    k=3
                )
                tck_y, _ = interpolate.splprep(
                    [extended_points[:, 1]], 
                    u=extended_t,
                    s=self.config.spline_smoothing_factor,
                    per=True,
                    k=3
                )
            else:
                tck_x, _ = interpolate.splprep(
                    [points[:, 0]], 
                    u=t,
                    s=self.config.spline_smoothing_factor,
                    k=3
                )
                tck_y, _ = interpolate.splprep(
                    [points[:, 1]], 
                    u=t,
                    s=self.config.spline_smoothing_factor,
                    k=3
                )
            
            # Evaluate spline at uniform intervals
            t_new = np.linspace(0, 1, self.config.spline_num_points)
            x_new = interpolate.splev(t_new, tck_x)[0]
            y_new = interpolate.splev(t_new, tck_y)[0]
            
            return np.column_stack([x_new, y_new])
            
        except (ValueError, TypeError):
            # Fall back to simple Gaussian smoothing if spline fails
            return self._gaussian_smooth(points)
    
    def _gaussian_smooth(
        self,
        points: np.ndarray,
        sigma: float = 2.0
    ) -> np.ndarray:
        """
        Fallback smoothing using Gaussian filter.
        """
        x_smooth = gaussian_filter1d(points[:, 0], sigma, mode='wrap')
        y_smooth = gaussian_filter1d(points[:, 1], sigma, mode='wrap')
        return np.column_stack([x_smooth, y_smooth])
    
    def _transform_to_mm(
        self,
        points_px: np.ndarray,
        calibration: CalibrationResult
    ) -> np.ndarray:
        """
        Transform pixel coordinates to millimeter coordinates.
        
        Uses perspective correction if available, otherwise simple scaling.
        """
        if not calibration.is_valid:
            # No calibration - return pixels as-is
            return points_px.copy()
        
        if calibration.perspective_matrix is not None:
            # Apply homography transformation
            points_h = np.hstack([
                points_px, 
                np.ones((len(points_px), 1))
            ])
            transformed = (calibration.perspective_matrix @ points_h.T).T
            # Normalize homogeneous coordinates
            points_mm = transformed[:, :2] / transformed[:, 2:3]
        else:
            # Simple scaling
            points_mm = points_px * calibration.scale_mm_per_pixel
        
        return points_mm
    
    def _compute_measurements(
        self,
        points_px: np.ndarray,
        points_mm: np.ndarray
    ) -> ContourResult:
        """
        Compute geometric measurements of the contour.
        """
        # Perimeter in mm
        diffs = np.diff(np.vstack([points_mm, points_mm[0:1]]), axis=0)
        perimeter_mm = np.sqrt((diffs ** 2).sum(axis=1)).sum()
        
        # Area using shoelace formula
        x = points_mm[:, 0]
        y = points_mm[:, 1]
        area_mm2 = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        # Bounding box
        min_x, min_y = points_mm.min(axis=0)
        max_x, max_y = points_mm.max(axis=0)
        bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
        
        # Centroid
        centroid = points_mm.mean(axis=0)
        
        # Check if closed
        dist_to_close = np.linalg.norm(points_mm[0] - points_mm[-1])
        is_closed = dist_to_close < 1.0  # mm
        
        return ContourResult(
            contour_pixels=points_px,
            contour_mm=points_mm,
            perimeter_mm=perimeter_mm,
            area_mm2=area_mm2,
            bounding_box_mm=bbox,
            centroid_mm=tuple(centroid),
            is_closed=is_closed
        )


class AdvancedContourProcessor(ContourProcessor):
    """
    Extended processor with additional refinement techniques.
    """
    
    def _subpixel_refine(
        self,
        points: np.ndarray,
        gray_image: np.ndarray
    ) -> np.ndarray:
        """
        Enhanced subpixel refinement using gradient-based method.
        
        For each point, finds the local edge direction and refines
        position perpendicular to the edge.
        """
        # First apply standard refinement
        refined = super()._subpixel_refine(points, gray_image)
        
        # Then apply gradient-based refinement
        # Compute image gradients
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        h, w = gray_image.shape
        
        for i, (x, y) in enumerate(refined):
            # Skip points near image boundary
            if x < 2 or x >= w - 2 or y < 2 or y >= h - 2:
                continue
            
            # Get local gradient
            ix, iy = int(x), int(y)
            gx = grad_x[iy, ix]
            gy = grad_y[iy, ix]
            
            # Gradient magnitude
            mag = np.sqrt(gx**2 + gy**2)
            if mag < 1e-6:
                continue
            
            # Normalize gradient direction
            nx, ny = gx / mag, gy / mag
            
            # Search along gradient direction for maximum gradient
            best_offset = 0
            best_mag = mag
            
            for offset in np.linspace(-1, 1, 21):  # Search ±1 pixel
                px = x + offset * nx
                py = y + offset * ny
                
                if px < 0 or px >= w - 1 or py < 0 or py >= h - 1:
                    continue
                
                # Bilinear interpolation of gradient magnitude
                px0, py0 = int(px), int(py)
                fx, fy = px - px0, py - py0
                
                m00 = np.sqrt(grad_x[py0, px0]**2 + grad_y[py0, px0]**2)
                m01 = np.sqrt(grad_x[py0, px0+1]**2 + grad_y[py0, px0+1]**2)
                m10 = np.sqrt(grad_x[py0+1, px0]**2 + grad_y[py0+1, px0]**2)
                m11 = np.sqrt(grad_x[py0+1, px0+1]**2 + grad_y[py0+1, px0+1]**2)
                
                interp_mag = (
                    m00 * (1-fx) * (1-fy) +
                    m01 * fx * (1-fy) +
                    m10 * (1-fx) * fy +
                    m11 * fx * fy
                )
                
                if interp_mag > best_mag:
                    best_mag = interp_mag
                    best_offset = offset
            
            # Update point position
            refined[i, 0] = x + best_offset * nx
            refined[i, 1] = y + best_offset * ny
        
        return refined
