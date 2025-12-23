"""
Edge-Lit Lens Detection Module.

Implements lens edge detection optimized for edge-lighting setup where:
- Lens is placed on matte black background
- Low-angle lighting causes lens edges to appear bright
- Interior of lens may show distortions (ignored)

The detection pipeline:
1. Grayscale conversion
2. Gaussian blur for noise reduction
3. Canny edge detection
4. Morphological closing to connect edge fragments
5. Contour extraction and filtering
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from .config import ScannerConfig


class EdgeLitLensDetector:
    """
    Detects lens contours using edge-lighting principles.
    """
    
    def __init__(self, config: ScannerConfig):
        self.config = config
        self._debug_images = {}
    
    def detect_lens_contour(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Detect the main lens contour in the image.
        
        Args:
            image: Input BGR image
            mask: Optional mask to exclude regions (e.g., marker areas)
            
        Returns:
            Tuple of (contour, debug_info)
            - contour: Nx1x2 array of contour points, or None if not found
            - debug_info: Dictionary with intermediate results for debugging
        """
        debug_info = {}
        
        # Step 1: Convert to grayscale
        gray = self._to_grayscale(image)
        debug_info['grayscale'] = gray
        
        # Step 2: Apply Gaussian blur
        blurred = self._apply_blur(gray)
        debug_info['blurred'] = blurred
        
        # Step 3: Edge detection
        edges = self._detect_edges(blurred)
        debug_info['edges'] = edges
        
        # Step 4: Morphological closing
        closed = self._morphological_close(edges)
        debug_info['closed'] = closed
        
        # Step 5: Apply mask if provided
        if mask is not None:
            closed = cv2.bitwise_and(closed, closed, mask=mask)
            debug_info['masked'] = closed
        
        # Step 6: Find contours
        contours = self._find_contours(closed)
        debug_info['num_contours'] = len(contours)
        
        # Step 7: Filter and select best contour
        best_contour = self._select_lens_contour(contours, image.shape[:2])
        
        if best_contour is not None:
            debug_info['contour_points'] = len(best_contour)
            debug_info['contour_area'] = cv2.contourArea(best_contour)
            debug_info['contour_perimeter'] = cv2.arcLength(best_contour, True)
        
        return best_contour, debug_info
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()
    
    def _apply_blur(self, gray: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur for noise reduction."""
        k = self.config.gaussian_blur_kernel
        return cv2.GaussianBlur(gray, (k, k), 0)
    
    def _detect_edges(self, blurred: np.ndarray) -> np.ndarray:
        """
        Detect edges using Canny algorithm.
        
        Parameters are tuned for edge-lit setup where lens edges
        appear as bright lines on dark background.
        """
        return cv2.Canny(
            blurred,
            self.config.canny_low_threshold,
            self.config.canny_high_threshold
        )
    
    def _morphological_close(self, edges: np.ndarray) -> np.ndarray:
        """
        Apply morphological closing to connect edge fragments.
        
        This helps create continuous contours from fragmented edges.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morph_kernel_size, self.config.morph_kernel_size)
        )
        return cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.config.morph_iterations
        )
    
    def _find_contours(self, binary: np.ndarray) -> List[np.ndarray]:
        """Find all contours in binary image."""
        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,  # Only external contours
            cv2.CHAIN_APPROX_NONE  # Keep all points for accuracy
        )
        return list(contours)
    
    def _select_lens_contour(
        self,
        contours: List[np.ndarray],
        image_shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Select the contour most likely to be the lens.
        
        Selection criteria:
        1. Area within expected range
        2. Reasonable circularity (lens shapes are roughly elliptical)
        3. Closed contour
        4. Largest valid contour wins
        """
        image_area = image_shape[0] * image_shape[1]
        min_area = image_area * self.config.min_contour_area_ratio
        max_area = image_area * self.config.max_contour_area_ratio
        
        valid_contours = []
        
        for contour in contours:
            # Check area
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < self.config.min_contour_circularity:
                continue
            
            # Check if contour is reasonably convex (lens shouldn't have deep concavities)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.5:  # Too many concavities
                    continue
            
            valid_contours.append((contour, area, circularity))
        
        if not valid_contours:
            return None
        
        # Select largest valid contour
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        return valid_contours[0][0]
    
    def create_marker_exclusion_mask(
        self,
        image_shape: Tuple[int, int],
        marker_centers: np.ndarray,
        exclusion_radius: int = 50
    ) -> np.ndarray:
        """
        Create a mask that excludes areas around reference markers.
        
        This prevents marker detection from interfering with lens detection.
        
        Args:
            image_shape: (height, width) of image
            marker_centers: Nx2 array of marker center coordinates
            exclusion_radius: Radius around each marker to exclude
            
        Returns:
            Binary mask (255 = valid area, 0 = excluded)
        """
        mask = np.ones((image_shape[0], image_shape[1]), dtype=np.uint8) * 255
        
        for center in marker_centers:
            cv2.circle(
                mask,
                (int(center[0]), int(center[1])),
                exclusion_radius,
                0,
                -1  # Filled circle
            )
        
        return mask


class AdaptiveEdgeDetector(EdgeLitLensDetector):
    """
    Extended detector with adaptive parameter tuning.
    
    Automatically adjusts Canny thresholds based on image statistics.
    Use when lighting conditions vary significantly.
    """
    
    def _detect_edges(self, blurred: np.ndarray) -> np.ndarray:
        """
        Detect edges with automatically computed thresholds.
        
        Uses median-based threshold computation for robustness.
        """
        # Compute median intensity
        median = np.median(blurred)
        
        # Compute thresholds based on median
        # This adapts to overall image brightness
        sigma = 0.33
        low = int(max(0, (1.0 - sigma) * median))
        high = int(min(255, (1.0 + sigma) * median))
        
        # Ensure minimum separation between thresholds
        if high - low < 30:
            low = max(0, int(median - 30))
            high = min(255, int(median + 30))
        
        return cv2.Canny(blurred, low, high)
