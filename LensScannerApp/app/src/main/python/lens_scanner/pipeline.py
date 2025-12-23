"""
Main Processing Pipeline.

Orchestrates the complete lens scanning workflow:
1. Reference marker detection and calibration
2. Lens edge detection
3. Contour processing and refinement
4. SVG export

This module provides the main entry point for Android integration.
"""

import cv2
import numpy as np
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from .config import ScannerConfig, CalibrationResult, ContourResult
from .reference_markers import ReferenceMarkerDetector
from .edge_detection import EdgeLitLensDetector, AdaptiveEdgeDetector
from .contour_processing import ContourProcessor, AdvancedContourProcessor
from .svg_export import SVGExporter


@dataclass
class ScanResult:
    """
    Complete result of a lens scan operation.
    
    Attributes:
        success: Whether scan completed successfully
        error_message: Error description if failed
        svg_path: Path to generated SVG file
        calibration: Calibration results
        contour: Contour extraction results
        processing_time_ms: Total processing time in milliseconds
        debug_info: Additional diagnostic information
    """
    success: bool = False
    error_message: str = ""
    svg_path: str = ""
    calibration: Optional[CalibrationResult] = None
    contour: Optional[ContourResult] = None
    processing_time_ms: float = 0.0
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Android interop."""
        result = {
            "success": self.success,
            "error_message": self.error_message,
            "svg_path": self.svg_path,
            "processing_time_ms": self.processing_time_ms,
        }
        
        if self.calibration:
            result["calibration"] = {
                "scale_mm_per_pixel": self.calibration.scale_mm_per_pixel,
                "markers_detected": self.calibration.markers_detected,
                "calibration_error_mm": self.calibration.calibration_error_mm,
                "is_valid": self.calibration.is_valid,
            }
        
        if self.contour:
            result["contour"] = {
                "perimeter_mm": self.contour.perimeter_mm,
                "area_mm2": self.contour.area_mm2,
                "bounding_box_mm": self.contour.bounding_box_mm,
                "centroid_mm": self.contour.centroid_mm,
                "is_closed": self.contour.is_closed,
                "num_points": len(self.contour.contour_mm) if self.contour.contour_mm is not None else 0,
            }
        
        return result


class LensScannerPipeline:
    """
    Main pipeline for lens contour extraction.
    
    Usage:
        pipeline = LensScannerPipeline()
        result = pipeline.process_image(image_path, output_dir)
        
        # Or with numpy array:
        result = pipeline.process_array(image_array, output_dir)
    """
    
    def __init__(self, config: Optional[ScannerConfig] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Scanner configuration. Uses defaults if None.
        """
        self.config = config or ScannerConfig()
        
        # Initialize processing modules
        self.marker_detector = ReferenceMarkerDetector(self.config)
        self.edge_detector = EdgeLitLensDetector(self.config)
        self.contour_processor = ContourProcessor(self.config)
        self.svg_exporter = SVGExporter(self.config)
    
    def process_image(
        self,
        image_path: str,
        output_dir: str,
        output_filename: str = "lens_contour.svg",
        expected_marker_positions_mm: Optional[np.ndarray] = None
    ) -> ScanResult:
        """
        Process an image file and extract lens contour.
        
        Args:
            image_path: Path to input image
            output_dir: Directory for output files
            output_filename: Name for output SVG file
            expected_marker_positions_mm: Known marker positions (optional)
            
        Returns:
            ScanResult with processing results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return ScanResult(
                success=False,
                error_message=f"Failed to load image: {image_path}"
            )
        
        return self.process_array(
            image,
            output_dir,
            output_filename,
            expected_marker_positions_mm
        )
    
    def process_array(
        self,
        image: np.ndarray,
        output_dir: str,
        output_filename: str = "lens_contour.svg",
        expected_marker_positions_mm: Optional[np.ndarray] = None
    ) -> ScanResult:
        """
        Process a numpy array image and extract lens contour.
        
        Args:
            image: Input BGR image as numpy array
            output_dir: Directory for output files
            output_filename: Name for output SVG file
            expected_marker_positions_mm: Known marker positions (optional)
            
        Returns:
            ScanResult with processing results
        """
        start_time = time.time()
        debug_info = {}
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Step 1: Detect reference markers and calibrate
            calibration = self.marker_detector.detect_and_calibrate(
                image,
                expected_marker_positions_mm
            )
            debug_info['calibration'] = {
                'markers_detected': calibration.markers_detected,
                'scale': calibration.scale_mm_per_pixel,
                'is_valid': calibration.is_valid
            }
            
            # Handle calibration failure
            if not calibration.is_valid:
                # Try with fallback scale (assume typical phone camera setup)
                # ~0.1 mm/pixel at typical scanning distance
                calibration = CalibrationResult(
                    scale_mm_per_pixel=0.1,
                    is_valid=True,
                    markers_detected=0,
                    calibration_error_mm=float('inf')
                )
                debug_info['calibration_fallback'] = True
            
            # Step 2: Detect lens contour
            contour, edge_debug = self.edge_detector.detect_lens_contour(image)
            debug_info['edge_detection'] = {
                'num_contours': edge_debug.get('num_contours', 0),
                'contour_found': contour is not None
            }
            
            if contour is None:
                return ScanResult(
                    success=False,
                    error_message="No valid lens contour detected",
                    calibration=calibration,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    debug_info=debug_info
                )
            
            # Step 3: Process and refine contour
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contour_result = self.contour_processor.process_contour(
                contour,
                gray,
                calibration
            )
            debug_info['contour_processing'] = {
                'num_points': len(contour_result.contour_mm),
                'perimeter_mm': contour_result.perimeter_mm,
                'area_mm2': contour_result.area_mm2
            }
            
            # Step 4: Export to SVG
            svg_path = os.path.join(output_dir, output_filename)
            self.svg_exporter.export(
                contour_result,
                svg_path,
                include_metadata=True,
                include_dimensions=False
            )
            
            # Save debug images if configured
            if self.config.save_debug_images:
                self._save_debug_images(image, contour, edge_debug, output_dir)
            
            processing_time = (time.time() - start_time) * 1000
            
            return ScanResult(
                success=True,
                svg_path=svg_path,
                calibration=calibration,
                contour=contour_result,
                processing_time_ms=processing_time,
                debug_info=debug_info
            )
            
        except Exception as e:
            return ScanResult(
                success=False,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
                debug_info=debug_info
            )
    
    def process_bytes(
        self,
        image_bytes: bytes,
        output_dir: str,
        output_filename: str = "lens_contour.svg"
    ) -> ScanResult:
        """
        Process image from byte array (for Android camera integration).
        
        Args:
            image_bytes: Image data as bytes (JPEG, PNG, etc.)
            output_dir: Directory for output files
            output_filename: Name for output SVG file
            
        Returns:
            ScanResult with processing results
        """
        # Decode image bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return ScanResult(
                success=False,
                error_message="Failed to decode image bytes"
            )
        
        return self.process_array(image, output_dir, output_filename)
    
    def _save_debug_images(
        self,
        image: np.ndarray,
        contour: np.ndarray,
        edge_debug: Dict,
        output_dir: str
    ):
        """Save intermediate processing images for debugging."""
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save edge detection stages
        for name in ['grayscale', 'blurred', 'edges', 'closed']:
            if name in edge_debug:
                cv2.imwrite(
                    os.path.join(debug_dir, f"{name}.png"),
                    edge_debug[name]
                )
        
        # Save contour overlay
        overlay = image.copy()
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "contour_overlay.png"), overlay)


# ============================================================================
# Android Entry Points
# ============================================================================

def scan_lens_from_file(
    image_path: str,
    output_dir: str,
    marker_distance_mm: float = 100.0
) -> Dict[str, Any]:
    """
    Main entry point for Android - process image file.
    
    Args:
        image_path: Path to input image
        output_dir: Directory for output SVG
        marker_distance_mm: Known distance between reference markers
        
    Returns:
        Dictionary with scan results (for Chaquopy interop)
    """
    config = ScannerConfig(marker_known_distance_mm=marker_distance_mm)
    pipeline = LensScannerPipeline(config)
    result = pipeline.process_image(image_path, output_dir)
    return result.to_dict()


def scan_lens_from_bytes(
    image_bytes: bytes,
    output_dir: str,
    marker_distance_mm: float = 100.0
) -> Dict[str, Any]:
    """
    Main entry point for Android - process image bytes.
    
    Args:
        image_bytes: Image data as bytes
        output_dir: Directory for output SVG
        marker_distance_mm: Known distance between reference markers
        
    Returns:
        Dictionary with scan results (for Chaquopy interop)
    """
    config = ScannerConfig(marker_known_distance_mm=marker_distance_mm)
    pipeline = LensScannerPipeline(config)
    result = pipeline.process_bytes(image_bytes, output_dir)
    return result.to_dict()


def get_version() -> str:
    """Return pipeline version string."""
    return "1.0.0-MVP"
