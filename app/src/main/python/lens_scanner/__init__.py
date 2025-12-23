"""
Lens Scanner - Vision-based glasses lens contour extraction system.

This package provides classical computer vision algorithms for extracting
precise lens contours from images captured under edge-lighting conditions.

Target accuracy: ~0.03-0.05mm with standard phone camera
Processing: Single-frame, < 1 second

Modules:
    - reference_markers: Scale calibration and perspective correction
    - edge_detection: Edge-lit lens contour detection
    - contour_processing: Subpixel refinement and smoothing
    - svg_export: Vector output generation
    - pipeline: Main processing orchestration
"""

__version__ = "1.0.0-MVP"
__author__ = "Lens Scanner Team"

from .pipeline import LensScannerPipeline, ScanResult
from .config import ScannerConfig

__all__ = [
    "LensScannerPipeline",
    "ScanResult", 
    "ScannerConfig",
]
