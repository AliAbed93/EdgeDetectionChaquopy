"""
Test script for lens scanner pipeline.

This script creates a synthetic test image and runs the full pipeline
to verify all components work correctly.

Usage:
    python test_pipeline.py [--output-dir ./output]
"""

import numpy as np
import cv2
import os
import argparse
from lens_scanner import LensScannerPipeline, ScannerConfig


def create_synthetic_lens_image(
    width: int = 1920,
    height: int = 1440,
    lens_center: tuple = None,
    lens_axes: tuple = (300, 200),
    marker_positions: list = None
) -> np.ndarray:
    """
    Create a synthetic test image with lens and markers.
    
    Simulates edge-lit lens on dark background.
    """
    # Dark background
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (20, 20, 20)  # Near black
    
    # Default lens center
    if lens_center is None:
        lens_center = (width // 2, height // 2)

    # Default marker positions (corners of rectangle)
    if marker_positions is None:
        margin = 100
        marker_positions = [
            (margin, margin),
            (width - margin, margin),
            (width - margin, height - margin),
            (margin, height - margin),
        ]
    
    # Draw reference markers (bright circles)
    for pos in marker_positions:
        cv2.circle(image, pos, 30, (255, 255, 255), -1)
    
    # Draw lens edge (bright ellipse outline - simulating edge lighting)
    cv2.ellipse(
        image,
        lens_center,
        lens_axes,
        angle=15,  # Slight rotation
        startAngle=0,
        endAngle=360,
        color=(200, 200, 200),
        thickness=3
    )
    
    # Add some noise for realism
    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


def run_test(output_dir: str = "./test_output"):
    """Run pipeline test with synthetic image."""
    print("=" * 60)
    print("Lens Scanner Pipeline Test")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic test image
    print("\n1. Creating synthetic test image...")
    image = create_synthetic_lens_image()
    test_image_path = os.path.join(output_dir, "test_input.png")
    cv2.imwrite(test_image_path, image)
    print(f"   Saved: {test_image_path}")
    print(f"   Size: {image.shape[1]}x{image.shape[0]}")

    # Configure pipeline
    print("\n2. Configuring pipeline...")
    config = ScannerConfig(
        marker_known_distance_mm=100.0,
        save_debug_images=True,
        debug_output_dir=output_dir,
    )
    
    # Create pipeline
    pipeline = LensScannerPipeline(config)
    print(f"   Marker distance: {config.marker_known_distance_mm} mm")
    
    # Process image
    print("\n3. Processing image...")
    result = pipeline.process_image(
        image_path=test_image_path,
        output_dir=output_dir,
        output_filename="test_lens_contour.svg"
    )
    
    # Report results
    print("\n4. Results:")
    print(f"   Success: {result.success}")
    
    if result.success:
        print(f"   SVG Path: {result.svg_path}")
        print(f"   Processing Time: {result.processing_time_ms:.1f} ms")
        
        if result.calibration:
            print(f"\n   Calibration:")
            print(f"     Markers detected: {result.calibration.markers_detected}")
            print(f"     Scale: {result.calibration.scale_mm_per_pixel:.6f} mm/pixel")
        
        if result.contour:
            print(f"\n   Contour:")
            print(f"     Points: {len(result.contour.contour_mm)}")
            print(f"     Perimeter: {result.contour.perimeter_mm:.2f} mm")
            print(f"     Area: {result.contour.area_mm2:.2f} mm²")
            print(f"     Bounding box: {result.contour.bounding_box_mm[2]:.2f} x {result.contour.bounding_box_mm[3]:.2f} mm")
            print(f"     Closed: {result.contour.is_closed}")
    else:
        print(f"   Error: {result.error_message}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test lens scanner pipeline")
    parser.add_argument("--output-dir", default="./test_output", help="Output directory")
    args = parser.parse_args()
    
    run_test(args.output_dir)
