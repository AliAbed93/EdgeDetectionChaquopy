# Lens Scanner - Vision-Based Glasses Lens Contour Extraction

An MVP Android application for extracting precise vector (SVG) contours from eyeglass lenses using classical computer vision techniques. Designed for manufacturing purposes (CNC/mold making).

## Overview

This system captures images of eyeglass lenses placed on a reference board under edge-lighting conditions and outputs precise SVG contours suitable for CNC machining or mold making.

**Target Accuracy:** ~0.03-0.05mm with standard phone camera  
**Processing Time:** < 1 second per frame

## Hardware Setup Requirements

```
┌─────────────────────────────────────────┐
│           Camera (phone/webcam)          │
│                   ▼                      │
│    ┌─────────────────────────────┐      │
│    │   ○               ○         │      │  ← Reference markers
│    │                             │      │    (outside lens area)
│    │       ╭─────────╮           │      │
│    │      ╱           ╲          │      │
│    │     │   LENS      │         │      │  ← Lens on matte black
│    │      ╲           ╱          │      │    background
│    │       ╰─────────╯           │      │
│    │                             │      │
│    │   ○               ○         │      │  ← Reference markers
│    └─────────────────────────────┘      │
│         Matte Black Board               │
│                                         │
│    ═══════════════════════════════      │  ← Low-angle edge lighting
│         Edge Lighting (LED strip)       │
└─────────────────────────────────────────┘
```

### Key Setup Points:
- **Camera:** Mounted above, looking down at the board
- **Background:** Matte black to maximize contrast
- **Lighting:** Low-angle (edge lighting) so lens edges appear bright
- **Reference Markers:** Placed OUTSIDE the lens area (never under glass)
- **Lens Position:** Centered on the board

## Project Structure

```
LensScannerApp/
├── app/
│   ├── build.gradle.kts          # App build config with Chaquopy
│   ├── src/main/
│   │   ├── AndroidManifest.xml
│   │   ├── java/.../MainActivity.kt   # Android UI
│   │   ├── python/lens_scanner/       # Python CV pipeline
│   │   │   ├── __init__.py
│   │   │   ├── config.py              # Configuration parameters
│   │   │   ├── reference_markers.py   # Marker detection & calibration
│   │   │   ├── edge_detection.py      # Lens edge detection
│   │   │   ├── contour_processing.py  # Subpixel refinement & smoothing
│   │   │   ├── svg_export.py          # SVG generation
│   │   │   └── pipeline.py            # Main orchestration
│   │   └── res/                       # Android resources
├── build.gradle.kts              # Root build config
├── settings.gradle.kts
└── gradle.properties
```

## Image Processing Pipeline

```
Input Image
    │
    ▼
┌─────────────────────────────────┐
│  1. Reference Marker Detection  │  → Scale (mm/pixel)
│     - Circle/ArUco detection    │  → Perspective matrix
│     - Calibration computation   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  2. Grayscale Conversion        │
│     - BGR → Gray                │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  3. Gaussian Blur               │
│     - Noise reduction           │
│     - Kernel: 5x5               │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  4. Canny Edge Detection        │
│     - Low threshold: 50         │
│     - High threshold: 150       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  5. Morphological Closing       │
│     - Connect edge fragments    │
│     - Kernel: 3x3 ellipse       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  6. Contour Extraction          │
│     - Find external contours    │
│     - Filter by area/shape      │
│     - Select largest valid      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  7. Subpixel Refinement         │
│     - cornerSubPix algorithm    │
│     - Gradient-based adjustment │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  8. B-Spline Smoothing          │
│     - Periodic spline fitting   │
│     - 500 output points         │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  9. Coordinate Transform        │
│     - Pixels → Millimeters      │
│     - Apply perspective if any  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  10. SVG Export                 │
│     - Cubic Bezier curves       │
│     - Real-world units (mm)     │
│     - Metadata embedding        │
└─────────────────────────────────┘
    │
    ▼
Output SVG File
```

## Building the Project

### Prerequisites
- Android Studio Arctic Fox or later
- JDK 17
- Android SDK 34

### Build Steps

1. Open the project in Android Studio
2. Sync Gradle (Chaquopy will download Python packages)
3. Build and run on device/emulator

```bash
# Command line build
cd LensScannerApp
./gradlew assembleDebug
```

## Usage

### Android App
1. Launch the app
2. Position lens on reference board under edge lighting
3. Tap "Capture" to take photo, or "Load" to select existing image
4. Tap "Scan" to process
5. SVG file is saved to app's internal storage

### Python API (Direct Usage)

```python
from lens_scanner import LensScannerPipeline, ScannerConfig

# Configure scanner
config = ScannerConfig(
    marker_known_distance_mm=100.0,  # Distance between markers
    canny_low_threshold=50,
    canny_high_threshold=150,
)

# Create pipeline
pipeline = LensScannerPipeline(config)

# Process image
result = pipeline.process_image(
    image_path="lens_photo.jpg",
    output_dir="./output",
    output_filename="lens_contour.svg"
)

if result.success:
    print(f"SVG saved to: {result.svg_path}")
    print(f"Perimeter: {result.contour.perimeter_mm:.2f} mm")
    print(f"Area: {result.contour.area_mm2:.2f} mm²")
else:
    print(f"Error: {result.error_message}")
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `marker_known_distance_mm` | 100.0 | Distance between reference markers |
| `gaussian_blur_kernel` | 5 | Blur kernel size (must be odd) |
| `canny_low_threshold` | 50 | Canny lower threshold |
| `canny_high_threshold` | 150 | Canny upper threshold |
| `morph_kernel_size` | 3 | Morphological kernel size |
| `min_contour_area_ratio` | 0.01 | Min contour area (% of image) |
| `max_contour_area_ratio` | 0.5 | Max contour area (% of image) |
| `spline_num_points` | 500 | Points in smoothed contour |
| `svg_decimal_precision` | 3 | Decimal places in SVG |

## Output SVG Format

The generated SVG uses:
- Real-world units (millimeters)
- Cubic Bezier curves for smooth paths
- Embedded metadata (perimeter, area, etc.)

Example SVG structure:
```xml
<?xml version="1.0" encoding="utf-8" ?>
<svg width="75.5mm" height="45.2mm" viewBox="0 0 75.5 45.2">
  <desc>
    Lens Contour - Generated by LensScanner
    Perimeter: 156.234 mm
    Area: 1523.456 mm²
  </desc>
  <path d="M 5.000 22.600 C 5.123 21.456 ... Z" 
        stroke="black" stroke-width="0.1mm" fill="none"/>
</svg>
```

## Engineering Assumptions

1. **Lighting:** Edge lighting creates bright lens edges on dark background
2. **Markers:** Reference markers are circular, white/bright on dark background
3. **Lens Shape:** Roughly elliptical with circularity > 0.3
4. **Camera:** Approximately orthogonal to scanning surface
5. **Scale:** If no markers detected, assumes ~0.1 mm/pixel (typical phone setup)

## Accuracy Considerations

- **Subpixel refinement** improves edge localization to ~0.1 pixel
- **B-spline smoothing** reduces noise while preserving shape
- **Perspective correction** compensates for non-orthogonal camera angles
- **Target accuracy:** 0.03-0.05mm achievable with proper setup

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No contour detected | Improve edge lighting, check lens contrast |
| Noisy contour | Increase blur kernel, adjust Canny thresholds |
| Wrong contour selected | Adjust area ratio filters |
| Scale incorrect | Verify marker distance setting |
| Perspective distortion | Ensure 4 markers are visible |

## Dependencies

### Python (via Chaquopy)
- numpy
- opencv-python
- scipy
- svgwrite

### Android
- CameraX
- Kotlin Coroutines
- Material Components

## License

MIT License - See LICENSE file for details.
