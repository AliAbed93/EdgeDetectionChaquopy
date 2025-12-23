# Codemagic Setup Guide for Lens Scanner

## Quick Start

### 1. Repository Structure ✅
Your project is now correctly structured with Android files at the root:
```
.
├── app/                    # Android app module
├── gradle/                 # Gradle wrapper
├── build.gradle.kts        # Root build file
├── settings.gradle.kts     # Project settings
├── gradle.properties       # Gradle properties
├── codemagic.yaml         # CI/CD configuration
└── README.md
```

### 2. Connect to Codemagic

1. Go to [codemagic.io](https://codemagic.io)
2. Sign in with your Git provider (GitHub, GitLab, Bitbucket)
3. Click **"Add application"**
4. Select your repository
5. Codemagic will scan and detect the Android project

### 3. Project Detection

When Codemagic scans your repository, it should now automatically detect:
- **Project type:** Android
- **Build file:** `build.gradle.kts` at root
- **App module:** `app/`

If it still doesn't detect:
1. Click **"Set up build manually"**
2. Select **"Android"** as project type
3. Set project path to: `.` (root)
4. Click **"Finish setup"**

### 4. Build Configuration

The `codemagic.yaml` file is already configured with:

```yaml
workflows:
  android-lens-scanner:
    name: Lens Scanner Android Build
    instance_type: mac_mini_m1
    environment:
      java: 17
    scripts:
      - ./gradlew assembleDebug
```

### 5. First Build

**Important:** The first build will take 10-15 minutes because:
- Chaquopy downloads Python 3.11
- Python packages are installed (numpy, opencv-python, scipy, svgwrite)
- OpenCV is large (~100MB)

Subsequent builds are much faster with caching.

### 6. Build Triggers

You can trigger builds:
- **Manual:** Click "Start new build" in Codemagic dashboard
- **Automatic:** Push to your repository (configure in Codemagic settings)
- **Pull Request:** Automatically build PRs

### 7. Artifacts

After successful build, download:
- `app/build/outputs/apk/debug/app-debug.apk`

## Troubleshooting

### Issue: "Repository doesn't contain a mobile application"

**Solution:** Ensure these files are at repository root (not in subfolder):
- ✅ `build.gradle.kts`
- ✅ `settings.gradle.kts`
- ✅ `app/` folder

### Issue: Build timeout

**Solution:** Increase timeout in `codemagic.yaml`:
```yaml
max_build_duration: 90  # Increase to 90 minutes
```

### Issue: Chaquopy package download fails

**Solution:** Check network connectivity. Chaquopy downloads from PyPI. If persistent, try:
```yaml
scripts:
  - name: Pre-download Python packages
    script: |
      pip3 install numpy opencv-python scipy svgwrite
```

### Issue: Out of memory during build

**Solution:** Use larger instance type:
```yaml
instance_type: mac_pro  # Instead of mac_mini_m1
```

## Advanced Configuration

### Environment Variables

Add to `codemagic.yaml`:
```yaml
environment:
  vars:
    MARKER_DISTANCE_MM: "100.0"
    DEBUG_MODE: "false"
```

### Signing for Release

1. Generate keystore
2. Upload to Codemagic (Settings → Code signing)
3. Update `codemagic.yaml`:
```yaml
environment:
  android_signing:
    - keystore_reference
scripts:
  - ./gradlew assembleRelease
```

### Publishing to Google Play

```yaml
publishing:
  google_play:
    credentials: $GCLOUD_SERVICE_ACCOUNT_CREDENTIALS
    track: internal
```

## Build Status Badge

Add to your README.md:
```markdown
[![Codemagic build status](https://api.codemagic.io/apps/<app-id>/status_badge.svg)](https://codemagic.io/apps/<app-id>/latest_build)
```

## Support

- [Codemagic Docs](https://docs.codemagic.io/)
- [Chaquopy Docs](https://chaquo.com/chaquopy/doc/current/)
- [Project README](README.md)
