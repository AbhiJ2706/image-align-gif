# Image Aligner

A powerful Python tool for creating aligned GIFs from a series of images by automatically detecting and centering a subject across all frames. The tool includes intelligent subject detection, user correction capabilities, and multiple output formats.

## Features

- **Automatic Subject Detection**: Uses multi-scale template matching and feature-based detection to identify subjects across images
- **Interactive Correction**: Manual override system for low-confidence detections

## Installation

Clone the repository, then run

```bash
pip install image-align
```

## Usage

### Basic Usage

Manually:

```bash
hatch shell
python src/image_align/image_align.py [path/to/images]
```

Using Hatch:

```bash
hatch run image_align.py [path/to/images]
```

### Interactive Workflow

1. **Subject Selection**: When you run the script, it will display the first image and ask you to draw a rectangle around the subject you want to track.

2. **Automatic Processing**: The tool will process each subsequent image, attempting to locate the same subject using:
   - Multi-scale template matching
   - Feature-based matching (SIFT/ORB)
   - Fallback positioning

3. **Manual Correction**: For low-confidence detections, the tool will show you the current frame with the detected subject box and allow you to:
   - **ENTER**: Accept the current detection
   - **SPACE**: Skip correction (use current detection)
   - **Mouse drag**: Draw a new rectangle around the correct subject location
   - **ESC**: Exit the program

### Detection Methods Visualization

The tool shows different colored boxes based on detection confidence:
- **Green**: High confidence detection or reference frame
- **Yellow**: Medium confidence detection
- **Red**: Low confidence or fallback detection
- **Magenta**: User-corrected detection

## Output Files

The script generates three types of GIF files:

### 1. Main Aligned GIF (`enhanced_aligned.gif`)
The primary output showing all images with the subject centered.

![Final Aligned GIF Placeholder](examples/enhanced_aligned.gif)

### 2. Detection Visualization (`enhanced_aligned_detection_boxes.gif`)
Shows the original images with colored detection boxes overlaid, helping you understand how well the subject detection worked.

![Detection Boxes GIF Placeholder](examples/enhanced_aligned_detection_boxes.gif)

### 3. Cropped Version (`enhanced_aligned_cropped.gif`)
A version with unnecessary black borders automatically removed while maintaining subject centering.
