# Bean Measurement Project

This tool analyzes an image containing green beans and a Post-it note of known size to measure each bean’s length and width.

> **Note**: All files under `not-working/` are excluded and ignored.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Function Reference](#function-reference)
6. [Project Structure](#project-structure)
7. [Contributing](#contributing)

---

## Overview

`measure_beans.py` loads an image, detects a Post-it for scale calibration, segments green beans, then computes and annotates each bean’s maximum length and width in millimeters. The result is saved as a timestamped image in the working directory.

## Requirements

- Python 3.6 or newer
- OpenCV (`opencv-python`)
- NumPy

Dependencies are listed in `requirements.txt`.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python measure_beans.py <input_image> [--debug]
```

- `<input_image>`: Path to your image file (contains beans + Post-it note).
- `--debug`: Enable verbose console output and intermediate mask displays.

**Output**: An annotated image named `<YYYYMMDD_HHMMSS>_measured.jpg`, e.g., `20250429_130844_measured.jpg`.

## Function Reference

### `order_corners(pts)`
- **Purpose**: Sorts four corner points into top-left, top-right, bottom-right, bottom-left order.
- **Input**: `pts` — List/array of four `(x, y)` points.
- **Output**: `np.ndarray` of shape `(4,2)` in TL, TR, BR, BL order.

### `measure_beans(image_path, postit_side_mm=76.0, bean_lower=(30,40,40), bean_upper=(85,255,255), debug=False)`
- **Purpose**: Detects beans and measures each bean’s maximum length & width.
- **Parameters**:
  - `image_path` (str): Filepath to the input image.
  - `postit_side_mm` (float): Physical side length of the Post-it square (default 76 mm).
  - `bean_lower`, `bean_upper` (tuple): HSV color thresholds for segmenting beans.
  - `debug` (bool): If `True`, prints intermediate info and masks.
- **Returns**: `(annotated_img, results_list)`
  - `annotated_img`: Image array with drawn contours, measurements, and labels.
  - `results_list`: List of tuples `(bean_id, "length"|"width", measurement_mm)`.

## Project Structure

```
beans/
├─ docs/              # Documentation (this folder)
│  └─ README.md
├─ measure_beans.py    # Main script
├─ requirements.txt    # Python dependencies
└─ not-working/        # Ignored experimental scripts
```

## Contributing

- Do **not** modify files under `not-working/`. They are deprecated or experimental.
- Feel free to open issues or pull requests for improvements, bug fixes, or new features.

---

Happy measuring! 🫘
