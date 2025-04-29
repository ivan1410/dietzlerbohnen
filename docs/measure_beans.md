# measure_beans.py Documentation

## 1. Introduction

`measure_beans.py` is the main entry point for measuring the maximum length and width of green beans in an image. It uses a Post-it note of known side length for scale calibration, segments beans by color, computes each bean’s dimensions in millimeters, annotates the image, and saves a timestamped output.

---

## 2. Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- NumPy

Dependencies are listed in `requirements.txt`.

---

## 3. Command-Line Interface

```bash
python measure_beans.py <input_image> [--debug]
```

- `<input_image>`: Path to an image containing green beans and a Post-it note.
- `--debug`: If set, prints detailed info and shows intermediate mask windows.

---

## 4. Processing Pipeline

### 4.1 Load Image
- Uses `cv2.imread()` to load the input. Fails with `IOError` on error.
- Copies the original for annotation.

### 4.2 Color Space Conversion
- Converts BGR→HSV and BGR→LAB for robust thresholding.

### 4.3 Post-it Detection & Scale Calibration
1. Threshold HSV by a yellowish range to isolate the Post-it.
2. Morphologically clean mask and find contours.
3. Select the largest rectangular contour.
4. Order its four corners via `order_corners()`.
5. Compute perspective transform matrix `M` and inverse `M_inv`.
6. Map square size (in px) to known side length (`postit_side_mm`) → mm/px scale.

### 4.4 Bean Segmentation
1. Threshold HSV by `bean_lower`/`bean_upper` to isolate bean regions.
2. Clean with erosion/dilation.
3. Find bean contours.

### 4.5 Bean Measurement
For each bean contour:
1. Warp points to "top-down" view (1 px = 1 mm) using `M`.
2. Compute covariance matrix of warped points.
3. Eigen-decomposition yields major/minor axes directions.
4. Project points on each axis to get min/max → length_mm and width_mm.
5. Draw lines and annotate each measurement on the original image.
6. Collect `(bean_id, "length", length_mm)` and `(bean_id, "width", width_mm)`.

### 4.6 Save & Output
- Generate timestamp: `datetime.now().strftime("%Y%m%d_%H%M%S")`.
- Filename: `<timestamp>_measured.jpg`.
- `cv2.imwrite()` saves annotated image.
- Returns `(annotated_img, results_list)`.

---

## 5. Function Reference

### `order_corners(pts)`
```python
def order_corners(pts: array-like) -> np.ndarray
``` 
- **Purpose**: Sorts four (x,y) points to TL, TR, BR, BL order.
- **Input**: List/array of four points.
- **Returns**: `np.ndarray` shape (4,2).

### `measure_beans(image_path, postit_side_mm=76.0, bean_lower=(30,40,40), bean_upper=(85,255,255), debug=False)`
```python
def measure_beans(image_path: str,
                  postit_side_mm: float,
                  bean_lower: Tuple[int,int,int],
                  bean_upper: Tuple[int,int,int],
                  debug: bool) -> Tuple[np.ndarray,List]
```  
- **Loads** the image and calibrates scale via Post-it.
- **Segments** beans by HSV range.
- **Computes** dimensions with linear algebra on warped contours.
- **Annotates** and **saves** the result.
- **Returns**: 
  - `annotated_img`: BGR image with overlays.
  - `results_list`: Measurement tuples.

---

## 6. Design Rationale

- **HSV Thresholding**: isolates colored objects robustly under varying lighting.
- **Perspective Transform**: corrects for camera angle, ensures metric measurements.
- **Covariance & Eigen**: finds principal axes of shape to measure true length/width.
- **Timestamped Output**: prevents overwrites and orders results chronologically.

---

## 7. Example

```bash
python measure_beans.py images/beans_sample.jpg --debug
```

Produces `20250429_131200_measured.jpg` with visual annotations and prints:
```
Finished measuring beans!
Bean info: [(1,'length',...),(1,'width',...),...]
```

---

## 8. Notes

- Tweak `bean_lower`/`bean_upper` HSV values for different bean colors.
- Ensure the Post-it is fully visible in the frame.
- Use `--debug` to fine-tune thresholds by inspecting masks.

---

## 9. Deep Dive: Algorithm Details

### 9.1 Color Space Selection & Thresholding
- Convert BGR→HSV for hue-based color isolation:
  ```python
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  mask_postit = cv2.inRange(hsv, postit_lower, postit_upper)
  ```
- HSV separates chromatic (H) from intensity (V), robust to lighting.

### 9.2 Morphological Cleaning
- Remove noise via erosion then dilation:
  ```python
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
  mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  ```
- Ensures contiguous regions for reliable contouring.

### 9.3 Contour Extraction & Filtering
- `contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`
- Filter by area: discard small blobs
- For Post-it, select the largest quadrilateral (4 vertices) after `cv2.approxPolyDP`.

### 9.4 Corner Ordering & Homography
- `order_corners()` sorts by sum and y-coordinate to TL, TR, BR, BL.
- Compute homography:
  ```python
  dst = np.array([[0,0],[px,0],[px,px],[0,px]], dtype=np.float32)
  M = cv2.getPerspectiveTransform(src_pts, dst)
  M_inv = np.linalg.inv(M)
  ```
- `px` = side length in pixels → mapped to `postit_side_mm` in mm.

### 9.5 Metric Conversion & Warping
- Warp bean contours: 1 px→1 mm in warped space
  ```python
  cnt_mm = cv2.perspectiveTransform(cnt_float, M)
  ```
- Now measurements on `cnt_mm` are in mm units directly.

### 9.6 Eigen Decomposition for Dimensions
- Compute covariance: `cov = np.cov(cnt_mm.T)`
- Eigen decomposition: `eigvals, eigvecs = np.linalg.eigh(cov)`
- Major axis = `eigvecs[:,1]`, minor = `eigvecs[:,0]`
- Project points: `proj = (cnt_mm - mean) @ axis` → compute min/max → length_mm, width_mm.

### 9.7 Annotation & Inverse Mapping
- Draw lines in orig image:
  ```python
  p1_img = tuple(int(x) for x in cv2.perspectiveTransform([[p1_mm]], M_inv)[0][0])
  cv2.line(orig, p1_img, p2_img, color, 2)
  ```
- Place labels at midpoint with `cv2.putText`.

### 9.8 Debug Visualizations
- If `debug`, display intermediate masks and warped contours:
  ```python
  cv2.imshow("Post-it Mask", mask_postit)
  cv2.waitKey(0)
  ```
- Helps tune thresholds and validate geometry.

### 9.9 Performance & Tuning
- Kernel sizes, HSV ranges, and contour approximation epsilon affect accuracy.
- For large batches, consider multi-threading per bean.
