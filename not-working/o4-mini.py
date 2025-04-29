import cv2
import numpy as np

def order_corners(pts):
    """
    Sorts 4 corner points in the order:
    top-left, top-right, bottom-right, bottom-left.
    This is needed before computing the perspective transform.
    """
    # Sort by sum of coordinates
    pts_sorted = sorted(pts, key=lambda p: p[0] + p[1])
    tl, br = pts_sorted[0], pts_sorted[3]
    tr, bl = pts_sorted[1], pts_sorted[2]
    # Ensure tr is top-right by comparing y-values
    if tr[1] > bl[1]:
        tr, bl = bl, tr
    return np.array([tl, tr, br, bl], dtype=np.float32)

def measure_beans(image_path,
                  postit_side_mm=76.0,
                  postit_lower=(20, 80, 80),
                  postit_upper=(40, 255, 255),
                  bean_lower=(35, 50, 50),
                  bean_upper=(90, 255, 255),
                  debug=False):
    """
    Measures the maximum internal width and length of green beans in an image using 
    a known-size Post-it for calibration. Returns an annotated image and 
    a list of (bean_id, dimension_type, measurement_mm).
    """

    # 1. Load Image
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load image: {image_path}")

    orig = image.copy()  # Keep an original copy for final annotation

    # 2. Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 3. Detect the Post-it note for calibration
    mask_postit = cv2.inRange(hsv, postit_lower, postit_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask_postit = cv2.morphologyEx(mask_postit, cv2.MORPH_CLOSE, kernel)

    contours_postit, _ = cv2.findContours(mask_postit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    square_contour = None
    max_area = 0
    for cnt in contours_postit:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > max_area:
            max_area = area
            square_contour = approx

    if square_contour is None:
        raise Exception("Could not find a valid Post-it note contour. Check HSV range or image setup.")

    square_contour = square_contour.reshape((4, 2))
    corners = order_corners(square_contour)

    # 4. Compute perspective transform
    dst_pts = np.array([
        [0, 0],
        [postit_side_mm, 0],
        [postit_side_mm, postit_side_mm],
        [0, postit_side_mm]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst_pts)
    M_inv = np.linalg.inv(M)

    # 5. Detect beans
    mask_beans = cv2.inRange(hsv, bean_lower, bean_upper)
    mask_beans = cv2.bitwise_and(mask_beans, cv2.bitwise_not(mask_postit))
    mask_beans = cv2.morphologyEx(mask_beans, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    mask_beans = cv2.morphologyEx(mask_beans, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    bean_contours, _ = cv2.findContours(mask_beans, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bean_contours = [cnt for cnt in bean_contours if cv2.contourArea(cnt) > 50]

    # 6. For each bean contour, measure width and length
    results = []
    bean_id = 0

    for cnt in bean_contours:
        bean_id += 1

        cnt_float = cnt.astype(np.float32)
        cnt_reshaped = cnt_float.reshape(-1, 1, 2)
        cnt_warped = cv2.perspectiveTransform(cnt_reshaped, M).reshape(-1, 2)

        num_points = len(cnt_warped)
        if num_points < 4:
            print(f"Skipping bean {bean_id} - too few contour points ({num_points})")
            continue

        # --- NEW MAX-WIDTH CALCULATION USING STRAIGHT-SEGMENT INSIDE CHECK ---
        max_width_mm = 0.0
        best_p1_warped = None
        best_p2_warped = None

        def segment_inside(contour, p1, p2, samples=10):
            """Returns True if all sampled points along [p1,p2] lie inside the contour."""
            for t in np.linspace(0, 1, samples):
                pt = ((1 - t) * p1 + t * p2).astype(np.float32)
                if cv2.pointPolygonTest(contour, tuple(pt), False) < 0:
                    return False
            return True

        # Search all chords for the longest one fully inside
        for i in range(num_points):
            p1 = cnt_warped[i]
            for j in range(i + 1, num_points):
                p2 = cnt_warped[j]
                d = np.linalg.norm(p1 - p2)
                if d <= max_width_mm:
                    continue
                if segment_inside(cnt_warped, p1, p2, samples=10):
                    max_width_mm = d
                    best_p1_warped = p1
                    best_p2_warped = p2

        # If no valid width chord found, skip
        if best_p1_warped is None or best_p2_warped is None:
            print(f"Warning: Could not determine valid width for bean {bean_id}")
            continue

        # Transform width endpoints back to original image
        p1_t = np.array([[best_p1_warped]], dtype=np.float32)
        p2_t = np.array([[best_p2_warped]], dtype=np.float32)
        p1_orig = tuple(np.intp(cv2.perspectiveTransform(p1_t, M_inv)[0][0]))
        p2_orig = tuple(np.intp(cv2.perspectiveTransform(p2_t, M_inv)[0][0]))

        # Draw the max width line in green
        cv2.line(orig, p1_orig, p2_orig, (0, 255, 0), 2)
        width_mid = tuple(np.intp((np.array(p1_orig) + np.array(p2_orig)) / 2))
        width_label = f"{max_width_mm:.1f} mm"
        cv2.putText(orig, width_label, width_mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(orig, width_label, width_mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        results.append((bean_id, "width", max_width_mm))
        if debug:
            print(f"[DEBUG] Bean {bean_id}: Width={max_width_mm:.2f} mm")

        # --- LENGTH CALCULATION (UNCHANGED) ---
        # Calculate length as the longest chord roughly perpendicular to width
        width_vec = (best_p2_warped - best_p1_warped)
        width_dir = width_vec / np.linalg.norm(width_vec)

        max_length_mm = 0.0
        best_len_p1 = None
        best_len_p2 = None
        for i in range(num_points):
            p1 = cnt_warped[i]
            for j in range(i + 1, num_points):
                p2 = cnt_warped[j]
                vec = p2 - p1
                length = np.linalg.norm(vec)
                if length <= max_length_mm:
                    continue
                if length == 0:
                    continue
                dir_vec = vec / length
                align = abs(np.dot(width_dir, dir_vec))
                mid = (p1 + p2) / 2
                if align < 0.3 and cv2.pointPolygonTest(cnt_warped, tuple(mid), False) >= 0:
                    max_length_mm = length
                    best_len_p1 = p1
                    best_len_p2 = p2

        if best_len_p1 is not None and best_len_p2 is not None:
            lp1_t = np.array([[best_len_p1]], dtype=np.float32)
            lp2_t = np.array([[best_len_p2]], dtype=np.float32)
            lp1_orig = tuple(np.intp(cv2.perspectiveTransform(lp1_t, M_inv)[0][0]))
            lp2_orig = tuple(np.intp(cv2.perspectiveTransform(lp2_t, M_inv)[0][0]))

            # Draw the max length line in blue
            cv2.line(orig, lp1_orig, lp2_orig, (255, 0, 0), 2)
            length_mid = tuple(np.intp((np.array(lp1_orig) + np.array(lp2_orig)) / 2))
            length_label = f"{max_length_mm:.1f} mm"
            cv2.putText(orig, length_label, length_mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(orig, length_label, length_mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            results.append((bean_id, "length", max_length_mm))
            if debug:
                print(f"[DEBUG] Bean {bean_id}: Length={max_length_mm:.2f} mm)")
        else:
            print(f"Warning: Could not determine valid length for bean {bean_id}")

    # 7. Save and return
    out_name = "output_width_length.jpg"
    cv2.imwrite(out_name, orig)
    if debug:
        print(f"[INFO] Processed image saved as: {out_name}")
        print("[INFO] Bean measurements (mm):", results)

    return orig, results


if __name__ == "__main__":
    input_image = "C:\\beans\\beans_example.jpg"  # <-- Replace with your image path
    annotated_img, bean_info = measure_beans(
        image_path=input_image,
        postit_side_mm=76.0,
        postit_lower=(35, 0, 180),
        postit_upper=(55, 80, 255),
        bean_lower=(20, 100, 100),
        bean_upper=(70, 255, 255),
        debug=True
    )
    print("Finished measuring beans!")
    print("Bean info:", bean_info)