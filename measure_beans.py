import cv2
import numpy as np
import argparse
import datetime


def order_corners(pts):
    """Return corners ordered TL, TR, BR, BL."""
    pts_sorted = sorted(pts, key=lambda p: p[0] + p[1])
    tl, br = pts_sorted[0], pts_sorted[3]
    tr, bl = pts_sorted[1], pts_sorted[2]
    if tr[1] > bl[1]:
        tr, bl = bl, tr
    return np.array([tl, tr, br, bl], dtype=np.float32)


def measure_beans(image_path,
                  postit_side_mm=76.0,
                  bean_lower=(30, 40, 40),
                  bean_upper=(85, 255, 255),
                  debug=False):
    """
    Measure straight-line length and maximum thickness of each bean.
    The Post-it square supplies the pixel-to-mm scale.
    """

    # 1 · load
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load image: {image_path}")
    orig = image.copy()

    # 2 · HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # --- 3 · POST-IT DETECTION ----------------------------------------------
    mask_hsv = cv2.inRange(hsv, (15, 25, 60), (45, 255, 255))
    mask_lab = cv2.inRange(lab, (0, 0, 120), (255, 120, 255))

    mask_postit = cv2.bitwise_or(mask_hsv, mask_lab)
    mask_postit = cv2.morphologyEx(
        mask_postit, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    square_cnt, max_area = None, 0
    cnts_p, _ = cv2.findContours(mask_postit, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts_p:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            # discard shapes that touch the picture edge
            if cv2.pointPolygonTest(approx, (0, 0), False) >= 0 or \
               cv2.pointPolygonTest(approx, (image.shape[1] - 1, 0), False) >= 0 or \
               cv2.pointPolygonTest(approx, (image.shape[1] - 1, image.shape[0] - 1), False) >= 0 or \
               cv2.pointPolygonTest(approx, (0, image.shape[0] - 1), False) >= 0:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            if 0.85 < w / float(h) < 1.15 and area > max_area:
                square_cnt, max_area = approx, area

    if square_cnt is None and cnts_p:
        cnts4 = [cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
                 for c in cnts_p]
        cnts4 = [c for c in cnts4 if len(c) == 4]
        if cnts4:
            square_cnt = max(cnts4, key=cv2.contourArea)

    if debug and square_cnt is not None:
        mask_only_note = np.zeros(mask_postit.shape, np.uint8)
        cv2.drawContours(mask_only_note, [square_cnt], -1, 255, -1)
        h_note = hsv[..., 0][mask_only_note == 255]
        s_note = hsv[..., 1][mask_only_note == 255]
        v_note = hsv[..., 2][mask_only_note == 255]
        print(f"[DEBUG] Post-it HSV ranges: "
              f"H {h_note.min()}-{h_note.max()}, "
              f"S {s_note.min()}-{s_note.max()}, "
              f"V {v_note.min()}-{v_note.max()}")

    if square_cnt is None:
        raise ValueError("Post-it not found in image")

    # build a tight mask containing only the chosen note
    mask_postit = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_postit, [square_cnt], -1, 255, -1)

    # --- 4 · SCALE CALCULATION ----------------------------------------------
    M = cv2.getPerspectiveTransform(
        order_corners(square_cnt.reshape(4, 2)),
        np.array([[0, 0],
                  [postit_side_mm, 0],
                  [postit_side_mm, postit_side_mm],
                  [0, postit_side_mm]], dtype=np.float32))
    M_inv = np.linalg.inv(M)

    # 5 · bean mask
    mask_beans = cv2.inRange(hsv, bean_lower, bean_upper)
    mask_beans = cv2.bitwise_and(mask_beans, cv2.bitwise_not(mask_postit))
    mask_beans = cv2.morphologyEx(mask_beans, cv2.MORPH_CLOSE,
                                  np.ones((5, 5), np.uint8))
    mask_beans = cv2.morphologyEx(mask_beans, cv2.MORPH_OPEN,
                                  np.ones((3, 3), np.uint8))

    if debug:
        cv2.imshow("tight post-it mask", mask_postit)
        cv2.imshow("bean mask", mask_beans)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    bean_cnts, _ = cv2.findContours(mask_beans, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    bean_cnts = [c for c in bean_cnts if cv2.contourArea(c) > 50]

    results, bean_id = [], 0

    # ------------------------------------------------------------------
    # 6 · measure each bean
    # ------------------------------------------------------------------
    for cnt in bean_cnts:
        bean_id += 1

        pts_warp = cv2.perspectiveTransform(
            cnt.astype(np.float32).reshape(-1, 1, 2), M).reshape(-1, 2)
        if len(pts_warp) < 5:
            continue

        mean = pts_warp.mean(axis=0)
        cov = np.cov((pts_warp - mean).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_dir = eigvecs[:, 1] / np.linalg.norm(eigvecs[:, 1])
        minor_dir = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])

        major_proj = (pts_warp - mean) @ major_dir
        minor_proj = (pts_warp - mean) @ minor_dir

        len_min, len_max = major_proj.min(), major_proj.max()
        length_mm = len_max - len_min
        p_len_1 = mean + major_dir * len_min
        p_len_2 = mean + major_dir * len_max

        bin_size = 0.5
        bins = np.floor(major_proj / bin_size).astype(int)
        max_w, x_at_max, y_min_at, y_max_at = 0, None, None, None
        for b in np.unique(bins):
            idx = bins == b
            if idx.sum() < 2:
                continue
            y_min, y_max = minor_proj[idx].min(), minor_proj[idx].max()
            w = y_max - y_min
            if w > max_w:
                max_w = w
                x_at_max = major_proj[idx].mean()
                y_min_at, y_max_at = y_min, y_max

        p_wid_1 = mean + major_dir * x_at_max + minor_dir * y_min_at
        p_wid_2 = mean + major_dir * x_at_max + minor_dir * y_max_at
        width_mm = max_w

        def _back(pt):
            return tuple(np.int32(cv2.perspectiveTransform(
                np.array([[pt]], np.float32), M_inv)[0][0]))

        len_p1, len_p2 = _back(p_len_1), _back(p_len_2)
        wid_p1, wid_p2 = _back(p_wid_1), _back(p_wid_2)

        cv2.line(orig, len_p1, len_p2, (255, 0, 0), 2)
        cv2.line(orig, wid_p1, wid_p2, (0, 255, 0), 2)

        mid_l = ((len_p1[0] + len_p2[0]) // 2,
                 (len_p1[1] + len_p2[1]) // 2)
        mid_w = ((wid_p1[0] + wid_p2[0]) // 2,
                 (wid_p1[1] + wid_p2[1]) // 2)

        cv2.putText(orig, f"{length_mm:.1f} mm", mid_l,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(orig, f"{length_mm:.1f} mm", mid_l,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(orig, f"{width_mm:.1f} mm", mid_w,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(orig, f"{width_mm:.1f} mm", mid_w,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        results.extend([(bean_id, "length", length_mm),
                        (bean_id, "width", width_mm)])

        if debug:
            print(f"[DEBUG] Bean {bean_id}: length={length_mm:.2f} mm, "
                  f"width={width_mm:.2f} mm")

    # 7 · save output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{timestamp}_measured.jpg"
    cv2.imwrite(out_name, orig)
    if debug:
        print(f"[INFO] Output saved to: {out_name}")
        print("[INFO] Bean measurements:", results)

    return orig, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure bean length and width from an image using a Post-it for scale.")
    parser.add_argument("input_image", type=str,
                        help="Path to the input image file containing beans and a Post-it note.")
    parser.add_argument("--debug", action="store_true",
                        help="Show intermediate masks for debugging")
    args = parser.parse_args()

    annotated_img, bean_info = measure_beans(
        image_path=args.input_image,
        postit_side_mm=76.0,
        bean_lower=(30, 40, 40),
        bean_upper=(85, 255, 255),
        debug=args.debug
    )
    print("Finished measuring beans!")
    print("Bean info:", bean_info)