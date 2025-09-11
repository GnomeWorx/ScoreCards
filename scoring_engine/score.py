import argparse
import cv2
import numpy as np

def order_points(pts):
    """
    Orders a list of 4 points corresponding to a rectangle in top-left,
    top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left has the smallest sum
    rect[2] = pts[np.argmax(s)] # Bottom-right has the largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right has the smallest difference
    rect[3] = pts[np.argmax(diff)] # Bottom-left has the largest difference

    return rect

def localize_target(image, output_size=600):
    """
    Finds the target card in the image and applies perspective correction.
    """
    print("Stage 1: Localizing target and correcting perspective...")

    # Pre-processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Lower Canny thresholds to detect the low-contrast edge of the card
    edged = cv2.Canny(blurred, 30, 150)

    # Find contours and sort them by area
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("  - No contours found. Returning original image.")
        return image

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Loop through the largest contours and find the first one with 4 points
    target_contour_approx = None
    for c in contours[:5]: # Check the 5 largest contours
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            target_contour_approx = approx
            break

    if target_contour_approx is not None:
        print("  - Found a 4-point contour, applying perspective transform.")
        # Order the points
        ordered_pts = order_points(target_contour_approx.reshape(4, 2))

        # Define the destination points for the warp
        # We'll warp it to a square image of size `output_size`
        dst_pts = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]], dtype="float32")

        # Compute the perspective transform matrix and apply it
        matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (output_size, output_size))

        return warped
    else:
        print("  - Could not find a 4-point contour in the largest contours. Returning original image.")
        return image

def find_center_and_calibrate_scale(corrected_image):
    """
    Finds the bullseye's center and determines the image scale using Hough Circle Transform.
    """
    print("Stage 2: Finding center and calibrating scale...")

    # Convert to grayscale if it's not already
    if len(corrected_image.shape) == 3:
        gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = corrected_image

    # Apply Gaussian blur to reduce noise and improve circle detection
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use Hough Circle Transform to find circles
    # These parameters were tuned for the synthetic target. They may not work on real images.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=10, maxRadius=0) # maxRadius=0 means it will find all sizes

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # We assume the largest detected circle is the main aiming mark
        largest_circle = sorted(circles, key=lambda x: x[2], reverse=True)[0]
        center = (largest_circle[0], largest_circle[1])
        radius_px = largest_circle[2]

        # Calibrate scale: NSRA PL14 aiming mark is 87.9mm in diameter
        AIMING_MARK_DIAMETER_MM = 87.9
        pixels_per_mm = (radius_px * 2) / AIMING_MARK_DIAMETER_MM

        print(f"  - Detected center at: {center}")
        print(f"  - Detected aiming mark radius: {radius_px}px")
        print(f"  - Calculated scale: {pixels_per_mm:.2f} pixels/mm")

        return center, pixels_per_mm
    else:
        print("  - No circles found! Using fallback values.")
        # Fallback if no circles are found
        height, width = corrected_image.shape[:2]
        center = (width // 2, height // 2)
        pixels_per_mm = 1.0
        return center, pixels_per_mm

def detect_shot_holes(corrected_image, calibre, pixels_per_mm):
    """
    Detects shot holes on the target face using Blob Detection.
    Filters blobs based on the expected size of the given calibre.
    """
    print(f"Stage 3: Detecting shot holes for calibre {calibre}...")

    # Convert to grayscale and apply inverse binary threshold
    gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
    # Use a moderate threshold to isolate the black holes.
    # A value too high might eliminate anti-aliased edges.
    _, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # --- Morphological Opening to remove rings ---
    # The rings are thin lines. An opening operation (erosion followed by dilation)
    # will remove them while preserving the larger, solid shot holes.
    # We use a kernel larger than the ring thickness and run it twice to be sure.
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # We are looking for white blobs on a black background.
    params.blobColor = 255

    # Filter by Area
    params.filterByArea = True
    CALIBRE_MM = 4.5 if calibre == 0.177 else 5.6
    hole_radius_px = (CALIBRE_MM / 2) * pixels_per_mm
    hole_area_px = np.pi * (hole_radius_px ** 2)
    # Allow for some tolerance in detected area
    params.minArea = hole_area_px * 0.7
    params.maxArea = hole_area_px * 1.3
    print(f"  - Filtering by area: {params.minArea:.2f}px - {params.maxArea:.2f}px")

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.9

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.6

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(binary_image)

    print(f"  - Detected {len(keypoints)} potential shot holes.")

    # Extract coordinates from keypoints
    shot_hole_coords = [ (int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

    return shot_hole_coords

def calculate_score(shot_holes, center, pixels_per_mm, calibre):
    """
    Assigns a score to each detected shot hole based on NSRA inward gauging rules.
    """
    print(f"Stage 4: Calculating score for calibre {calibre}...")

    SCORING_RINGS_MM = [
        (10, 9.144), (9, 18.288), (8, 27.432), (7, 36.576), (6, 45.72),
        (5, 54.864), (4, 64.008), (3, 73.152), (2, 82.296), (1, 91.44)
    ]
    CALIBRE_MM = 4.5 if calibre == 0.177 else 5.6
    shot_radius_mm = CALIBRE_MM / 2

    scores = []
    for hole_coords in shot_holes:
        dist_px = np.sqrt((hole_coords[0] - center[0])**2 + (hole_coords[1] - center[1])**2)
        dist_mm = dist_px / pixels_per_mm
        gauged_dist_mm = dist_mm - shot_radius_mm

        shot_score = 0
        for score_value, ring_radius_mm in SCORING_RINGS_MM:
            if gauged_dist_mm <= ring_radius_mm:
                shot_score = score_value
                break

        scores.append(shot_score)
        print(f"  - Shot at {hole_coords}: dist={dist_mm:.2f}mm, gauged_dist={gauged_dist_mm:.2f}mm, score={shot_score}")

    total_score = sum(scores)
    return total_score, scores

def draw_results_on_image(image, shot_holes, scores, pixels_per_mm, calibre):
    """Draws the detected shot holes and their scores on the image."""
    annotated_image = image.copy()
    CALIBRE_MM = 4.5 if calibre == 0.177 else 5.6
    shot_radius_px = int((CALIBRE_MM / 2) * pixels_per_mm)

    for i, coords in enumerate(shot_holes):
        # Draw a circle around the shot
        cv2.circle(annotated_image, coords, shot_radius_px + 3, (0, 255, 0), 2) # Green circle

        # Put the score text next to the shot
        score_text = str(scores[i])
        text_coords = (coords[0] + shot_radius_px, coords[1] - shot_radius_px)
        cv2.putText(annotated_image, score_text, text_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Red text

    return annotated_image

def run_scoring_pipeline(image_path, calibre):
    """
    Runs the full CV pipeline on an image and returns the results.
    This function is designed to be called from other modules (like the GUI).
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from path: {image_path}")
        return None

    # Run the pipeline
    corrected_image = localize_target(image)
    center, pixels_per_mm = find_center_and_calibrate_scale(corrected_image)
    shot_holes = detect_shot_holes(corrected_image, calibre, pixels_per_mm)
    total_score, individual_scores = calculate_score(shot_holes, center, pixels_per_mm, calibre)

    # Annotate the image with results
    annotated_image = draw_results_on_image(corrected_image, shot_holes, individual_scores, pixels_per_mm, calibre)

    return {
        "total_score": total_score,
        "individual_scores": individual_scores,
        "annotated_image": annotated_image,
        "shot_holes": shot_holes,
    }

def create_synthetic_target(width=600, height=600, shot_holes_info=None, is_test=False):
    """
    Generates a synthetic target image for testing, optionally with shot holes.
    The aiming mark (largest circle) is created with a known pixel diameter
    to allow for scale calibration verification.
    shot_holes_info: A list of tuples, where each tuple is (x, y, radius) for a shot hole.
    is_test: If True, skips drawing the black aiming mark background to ensure all shots are visible.
    """
    print("Creating synthetic target image...")
    # Create a white background
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    center = (width // 2, height // 2)

    # Define rings (we'll just draw a few for visual effect)
    # The key is the aiming mark, which we'll set to a specific pixel radius.
    # Let's make the aiming mark diameter 450px.
    # Known real diameter is 87.9mm.
    # So, scale should be ~ 450 / 87.9 = 5.12 px/mm
    aiming_mark_radius_px = 225

    rings = [
        (aiming_mark_radius_px, (0, 0, 0), -1),  # The black aiming mark (filled)
        (aiming_mark_radius_px, (0,0,0), 2), # Draw the outer ring line even in test mode
        (150, (0, 0, 0), 2),
        (100, (0, 0, 0), 2),
        (50, (0, 0, 0), 2),
        (25, (255, 255, 255), -1), # small white dot in center
    ]

    # In test mode, don't draw the big black background, just the rings.
    if is_test:
        rings.pop(0) # Remove the filled aiming mark

    for radius, color, thickness in rings:
        cv2.circle(image, center, radius, color, thickness)

    # Add simulated shot holes
    if shot_holes_info:
        print(f"  - Adding {len(shot_holes_info)} synthetic shot holes...")
        for x, y, r in shot_holes_info:
            cv2.circle(image, (x, y), r, (0, 0, 0), -1) # Black, filled circles

    return image

def main():
    """
    Main function for command-line execution.
    """
    parser = argparse.ArgumentParser(description="Score NSRA PL14 target cards.")
    parser.add_argument("--image", required=True, help="Path to the target image.")
    parser.add_argument("--calibre", required=True, type=float, choices=[0.177, 0.22], help="Calibre of the shots (.177 or .22).")
    args = parser.parse_args()

    print(f"Processing image: {args.image} for calibre: {args.calibre}")

    results = run_scoring_pipeline(args.image, args.calibre)

    if results:
        print(f"\n--- Scoring Complete ---")
        print(f"Total Score: {results['total_score']}")
        print(f"Individual Scores: {results['individual_scores']}")
        print("------------------------")
        # In a real application, you would display or save the annotated_image
        # cv2.imshow("Annotated Image", results['annotated_image'])
        # cv2.waitKey(0)

if __name__ == "__main__":
    main()
