import argparse
import cv2
import numpy as np

def localize_target(image):
    """
    Finds the target card in the image and applies perspective correction.
    """
    # Placeholder
    print("Stage 1: Localizing target and correcting perspective...")
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
    # These parameters may need tuning for real-world images
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

    # Scaled radii for PL14 target (derived from ISSF 50m Pistol target)
    # Each tuple is (score, max_radius_for_score_in_mm)
    SCORING_RINGS_MM = [
        (10, 9.144),
        (9, 18.288),
        (8, 27.432),
        (7, 36.576),
        (6, 45.72),
        (5, 54.864),
        (4, 64.008),
        (3, 73.152),
        (2, 82.296),
        (1, 91.44)
    ]

    # Get shot radius in mm for inward gauging
    CALIBRE_MM = 4.5 if calibre == 0.177 else 5.6
    shot_radius_mm = CALIBRE_MM / 2

    scores = []
    for hole_coords in shot_holes:
        # Calculate distance from target center to shot center in pixels
        dist_px = np.sqrt((hole_coords[0] - center[0])**2 + (hole_coords[1] - center[1])**2)

        # Convert distance to mm
        dist_mm = dist_px / pixels_per_mm

        # Apply inward gauging: subtract the shot's radius from the distance
        gauged_dist_mm = dist_mm - shot_radius_mm

        # Determine score
        shot_score = 0  # Default score is 0 if it's outside the 1-ring
        for score_value, ring_radius_mm in SCORING_RINGS_MM:
            if gauged_dist_mm <= ring_radius_mm:
                shot_score = score_value
                break # Stop at the first (highest) ring it touches

        scores.append(shot_score)
        print(f"  - Shot at {hole_coords}: dist={dist_mm:.2f}mm, gauged_dist={gauged_dist_mm:.2f}mm, score={shot_score}")

    total_score = sum(scores)
    return total_score, scores

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
    Main function to run the scoring process.
    """
    parser = argparse.ArgumentParser(description="Score NSRA PL14 target cards.")
    parser.add_argument("--image", help="Path to the target image (currently ignored, uses synthetic image).")
    parser.add_argument("--calibre", required=True, type=float, choices=[0.177, 0.22], help="Calibre of the shots (.177 or .22).")
    args = parser.parse_args()

    # Create a synthetic target and add some test shots to it.
    # To do this, we need to know the approximate size of the shot holes in pixels.
    # We use a pre-calculated approximate scale for this, which is a bit of a simplification
    # for this testing setup.
    CALIBRE_MM = 4.5 if args.calibre == 0.177 else 5.6
    APPROX_SCALE_PX_PER_MM = 5.1  # Based on our synthetic target's design (450px / 87.9mm)
    shot_radius_px = int((CALIBRE_MM / 2) * APPROX_SCALE_PX_PER_MM)

    # Place shots clearly on the white background, outside the black aiming mark.
    # The aiming mark has radius 225 centered at (300,300).
    # A shot at (100, 500) is ~282 pixels from the center.
    # A shot at (500, 100) is also ~282 pixels from the center.
    synthetic_shot_coords = [(100, 500), (500, 100)]
    synthetic_shots_info = [(x, y, shot_radius_px) for x, y in synthetic_shot_coords]

    synthetic_target_image = create_synthetic_target(shot_holes_info=synthetic_shots_info)

    # The --image argument is ignored, but we print it to show it's received.
    image_path = args.image if args.image else "synthetic_image"
    print(f"Processing image: {image_path} for calibre: {args.calibre}")


    # CV Pipeline
    # Stage 1 is skipped for now, as we are using a perfect, flat image.
    corrected_image = localize_target(synthetic_target_image)
    center, pixels_per_mm = find_center_and_calibrate_scale(corrected_image)
    shot_holes = detect_shot_holes(corrected_image, args.calibre, pixels_per_mm)
    total_score, individual_scores = calculate_score(shot_holes, center, pixels_per_mm, args.calibre)

    print(f"\n--- Scoring Complete ---")
    print(f"Total Score: {total_score}")
    print(f"Individual Scores: {individual_scores}")
    print("------------------------")


if __name__ == "__main__":
    main()
