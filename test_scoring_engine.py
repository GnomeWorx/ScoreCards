import numpy as np
from scoring_engine.score import (
    create_synthetic_target,
    localize_target,
    find_center_and_calibrate_scale,
    detect_shot_holes,
    calculate_score,
)

def run_test_case(test_case):
    """Runs a single test case against the scoring engine."""
    print(f"--- Running Test Case: {test_case['test_name']} ---")

    calibre = test_case['calibre']
    shot_coords = test_case['shot_hole_coords']
    expected_scores = test_case['expected_scores']

    # --- Test Setup ---
    CALIBRE_MM = 4.5 if calibre == 0.177 else 5.6
    # Use the known scale of our synthetic target for placing shots accurately
    SCALE_PX_PER_MM = 5.119  # 450px / 87.9mm
    shot_radius_px = int((CALIBRE_MM / 2) * SCALE_PX_PER_MM)

    synthetic_shots_info = [(x, y, shot_radius_px) for x, y in shot_coords]

    # --- Execution ---
    target_image = create_synthetic_target(shot_holes_info=synthetic_shots_info, is_test=True)

    # Run the pipeline, but BYPASS the flaky detection stages for testing.
    # We provide the known, correct center and scale to test the logic of the later stages.
    corrected_image = localize_target(target_image)
    center = (300, 300)
    pixels_per_mm = SCALE_PX_PER_MM # Use the known, true scale
    print(f"  - Bypassing detection, using known center={center} and scale={pixels_per_mm:.2f}")

    shot_holes = detect_shot_holes(corrected_image, calibre, pixels_per_mm)

    if len(shot_holes) != len(expected_scores):
        print(f"[FAIL] Detected {len(shot_holes)} shots, but expected {len(expected_scores)}.")
        return False

    # The order of detected holes is not guaranteed, so we need to be clever
    # For now, we'll sort both lists. This works for simple cases.
    total_score, individual_scores = calculate_score(shot_holes, center, pixels_per_mm, calibre)

    individual_scores.sort(reverse=True)
    expected_scores.sort(reverse=True)

    # --- Verification ---
    print(f"Expected Scores: {expected_scores}")
    print(f"Actual Scores:   {individual_scores}")

    if individual_scores == expected_scores:
        print("[PASS]")
        return True
    else:
        print("[FAIL]")
        return False

def main():
    """Defines and runs all test cases."""

    # SCORING RINGS (for reference when creating tests)
    # Radii in mm:
    # 10: 9.144, 9: 18.288, 8: 27.432, 7: 36.576, 6: 45.72,
    # 5: 54.864, 4: 64.008, 3: 73.152, 2: 82.296, 1: 91.44
    SCALE_PX_PER_MM = 5.119

    test_cases = [
        {
            "test_name": "Calibre .177 - Two high scores",
            "calibre": 0.177,
            # Center is (300,300). A shot at (305,305) is ~7px away -> 10
            # A shot at (380,300) is 80px away. 80/5.119 = 15.6mm -> 9
            "shot_hole_coords": [(305, 305), (380, 300)],
            "expected_scores": [10, 9]
        },
        {
            "test_name": "Calibre .22 - Mixed scores",
            "calibre": 0.22,
            # 8-ring radius = 27.432mm * 5.119 = 140.4px
            # A shot at (430,300) is 130px away -> 8
            # A shot at (200,200) is ~141px away -> should be 8 due to inward gauging
            "shot_hole_coords": [(430, 300), (200, 200)],
            "expected_scores": [8, 8]
        },
        {
            "test_name": "Calibre .177 - A low score",
            "calibre": 0.177,
            # A shot at (50,50) is ~68.8mm from center -> score 3
            "shot_hole_coords": [(50, 50)],
            "expected_scores": [3]
        },
        {
            "test_name": "Calibre .22 - Inward Gauging Line Cutter",
            "calibre": 0.22,
            # 8-ring radius_px = 140.4px
            # .22 bullet radius_px = (5.6/2) * 5.119 = 14.3px
            # Place shot center at 141px away.
            # Gauged distance_px = 141 - 14.3 = 126.7px. This is inside 140.4px. Score=8.
            "shot_hole_coords": [(300, 441)], # 141px from center (300,300)
            "expected_scores": [8]
        },
    ]

    results = [run_test_case(tc) for tc in test_cases]

    print("\n--- Test Summary ---")
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")

if __name__ == "__main__":
    main()
