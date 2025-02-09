import cv2
import pyautogui
from gaze_tracking import GazeTracking
import time
import numpy as np
from collections import deque

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize gaze tracking
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# Calibration points on the screen
calibration_points = [
    (screen_width // 2, screen_height // 2),  # Center
    (20, 20),  # Top-left
    (screen_width - 20, 20),  # Top-right
    (20, screen_height - 20),  # Bottom-left
    (screen_width - 20, screen_height - 20),  # Bottom-right
]

# Calibration data
calibrated_horizontal = []
calibrated_vertical = []

# Buffers for gaze ratios and exponential moving average state
gaze_buffer = {"horizontal": deque(maxlen=30), "vertical": deque(maxlen=30)}
cursor_position = {"x": screen_width // 2, "y": screen_height // 2}  # Start at screen center

# Dead zone radius in pixels
DEAD_ZONE_RADIUS = 40  # Adjust to control sensitivity


def smooth_with_ema(new_value, current_value, alpha=0.2):
    """
    Smooth the cursor position using exponential moving average (EMA).

    Args:
        new_value (float): New position value.
        current_value (float): Current smoothed value.
        alpha (float): Smoothing factor (0 < alpha <= 1). Higher values make it more responsive.

    Returns:
        float: Smoothed value.
    """
    return alpha * new_value + (1 - alpha) * current_value


def calibrate():
    """Calibrate gaze ratios to screen coordinates."""
    calibrated_horizontal.clear()
    calibrated_vertical.clear()
    print("Starting calibration...")

    for i, (screen_x, screen_y) in enumerate(calibration_points):
        # Move cursor to calibration point
        pyautogui.moveTo(screen_x, screen_y)
        print(f"Step {i+1}/{len(calibration_points)}: Look at ({screen_x}, {screen_y})")
        time.sleep(2)

        # Capture gaze ratios
        horizontal_ratio, vertical_ratio = None, None
        while horizontal_ratio is None or vertical_ratio is None:
            _, frame = webcam.read()
            gaze.refresh(frame)
            horizontal_ratio = gaze.horizontal_ratio()
            vertical_ratio = gaze.vertical_ratio()

        # Ensure valid ratios before adding to calibration
        if horizontal_ratio is not None and vertical_ratio is not None:
            calibrated_horizontal.append((horizontal_ratio, screen_x))
            calibrated_vertical.append((vertical_ratio, screen_y))

    if len(calibrated_horizontal) == len(calibration_points) and len(calibrated_vertical) == len(calibration_points):
        print("Calibration complete!")
    else:
        print("Calibration incomplete. Check camera and retry.")


def map_gaze_to_screen(horizontal_ratio, vertical_ratio):
    """Map gaze ratios to screen coordinates using weighted interpolation."""
    if not calibrated_horizontal or not calibrated_vertical:
        print("Calibration data is missing. Skipping gaze mapping.")
        return None, None

    # Weighted interpolation for horizontal
    weights_h = [(1 / abs(horizontal_ratio - h_ratio) if horizontal_ratio != h_ratio else 1e6, screen_x)
                 for h_ratio, screen_x in calibrated_horizontal]
    total_weight_h = sum(w for w, _ in weights_h)
    if total_weight_h == 0:
        return None, None
    screen_x = sum(w * x for w, x in weights_h) / total_weight_h

    # Weighted interpolation for vertical
    weights_v = [(1 / abs(vertical_ratio - v_ratio) if vertical_ratio != v_ratio else 1e6, screen_y)
                 for v_ratio, screen_y in calibrated_vertical]
    total_weight_v = sum(w for w, _ in weights_v)
    if total_weight_v == 0:
        return None, None
    screen_y = sum(w * y for w, y in weights_v) / total_weight_v

    # Ensure valid screen coordinates
    if np.isnan(screen_x) or np.isnan(screen_y):
        return None, None

    return int(screen_x), int(screen_y)


def is_outside_dead_zone(new_x, new_y, current_x, current_y, radius):
    """
    Check if the new position is outside the dead zone radius.

    Args:
        new_x, new_y (int): New cursor coordinates.
        current_x, current_y (int): Current cursor coordinates.
        radius (int): Dead zone radius.

    Returns:
        bool: True if outside the dead zone, False otherwise.
    """
    distance = np.sqrt((new_x - current_x) ** 2 + (new_y - current_y) ** 2)
    return distance > radius


# Start calibration
calibrate()

# Main loop for gaze tracking
while True:
    _, frame = webcam.read()
    gaze.refresh(frame)

    # Add gaze ratios to buffers
    gaze_buffer["horizontal"].append(gaze.horizontal_ratio())
    gaze_buffer["vertical"].append(gaze.vertical_ratio())

    # Get averaged gaze ratios
    horizontal_ratio = np.median(gaze_buffer["horizontal"]) if gaze_buffer["horizontal"] else None
    vertical_ratio = np.median(gaze_buffer["vertical"]) if gaze_buffer["vertical"] else None

    # Map gaze to screen position
    if horizontal_ratio is not None and vertical_ratio is not None:
        target_x, target_y = map_gaze_to_screen(horizontal_ratio, vertical_ratio)
        if target_x is not None and target_y is not None:
            # Check if the new position is outside the dead zone
            if is_outside_dead_zone(target_x, target_y, cursor_position["x"], cursor_position["y"], DEAD_ZONE_RADIUS):
                # Smooth the cursor movement using EMA
                cursor_position["x"] = smooth_with_ema(target_x, cursor_position["x"])
                cursor_position["y"] = smooth_with_ema(target_y, cursor_position["y"])
                pyautogui.moveTo(int(cursor_position["x"]), int(cursor_position["y"]), duration=0.01)
                time.sleep(0.3)

    # Display the annotated frame
    frame = gaze.annotated_frame()
    cv2.imshow("Gaze Tracking", frame)

    # Exit on 'ESC' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()