import cv2
import pyautogui
from gaze_tracking import GazeTracking
import numpy as np

# Initialize gaze tracking
gaze = GazeTracking()

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Smoothing class to reduce jitter
class ExponentialSmoother:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.previous_x = None
        self.previous_y = None

    def smooth(self, x, y):
        if self.previous_x is None or self.previous_y is None:
            self.previous_x, self.previous_y = x, y

        smoothed_x = self.alpha * x + (1 - self.alpha) * self.previous_x
        smoothed_y = self.alpha * y + (1 - self.alpha) * self.previous_y

        self.previous_x, self.previous_y = smoothed_x, smoothed_y
        return int(smoothed_x), int(smoothed_y)

# Initialize smoother for cursor
smoother = ExponentialSmoother(alpha=0.2)

def map_gaze_to_screen(horizontal_ratio, vertical_ratio):
    """
    Map gaze direction ratios to screen coordinates.

    Args:
        horizontal_ratio (float): Gaze direction ratio horizontally (0.0-1.0).
        vertical_ratio (float): Gaze direction ratio vertically (0.0-1.0).

    Returns:
        (int, int): Screen coordinates corresponding to the gaze direction.
    """
    x = max(0, min(screen_width, int(horizontal_ratio * screen_width)))
    y = max(0, min(screen_height, int(vertical_ratio * screen_height)))
    return x, y

def main():
    # Open webcam
    webcam = cv2.VideoCapture(0)

    print("Starting gaze-based cursor control. Press 'q' to exit.")
    while True:
        # Read frame from webcam
        ret, frame = webcam.read()
        if not ret:
            print("Webcam not detected.")
            break

        # Update gaze tracker with the current frame
        gaze.refresh(frame)

        # Get gaze direction ratios
        horizontal_ratio = gaze.horizontal_ratio()
        vertical_ratio = gaze.vertical_ratio()

        if horizontal_ratio is not None and vertical_ratio is not None:
            # Map gaze ratios to screen coordinates
            screen_x, screen_y = map_gaze_to_screen(horizontal_ratio, vertical_ratio)

            # Smooth cursor movement
            smooth_x, smooth_y = smoother.smooth(screen_x, screen_y)

            # Move the cursor to the smoothed coordinates
            pyautogui.moveTo(smooth_x, smooth_y, duration=0.05)

        # Annotate and display the frame
        annotated_frame = gaze.annotated_frame()
        cv2.imshow("Gaze Tracking", annotated_frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
