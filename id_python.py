"""
ID Card Detector using OpenCV
-------------------------------
Detects a rectangular ID/document card in an image or live webcam feed,
then extracts and perspective-corrects it into a clean top-down scan.

Usage:
    python id_card_detector.py --image path/to/photo.jpg
    python id_card_detector.py --webcam

Controls (webcam mode):
    q  - quit
    s  - save the current detected/warped card as 'scanned_id.png'
"""

import cv2
import numpy as np
import argparse
import os
from datetime import datetime


def get_downloads_path():
    """Get the user's Downloads folder path (cross-platform)."""
    home = os.path.expanduser("~")
    downloads = os.path.join(home, "Downloads")
    os.makedirs(downloads, exist_ok=True)
    return downloads


def save_to_downloads(image, prefix="scanned_id"):
    """Save an image to the Downloads folder with a timestamped filename."""
    downloads = get_downloads_path()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(downloads, filename)
    cv2.imwrite(filepath, image)
    return filepath


def order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]      # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right has smallest diff
    rect[3] = pts[np.argmax(diff)]   # bottom-left has largest diff
    return rect


def four_point_transform(image, pts):
    """Warp the quadrilateral region defined by pts into a top-down view."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def find_card_contour(frame, min_area_ratio=0.1):
    """
    Find the largest 4-point contour in the frame that looks like a card.
    Returns the 4 corner points (or None if not found).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Close gaps in edges so the card outline is a solid contour
    kernel = np.ones((5, 5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    edged = cv2.erode(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = frame.shape[0] * frame.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours[:10]:
        area = cv2.contourArea(c)
        if area < frame_area * min_area_ratio:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None


def process_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f"Could not load image: {path}")
        return

    display = image.copy()
    corners = find_card_contour(image)

    if corners is not None:
        cv2.polylines(display, [corners.astype(int)], True, (0, 255, 0), 3)
        warped = four_point_transform(image, corners)

        saved_path = save_to_downloads(warped)
        print(f"ID card detected! Saved scanned copy to: {saved_path}")

        cv2.imshow("Detected Card", display)
        cv2.imshow("Scanned / Warped", warped)
    else:
        print("No card-like rectangle detected. Try a photo with better contrast "
              "against the background.")
        cv2.imshow("Original (no detection)", display)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Press 'q' to quit, 's' to save the current scan.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        corners = find_card_contour(frame)
        warped = None

        if corners is not None:
            cv2.polylines(display, [corners.astype(int)], True, (0, 255, 0), 3)
            warped = four_point_transform(frame, corners)
            cv2.imshow("Scanned Preview", warped)

        cv2.imshow("ID Card Detector - press q to quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and warped is not None:
            saved_path = save_to_downloads(warped)
            print(f"Saved to: {saved_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and scan ID cards using OpenCV")
    parser.add_argument("--image", type=str, help="Path to an image file containing an ID card")
    parser.add_argument("--webcam", action="store_true", help="Use live webcam feed instead")
    args = parser.parse_args()

    if args.webcam:
        process_webcam()
    elif args.image:
        process_image(args.image)
    else:
        print("Please provide --image <path> or use --webcam. Run with -h for help.")