import cv2, os, math, urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# FIX: auto-download model so it never crashes on a missing file
MODEL = "hand_landmarker.task"
if not os.path.exists(MODEL):
    print("Downloading hand model (one time only)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL)

detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL),
        num_hands=2,
        running_mode=vision.RunningMode.IMAGE))

EFFECTS = ["glitch", "edges", "thermal", "night_vision", "invert", "none"]
effect_index = 0

def apply_effect(img, fx):
    if img is None or img.size == 0:
        return img
    if fx == "invert":
        return cv2.bitwise_not(img)
    if fx == "edges":
        return cv2.cvtColor(cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200), cv2.COLOR_GRAY2BGR)
    if fx == "thermal":
        return cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    if fx == "night_vision":
        g = img.astype(np.float32)
        g[:, :, 0] *= 0.2
        g[:, :, 1] = np.clip(g[:, :, 1] * 1.5 + 30, 0, 255)
        g[:, :, 2] *= 0.2
        return g.astype(np.uint8)
    if fx == "glitch":
        b, g, r = cv2.split(img)
        res = cv2.merge([np.roll(b, 8, axis=1), g, np.roll(r, -8, axis=1)])
        res[::4] = 0
        return res
    return img

PINCH = 0.12            # FIX: was 0.05 -> fingers had to be crushed together
HOLD = 5
counter, locked = 0, False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    if res.hand_landmarks and len(res.hand_landmarks) == 2:
        h1, h2 = res.hand_landmarks
        L, R = (h1, h2) if h1[0].x < h2[0].x else (h2, h1)

        size = lambda hd: max(math.hypot(hd[0].x - hd[9].x, hd[0].y - hd[9].y), 1e-3)
        dL = math.hypot(L[4].x - L[8].x, L[4].y - L[8].y) / size(L)
        dR = math.hypot(R[4].x - R[8].x, R[4].y - R[8].y) / size(R)
        pinching = dL < PINCH and dR < PINCH

        # FIX: one effect switch per pinch (release to re-arm)
        if pinching and not locked:
            counter += 1
            if counter >= HOLD:
                effect_index = (effect_index + 1) % len(EFFECTS)
                locked = True
        else:
            counter = 0
            if not pinching:
                locked = False

        # pinch feedback rings (green = pinched)
        for hd, d in ((L, dL), (R, dR)):
            c = (int(hd[8].x * w), int(hd[8].y * h))
            cv2.circle(frame, c, 10, (0, 255, 150) if d < PINCH else (160, 160, 160), 2, cv2.LINE_AA)

        quad = np.float32([[L[8].x * w, L[8].y * h],
                           [R[8].x * w, R[8].y * h],
                           [R[4].x * w, R[4].y * h],
                           [L[4].x * w, L[4].y * h]])

        # FIX: skip degenerate quads (singular matrix crash) and
        # reorder points so the quad can never self-intersect
        if cv2.contourArea(quad) > 1500:
            cxy = quad.mean(axis=0)
            quad = quad[np.argsort(np.arctan2(quad[:, 1] - cxy[1], quad[:, 0] - cxy[0]))]
            x, y = quad[:, 0], quad[:, 1]
            if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:   # fix winding
                quad = quad[::-1]

            bw, bh = 400, 200
            dst = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
            warped = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(quad, dst), (bw, bh))
            fx_img = apply_effect(warped, EFFECTS[effect_index])
            back = cv2.warpPerspective(fx_img, cv2.getPerspectiveTransform(dst, quad), (w, h))

            mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
            frame = np.where(mask[:, :, None] == 255, back, frame)
            cv2.polylines(frame, [quad.astype(np.int32)], True, (255, 220, 120), 2, cv2.LINE_AA)

    if res.hand_landmarks:
        for hand in res.hand_landmarks:
            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (120, 120, 255), -1)

    cv2.putText(frame, f"Effect: {EFFECTS[effect_index]}  |  pinch both hands to switch",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Hologram Glitch Band", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("n"):
        effect_index = (effect_index + 1) % len(EFFECTS)

cap.release()
cv2.destroyAllWindows()