import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

GREEN = (80, 220, 100)
GRAY = (200, 200, 200)

clahe = cv2.createCLAHE(2.0, (8, 8))  # CHANGE 1: contrast boost for distant faces

def corners(img, x, y, w, h, color, t=2, L=18):
    """Draw modern corner brackets instead of a full rectangle."""
    pts = [(x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)]
    for (cx, cy, dx, dy) in pts:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, t)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, t)

def top_bar(img, text, color):
    """Semi-transparent status bar at the top."""
    h, w = img.shape[:2]
    overlay = img[0:50, 0:w].copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img[0:50, 0:w], 0.4, 0, img[0:50, 0:w])
    cv2.putText(img, "SMILE DETECTOR [LONG-RANGE]", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, text, (w - 260, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)

    # CHANGE 2: zoom small frames so distant faces become visible
    scale = 1.6 if frame.shape[1] < 900 else 1.0
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

    smiling_any = False
    for (fx, fy, fw, fh) in faces:
        # map box back to original frame size
        x, y, w, h = int(fx / scale), int(fy / scale), int(fw / scale), int(fh / scale)

        mouth = gray[fy + fh // 2: fy + fh, fx: fx + fw]

        # CHANGE 3: zoom tiny far-away mouths before checking
        if mouth.shape[1] < 120:
            mouth = cv2.resize(mouth, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        smiles = smile_cascade.detectMultiScale(mouth, 1.2, 15)

        if len(smiles) > 0:
            smiling_any = True
            corners(frame, x, y, w, h, GREEN)
            cv2.putText(frame, "Smiling :)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
        else:
            corners(frame, x, y, w, h, GRAY)
            cv2.putText(frame, "Not smiling", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GRAY, 2)

    top_bar(frame, "SMILING!" if smiling_any else "SCANNING...",
            GREEN if smiling_any else (100, 180, 255))

    cv2.imshow("Smile Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()