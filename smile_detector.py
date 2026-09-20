import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

GREEN = (80, 220, 100)
GRAY = (200, 200, 200)
ORANGE = (100, 180, 255)

def corners(img, x, y, w, h, color):
    L, t = 18, 2
    for cx, cy, dx, dy in [
        (x, y, 1, 1), (x + w, y, -1, 1),
        (x, y + h, 1, -1), (x + w, y + h, -1, -1)
    ]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, t)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, t)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray, 1.05, 6, minSize=(60, 60)
    )

    smiling = 0

    for x, y, w, h in faces:
        face = gray[y:y + h, x:x + w]
        mouth = face[int(h * 0.48):int(h * 0.95), :]

        if mouth.size == 0:
            continue

        if mouth.shape[1] < 180:
            scale = 180 / mouth.shape[1]
            mouth = cv2.resize(
                mouth, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        smiles = smile_cascade.detectMultiScale(
            mouth, 1.05, 18, minSize=(25, 15)
        )

        is_smiling = False

        if len(smiles):
            sx, sy, sw, sh = max(
                smiles, key=lambda r: r[2] * r[3]
            )
            if sw * sh > mouth.shape[1] * mouth.shape[0] * 0.005:
                is_smiling = True

        color = GREEN if is_smiling else GRAY
        text = "Smiling :)" if is_smiling else "Not smiling"

        if is_smiling:
            smiling += 1

        corners(frame, x, y, w, h, color)
        cv2.putText(
            frame, text, (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    status = f"{smiling} PEOPLE SMILING" if smiling else "SCANNING..."
    color = GREEN if smiling else ORANGE

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)
    cv2.putText(
        frame, "SMILE DETECTOR", (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        frame, status, (frame.shape[1] - 260, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )

    cv2.imshow("Smile Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()