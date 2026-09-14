#Sir. homer Project 

# Capili, Justine Kane
# Gonzales, Norman 
# Prince Ram Roydlikent F. Igna

import cv2
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Smile detector running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improves detection in varied lighting

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80),
        )

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Restrict smile search to the lower half of the face
            # (smiles live in the mouth area, and this avoids false positives on eyes)
            roi_gray = gray[y + h // 2 : y + h, x : x + w]
            roi_color = frame[y + h // 2 : y + h, x : x + w]

            smiles = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.7,
                minNeighbors=22,
                minSize=(25, 25),
            )

            if len(smiles) > 0:
                cv2.putText(
                    frame,
                    "Smiling :)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(
                        roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2
                    )

        cv2.imshow("Smile Detector - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()