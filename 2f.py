import os
import pickle
import urllib.request
import cv2
import numpy as np

# ==========================================
# CONFIGURATION & THRESHOLDS
# ==========================================
# SFace cosine similarity threshold: >= 0.363 means SAME person
COSINE_THRESHOLD = 0.363

# SFace L2 distance threshold (alternative metric): <= 1.128 means SAME person
L2_THRESHOLD = 1.128

# Distance metric selection: 'cosine' or 'l2'
MATCH_METRIC = 'cosine'

# Database file for saving enrolled faces
DB_FILE = "known_faces.pkl"

# Model filenames and OpenCV Zoo URL endpoints
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"

YUNET_URL = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{YUNET_MODEL}"
SFACE_URL = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/{SFACE_MODEL}"


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def download_model(filename, url):
    """Auto-download ONNX models from OpenCV Zoo GitHub if missing."""
    if not os.path.exists(filename):
        print(f"[INFO] Downloading {filename} from OpenCV Zoo...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"[SUCCESS] Downloaded {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            exit(1)


def load_database():
    """Load enrolled face signatures from pickle file."""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            print(f"[INFO] Loaded database from '{DB_FILE}'")
            return pickle.load(f)
    print("[INFO] No database found. Starting fresh.")
    return {}  # Format: {'Name': [vector_1, vector_2, ...]}


def save_database(database):
    """Save enrolled face signatures to pickle file."""
    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)
    print(f"[SUCCESS] Saved database to '{DB_FILE}'")


def match_face(recognizer, feature, database):
    """
    Compares extracted 128-D feature vector against all enrolled features.
    Returns: (best_match_name, best_score, is_match)
    """
    if not database:
        return "Unknown", 0.0, False

    best_name = "Unknown"
    best_score = -1.0 if MATCH_METRIC == 'cosine' else float('inf')

    for name, features_list in database.items():
        for db_feature in features_list:
            if MATCH_METRIC == 'cosine':
                # Fixed: Called .match() on the recognizer instance
                score = recognizer.match(feature, db_feature, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_score = score
                    best_name = name
            else:
                score = recognizer.match(feature, db_feature, cv2.FaceRecognizerSF_FR_NORM_L2)
                if score < best_score:
                    best_score = score
                    best_name = name

    # Determine match based on official threshold
    is_match = False
    if MATCH_METRIC == 'cosine':
        if best_score >= COSINE_THRESHOLD:
            is_match = True
    else:
        if best_score <= L2_THRESHOLD:
            is_match = True

    return (best_name if is_match else "Unknown"), best_score, is_match


# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    # 1. Download models if necessary
    download_model(YUNET_MODEL, YUNET_URL)
    download_model(SFACE_MODEL, SFACE_URL)

    # 2. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access webcam.")
        return

    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 3. Initialize YuNet Face Detector and SFace Face Recognizer
    detector = cv2.FaceDetectorYN.create(
        model=YUNET_MODEL,
        config="",
        input_size=(frame_width, frame_height),
        score_threshold=0.8,    # Confidence threshold to filter weak faces
        nms_threshold=0.3,      # Non-maximum suppression threshold
        top_k=5000
    )

    recognizer = cv2.FaceRecognizerSF.create(
        model=SFACE_MODEL,
        config=""
    )

    # 4. Load Database
    database = load_database()

    # State Variables
    mode = "SCAN"           # Modes: "SCAN", "VERIFY"
    claimed_identity = ""   # Used during VERIFY mode
    
    # FPS Counter variables
    fps_eval_time = cv2.getTickCount()
    fps = 0.0

    print("\n--- CONTROLS ---")
    print("Press 'e' : Enroll a new person")
    print("Press 'v' : Toggle VERIFY mode (prompt for claimed identity)")
    print("Press 's' : Return to SCAN mode")
    print("Press 'q' : Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture video frame.")
            break

        # 1. Mirror View
        frame = cv2.flip(frame, 1)

        # Update YuNet input size in case camera resolution changes dynamically
        detector.setInputSize((frame.shape[1], frame.shape[0]))

        # 2. Detect Faces using YuNet
        _, faces = detector.detect(frame)

        # Convert to list safely if faces are found
        faces = faces if faces is not None else []

        # 3. Process Each Detected Face
        for face in faces:
            bbox = list(map(int, face[0:4]))
            x, y, w, h = bbox

            # Align face and extract 128-D feature vector using SFace
            aligned_face = recognizer.alignCrop(frame, face)
            feature = recognizer.feature(aligned_face)

            # Match extracted face against database (Fixed call signature)
            matched_name, score, is_match = match_face(recognizer, feature, database)

            # MODE-SPECIFIC LOGIC
            if mode == "SCAN":
                if is_match:
                    color = (0, 255, 0)  # Green
                    label_top = f"{matched_name} - PRESENT"
                    label_sub = f"Match Score: {score:.3f}"
                else:
                    color = (0, 0, 255)  # Red
                    label_top = "IMPOSTOR DETECTED!"
                    label_sub = "FACE NOT REGISTERED!"

            elif mode == "VERIFY":
                # Check if camera face matches the user-typed claim
                if is_match and (matched_name.lower() == claimed_identity.lower()):
                    color = (0, 255, 0)  # Green
                    label_top = "ACCESS GRANTED"
                    label_sub = f"Verified: {matched_name} ({score:.3f})"
                else:
                    color = (0, 0, 255)  # Red
                    label_top = "ERROR: IMPOSTOR!"
                    label_sub = f"NOT {claimed_identity.upper()}"

            # Draw bounding box (Anti-aliased)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, lineType=cv2.LINE_AA)

            # Render identity labels above bounding box
            cv2.putText(frame, label_top, (x, max(y - 25, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.putText(frame, label_sub, (x, max(y - 8, 35)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 4. Render UI Header (Mode & Controls)
        cv2.putText(frame, f"MODE: {mode}" + (f" (Claiming: {claimed_identity})" if mode == "VERIFY" else ""),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Keys: 'e'=Enroll | 'v'=Verify | 's'=Scan | 'q'=Quit",
                    (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # 5. Calculate & Render Live FPS Counter
        tick_now = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (tick_now - fps_eval_time)
        fps_eval_time = tick_now
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Show Output Frame
        cv2.imshow("Bading lang nag a-ai", frame)

        # 6. Handle Keyboard Shortcuts
        key = cv2.waitKey(1) & 0xFF

        # --- QUIT ---
        if key == ord('q'):
            break

        # --- RETURN TO SCAN MODE ---
        elif key == ord('s'):
            mode = "SCAN"
            print("[INFO] Switched to SCAN mode.")

        # --- TOGGLE VERIFY MODE ---
        elif key == ord('v'):
            print("\n[VERIFY MODE] Enter person's claimed identity: ", end="")
            claimed_identity = input().strip()
            if claimed_identity:
                mode = "VERIFY"
                print(f"[INFO] Verification active for claim: '{claimed_identity}'")
            else:
                print("[WARNING] Empty claim. Reverting to SCAN mode.")
                mode = "SCAN"

        # --- ENROLLMENT MODE ---
        elif key == ord('e'):
            print("\n[ENROLLMENT] Enter name of person to enroll: ", end="")
            person_name = input().strip()

            if not person_name:
                print("[WARNING] Name cannot be empty. Enrollment cancelled.")
                continue

            print(f"[INFO] Look at the camera. Capturing 5 samples for '{person_name}'...")
            captured_features = []
            
            while len(captured_features) < 5:
                ret, frame_e = cap.read()
                if not ret:
                    break
                frame_e = cv2.flip(frame_e, 1)

                _, faces_e = detector.detect(frame_e)
                faces_e = faces_e if faces_e is not None else []

                if len(faces_e) == 1:  # Ensure exactly one face is visible
                    aligned_face = recognizer.alignCrop(frame_e, faces_e[0])
                    feat = recognizer.feature(aligned_face)
                    captured_features.append(feat)

                    # Draw green visual box indicating capture progress
                    x, y, w, h = list(map(int, faces_e[0][0:4]))
                    cv2.rectangle(frame_e, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame_e, f"Captured Sample {len(captured_features)}/5", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif len(faces_e) > 1:
                    cv2.putText(frame_e, "Ensure ONLY ONE face is in frame!", 
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame_e, "No face detected!", 
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Bading lang nag a-ai", frame_e)
                cv2.waitKey(200)  # Delay between captures

            # Save extracted feature representations into database
            if len(captured_features) == 5:
                if person_name not in database:
                    database[person_name] = []
                database[person_name].extend(captured_features)
                save_database(database)
                print(f"[SUCCESS] Enrolled '{person_name}' with 5 samples successfully.")
            else:
                print("[ERROR] Enrollment failed. Couldn't capture enough face samples.")

    # Clean Up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()