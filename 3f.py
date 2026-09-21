import os
import pickle
import urllib.request
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# ==========================================
# CONFIGURATION & THRESHOLDS
# ==========================================
COSINE_THRESHOLD = 0.363   # >= means SAME person (cosine)
L2_THRESHOLD = 1.128       # <= means SAME person (L2)
MATCH_METRIC = 'cosine'    # 'cosine' or 'l2'

DB_FILE = "known_faces.pkl"
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
YUNET_URL = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{YUNET_MODEL}"
SFACE_URL = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/{SFACE_MODEL}"

ENROLL_SAMPLES = 5
SAMPLE_DELAY_MS = 300

# Theme colors
BG_DARK   = "#14141f"
BG_PANEL  = "#1f1f2e"
BG_INPUT  = "#2a2a3c"
ACCENT    = "#4f8cff"
GREEN     = "#2ecc71"
RED       = "#e74c3c"
PURPLE    = "#9b59b6"
TXT_MAIN  = "#ffffff"
TXT_DIM   = "#8a8fa3"


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
            raise SystemExit(1)


def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            print(f"[INFO] Loaded database from '{DB_FILE}'")
            return pickle.load(f)
    print("[INFO] No database found. Starting fresh.")
    return {}  # {'Name': [vector_1, vector_2, ...]}


def save_database(database):
    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)
    print(f"[SUCCESS] Saved database to '{DB_FILE}'")


# ==========================================
# MAIN GUI APPLICATION
# ==========================================
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.ok = True
        self.root.title("Bading lang nag a-ai  |  Face Recognition Console")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Escape>", lambda e: self.on_close())

        # ---- Models ----
        try:
            download_model(YUNET_MODEL, YUNET_URL)
            download_model(SFACE_MODEL, SFACE_URL)
        except SystemExit:
            self.ok = False
            return

        # ---- Camera ----
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Could not access webcam.")
            self.ok = False
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ---- YuNet + SFace ----
        self.detector = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL, config="", input_size=(640, 480),
            score_threshold=0.8, nms_threshold=0.3, top_k=5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(model=SFACE_MODEL, config="")

        self.database = load_database()

        # ---- State ----
        self.mode = "SCAN"            # SCAN / VERIFY
        self.claimed_identity = ""
        self.enrolling = False
        self.enroll_name = ""
        self.enroll_features = []
        self.last_sample_tick = cv2.getTickCount()
        self.fps_eval_time = cv2.getTickCount()
        self.fps = 0.0

        self._build_ui()
        self._refresh_name_list()
        self.set_status("Ready. Press SCAN, VERIFY, ENROLL or DELETE.")
        self.update_frame()

    # --------------------------------------
    # UI CONSTRUCTION
    # --------------------------------------
    def _build_ui(self):
        # Video area
        self.video_label = tk.Label(self.root, bg="black", bd=0)
        self.video_label.grid(row=0, column=0, padx=(12, 6), pady=12)

        # Control panel (right side)
        panel = tk.Frame(self.root, bg=BG_PANEL, width=290)
        panel.grid(row=0, column=1, sticky="ns", padx=(6, 12), pady=12)
        panel.grid_propagate(False)

        def section_title(text):
            tk.Label(panel, text=text, fg=TXT_DIM, bg=BG_PANEL,
                     font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(14, 4))

        def make_button(text, cmd, color):
            return tk.Button(panel, text=text, command=cmd, bg=color, fg="white",
                             activebackground=color, activeforeground="white",
                             relief="flat", font=("Segoe UI", 10, "bold"),
                             cursor="hand2", pady=6)

        tk.Label(panel, text="CONTROL PANEL", fg=TXT_MAIN, bg=BG_PANEL,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")

        # --- MODE section ---
        section_title("MODE")
        make_button("SCAN MODE", self.set_mode_scan, ACCENT).pack(fill="x", pady=2)

        tk.Label(panel, text="Claimed identity:", fg="#c7cad4", bg=BG_PANEL,
                 font=("Segoe UI", 9)).pack(anchor="w")
        self.claim_entry = tk.Entry(panel, font=("Segoe UI", 10), bg=BG_INPUT,
                                    fg="white", insertbackground="white", relief="flat")
        self.claim_entry.pack(fill="x", pady=(2, 4))
        make_button("VERIFY MODE", self.start_verify, PURPLE).pack(fill="x", pady=2)

        # --- ENROLL section ---
        section_title("ENROLL NEW PERSON")
        self.enroll_entry = tk.Entry(panel, font=("Segoe UI", 10), bg=BG_INPUT,
                                     fg="white", insertbackground="white", relief="flat")
        self.enroll_entry.pack(fill="x", pady=(2, 4))
        make_button("ENROLL PERSON", self.start_enroll, GREEN).pack(fill="x", pady=2)
        tk.Label(panel, text="(auto-captures 5 samples)", fg=TXT_DIM, bg=BG_PANEL,
                 font=("Segoe UI", 8)).pack(anchor="w")

        # --- DELETE section ---
        section_title("DELETE PERSON DATA")
        self.delete_combo = ttk.Combobox(panel, state="readonly", font=("Segoe UI", 10))
        self.delete_combo.pack(fill="x", pady=(2, 4))
        make_button("DELETE PERSON", self.delete_person, RED).pack(fill="x", pady=2)

        # --- STATUS ---
        section_title("STATUS")
        self.status_label = tk.Label(panel, text="...", fg="#c7cad4", bg=BG_PANEL,
                                     font=("Segoe UI", 9), wraplength=250, justify="left")
        self.status_label.pack(anchor="w")

        # --- QUIT (bottom) ---
        make_button("QUIT", self.on_close, "#55555f").pack(fill="x", side="bottom")

    # --------------------------------------
    # BUTTON ACTIONS
    # --------------------------------------
    def set_mode_scan(self):
        self.mode = "SCAN"
        self.enrolling = False
        self.set_status("SCAN mode — detecting known vs unknown faces.")

    def start_verify(self):
        claim = self.claim_entry.get().strip()
        if not claim:
            messagebox.showwarning("Verify", "Please enter the claimed identity.")
            return
        if claim not in self.database:
            messagebox.showwarning("Not Enrolled",
                                   f"'{claim}' is not in the database.\nEnroll this person first.")
            return
        self.claimed_identity = claim
        self.mode = "VERIFY"
        self.enrolling = False
        self.set_status(f"VERIFY active — face must match '{claim}'.", success=True)

    def start_enroll(self):
        if self.enrolling:
            messagebox.showinfo("Enroll", "Enrollment already in progress.")
            return
        name = self.enroll_entry.get().strip()
        if not name:
            messagebox.showwarning("Enroll", "Please enter a name first.")
            return
        if name in self.database and not messagebox.askyesno(
                "Person Exists",
                f"'{name}' already has {len(self.database[name])} sample(s).\nAdd {ENROLL_SAMPLES} more?"):
            return
        self.enroll_name = name
        self.enroll_features = []
        self.enrolling = True
        self.last_sample_tick = cv2.getTickCount()
        self.enroll_entry.delete(0, tk.END)
        self.set_status(f"Look at the camera — capturing {ENROLL_SAMPLES} samples for '{name}'...")

    def delete_person(self):
        name = self.delete_combo.get()
        if not name:
            messagebox.showwarning("Delete", "Select a person from the dropdown first.")
            return
        if name not in self.database:
            messagebox.showerror("Delete", f"'{name}' not found in database.")
            return
        if messagebox.askyesno("Confirm Delete",
                               f"Delete ALL stored data for '{name}'?\nThis cannot be undone."):
            del self.database[name]
            save_database(self.database)
            self._refresh_name_list()
            self.set_status(f"Deleted all data for '{name}'.", success=True)

    def on_close(self):
        self.enrolling = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    # --------------------------------------
    # UI HELPERS
    # --------------------------------------
    def set_status(self, text, success=False, error=False):
        color = "#c7cad4"
        if success:
            color = GREEN
        if error:
            color = RED
        self.status_label.config(text=text, fg=color)

    def _refresh_name_list(self):
        values = sorted(self.database.keys())
        self.delete_combo['values'] = values
        if values:
            self.delete_combo.current(0)
        else:
            self.delete_combo.set("")

    # --------------------------------------
    # RECOGNITION LOGIC
    # --------------------------------------
    def match_face(self, feature):
        if not self.database:
            return "Unknown", 0.0, False
        best_name = "Unknown"
        best_score = -1.0 if MATCH_METRIC == 'cosine' else float('inf')

        for name, features_list in self.database.items():
            for db_feature in features_list:
                if MATCH_METRIC == 'cosine':
                    score = self.recognizer.match(feature, db_feature, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > best_score:
                        best_score, best_name = score, name
                else:
                    score = self.recognizer.match(feature, db_feature, cv2.FaceRecognizerSF_FR_NORM_L2)
                    if score < best_score:
                        best_score, best_name = score, name

        is_match = (best_score >= COSINE_THRESHOLD if MATCH_METRIC == 'cosine'
                    else best_score <= L2_THRESHOLD)
        return (best_name if is_match else "Unknown"), best_score, is_match

    def _draw_face_info(self, frame, face):
        x, y, w, h = map(int, face[0:4])
        aligned_face = self.recognizer.alignCrop(frame, face)
        feature = self.recognizer.feature(aligned_face)
        matched_name, score, is_match = self.match_face(feature)

        if self.mode == "SCAN":
            if is_match:
                color, label_top = (0, 255, 0), f"{matched_name} - PRESENT"
                label_sub = f"Match Score: {score:.3f}"
            else:
                color, label_top = (0, 0, 255), "IMPOSTOR DETECTED!"
                label_sub = "FACE NOT REGISTERED!"
        else:  # VERIFY
            if is_match and matched_name.lower() == self.claimed_identity.lower():
                color, label_top = (0, 255, 0), "ACCESS GRANTED"
                label_sub = f"Verified: {matched_name} ({score:.3f})"
            else:
                color, label_top = (0, 0, 255), "ERROR: IMPOSTOR!"
                label_sub = f"NOT {self.claimed_identity.upper()}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        cv2.putText(frame, label_top, (x, max(y - 25, 55)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(frame, label_sub, (x, max(y - 8, 72)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # --------------------------------------
    # ENROLLMENT CAPTURE (in-window)
    # --------------------------------------
    def _process_enrollment(self, frame):
        now = cv2.getTickCount()
        elapsed_ms = (now - self.last_sample_tick) / cv2.getTickFrequency() * 1000.0

        _, faces = self.detector.detect(frame)
        faces = faces if faces is not None else []

        if len(faces) > 1:
            cv2.putText(frame, "Ensure ONLY ONE face is in frame!", (30, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            return
        if len(faces) == 0:
            cv2.putText(frame, "No face detected!", (30, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            return

        face = faces[0]
        x, y, w, h = map(int, face[0:4])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Capturing sample {min(len(self.enroll_features) + 1, ENROLL_SAMPLES)}/{ENROLL_SAMPLES}",
                    (x, max(y - 12, 90)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        self._draw_progress_bar(frame)
        if elapsed_ms < SAMPLE_DELAY_MS:
            return

        aligned_face = self.recognizer.alignCrop(frame, face)
        self.enroll_features.append(self.recognizer.feature(aligned_face))
        self.last_sample_tick = now
        self.set_status(f"Captured {len(self.enroll_features)}/{ENROLL_SAMPLES} samples for '{self.enroll_name}'")

        if len(self.enroll_features) >= ENROLL_SAMPLES:
            self.database.setdefault(self.enroll_name, []).extend(self.enroll_features)
            save_database(self.database)
            self._refresh_name_list()
            self.enrolling = False
            self.mode = "SCAN"
            messagebox.showinfo("Enrollment Complete",
                                f"Enrolled '{self.enroll_name}' with {ENROLL_SAMPLES} samples.")
            self.set_status(f"Enrolled '{self.enroll_name}' successfully.", success=True)

    def _draw_progress_bar(self, frame):
        fh, fw = frame.shape[:2]
        done = len(self.enroll_features)
        bar_w = int(fw * 0.6)
        x0, y0 = (fw - bar_w) // 2, fh - 45
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 18), (70, 70, 70), -1)
        fill = int(bar_w * done / ENROLL_SAMPLES)
        if fill > 0:
            cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + 18), (0, 200, 0), -1)
        cv2.putText(frame, f"{done}/{ENROLL_SAMPLES}", (x0 + bar_w + 10, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    # --------------------------------------
    # MAIN VIDEO LOOP
    # --------------------------------------
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.set_status("Camera frame capture failed.", error=True)
            self.root.after(100, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))

        # FPS
        tick_now = cv2.getTickCount()
        dt = tick_now - self.fps_eval_time
        if dt > 0:
            self.fps = cv2.getTickFrequency() / dt
        self.fps_eval_time = tick_now

        # Detection / enrollment
        if self.enrolling:
            self._process_enrollment(frame)
        else:
            _, faces = self.detector.detect(frame)
            for face in (faces if faces is not None else []):
                self._draw_face_info(frame, face)

        # HUD header
        if self.enrolling:
            hud = f"ENROLLING: {self.enroll_name}  ({len(self.enroll_features)}/{ENROLL_SAMPLES})"
        else:
            hud = f"MODE: {self.mode}"
            if self.mode == "VERIFY":
                hud += f"  |  Claiming: {self.claimed_identity}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (20, 20, 30), -1)
        cv2.putText(frame, hud, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (frame.shape[1] - 105, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        # Show frame in Tkinter window
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.imgtk = imgtk  # keep reference (prevents garbage collection)
        self.video_label.configure(image=imgtk)

        self.root.after(15, self.update_frame)


# ==========================================
# ENTRY POINT
# ==========================================
def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    if app.ok:
        root.mainloop()


if __name__ == "__main__":
    main()