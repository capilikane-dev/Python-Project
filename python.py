#!/usr/bin/env python3
"""
smile_detector.py — Professional real-time smile detection system.

Pipeline:
    Webcam frame -> grayscale + histogram equalization
        -> Haar cascade face detection
        -> mouth-region ROI extraction (per face)
        -> Haar cascade smile detection
        -> persistence filter (N consecutive frames)
        -> HUD rendering / optional auto-capture

Hotkeys:
    q  quit   |   s  manual snapshot   |   c  toggle auto-capture
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install them with:\n  pip install opencv-python numpy"
    ) from exc

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smile_detector")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DetectorConfig:
    """All tunable parameters for the detection pipeline."""

    # -- face detection --
    face_scale_factor: float = 1.1
    face_min_neighbors: int = 5
    face_min_size: Tuple[int, int] = (120, 120)

    # -- smile detection --
    smile_scale_factor: float = 1.4
    smile_min_size: Tuple[int, int] = (35, 35)

    # -- mouth ROI (fractions of the face box) --
    roi_y_offset: float = 0.55        # search the lower 45% of the face
    roi_x_margin: float = 0.15        # trim 15% from each side

    # -- behaviour --
    sensitivity: float = 0.5          # 0.0 = very sensitive, 1.0 = very strict
    persistence_frames: int = 4       # consecutive frames required to confirm

    # -- auto capture --
    capture_dir: Optional[Path] = None
    capture_cooldown: float = 3.0     # seconds between automatic captures

    # -- display --
    window_name: str = "Smile Detection System"
    mirror: bool = True
    show_smile_roi: bool = True


# --------------------------------------------------------------------------- #
# Detection engine
# --------------------------------------------------------------------------- #
class SmileDetector:
    """Real-time face + smile detection engine (Haar cascade based)."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._face_cascade = self._load_cascade("haarcascade_frontalface_default.xml")
        self._smile_cascade = self._load_cascade("haarcascade_smile.xml")

        # Runtime state
        self._smile_streak: int = 0
        self._auto_capture: bool = config.capture_dir is not None
        self._last_capture_ts: float = 0.0
        self._prev_time: float = time.perf_counter()
        self._fps: float = 0.0

        logger.info(
            "Engine ready | sensitivity=%.2f | persistence=%d frames | auto_capture=%s",
            config.sensitivity, config.persistence_frames, self._auto_capture,
        )

    # -- setup -------------------------------------------------------------- #

    @staticmethod
    def _load_cascade(name: str) -> cv2.CascadeClassifier:
        """Load a bundled Haar cascade and fail fast if unavailable."""
        path = Path(cv2.data.haarcascades) / name
        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            raise RuntimeError(f"Failed to load cascade file: {path}")
        logger.debug("Loaded cascade: %s", path.name)
        return cascade

    # -- public API ---------------------------------------------------------- #

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run the full detection pipeline on one BGR frame, return annotated frame."""
        cfg = self.config
        if cfg.mirror:
            frame = cv2.flip(frame, 1)

        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=cfg.face_scale_factor,
            minNeighbors=cfg.face_min_neighbors,
            minSize=cfg.face_min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        raw_smile = False
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 200, 120), 2)

            # Extract mouth region: lower part of the face, side-trimmed
            x_off = int(w * cfg.roi_x_margin)
            y_off = int(h * cfg.roi_y_offset)
            roi = gray[y + y_off: y + h, x + x_off: x + w - x_off]
            if roi.size == 0:
                continue

            smiles = self._smile_cascade.detectMultiScale(
                roi,
                scaleFactor=cfg.smile_scale_factor,
                minNeighbors=self._smile_min_neighbors(),
                minSize=cfg.smile_min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            if len(smiles) > 0:
                raw_smile = True
                if cfg.show_smile_roi:
                    sx, sy, sw, sh = max(smiles, key=lambda s: s[2] * s[3])
                    cv2.rectangle(
                        frame,
                        (x + x_off + sx, y + y_off + sy),
                        (x + x_off + sx + sw, y + y_off + sy + sh),
                        (0, 255, 255), 2,
                    )

        # Persistence filter: reduces flicker / false positives
        self._smile_streak = self._smile_streak + 1 if raw_smile else 0
        smiling = self._smile_streak >= cfg.persistence_frames

        if smiling:
            self._maybe_capture(frame)

        self._draw_hud(frame, face_count=len(faces), smiling=smiling)
        return frame

    def save_snapshot(self, frame: np.ndarray) -> Optional[Path]:
        """Manually save a snapshot (ignores cooldown)."""
        if self.config.capture_dir is None:
            logger.warning("No capture directory configured (use --capture-dir).")
            return None
        return self._write_capture(frame)

    def toggle_auto_capture(self) -> None:
        self._auto_capture = not self._auto_capture
        logger.info("Auto-capture %s", "ENABLED" if self._auto_capture else "DISABLED")

    # -- internals ------------------------------------------------------------ #

    def _smile_min_neighbors(self) -> int:
        """Map sensitivity (0..1) to minNeighbors: 8 (sensitive) .. 25 (strict)."""
        return int(8 + 17 * self.config.sensitivity)

    def _maybe_capture(self, frame: np.ndarray) -> None:
        if not self._auto_capture or self.config.capture_dir is None:
            return
        now = time.time()
        if now - self._last_capture_ts < self.config.capture_cooldown:
            return
        self._last_capture_ts = now
        path = self._write_capture(frame)
        logger.info("Smile captured -> %s", path)

    def _write_capture(self, frame: np.ndarray) -> Path:
        assert self.config.capture_dir is not None
        self.config.capture_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.capture_dir / f"smile_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        cv2.imwrite(str(path), frame)
        return path

    def _draw_hud(self, frame: np.ndarray, face_count: int, smiling: bool) -> None:
        # Smoothed FPS (exponential moving average)
        now = time.perf_counter()
        dt = now - self._prev_time
        if dt > 0:
            self._fps = self._fps * 0.9 + (1.0 / dt) * 0.1
        self._prev_time = now

        status = "SMILING :)" if smiling else "NEUTRAL"
        color = (0, 220, 100) if smiling else (60, 60, 230)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (25, 25, 25), -1)
        cv2.putText(frame, f"FPS: {self._fps:5.1f}", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"FACES: {face_count}", (140, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, status, (frame.shape[1] - 150, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# --------------------------------------------------------------------------- #
# Application layer
# --------------------------------------------------------------------------- #
class SmileApp:
    """Owns the video-capture lifecycle and main event loop."""

    def __init__(self, config: DetectorConfig, camera_index: int = 0) -> None:
        self.detector = SmileDetector(config)
        self.camera_index = camera_index
        self._capture: Optional[cv2.VideoCapture] = None

    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        logger.info("Camera %d opened.", self.camera_index)
        return cap

    def run(self) -> None:
        self._capture = self._open_camera()
        logger.info("Running — press 'q' to quit, 's' snapshot, 'c' toggle auto-capture.")
        try:
            while True:
                ok, frame = self._capture.read()
                if not ok:
                    logger.warning("Frame grab failed — retrying.")
                    continue

                cv2.imshow(self.detector.config.window_name,
                           self.detector.process_frame(frame))

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested.")
                    break
                elif key == ord("s"):
                    self.detector.save_snapshot(frame)
                elif key == ord("c"):
                    self.detector.toggle_auto_capture()
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        if self._capture is not None:
            self._capture.release()
        cv2.destroyAllWindows()
        logger.info("Resources released. Goodbye.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time smile detection system")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--sensitivity", type=float, default=0.5,
                   help="0.0 = very sensitive, 1.0 = very strict (default: 0.5)")
    p.add_argument("--capture-dir", type=Path, default=None,
                   help="Directory for auto-captured smiles")
    p.add_argument("--no-mirror", action="store_true", help="Disable mirrored view")
    p.add_argument("--debug", action="store_true", help="Verbose logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = DetectorConfig(
        sensitivity=max(0.0, min(1.0, args.sensitivity)),
        capture_dir=args.capture_dir,
        mirror=not args.no_mirror,
    )

    try:
        SmileApp(config, camera_index=args.camera).run()
    except RuntimeError as exc:
        logger.error("Fatal: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()