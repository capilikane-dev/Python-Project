# hologram_viewer.py — Iron-Man style hologram viewer (cv2 + mediapipe hands)
# PINCH = grab & move | HAND = rotate | 2 HANDS = zoom | N/P = switch model
import math, os, time, urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

# ---------------- 5 procedural 3D models ----------------
def ring(y, r, n, V, E, ph=0.0):
    b = len(V)
    for k in range(n):
        a = 2 * math.pi * k / n + ph
        V.append((r * math.cos(a), y, r * math.sin(a)))
    for k in range(n):
        E.append((b + k, b + (k + 1) % n))
    return b

def m_rocket():
    V, E = [], []
    V.append((0, -1.7, 0))                                              # nose tip
    bs = [ring(y, r, 10, V, E) for y, r in
          ((-1.3, .25), (-.7, .38), (0, .38), (.7, .38), (1.1, .38), (1.35, .2))]
    E += [(0, bs[0] + k) for k in range(10)]                            # nose cone
    for a, b in zip(bs, bs[1:]):
        E += [(a + k, b + k) for k in range(10)]                        # body stringers
    for f in range(4):                                                  # 4 fins
        a = f * math.pi / 2; t = len(V)
        V += [(math.cos(a) * .95, 1.05, math.sin(a) * .95),
              (math.cos(a) * .38, .4, math.sin(a) * .38)]
        E += [(t, t + 1), (t, bs[4] + f * 10 // 4), (t + 1, bs[3] + f * 10 // 4)]
    fl = len(V); V.append((0, 1.95, 0))                                 # exhaust plume
    E += [(bs[-1] + k, fl) for k in range(10)]
    return np.float32(V), E

def m_tesseract():
    V, E = [], []
    def cube(s):
        nonlocal V, E                                                   # FIX: closure needs this
        b = len(V)
        V += [(x * s, y * s, z * s) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        E += [(b + i, b + j) for i in range(8) for j in range(i + 1, 8)
              if bin(i ^ j).count("1") == 1]                            # bit-trick cube edges
        return b
    o, i = cube(1), cube(.45)
    E += [(o + k, i + k) for k in range(8)]                             # hyper-struts
    return np.float32(V), E

def m_globe():
    V, E = [], []
    for la in range(7):
        th = math.pi * la / 6; b = len(V)
        for lo in range(14):
            a = 2 * math.pi * lo / 14
            V.append((.95 * math.sin(th) * math.cos(a),
                      .95 * math.cos(th), .95 * math.sin(th) * math.sin(a)))
        E += [(b + k, b + (k + 1) % 14) for k in range(14)]             # latitude ring
        if la:
            E += [(b + k, b - 14 + k) for k in range(14)]               # meridians
    ring(0, 1.3, 22, V, E)                                              # orbit ring
    return np.float32(V), E

def m_dna():
    V, E, A, B = [], [], [], []
    for k in range(22):
        t = k / 21; y = -1.3 + 2.6 * t; a = t * 3.5 * math.pi
        A.append(len(V)); V.append((.6 * math.cos(a), y, .6 * math.sin(a)))
        B.append(len(V)); V.append((.6 * math.cos(a + math.pi), y, .6 * math.sin(a + math.pi)))
        E.append((A[-1], B[-1]))                                        # base-pair rung
    E += [(A[k], A[k + 1]) for k in range(21)] + [(B[k], B[k + 1]) for k in range(21)]
    return np.float32(V), E

def m_diamond():
    V, E = [], []
    t = ring(-.8, .45, 8, V, E, math.pi / 8)                            # table facet
    g = ring(-.15, .95, 8, V, E)                                        # girdle
    E += [(t + k, g + k) for k in range(8)] + [(t + k, g + (k + 1) % 8) for k in range(8)]
    c = len(V); V.append((0, 1, 0))                                     # culet point
    E += [(g + k, c) for k in range(8)]
    return np.float32(V), E

MODELS = [("ROCKET", m_rocket, (60, 160, 255)), ("TESSERACT", m_tesseract, (255, 220, 80)),
          ("GLOBE", m_globe, (255, 200, 40)), ("DNA HELIX", m_dna, (230, 100, 255)),
          ("DIAMOND", m_diamond, (255, 245, 190))]
CACHE = {name: fn() for name, fn, _ in MODELS}

# ---------------- 3D math ----------------
def rotate(V, rx, ry):
    rx, ry = math.radians(rx), math.radians(ry)
    cx, sx, cy, sy = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry)
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    y, z = y * cx - z * sx, y * sx + z * cx                             # pitch
    x, z = x * cy + z * sy, -x * sy + z * cy                            # yaw
    return np.stack([x, y, z], 1)

def project(V, cx, cy, s, f=4.0):
    p = f / np.clip(V[:, 2] + f, .4, None)
    return np.stack([V[:, 0] * p * s + cx, V[:, 1] * p * s + cy], 1), V[:, 2] + f

# ---------------- hologram renderer ----------------
def draw_holo(frame, glow, name, color, cx, cy, s, rx, ry):
    V, E = CACHE[name]
    P, z = project(rotate(V, rx, ry), cx, cy, s)
    Pi = P.astype(int); lo, hi = z.min(), z.max()
    by, br = int(cy + s * 1.05), int(s * 1.1)                           # projector base
    ov = frame.copy()
    cv2.ellipse(ov, (int(cx), by), (br, max(4, br // 5)), 0, 0, 360, color, -1)
    cv2.addWeighted(ov, .22, frame, .78, 0, frame)
    cv2.ellipse(frame, (int(cx), by), (br, max(4, br // 5)), 0, 0, 360, color, 1, cv2.LINE_AA)
    cv2.ellipse(frame, (int(cx), by), (int(br * .6), max(3, br // 8)), 0, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)
    for i in np.argsort(P[:, 1])[-4:]:                                  # emitter beams
        cv2.line(frame, (int(cx), by), tuple(Pi[i]), tuple(int(c * .4) for c in color), 1, cv2.LINE_AA)
    for a, b in sorted(E, key=lambda e: -(z[e[0]] + z[e[1]])):          # depth-sorted wireframe
        t = 1 - ((z[a] + z[b]) / 2 - lo) / max(hi - lo, 1e-3)
        c = tuple(int(ch * (.35 + .65 * t)) for ch in color)
        cv2.line(frame, tuple(Pi[a]), tuple(Pi[b]), c, 2 if t > .6 else 1, cv2.LINE_AA)
        cv2.line(glow, tuple(Pi[a]), tuple(Pi[b]), c, 4, cv2.LINE_AA)
    for i, p in enumerate(Pi):                                          # glowing vertex nodes
        t = 1 - (z[i] - lo) / max(hi - lo, 1e-3)
        cv2.circle(frame, tuple(p), 2, tuple(int(ch * (.5 + .5 * t)) for ch in color), -1, cv2.LINE_AA)
    sy = int(cy - s + (time.time() * .5 % 1) * 2 * s)                   # scan sweep
    cv2.line(frame, (int(cx - br), sy), (int(cx + br), sy), color, 1, cv2.LINE_AA)

# ---------------- hand tracker (reuses your hand_landmarker.task) ----------------
MF = "hand_landmarker.task"
if not os.path.exists(MF):
    print("Downloading hand model (one time)...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                               "hand_landmarker/float16/1/hand_landmarker.task", MF)
hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=MF), num_hands=2))

# ---------------- main loop ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
mi, fps, prev = 0, 0.0, time.time()
yaw = pitch = t_yaw = t_pitch = 0.0
scale = t_scale = 150.0
pos, t_pos = [0.0, 0.0], [0.0, 0.0]
lerp = lambda a, b, k=.18: a + (b - a) * k

while True:
    ok, cam = cap.read()
    if not ok: break
    cam = cv2.flip(cam, 1)
    H, W = cam.shape[:2]
    frame = (cam * .5).astype(np.uint8)                                 # dim feed, neon pops
    glow = np.zeros_like(frame)
    now = time.time(); fps = fps * .9 + .1 / max(now - prev, 1e-6); prev = now

    res = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)))
    tips = []                                                           # pinch points per hand
    for hd in (res.hand_landmarks or []):
        d = (math.dist((hd[4].x, hd[4].y), (hd[8].x, hd[8].y))
             / max(math.dist((hd[0].x, hd[0].y), (hd[9].x, hd[9].y)), 1e-3))
        tips.append(((hd[8].x * W, hd[8].y * H), d < .18))

    if len(tips) == 2:                                                  # ZOOM: two-hand distance
        t_scale = float(np.clip(math.dist(tips[0][0], tips[1][0]) * .8, 60, 300))
        t_pos = [0, 0]
    elif len(tips) == 1:
        (px, py), pinched = tips[0]
        if pinched:                                                     # GRAB & MOVE
            t_pos = [px - W / 2, py - H / 2]
        else:                                                           # ROTATE: hand position
            t_yaw, t_pitch = (px / W - .5) * 360, (py / H - .5) * 180
            t_pos = [0, 0]
    else:
        t_yaw += .6                                                     # idle auto-spin
    for (px, py), pinched in tips:
        cv2.circle(frame, (int(px), int(py)), 12, (0, 255, 180) if pinched else (200, 200, 200), 2, cv2.LINE_AA)

    yaw, pitch = lerp(yaw, t_yaw), lerp(pitch, t_pitch)
    scale = lerp(scale, t_scale)
    pos = [lerp(pos[0], t_pos[0]), lerp(pos[1], t_pos[1])]
    name, _, color = MODELS[mi]
    draw_holo(frame, glow, name, color, W / 2 + pos[0], H / 2 + pos[1], scale, pitch, yaw)
    frame = cv2.addWeighted(frame, 1, cv2.GaussianBlur(glow, (9, 9), 0), .85, 0)   # neon bloom

    # ---------------- HUD ----------------
    for y, h in ((0, 44), (H - 36, 36)):                                # glass bars
        ov = frame.copy(); cv2.rectangle(ov, (0, y), (W, y + h), (0, 0, 0), -1)
        cv2.addWeighted(ov, .45, frame, .55, 0, frame)
    txt = lambda s, x, y, c, sc=.6, th=2: cv2.putText(frame, s, (x, y),
                                                      cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)
    txt("HOLOGRAM PROJECTOR", 16, 29, (240, 240, 240))
    (tw, _), _ = cv2.getTextSize(f"{name} {mi+1}/5", cv2.FONT_HERSHEY_SIMPLEX, .8, 2)
    txt(f"{name} {mi+1}/5", (W - tw) // 2, 31, color, .8, 2)            # colored model title
    txt(f"{fps:.0f} FPS   ZOOM {int(scale / 1.5)}%", W - 240, 29, (240, 240, 240), .6, 1)
    for i in range(5):                                                  # model selector dots
        cv2.circle(frame, (W // 2 - 64 + i * 32, H - 50), 8 if i == mi else 4,
                   color if i == mi else (120, 120, 120), -1, cv2.LINE_AA)
    txt("PINCH grab+move   HAND rotate   2 HANDS zoom   [N]ext [P]rev [R]eset [Q]uit",
        16, H - 12, (190, 190, 190), .5, 1)

    cv2.imshow("Hologram Viewer", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    elif k == ord("n"): mi = (mi + 1) % 5; scale = 40                   # pop-in transition
    elif k == ord("p"): mi = (mi - 1) % 5; scale = 40
    elif k == ord("r"): t_yaw = t_pitch = 0.0; t_scale = 150.0; t_pos = [0, 0]

cap.release(); cv2.destroyAllWindows()