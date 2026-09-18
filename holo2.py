# hologram_viewer.py — Iron-Man hologram viewer w/ upgraded HUD
# PINCH = grab & move | HAND = rotate | 2 HANDS = zoom | CLICK tabs or N/P = model
import math, os, time, urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

FONT, CYAN = cv2.FONT_HERSHEY_SIMPLEX, (255, 200, 90)

# ---------------- 5 procedural 3D models ----------------
def ring(y, r, n, V, E, ph=0.0):
    b = len(V)
    for k in range(n):
        a = 2 * math.pi * k / n + ph
        V.append((r * math.cos(a), y, r * math.sin(a)))
    for k in range(n): E.append((b + k, b + (k + 1) % n))
    return b

def m_rocket():
    V, E = [], []
    V.append((0, -1.7, 0))
    bs = [ring(y, r, 10, V, E) for y, r in
          ((-1.3, .25), (-.7, .38), (0, .38), (.7, .38), (1.1, .38), (1.35, .2))]
    E += [(0, bs[0] + k) for k in range(10)]
    for a, b in zip(bs, bs[1:]): E += [(a + k, b + k) for k in range(10)]
    for f in range(4):
        a = f * math.pi / 2; t = len(V)
        V += [(math.cos(a) * .95, 1.05, math.sin(a) * .95), (math.cos(a) * .38, .4, math.sin(a) * .38)]
        E += [(t, t + 1), (t, bs[4] + f * 10 // 4), (t + 1, bs[3] + f * 10 // 4)]
    fl = len(V); V.append((0, 1.95, 0)); E += [(bs[-1] + k, fl) for k in range(10)]
    return np.float32(V), E

def m_tesseract():
    V, E = [], []
    def cube(s):
        nonlocal V, E
        b = len(V)
        V += [(x * s, y * s, z * s) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        E += [(b + i, b + j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
        return b
    o, i = cube(1), cube(.45)
    E += [(o + k, i + k) for k in range(8)]
    return np.float32(V), E

def m_globe():
    V, E = [], []
    for la in range(7):
        th = math.pi * la / 6; b = len(V)
        for lo in range(14):
            a = 2 * math.pi * lo / 14
            V.append((.95 * math.sin(th) * math.cos(a), .95 * math.cos(th), .95 * math.sin(th) * math.sin(a)))
        E += [(b + k, b + (k + 1) % 14) for k in range(14)]
        if la: E += [(b + k, b - 14 + k) for k in range(14)]
    ring(0, 1.3, 22, V, E)
    return np.float32(V), E

def m_dna():
    V, E, A, B = [], [], [], []
    for k in range(22):
        t = k / 21; y = -1.3 + 2.6 * t; a = t * 3.5 * math.pi
        A.append(len(V)); V.append((.6 * math.cos(a), y, .6 * math.sin(a)))
        B.append(len(V)); V.append((.6 * math.cos(a + math.pi), y, .6 * math.sin(a + math.pi)))
        E.append((A[-1], B[-1]))
    E += [(A[k], A[k + 1]) for k in range(21)] + [(B[k], B[k + 1]) for k in range(21)]
    return np.float32(V), E

def m_diamond():
    V, E = [], []
    t = ring(-.8, .45, 8, V, E, math.pi / 8); g = ring(-.15, .95, 8, V, E)
    E += [(t + k, g + k) for k in range(8)] + [(t + k, g + (k + 1) % 8) for k in range(8)]
    c = len(V); V.append((0, 1, 0)); E += [(g + k, c) for k in range(8)]
    return np.float32(V), E

MODELS = [("ROCKET", "RKT", m_rocket, (60, 160, 255)), ("TESSERACT", "CUBE", m_tesseract, (255, 220, 80)),
          ("GLOBE", "GLB", m_globe, (255, 200, 40)), ("DNA HELIX", "DNA", m_dna, (230, 100, 255)),
          ("DIAMOND", "GEM", m_diamond, (255, 245, 190))]
CACHE = {m[0]: m[2]() for m in MODELS}

# ---------------- 3D math ----------------
def rotate(V, rx, ry):
    rx, ry = math.radians(rx), math.radians(ry)
    cx, sx, cy, sy = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry)
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    y, z = y * cx - z * sx, y * sx + z * cx
    x, z = x * cy + z * sy, -x * sy + z * cy
    return np.stack([x, y, z], 1)

def project(V, cx, cy, s, f=4.0):
    p = f / np.clip(V[:, 2] + f, .4, None)
    return np.stack([V[:, 0] * p * s + cx, V[:, 1] * p * s + cy], 1), V[:, 2] + f

# ---------------- hologram renderer ----------------
def draw_holo(frame, glow, name, color, cx, cy, s, rx, ry):
    V, E = CACHE[name]
    P, z = project(rotate(V, rx, ry), cx, cy, s)
    Pi = P.astype(int); lo, hi = z.min(), z.max()
    by, br = int(cy + s * 1.05), int(s * 1.1)
    ov = frame.copy()
    cv2.ellipse(ov, (int(cx), by), (br, max(4, br // 5)), 0, 0, 360, color, -1)
    cv2.addWeighted(ov, .22, frame, .78, 0, frame)
    cv2.ellipse(frame, (int(cx), by), (br, max(4, br // 5)), 0, 0, 360, color, 1, cv2.LINE_AA)
    cv2.ellipse(frame, (int(cx), by), (int(br * .6), max(3, br // 8)), 0, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)
    for i in np.argsort(P[:, 1])[-4:]:
        cv2.line(frame, (int(cx), by), tuple(Pi[i]), tuple(int(c * .4) for c in color), 1, cv2.LINE_AA)
    for a, b in sorted(E, key=lambda e: -(z[e[0]] + z[e[1]])):
        t = 1 - ((z[a] + z[b]) / 2 - lo) / max(hi - lo, 1e-3)
        c = tuple(int(ch * (.35 + .65 * t)) for ch in color)
        cv2.line(frame, tuple(Pi[a]), tuple(Pi[b]), c, 2 if t > .6 else 1, cv2.LINE_AA)
        cv2.line(glow, tuple(Pi[a]), tuple(Pi[b]), c, 4, cv2.LINE_AA)
    for i, p in enumerate(Pi):
        t = 1 - (z[i] - lo) / max(hi - lo, 1e-3)
        cv2.circle(frame, tuple(p), 2, tuple(int(ch * (.5 + .5 * t)) for ch in color), -1, cv2.LINE_AA)
    sy = int(cy - s + (time.time() * .5 % 1) * 2 * s)
    cv2.line(frame, (int(cx - br), sy), (int(cx + br), sy), color, 1, cv2.LINE_AA)

# ---------------- UI helpers ----------------
def txt(img, s, x, y, c, sc=.55, th=1):
    cv2.putText(img, s, (x, y), FONT, sc, c, th, cv2.LINE_AA)

def panel(img, x, y, w, h, a=.45):
    ov = img.copy(); cv2.rectangle(ov, (x, y), (x + w, y + h), (10, 14, 20), -1)
    cv2.addWeighted(ov, a, img, 1 - a, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 140, 190), 1, cv2.LINE_AA)

def chip(img, label, x, y, c, active):
    (tw, th), _ = cv2.getTextSize(label, FONT, .5, 1)
    w, h = tw + 18, th + 14
    a = .55 if active else .30
    ov = img.copy(); cv2.rectangle(ov, (x, y), (x + w, y + h), c if active else (60, 60, 60), -1)
    cv2.addWeighted(ov, a, img, 1 - a, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), c if active else (90, 90, 90), 1, cv2.LINE_AA)
    txt(img, label, x + 9, y + th + 7, (255, 255, 255) if active else (160, 160, 160))
    return x + w + 8

def corners(img, W, H, c, L=30):
    for x, y, dx, dy in ((0, 0, 1, 1), (W - 1, 0, -1, 1), (0, H - 1, 1, -1), (W - 1, H - 1, -1, -1)):
        cv2.line(img, (x, y), (x + L * dx, y), c, 2, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + L * dy), c, 2, cv2.LINE_AA)

def meter(img, x, y, w, frac, c, vert=False, h=6):
    x2, y2 = (x, y + w) if vert else (x + w, y)
    cv2.rectangle(img, (x, y), (x2, y2 + (0 if vert else h)), (60, 60, 60), -1, cv2.LINE_AA)
    f = int(w * np.clip(frac, 0, 1))
    cv2.rectangle(img, (x, y), ((x, y + f) if vert else (x + f, y + h)), c, -1, cv2.LINE_AA)

# ---------------- hand tracker ----------------
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
chip_rects = []

def on_mouse(e, x, y, flags, param):
    global mi, scale
    if e == cv2.EVENT_LBUTTONDOWN:
        for (rx, ry, rw, rh), idx in chip_rects:
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                mi, scale = idx, 40

cv2.namedWindow("Hologram Viewer"); cv2.setMouseCallback("Hologram Viewer", on_mouse)

while True:
    ok, cam = cap.read()
    if not ok: break
    cam = cv2.flip(cam, 1)
    H, W = cam.shape[:2]
    frame = (cam * .5).astype(np.uint8)
    glow = np.zeros_like(frame)
    now = time.time(); fps = fps * .9 + .1 / max(now - prev, 1e-6); prev = now

    res = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)))
    tips = []
    for hd in (res.hand_landmarks or []):
        d = (math.dist((hd[4].x, hd[4].y), (hd[8].x, hd[8].y))
             / max(math.dist((hd[0].x, hd[0].y), (hd[9].x, hd[9].y)), 1e-3))
        tips.append(((hd[8].x * W, hd[8].y * H), d < .18))

    if len(tips) == 2:
        t_scale = float(np.clip(math.dist(tips[0][0], tips[1][0]) * .8, 60, 300)); t_pos = [0, 0]
    elif len(tips) == 1:
        (px, py), pinched = tips[0]
        if pinched: t_pos = [px - W / 2, py - H / 2]
        else:
            t_yaw, t_pitch = (px / W - .5) * 360, (py / H - .5) * 180; t_pos = [0, 0]
    else:
        t_yaw += .6
    for (px, py), pinched in tips:
        cv2.circle(frame, (int(px), int(py)), 12, (0, 255, 180) if pinched else (200, 200, 200), 2, cv2.LINE_AA)

    yaw, pitch = lerp(yaw, t_yaw), lerp(pitch, t_pitch)
    scale = lerp(scale, t_scale)
    pos = [lerp(pos[0], t_pos[0]), lerp(pos[1], t_pos[1])]
    name, _, _, color = MODELS[mi]
    draw_holo(frame, glow, name, color, W / 2 + pos[0], H / 2 + pos[1], scale, pitch, yaw)
    frame = cv2.addWeighted(frame, 1, cv2.GaussianBlur(glow, (9, 9), 0), .85, 0)

    # ================= HUD =================
    corners(frame, W, H, (120, 200, 255))
    panel(frame, 10, 10, W - 20, 46, .5)                                    # top glass bar
    sx = int((now * 140) % (W - 20))
    cv2.line(frame, (10 + sx, 12), (10 + sx, 54), (40, 90, 120), 2, cv2.LINE_AA)  # shimmer
    txt(frame, "HOLOGRAM PROJECTOR", 24, 30, (240, 240, 240), .7, 2)
    txt(frame, "v2.0 // STARK OS", 24, 48, (130, 130, 130), .4, 1)
    (tw, _), _ = cv2.getTextSize(f"{name}  {mi+1}/5", FONT, .8, 2)
    txt(frame, f"{name}  {mi+1}/5", (W - tw) // 2, 40, color, .8, 2)
    st_col = (120, 255, 120) if tips else (90, 90, 255)                     # tracking light
    cv2.circle(frame, (W - 190, 33), 5, st_col, -1, cv2.LINE_AA)
    txt(frame, "TRACKING" if tips else "NO HANDS", W - 178, 38, st_col, .5, 1)
    txt(frame, f"{fps:.0f} FPS", W - 80, 38, (240, 240, 240), .55, 1)
    gmode = "ZOOM" if len(tips) == 2 else ("GRAB" if tips and tips[0][1] else ("ROTATE" if tips else "IDLE"))
    txt(frame, f"MODE: {gmode}", 24, H - 64, CYAN, .55, 1)

    panel(frame, 10, H // 2 - 50, 165, 100, .45)                            # rotation dials
    txt(frame, f"YAW   {int(yaw) % 360:>4}", 22, H // 2 - 26, (220, 220, 220))
    meter(frame, 22, H // 2 - 16, 140, (yaw % 360) / 360, color)
    txt(frame, f"PITCH {int(pitch):>4}", 22, H // 2 + 12, (220, 220, 220))
    meter(frame, 22, H // 2 + 22, 140, (pitch + 90) / 180, color)

    panel(frame, W - 30, H // 2 - 70, 16, 140, .5)                          # zoom meter
    meter(frame, W - 28, H // 2 + 68, 136, (scale - 60) / 240, color, vert=True)
    txt(frame, "Z", W - 27, H // 2 - 78, CYAN)

    panel(frame, 10, H - 52, W - 20, 40, .5)                                # bottom bar
    chip_rects = []
    cx = 24
    for i, (_, short, _, mc) in enumerate(MODELS):
        nx = chip(frame, short, cx, H - 46, mc, i == mi)
        chip_rects.append(((cx, H - 46, nx - cx - 8, 30), i)); cx = nx
    txt(frame, "PINCH grab | HAND rotate | 2 HANDS zoom | [N][P] [R]eset [Q]uit",
        cx + 16, H - 26, (170, 170, 170), .5, 1)

    if not tips:                                                            # idle prompt
        p = .5 + .5 * math.sin(now * 4)
        s_ = "RAISE YOUR HANDS TO PROJECT"
        (tw, _), _ = cv2.getTextSize(s_, FONT, .8, 2)
        panel(frame, (W - tw) // 2 - 20, H // 2 - 96, tw + 40, 40, .3 + .3 * p)
        txt(frame, s_, (W - tw) // 2, H // 2 - 70, (255, 255, int(120 + 135 * p)), .8, 2)

    cv2.imshow("Hologram Viewer", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    elif k == ord("n"): mi = (mi + 1) % 5; scale = 40
    elif k == ord("p"): mi = (mi - 1) % 5; scale = 40
    elif k == ord("r"): t_yaw = t_pitch = 0.0; t_scale = 150.0; t_pos = [0, 0]

cap.release(); cv2.destroyAllWindows()