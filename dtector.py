import cv2, os, math, time, random, threading, urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
try:
    import winsound
except ImportError:
    winsound = None

MODEL = "hand_landmarker.task"
if not os.path.exists(MODEL):
    print("Downloading hand model (one time)...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", MODEL)

landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL), num_hands=2))

TIPS = [4, 8, 12, 16, 20]
CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
THEMES  = ["Rainbow", "Cyberpunk", "Lava", "Ocean", "Galaxy"]
PALETTE = {"Cyberpunk": [(60,0,255),(255,240,0)], "Lava": [(30,60,255),(0,120,255)],
           "Ocean": [(255,220,0),(255,120,0)], "Galaxy": [(255,0,255),(200,150,255)]}
theme_i = 0

def col(i, t):
    if THEMES[theme_i] == "Rainbow":
        h = (t * 60 + i * 8) % 180
        b, g, r = cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        return (int(b), int(g), int(r))
    return PALETTE[THEMES[theme_i]][i % 2]

def zap():
    if winsound:
        threading.Thread(target=winsound.Beep, args=(900, 90), daemon=True).start()

def txt(img, s, x, y, c, sc=0.6, th=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)

particles, ripples, last_pinch = [], [], [False, False]
rain, vel_prev = None, None
cap = cv2.VideoCapture(0)
prev, fps, t = time.time(), 0.0, 0.0

while True:
    ok, cam = cap.read()
    if not ok:
        break
    cam = cv2.flip(cam, 1)
    H, W = cam.shape[:2]
    frame = (cam * 0.55).astype(np.uint8)                  # dim camera so neon pops
    t += 0.033
    now = time.time(); fps = fps * 0.9 + 0.1 / max(now - prev, 1e-6); prev = now

    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)))
    hands = res.hand_landmarks or []
    vel = 0.0
    if hands:
        p8 = (hands[0][8].x * W, hands[0][8].y * H)
        if vel_prev:
            vel = min(math.hypot(p8[0] - vel_prev[0], p8[1] - vel_prev[1]) / 40, 3)
        vel_prev = p8

    # ---- matrix rain background (speed reacts to hand motion) ----
    if rain is None:
        rain = np.random.rand(W // 16) * H / 16
        bg = np.zeros((H, W), np.uint8)
    bg = (bg * 0.90).astype(np.uint8)
    acc = np.array(col(1, t), np.float32)
    for _ in range(12):
        i = random.randrange(len(rain))
        txt(bg, chr(0x30A0 + random.randint(0, 90)), i * 16, int(rain[i] * 16), 255, 0.5, 1)
        rain[i] += 0.6 + vel * 2
        if rain[i] * 16 > H:
            rain[i] = 0
    frame = cv2.add(frame, (cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR).astype(np.float32) * acc / 255).astype(np.uint8))

    # ---- neon hands ----
    glow = np.zeros_like(frame)
    pt = lambda hd, i: (int(hd[i].x * W), int(hd[i].y * H))
    for hi, hd in enumerate(hands):
        c = col(hi, t)
        for a, b in CONN:
            cv2.line(glow, pt(hd, a), pt(hd, b), c, 5, cv2.LINE_AA)
        for j in range(21):
            cv2.circle(glow, pt(hd, j), 4, c, -1, cv2.LINE_AA)
        for ti, tip in enumerate(TIPS):
            p = pt(hd, tip)
            cv2.circle(glow, p, 6, (255, 255, 255), -1, cv2.LINE_AA)
            if random.random() < 0.8:
                particles.append([p[0], p[1], random.uniform(-3, 3), random.uniform(-4, 0), 1.0, col(ti, t)])
        d = math.dist(pt(hd, 4), pt(hd, 8)) / max(math.dist(pt(hd, 0), pt(hd, 9)), 1e-3)
        if d < 0.18 and not last_pinch[hi]:                # pinch -> shockwave + zap
            m = ((pt(hd, 4)[0] + pt(hd, 8)[0]) // 2, (pt(hd, 4)[1] + pt(hd, 8)[1]) // 2)
            ripples.append([m[0], m[1], 8, random.uniform(150, 260), 1.0, col(hi, t)])
            zap()
        last_pinch[hi] = d < 0.18

    if len(hands) == 2:
        for ti, tip in enumerate(TIPS):                    # gradient energy beams
            p1, p2 = pt(hands[0], tip), pt(hands[1], tip)
            for s in range(8):
                a = (int(p1[0] + (p2[0]-p1[0])*s/8), int(p1[1] + (p2[1]-p1[1])*s/8))
                b = (int(p1[0] + (p2[0]-p1[0])*(s+1)/8), int(p1[1] + (p2[1]-p1[1])*(s+1)/8))
                cv2.line(frame, a, b, col(ti + s, t), 2, cv2.LINE_AA)
            if math.dist(p1, p2) < 150 and random.random() < 0.5:    # lightning arcs
                mx, my = (p1[0]+p2[0])//2 + random.randint(-30, 30), (p1[1]+p2[1])//2 + random.randint(-30, 30)
                cv2.line(glow, p1, (mx, my), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(glow, (mx, my), p2, (255, 255, 255), 2, cv2.LINE_AA)
        c1, c2 = pt(hands[0], 9), pt(hands[1], 9)          # energy core (closer = bigger)
        mid, rad = ((c1[0]+c2[0])//2, (c1[1]+c2[1])//2), max(int(150 - math.dist(c1, c2)/4), 8)
        cv2.circle(glow, mid, rad, col(0, t), 2, cv2.LINE_AA)
        for i in range(5):                                 # mandala star web
            cv2.line(frame, pt(hands[0], TIPS[i]), pt(hands[1], TIPS[(i+2) % 5]), (170, 170, 170), 1, cv2.LINE_AA)

    # ---- physics: sparks + shockwaves ----
    for p in particles[:]:
        p[0] += p[2]; p[1] += p[3]; p[3] += 0.15; p[4] -= 0.03
        if p[4] <= 0:
            particles.remove(p); continue
        cv2.circle(frame, (int(p[0]), int(p[1])), 2, p[5], -1, cv2.LINE_AA)
    for r in ripples[:]:
        r[2] += (r[3] - r[2]) * 0.12; r[4] -= 0.04
        if r[4] <= 0:
            ripples.remove(r); continue
        cv2.circle(frame, (int(r[0]), int(r[1])), int(r[2]), r[5], max(int(3 * r[4]), 1), cv2.LINE_AA)

    frame = cv2.addWeighted(frame, 1, cv2.GaussianBlur(glow, (7, 7), 0), 0.9, 0)   # neon bloom

    # ---- HUD ----
    ov = frame.copy(); cv2.rectangle(ov, (0, 0), (W, 40), (0, 0, 0), -1)
    frame = cv2.addWeighted(ov, 0.45, frame, 0.55, 0)
    gesture = "None"
    if hands:
        gesture = "PINCH!" if any(last_pinch) else ("Open Hand" if math.dist(pt(hands[0], 8), pt(hands[0], 20)) > 110 else "Fist")
    txt(frame, f"{THEMES[theme_i]} | Hands: {len(hands)} | {gesture} | FPS {fps:.0f}", 14, 27, (255, 255, 255), 0.65, 2)
    txt(frame, "[T] theme  [Q] quit", W - 220, 27, (180, 180, 180), 0.55, 1)

    cv2.imshow("Neon Aura AR", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("t"):
        theme_i = (theme_i + 1) % len(THEMES)

cap.release()
cv2.destroyAllWindows()