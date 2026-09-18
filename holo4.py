# holo_sketch.py — air drawing with ORIGINAL viewer gestures
# POINT = draw | PINCH = grab & move | OPEN HAND = rotate | 2 HANDS = zoom
import math, os, time, urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

# ---------------- palette / brushes ----------------
PALETTE = [((255, 255, 0), "CYAN"), ((255, 0, 255), "MAGENTA"), ((0, 255, 255), "YELLOW"),
           ((80, 255, 120), "GREEN"), ((0, 165, 255), "ORANGE"), ((255, 255, 255), "WHITE")]
BRUSHES = [3, 6, 12]
ERASE_R = 32
HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),
             (11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

# ---------------- canvas space (strokes stored here, transform freely) ----------------
GRID = []
for v in range(-360, 361, 60):
    GRID += [((v, -240), (v, 240)), ((-240, v), (240, v))]
GRID = np.float32(GRID)
FRAME_RECT = np.float32([[-360, -240], [360, -240], [360, 240], [-360, 240]])

def to_screen(P, W, H, pos, ang, sc):
    ca, sa = math.cos(ang), math.sin(ang)
    X = (P[:, 0] * ca - P[:, 1] * sa) * sc + W / 2 + pos[0]
    Y = (P[:, 0] * sa + P[:, 1] * ca) * sc + H / 2 + pos[1]
    return np.stack([X, Y], 1)

def to_canvas(sx, sy, W, H, pos, ang, sc):
    dx, dy = sx - W / 2 - pos[0], sy - H / 2 - pos[1]
    ca, sa = math.cos(ang), math.sin(ang)
    return ((dx * ca + dy * sa) / sc, (-dx * sa + dy * ca) / sc)

# ---------------- hand tracker ----------------
MF = "hand_landmarker.task"
if not os.path.exists(MF):
    print("Downloading hand model (one time)...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                               "hand_landmarker/float16/1/hand_landmarker.task", MF)
hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=MF), num_hands=2))

def finger_ext(hd, tip, pip):                                       # rotation-invariant check
    w = (hd[0].x, hd[0].y)
    return math.dist((hd[tip].x, hd[tip].y), w) > math.dist((hd[pip].x, hd[pip].y), w) * 1.12

# ---------------- drawing state ----------------
strokes = []          # {"pts": (N,2) canvas, "color", "w"}
active = {}
cursor, pinch_was, pt_was, ui_hold = {}, {}, {}, {}
tf = None             # 2-hand zoom capture
ci, bi, eraser = 0, 1, False
pos, t_pos = [0.0, 0.0], [0.0, 0.0]
ang, t_ang = 0.0, 0.0
sc, t_sc = 1.0, 1.0
lerp = lambda a, b, k=.25: a + (b - a) * k

def erase_at(cp, r):
    global strokes
    out = []
    for s in strokes:
        P = s["pts"]
        inside = (P[:, 0] - cp[0]) ** 2 + (P[:, 1] - cp[1]) ** 2 <= r * r
        if len(P) == 1:
            if not inside[0]: out.append(s)
            continue
        st = 0
        for i in range(1, len(P) + 1):
            if i == len(P) or inside[i] != inside[st]:
                if not inside[st] and i - st >= 2:
                    out.append({"pts": P[st:i].copy(), "color": s["color"], "w": s["w"]})
                st = i
    strokes = out

def render_strokes(frame, glow, all_s, W, H):
    for s in all_s:
        P = to_screen(np.asarray(s["pts"], np.float32).reshape(-1, 2), W, H, pos, ang, sc)
        Pi = P.astype(np.int32); w = max(1, int(round(s["w"] * sc)))
        if len(Pi) == 1:
            cv2.circle(frame, tuple(Pi[0]), max(2, int(s["w"] * sc)), s["color"], -1, cv2.LINE_AA)
            cv2.circle(glow, tuple(Pi[0]), max(2, int(s["w"] * sc)) + 3, s["color"], -1, cv2.LINE_AA)
            continue
        cv2.polylines(frame, [Pi], False, s["color"], w, cv2.LINE_AA)
        cv2.polylines(glow, [Pi], False, s["color"], w + 4, cv2.LINE_AA)

def flush(hid_set):                                                 # close open strokes
    for hid in list(active):
        if hid not in hid_set: strokes.append(active.pop(hid))

# ---------------- main loop ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
fps, prev = 0.0, time.time()

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
    lms = res.hand_landmarks or []
    hinfo = []
    for i, hd in enumerate(lms):
        try: hid = res.handedness[i][0].category_name
        except Exception: hid = str(i)
        for a, b in HAND_CONN:
            cv2.line(frame, (int(hd[a].x * W), int(hd[a].y * H)),
                            (int(hd[b].x * W), int(hd[b].y * H)), (70, 70, 70), 1, cv2.LINE_AA)
        ix, iy = hd[8].x * W, hd[8].y * H
        tx, ty = hd[4].x * W, hd[4].y * H
        c = cursor.get(hid, (ix, iy))
        cursor[hid] = (lerp(c[0], ix, .5), lerp(c[1], iy, .5))
        d = (math.dist((hd[4].x, hd[4].y), (hd[8].x, hd[8].y))
             / max(math.dist((hd[0].x, hd[0].y), (hd[9].x, hd[9].y)), 1e-3))
        point = finger_ext(hd, 8, 6) and not finger_ext(hd, 12, 10) and not finger_ext(hd, 16, 14)
        hinfo.append({"id": hid, "c": cursor[hid], "m": ((tx + ix) / 2, (ty + iy) / 2),
                      "pin": d < .2, "pt": point})

    n = len(PALETTE) + 1
    sw_y, sw_r, sw_gap = H - 72, 12, 46
    sw = [(W // 2 + int((j - (n - 1) / 2) * sw_gap), sw_y) for j in range(n)]

    mode = "IDLE"
    if len(hinfo) >= 2:                                             # ==== 2 HANDS = ZOOM ====
        mode = "ZOOM"
        flush({}); pt_was.update({h["id"]: False for h in hinfo})
        p1, p2 = sorted(hinfo[:2], key=lambda h: h["m"][0])
        A, B = p1["m"], p2["m"]
        d = math.dist(A, B); a = math.atan2(B[1] - A[1], B[0] - A[0])
        if tf is None:
            tf = {"d": d, "a": a, "sc": t_sc, "an": t_ang}
        else:
            t_sc = float(np.clip(tf["sc"] * d / max(tf["d"], 1e-3), .3, 4.0))
            t_ang = tf["an"] + math.remainder(a - tf["a"], 2 * math.pi)   # tilt line = rotate too
        cv2.line(frame, (int(A[0]), int(A[1])), (int(B[0]), int(B[1])), (255, 255, 255), 1, cv2.LINE_AA)
        for P in (A, B): cv2.circle(frame, (int(P[0]), int(P[1])), 14, (255, 255, 255), 2, cv2.LINE_AA)
    elif len(hinfo) == 1:                                           # ==== ONE HAND ====
        h = hinfo[0]; hid = h["id"]; c = h["c"]
        if h["pin"] and not pinch_was.get(hid, False):              # pinch-start: swatch check
            hit = next((j for j, s in enumerate(sw) if math.dist(c, s) < sw_r + 8), None)
            if hit is not None and hit < len(PALETTE): ci, eraser = hit, False; ui_hold[hid] = True
            elif hit == len(PALETTE): eraser = not eraser; ui_hold[hid] = True
        if h["pin"] and not ui_hold.get(hid, False):                # ==== PINCH = MOVE ====
            mode = "MOVE"; flush({hid}); pt_was[hid] = False
            t_pos = [h["m"][0] - W / 2, h["m"][1] - H / 2]          # drawing follows pinch
        elif h["pt"]:                                               # ==== POINT = DRAW ====
            mode = "ERASE" if eraser else "DRAW"
            cp = to_canvas(c[0], c[1], W, H, pos, ang, sc)
            if eraser:
                erase_at(cp, ERASE_R / sc); pt_was[hid] = True
            else:
                if not pt_was.get(hid, False):
                    active[hid] = {"pts": np.float32([cp]), "color": PALETTE[ci][0], "w": BRUSHES[bi]}
                elif hid in active and math.dist(active[hid]["pts"][-1], cp) > 2.5 / sc:
                    active[hid]["pts"] = np.vstack([active[hid]["pts"], cp])
                pt_was[hid] = True
        else:                                                       # ==== OPEN HAND = ROTATE ====
            mode = "ROTATE"; flush({hid}); pt_was[hid] = False
            t_ang = (c[0] / W - .5) * math.radians(360)             # like original yaw mapping
        if not h["pin"]: ui_hold.pop(hid, None)
        pinch_was[hid] = h["pin"]
    else:
        tf = None
        if not strokes and not active: t_ang += .003                # idle spin (empty stage only)
    if len(hinfo) != 1:
        for hid in list(pinch_was): pinch_was[hid] = False
    flush({h["id"] for h in hinfo})
    for hid in list(cursor):
        if hid not in [h["id"] for h in hinfo]:
            cursor.pop(hid); pinch_was.pop(hid, None); pt_was.pop(hid, None); ui_hold.pop(hid, None)

    pos = [lerp(pos[0], t_pos[0]), lerp(pos[1], t_pos[1])]
    ang, sc = lerp(ang, t_ang), lerp(sc, t_sc)

    G = to_screen(GRID.reshape(-1, 2), W, H, pos, ang, sc).reshape(-1, 2, 2).astype(int)
    for a_, b_ in G: cv2.line(frame, tuple(a_), tuple(b_), (60, 60, 85), 1, cv2.LINE_AA)
    R = to_screen(FRAME_RECT, W, H, pos, ang, sc).astype(np.int32)
    cv2.polylines(frame, [R], True, (90, 90, 130), 1, cv2.LINE_AA)

    render_strokes(frame, glow, strokes + list(active.values()), W, H)
    frame = cv2.addWeighted(frame, 1, cv2.GaussianBlur(glow, (9, 9), 0), .85, 0)

    for h in hinfo:                                                 # cursors per gesture
        c = h["c"]
        if mode == "ZOOM": continue
        if h["pin"] and not ui_hold.get(h["id"]):                   # move
            cv2.circle(frame, (int(c[0]), int(c[1])), 16, (0, 255, 180), 2, cv2.LINE_AA)
            cv2.circle(frame, (int(c[0]), int(c[1])), 4, (0, 255, 180), -1, cv2.LINE_AA)
        elif h["pt"]:                                               # pen
            col = (60, 60, 230) if eraser else PALETTE[ci][0]
            if eraser:
                cv2.circle(frame, (int(c[0]), int(c[1])), ERASE_R, col, 1, cv2.LINE_AA)
                cv2.line(frame, (int(c[0]) - 6, int(c[1])), (int(c[0]) + 6, int(c[1])), col, 2, cv2.LINE_AA)
                cv2.line(frame, (int(c[0]), int(c[1]) - 6), (int(c[0]), int(c[1]) + 6), col, 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, (int(c[0]), int(c[1])), 10, col, 2, cv2.LINE_AA)
                cv2.circle(frame, (int(c[0]), int(c[1])), 3, col, -1, cv2.LINE_AA)
        else:                                                       # rotate spinner
            cv2.ellipse(frame, (int(c[0]), int(c[1])), (18, 18), 0, 0, (now * 240) % 360, (0, 140, 255), 2, cv2.LINE_AA)

    sy = int((time.time() * .25 % 1) * H)                           # scan sweep
    ov = frame.copy(); cv2.line(ov, (0, sy), (W, sy), PALETTE[ci][0], 1, cv2.LINE_AA)
    cv2.addWeighted(ov, .2, frame, .8, 0, frame)

    pad_w = n * sw_gap // 2 + 26                                    # pad deck + swatches
    ov = frame.copy(); cv2.ellipse(ov, (W // 2, sw_y), (pad_w, 14), 0, 0, 360, PALETTE[ci][0], -1)
    cv2.addWeighted(ov, .18, frame, .82, 0, frame)
    cv2.ellipse(frame, (W // 2, sw_y), (pad_w, 14), 0, 0, 360, PALETTE[ci][0], 1, cv2.LINE_AA)
    for j, (sx, syy) in enumerate(sw):
        if j < len(PALETTE):
            cv2.circle(frame, (sx, syy), sw_r, PALETTE[j][0], -1, cv2.LINE_AA)
            if j == ci and not eraser: cv2.circle(frame, (sx, syy), sw_r + 4, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (sx, syy), sw_r, (40, 40, 40), -1, cv2.LINE_AA)
            cv2.line(frame, (sx - 6, syy - 6), (sx + 6, syy + 6), (60, 60, 230), 2, cv2.LINE_AA)
            cv2.line(frame, (sx - 6, syy + 6), (sx + 6, syy - 6), (60, 60, 230), 2, cv2.LINE_AA)
            if eraser: cv2.circle(frame, (sx, syy), sw_r + 4, (60, 60, 230), 2, cv2.LINE_AA)

    for y, h_ in ((0, 44), (H - 46, 46)):                           # HUD
        ov = frame.copy(); cv2.rectangle(ov, (0, y), (W, y + h_), (0, 0, 0), -1)
        cv2.addWeighted(ov, .45, frame, .55, 0, frame)
    txt = lambda s, x, y, c2, s2=.6, th=2: cv2.putText(frame, s, (x, y),
                                                       cv2.FONT_HERSHEY_SIMPLEX, s2, c2, th, cv2.LINE_AA)
    txt("HOLOGRAM SKETCHPAD", 16, 29, (240, 240, 240))
    mc = {"DRAW": PALETTE[ci][0], "ERASE": (60, 60, 230), "MOVE": (0, 255, 180),
          "ROTATE": (0, 140, 255), "ZOOM": (255, 255, 255), "IDLE": (150, 150, 150)}[mode]
    (tw, _), _ = cv2.getTextSize(mode, cv2.FONT_HERSHEY_SIMPLEX, .8, 2)
    txt(mode, (W - tw) // 2, 30, mc, .8, 2)
    txt(f"{fps:.0f} FPS  ZOOM {int(sc*100)}%  ROT {int(math.degrees(ang))%360}", W - 330, 29, (240, 240, 240), .6, 1)
    txt("POINT = draw   PINCH = move   OPEN HAND = rotate   2 HANDS = zoom   (1 hand only to draw)",
        16, H - 28, (200, 220, 240), .5, 1)
    txt("[N/P] color [B]rush [E]raser [U]ndo [C]lear [R]eset view [F]latten [S]ave [Q]uit",
        16, H - 10, (190, 190, 190), .5, 1)
    if not lms:
        msg = "RAISE A HAND — POINT TO DRAW, PINCH TO MOVE, 2 HANDS TO ZOOM"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, .8, 2)
        txt(msg, (W - tw) // 2, H // 2, (180, 220, 255), .8, 2)

    cv2.imshow("Hologram Sketchpad", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    elif k == ord("c"): strokes.clear()
    elif k == ord("u") and strokes: strokes.pop()
    elif k == ord("e"): eraser = not eraser
    elif k == ord("b"): bi = (bi + 1) % len(BRUSHES)
    elif k == ord("n"): ci = (ci + 1) % len(PALETTE); eraser = False
    elif k == ord("p"): ci = (ci - 1) % len(PALETTE); eraser = False
    elif k == ord("s"): cv2.imwrite(f"holo_{int(time.time())}.png", frame); print("saved")
    elif k == ord("r"): t_pos = [0.0, 0.0]; t_ang = 0.0; t_sc = 1.0
    elif k == ord("f"): flush(set()); 
    if k == ord("f"):                                                 # bake view into strokes
        ca, sa = math.cos(ang), math.sin(ang)
        for s in strokes:
            P = s["pts"]
            s["pts"] = np.stack([(P[:, 0] * ca - P[:, 1] * sa) * sc + pos[0],
                                 (P[:, 0] * sa + P[:, 1] * ca) * sc + pos[1]], 1)
        pos = t_pos = [0.0, 0.0]; ang = t_ang = 0.0; sc = t_sc = 1.0

cap.release(); cv2.destroyAllWindows()