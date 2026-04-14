"""
Terminator Vision – T-800 HUD simulace přes webcam
Detekuje osoby a objekty pomocí YOLOv8-seg a zobrazuje T2-style overlay.
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False

# ---------------------------------------------------------------------------
# Audio engine
# ---------------------------------------------------------------------------

_SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")


class SoundEngine:
    """Přehrávač zvuků ze složky sounds/ pomocí pygame.mixer.

    Zvukové události:
      startup    – přehraje se jednou při startu aplikace
      ambient    – smyčka na pozadí (nízká hlasitost)
      target     – nový cíl (osoba) detekován
      new_target – alternativní zvuk nového cíle (náhodně střídán s target)
      scan       – periodický „skenování" zvuk (~každé 4 s)
      alert      – poplach při ≥ 3 cílech najednou
    """

    _VOL = {
        "startup":    0.80,
        "ambient":    0.22,
        "target":     0.70,
        "new_target": 0.70,
        "scan":       0.60,
        "alert":      0.75,
    }

    def __init__(self):
        self.enabled = False
        if not _PYGAME_OK:
            print("[WARN] pygame není nainstalován – zvuk nebude dostupný. "
                  "Instaluj: pip install pygame", file=sys.stderr)
            return
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            pygame.mixer.set_num_channels(4)
            self._sounds: dict = {}
            for name in self._VOL:
                path = os.path.join(_SOUNDS_DIR, f"{name}.wav")
                if os.path.exists(path):
                    snd = pygame.mixer.Sound(path)
                    snd.set_volume(self._VOL[name])
                    self._sounds[name] = snd
                else:
                    print(f"[WARN] Zvukový soubor nenalezen: {path}", file=sys.stderr)
            self._ch_ambient = pygame.mixer.Channel(0)
            self._ch_event   = pygame.mixer.Channel(1)
            self._ch_scan    = pygame.mixer.Channel(2)
            self.enabled = True
        except Exception as exc:
            print(f"[WARN] Audio init selhal: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    def play_startup(self) -> None:
        if self.enabled and "startup" in self._sounds:
            self._ch_event.play(self._sounds["startup"])

    def start_ambient(self) -> None:
        if self.enabled and "ambient" in self._sounds:
            self._ch_ambient.play(self._sounds["ambient"], loops=-1)

    def play_target(self) -> None:
        """Zvuk pro nově detekovaný cíl – náhodně vybere target nebo new_target."""
        if not self.enabled:
            return
        candidates = [n for n in ("target", "new_target") if n in self._sounds]
        if candidates and not self._ch_event.get_busy():
            self._ch_event.play(self._sounds[random.choice(candidates)])

    def play_scan(self) -> None:
        """Periodický scan – nehraje se, pokud kanál právě hraje."""
        if self.enabled and "scan" in self._sounds and not self._ch_scan.get_busy():
            self._ch_scan.play(self._sounds["scan"])

    def play_alert(self) -> None:
        """Poplach – nehraje se přes probíhající event."""
        if self.enabled and "alert" in self._sounds and not self._ch_event.get_busy():
            self._ch_event.play(self._sounds["alert"])

    def stop(self) -> None:
        if self.enabled:
            pygame.mixer.stop()
            pygame.mixer.quit()


# ---------------------------------------------------------------------------
# Konstanty
# ---------------------------------------------------------------------------
PERSON_PHRASES = [
    "TARGET ACQUIRED",
    "THREAT ASSESSMENT: HOSTILE",
    "INITIATING TARGETING",
    "SUBJECT IDENTIFIED",
    "TRACKING ACTIVE",
]

RED       = (220, 220, 220)
RED_DIM   = (160, 160, 160)
RED_TEXT  = (255, 255, 255)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_SM   = 0.40
FONT_MD   = 0.50

# Zóny těla pro částečný mesh (name, y_rel_start, y_rel_end, x_rel_start, x_rel_end)
BODY_ZONES = [
    ("HEAD",        0.00, 0.20, 0.22, 0.78),
    ("TORSO",       0.20, 0.62, 0.10, 0.90),
    ("LEFT ARM",    0.20, 0.75, 0.00, 0.28),
    ("RIGHT ARM",   0.20, 0.75, 0.72, 1.00),
    ("LEFT LEG",    0.62, 1.00, 0.05, 0.50),
    ("RIGHT LEG",   0.62, 1.00, 0.50, 0.95),
]
ZONE_DURATION = 0.7   # sekund na jednu zónu

SCAN_ACQUIRES = [
    "ACQUIRE TRANSPORT",
    "ACQUIRE HUMAN TARGET",
    "ACQUIRE VEHICLE",
    "THREAT ANALYSIS",
    "ENVIRONMENT SCAN",
    "BIOMETRIC SCAN",
    "PATTERN MATCH",
]


# ---------------------------------------------------------------------------
# Obrazové filtry
# ---------------------------------------------------------------------------

def apply_red_filter(frame: np.ndarray) -> np.ndarray:
    """Boost R, potlač G a B – terminator červený pohled."""
    f = frame.astype(np.float32)
    f[:, :, 0] = np.clip(f[:, :, 0] * 0.08,          0, 255)  # B
    f[:, :, 1] = np.clip(f[:, :, 1] * 0.25,          0, 255)  # G
    f[:, :, 2] = np.clip(f[:, :, 2] * 1.3 + 20,      0, 255)  # R
    return f.astype(np.uint8)


def apply_scanlines(frame: np.ndarray) -> np.ndarray:
    """Každý 3. řádek ztmavit na 55 % – CRT scanline efekt."""
    out = frame.copy()
    out[::3] = (out[::3] * 0.55).astype(np.uint8)
    return out


def apply_vignette(frame: np.ndarray) -> np.ndarray:
    """Gaussovský vignette – ztmavení okrajů."""
    h, w = frame.shape[:2]
    sigma_x = w * 0.55
    sigma_y = h * 0.55
    gx = cv2.getGaussianKernel(w, sigma_x)
    gy = cv2.getGaussianKernel(h, sigma_y)
    kernel = gy * gx.T
    mask = kernel / kernel.max()
    # blend: 0.55 váha vignette darkness
    vignette = np.clip(mask + 0.45, 0.0, 1.0).astype(np.float32)
    out = (frame.astype(np.float32) * vignette[:, :, np.newaxis])
    return np.clip(out, 0, 255).astype(np.uint8)


def add_noise(frame: np.ndarray) -> np.ndarray:
    """Per-pixel náhodný šum ±6."""
    noise = np.random.randint(-6, 7, frame.shape, dtype=np.int16)
    out = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Pomocné kreslení
# ---------------------------------------------------------------------------

def put_text_outlined(img, text, pos, font_scale=FONT_SM, color=RED_TEXT, thickness=1):
    """Text s černým outline pro čitelnost."""
    x, y = pos
    cv2.putText(img, text, (x - 1, y - 1), FONT, font_scale, (0, 0, 0),    thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x,     y),     FONT, font_scale, color,         thickness,     cv2.LINE_AA)


def draw_corner_bracket(img, x1, y1, x2, y2, size=14, color=RED_TEXT, thickness=2):
    """Rohové závorky (L-tvar) – ne celý obdélník."""
    pts = [
        # top-left
        ((x1, y1 + size), (x1, y1), (x1 + size, y1)),
        # top-right
        ((x2 - size, y1), (x2, y1), (x2, y1 + size)),
        # bottom-left
        ((x1, y2 - size), (x1, y2), (x1 + size, y2)),
        # bottom-right
        ((x2 - size, y2), (x2, y2), (x2, y2 - size)),
    ]
    for p0, p1, p2 in pts:
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Nové HUD prvky – mřížka, scan panel, zaměřovací kříž
# ---------------------------------------------------------------------------

def draw_global_grid(img: np.ndarray, tick: float) -> None:
    """Semitransparentní skenovací mřížka a pohyblivá sweep linka – T2 styl."""
    h, w = img.shape[:2]
    layer = np.zeros_like(img)
    spacing = 30
    gc = (38, 38, 38)
    for x in range(0, w + spacing, spacing):
        cv2.line(layer, (x, 0), (x, h), gc, 1)
    for y in range(0, h + spacing, spacing):
        cv2.line(layer, (0, y), (w, y), gc, 1)

    # pohyblivá sweep linka dolů
    sweep_period = 3.5
    sweep_y = int((tick % sweep_period) / sweep_period * h)
    cv2.line(layer, (0, sweep_y), (w, sweep_y), (140, 140, 140), 1)
    for off in range(1, 12):
        sy = sweep_y - off
        if sy >= 0:
            alpha = int(100 * (1 - off / 12))
            cv2.line(layer, (0, sy), (w, sy), (alpha, alpha, alpha), 1)

    cv2.add(img, layer, img)


def draw_scan_panel(img: np.ndarray, tick: float) -> None:
    """SCAN MODE datový panel – pravý dolní roh (jako v T2)."""
    h, w = img.shape[:2]

    scan_num = (int(tick / 2.3) * 137 + 3958) % 99999
    acq_idx  = int(tick / 4.5) % len(SCAN_ACQUIRES)
    priority = f"PRIORITY {int(tick * 1237) % 9999999:07d}D"

    rng = random.Random(int(tick / 1.8))
    data_rows = []
    for tag in ("VEHI", "MTRC", "TRCT"):
        a = rng.randint(10000, 99999)
        b = rng.randint(10000, 99999)
        c = rng.randint(0, 99)
        data_rows.append(f"{tag}  {a} {b} {c:02d}")

    lines = [
        f"SCAN MODE {scan_num:05d}",
        SCAN_ACQUIRES[acq_idx],
        priority,
    ] + data_rows

    line_h  = 13
    panel_h = len(lines) * line_h + 8
    panel_w = 196
    px = w - panel_w - 8
    py = h - panel_h - 28

    bg = img.copy()
    cv2.rectangle(bg, (px - 3, py - 4), (px + panel_w, py + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(bg, 0.45, img, 0.55, 0, img)

    for i, line in enumerate(lines):
        color = RED_TEXT if i < 3 else RED_DIM
        put_text_outlined(img, line, (px, py + (i + 1) * line_h), 0.33, color)


def draw_center_reticle(img: np.ndarray, tick: float) -> None:
    """Zaměřovací kříž se rotujícím vnějším kruhem ve středu obrazu."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    r_inner = 10
    r_outer = 28
    col = RED_DIM

    # kříž
    cv2.line(img, (cx - r_outer, cy), (cx - r_inner, cy), col, 1, cv2.LINE_AA)
    cv2.line(img, (cx + r_inner, cy), (cx + r_outer, cy), col, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - r_outer), (cx, cy - r_inner), col, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + r_inner), (cx, cy + r_outer), col, 1, cv2.LINE_AA)

    # statický vnitřní kruh
    cv2.circle(img, (cx, cy), r_inner, col, 1, cv2.LINE_AA)

    # rotující vnější tečky (8 bodů)
    angle_off = (tick * 25) % 360
    for i in range(8):
        a = np.radians(i * 45 + angle_off)
        x = int(cx + r_outer * 1.45 * np.cos(a))
        y = int(cy + r_outer * 1.45 * np.sin(a))
        cv2.circle(img, (x, y), 1, col, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Segmentační obrys – T2 styl
# ---------------------------------------------------------------------------

def draw_segmentation_outline(frame: np.ndarray, mask_pts, label: str) -> None:
    """Nakreslí T2-style obrys segmentační masky na frame (in-place)."""
    pts = np.array(mask_pts, dtype=np.int32).reshape((-1, 1, 2))
    is_person = (label == "person")
    bright = (255, 255, 255) if is_person else (180, 180, 180)
    glow   = ( 80,  80,  80) if is_person else ( 60,  60,  60)

    # glow halo (wider, darker)
    cv2.polylines(frame, [pts], True, glow,   thickness=5, lineType=cv2.LINE_AA)
    # hlavní obrys
    cv2.polylines(frame, [pts], True, bright, thickness=2, lineType=cv2.LINE_AA)
    # vnitřní semitransparentní fill – thermal glow
    fill = frame.copy()
    cv2.fillPoly(fill, [pts], (200, 200, 200))
    cv2.addWeighted(fill, 0.10, frame, 0.90, 0, frame)


# ---------------------------------------------------------------------------
# Mesh overlay – Delaunay triangulace
# ---------------------------------------------------------------------------

def sample_points_in_mask(mask_bin: np.ndarray, n_points: int = 120):
    """Náhodně vzorkuj n_points bodů uvnitř binární masky."""
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return []
    if len(xs) < n_points:
        return list(zip(xs.tolist(), ys.tolist()))
    idx = np.random.choice(len(xs), n_points, replace=False)
    return list(zip(xs[idx].tolist(), ys[idx].tolist()))


def get_zone_bbox(det_bbox: tuple, zone: tuple) -> tuple:
    """Absolutní souřadnice zóny těla z relativních zlomků bboxu."""
    x1, y1, x2, y2 = det_bbox
    w = x2 - x1
    h = y2 - y1
    _, yr0, yr1, xr0, xr1 = zone
    return (
        x1 + int(xr0 * w),
        y1 + int(yr0 * h),
        x1 + int(xr1 * w),
        y1 + int(yr1 * h),
    )


def sample_points_in_zone(mask_bin: np.ndarray, zone_bbox: tuple, n_points: int = 60):
    """Vzorkuj body uvnitř masky omezené na obdélník zóny."""
    zx1, zy1, zx2, zy2 = zone_bbox
    mh, mw = mask_bin.shape[:2]
    zx1 = max(0, zx1);  zy1 = max(0, zy1)
    zx2 = min(mw - 1, zx2); zy2 = min(mh - 1, zy2)
    if zx2 <= zx1 or zy2 <= zy1:
        return []
    region = mask_bin[zy1:zy2, zx1:zx2]
    ys, xs = np.where(region > 0)
    xs = xs + zx1
    ys = ys + zy1
    if len(xs) == 0:
        return []
    if len(xs) < n_points:
        return list(zip(xs.tolist(), ys.tolist()))
    idx = np.random.choice(len(xs), n_points, replace=False)
    return list(zip(xs[idx].tolist(), ys[idx].tolist()))


def draw_mesh_overlay(frame: np.ndarray, mask_bin: np.ndarray,
                      bbox: tuple, label: str, tick: float) -> None:
    """Delaunay triangulace mesh přes segmentační masku."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w < 4 or h < 4:
        return

    is_person = (label == "person")
    color     = (200, 200, 200) if is_person else (140, 140, 140)
    n_pts     = 150         if is_person else 80

    # vzorkuj body uvnitř masky
    points = sample_points_in_mask(mask_bin, n_pts)
    if len(points) < 4:
        return

    # přidej rohové kotevní body
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2),
               ((x1 + x2) // 2, y1), ((x1 + x2) // 2, y2),
               (x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2)]
    points = points + corners

    rect = (x1, y1, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        px, py = int(p[0]), int(p[1])
        if x1 <= px <= x2 and y1 <= py <= y2:
            try:
                subdiv.insert((float(px), float(py)))
            except cv2.error:
                pass

    try:
        triangles = subdiv.getTriangleList().astype(np.int32)
    except cv2.error:
        return

    mh, mw = mask_bin.shape[:2]
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        # filtruj trojúhelníky jejichž střed leží mimo masku
        cx = (pt1[0] + pt2[0] + pt3[0]) // 3
        cy = (pt1[1] + pt2[1] + pt3[1]) // 3
        if 0 <= cy < mh and 0 <= cx < mw and mask_bin[cy, cx] > 0:
            cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)
            cv2.line(frame, pt2, pt3, color, 1, cv2.LINE_AA)
            cv2.line(frame, pt3, pt1, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# HUD – globální prvky
# ---------------------------------------------------------------------------

def draw_global_hud(img: np.ndarray, fps: float, det_count: int, frame_no: int) -> None:
    """Stavové řádky nahoře/dole + rohové závorky celého obrazu."""
    h, w = img.shape[:2]
    now_str = datetime.now().strftime("%H:%M:%S")

    # rohové závorky celého obrazu
    draw_corner_bracket(img, 4, 4, w - 4, h - 4, size=20, color=RED_TEXT, thickness=2)

    # horní řádek
    put_text_outlined(img, "CYBERDYNE SYSTEMS  MODEL 101", (10, 22), FONT_SM, RED_TEXT)
    tw, _ = cv2.getTextSize(now_str, FONT, FONT_SM, 1)[0], None
    tw = cv2.getTextSize(now_str, FONT, FONT_SM, 1)[0][0]
    put_text_outlined(img, now_str, (w - tw - 10, 22), FONT_SM, RED_TEXT)

    # dolní řádek
    fps_str     = f"FPS: {fps:4.1f}"
    center_str  = "NEURAL NET PROCESSOR :: ACTIVE"
    targets_str = f"TARGETS: {det_count}"

    put_text_outlined(img, fps_str,     (10, h - 10), FONT_SM, RED_TEXT)
    ctw = cv2.getTextSize(center_str, FONT, FONT_SM, 1)[0][0]
    put_text_outlined(img, center_str,  ((w - ctw) // 2, h - 10), FONT_SM, RED_TEXT)
    ttw = cv2.getTextSize(targets_str, FONT, FONT_SM, 1)[0][0]
    put_text_outlined(img, targets_str, (w - ttw - 10, h - 10), FONT_SM, RED_TEXT)


# ---------------------------------------------------------------------------
# HUD – jednotlivá detekce
# ---------------------------------------------------------------------------

def _dashed_line(img, pt1, pt2, color, gap=8):
    """Čárkovaná linie pomocí segmentů."""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    steps = int(dist / gap)
    for i in range(0, steps, 2):
        xa = int(x1 + (x2 - x1) * (i / steps))
        ya = int(y1 + (y2 - y1) * (i / steps))
        xb = int(x1 + (x2 - x1) * (min(i + 1, steps) / steps))
        yb = int(y1 + (y2 - y1) * (min(i + 1, steps) / steps))
        cv2.line(img, (xa, ya), (xb, yb), color, 1, cv2.LINE_4)


def draw_detection(img: np.ndarray, det: dict, frame_shape: tuple,
                   tick: float, debug: bool = False) -> None:
    """
    Nakreslí HUD pro jednu detekci.
    det = {x1, y1, x2, y2, conf, label, idx}
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    conf  = det["conf"]
    label = det["label"]
    idx   = det["idx"]

    is_person = (label == "person")
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # rohové závorky kolem bboxu
    draw_corner_bracket(img, x1, y1, x2, y2, size=14, color=RED_TEXT, thickness=2)

    # čárkovaná linie do středu obrazu (jen pro osoby, alpha blend 30 %)
    if is_person:
        overlay = img.copy()
        _dashed_line(overlay, (cx, cy), (w // 2, h // 2), RED_DIM, gap=10)
        cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)

    # blikající kroužek ve středu pro osoby
    if is_person and int(tick * 2) % 2 == 0:
        cv2.circle(img, (cx, cy), 6, RED_TEXT, 2, cv2.LINE_AA)

    # info box
    display_name = "HUMAN" if is_person else label.upper()
    conf_str     = f"CONF: {int(conf * 100)}%"

    phrase = ""
    height_str = ""
    if is_person:
        phrase_idx = int(tick / 2) % len(PERSON_PHRASES)
        # per-objekt rotace frází (použij idx jako offset)
        phrase = PERSON_PHRASES[(phrase_idx + idx) % len(PERSON_PHRASES)]
        est_h  = int((y2 - y1) / h * 175)
        height_str = f"HEIGHT EST: {est_h}cm"

    lines = [display_name, conf_str]
    if is_person:
        lines.append(phrase)
        lines.append(height_str)

    line_h  = 16
    box_h   = len(lines) * line_h + 6
    box_w   = max(cv2.getTextSize(l, FONT, FONT_SM, 1)[0][0] for l in lines) + 10

    # umístění boxu – nad bboxem nebo pod ním (pokud je u horního okraje)
    if y1 - box_h - 4 >= 0:
        bx1, by1 = x1, y1 - box_h - 4
    else:
        bx1, by1 = x1, y2 + 4

    bx2 = bx1 + box_w
    by2 = by1 + box_h

    # tmavé poloprůhledné pozadí info boxu
    bg = img.copy()
    cv2.rectangle(bg, (bx1, by1), (bx2, by2), (0, 0, 40), -1)
    cv2.addWeighted(bg, 0.55, img, 0.45, 0, img)

    for i, line in enumerate(lines):
        put_text_outlined(img, line, (bx1 + 4, by1 + (i + 1) * line_h), FONT_SM, RED_TEXT)

    # debug: raw YOLO bbox
    if debug:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        put_text_outlined(img, f"{label} {conf:.2f}", (x1, y1 - 2), 0.35, (0, 255, 0))


# ---------------------------------------------------------------------------
# Hlavní smyčka
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Terminator Vision – T-800 HUD")
    parser.add_argument("--camera", type=int, default=0,   help="Index kamery")
    parser.add_argument("--video",  type=str, default=None, help="Cesta k video souboru (místo kamery)")
    parser.add_argument("--width",  type=int, default=640, help="Šířka obrazu")
    parser.add_argument("--height", type=int, default=480, help="Výška obrazu")
    args = parser.parse_args()

    # inicializace zvuku
    sound = SoundEngine()
    sound.play_startup()
    sound.start_ambient()

    # inicializace modelu
    model = YOLO("yolov8n-seg.pt")

    # otevření zdroje videa
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        label = f"soubor '{args.video}'" if args.video else f"kamera {args.camera}"
        print(f"[ERROR] Nelze otevřít {label}.", file=sys.stderr)
        sys.exit(1)

    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fullscreen = False
    debug_mode = False
    WIN_NAME   = "Terminator Vision"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    frame_no        = 0
    fps             = 0.0
    fps_timer       = time.time()
    fps_frame_count = 0

    # cache YOLO výsledků (každý 2. frame)
    cached_detections = []   # list of dicts
    cached_masks_bin  = []   # list of np.ndarray (binární masky)
    last_mesh_tick    = 0.0  # čas posledního resample meshe
    mesh_points_cache = {}   # idx -> points
    mesh_zone_state   = {}   # idx -> (zone_idx, zone_start_tick)

    # audio stav
    prev_det_count = 0
    last_scan_tick = time.time()
    _SCAN_INTERVAL = 4.0   # sekundy mezi scan zvuky
    _ALERT_THRESH  = 3     # počet cílů pro alert

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                # loop video souboru od začátku
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret:
                print("[WARN] Nelze číst frame.")
                break

        frame_no += 1
        tick = time.time()

        # ---------------------------------------------------------------
        # YOLO inference každý 2. frame
        # ---------------------------------------------------------------
        run_inference = (frame_no % 2 == 1)
        if run_inference:
            results = model(frame, conf=0.40, verbose=False)
            cached_detections = []
            cached_masks_bin  = []

            if results and results[0].boxes is not None:
                boxes  = results[0].boxes
                masks  = results[0].masks  # může být None pokud žádná maska

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf_val  = float(boxes.conf[i].cpu())
                    cls_id    = int(boxes.cls[i].cpu())
                    lbl       = model.names[cls_id]

                    det = dict(x1=x1, y1=y1, x2=x2, y2=y2,
                               conf=conf_val, label=lbl, idx=i,
                               mask_xy=None)

                    mask_bin = None
                    if masks is not None and i < len(masks.xy):
                        det["mask_xy"] = masks.xy[i]
                        # binární maska z masks.data pro mesh
                        try:
                            mdata = masks.data[i].cpu().numpy()
                            # resize na rozlišení framu
                            mdata_u8 = (mdata > 0.5).astype(np.uint8) * 255
                            mask_bin = cv2.resize(mdata_u8,
                                                  (frame.shape[1], frame.shape[0]),
                                                  interpolation=cv2.INTER_NEAREST)
                        except Exception:
                            mask_bin = None

                    cached_detections.append(det)
                    cached_masks_bin.append(mask_bin)

            # resample mesh bodů každých 0.4 s + zone cycling pro osoby
            if tick - last_mesh_tick > 0.4:
                last_mesh_tick = tick
                new_cache = {}
                for i, (det, mbin) in enumerate(zip(cached_detections, cached_masks_bin)):
                    if mbin is not None:
                        lbl      = det["label"]
                        bbox_det = (det["x1"], det["y1"], det["x2"], det["y2"])
                        if lbl == "person":
                            zone_idx, zone_start = mesh_zone_state.get(i, (0, tick))
                            if tick - zone_start > ZONE_DURATION:
                                zone_idx   = (zone_idx + 1) % len(BODY_ZONES)
                                zone_start = tick
                            mesh_zone_state[i] = (zone_idx, zone_start)
                            zbbox = get_zone_bbox(bbox_det, BODY_ZONES[zone_idx])
                            new_cache[i] = sample_points_in_zone(mbin, zbbox, n_points=60)
                        else:
                            new_cache[i] = sample_points_in_mask(mbin, 80)
                mesh_points_cache = new_cache

        # ---------------------------------------------------------------
        # Aplikuj obrazové filtry
        # ---------------------------------------------------------------
        filtered = apply_red_filter(frame)
        filtered = apply_scanlines(filtered)
        filtered = apply_vignette(filtered)
        filtered = add_noise(filtered)

        # ---------------------------------------------------------------
        # Kresli overlay – pracuj na kopii pro alpha blend
        # ---------------------------------------------------------------
        overlay = filtered.copy()

        for i, det in enumerate(cached_detections):
            mask_xy  = det.get("mask_xy")
            mask_bin = cached_masks_bin[i] if i < len(cached_masks_bin) else None
            lbl      = det["label"]
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

            # segmentační obrys
            if mask_xy is not None and len(mask_xy) > 2:
                draw_segmentation_outline(overlay, mask_xy, lbl)

            # mesh overlay
            if mask_bin is not None:
                pts = mesh_points_cache.get(i)
                if pts:
                    color     = (200, 200, 200) if lbl == "person" else (140, 140, 140)
                    zone_bbox = None
                    if lbl == "person":
                        zone_idx, _ = mesh_zone_state.get(i, (0, tick))
                        zone_bbox   = get_zone_bbox((x1, y1, x2, y2), BODY_ZONES[zone_idx])
                        # label zóny vedle bboxu
                        zone_name = BODY_ZONES[zone_idx][0]
                        put_text_outlined(overlay, f"SCAN: {zone_name}",
                                          (x2 + 4, zone_bbox[1] + 10), 0.30, RED_DIM)
                    _draw_cached_mesh(overlay, pts, mask_bin,
                                      (x1, y1, x2, y2), color, zone_bbox)

        # HUD per-detekce (na overlay)
        for det in cached_detections:
            draw_detection(overlay, det, frame.shape, tick, debug=debug_mode)

        # mřížka na overlay (před blendem)
        draw_global_grid(overlay, tick)

        # zkombinuj overlay s filtrovaným framem (85/15)
        out = cv2.addWeighted(overlay, 0.85, filtered, 0.15, 0)

        # globální HUD přímo na výsledný obraz
        fps_frame_count += 1
        if tick - fps_timer >= 0.5:
            fps = fps_frame_count / (tick - fps_timer)
            fps_frame_count = 0
            fps_timer = tick

        draw_global_hud(out, fps, len(cached_detections), frame_no)
        draw_scan_panel(out, tick)
        draw_center_reticle(out, tick)

        # ---------------------------------------------------------------
        # Zvukové události
        # ---------------------------------------------------------------
        det_count = len(cached_detections)
        if det_count > prev_det_count:
            # nový cíl přibyl
            sound.play_target()
        elif det_count >= _ALERT_THRESH and det_count > prev_det_count - 1:
            # stále hodně cílů – alert (jen pokud se count nezmenšil)
            sound.play_alert()
        prev_det_count = det_count

        if tick - last_scan_tick >= _SCAN_INTERVAL:
            last_scan_tick = tick
            sound.play_scan()

        cv2.imshow(WIN_NAME, out)

        # ---------------------------------------------------------------
        # Klávesy
        # ---------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts  = datetime.now().strftime("%H%M%S")
            fn  = f"terminator_{ts}.png"
            cv2.imwrite(fn, out)
            print(f"[INFO] Screenshot uložen: {fn}")
        elif key == ord('f'):
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"[INFO] Debug mode: {'ON' if debug_mode else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    sound.stop()


# ---------------------------------------------------------------------------
# Pomocná funkce – kreslení meshe z cachovaných bodů
# ---------------------------------------------------------------------------

def _draw_cached_mesh(frame: np.ndarray, points: list, mask_bin: np.ndarray,
                      bbox: tuple, color: tuple,
                      zone_bbox: tuple = None) -> None:
    """Delaunay triangulace z předpočítaných bodů. zone_bbox omezuje kreslení na část těla."""
    x1, y1, x2, y2 = bbox
    if len(points) < 4:
        return

    # oblast pro Subdiv2D – buď celý bbox nebo zóna
    if zone_bbox:
        rx1, ry1, rx2, ry2 = zone_bbox
    else:
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2
    rw = max(rx2 - rx1, 4)
    rh = max(ry2 - ry1, 4)

    corners = [
        (rx1, ry1), (rx2, ry1), (rx1, ry2), (rx2, ry2),
        ((rx1 + rx2) // 2, ry1), ((rx1 + rx2) // 2, ry2),
        (rx1, (ry1 + ry2) // 2), (rx2, (ry1 + ry2) // 2),
    ]
    all_pts = list(points) + corners

    rect   = (rx1, ry1, rw, rh)
    subdiv = cv2.Subdiv2D(rect)
    for p in all_pts:
        px, py = int(p[0]), int(p[1])
        if rx1 <= px <= rx2 and ry1 <= py <= ry2:
            try:
                subdiv.insert((float(px), float(py)))
            except cv2.error:
                pass

    try:
        triangles = subdiv.getTriangleList().astype(np.int32)
    except cv2.error:
        return

    mh, mw = mask_bin.shape[:2]
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        tcx = (pt1[0] + pt2[0] + pt3[0]) // 3
        tcy = (pt1[1] + pt2[1] + pt3[1]) // 3
        # musí ležet uvnitř masky
        if not (0 <= tcy < mh and 0 <= tcx < mw and mask_bin[tcy, tcx] > 0):
            continue
        # musí ležet uvnitř zóny (pokud je zadána)
        if zone_bbox and not (rx1 <= tcx <= rx2 and ry1 <= tcy <= ry2):
            continue
        cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.line(frame, pt2, pt3, color, 1, cv2.LINE_AA)
        cv2.line(frame, pt3, pt1, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
