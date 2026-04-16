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
from PIL import Image, ImageDraw, ImageFont
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

# ---------------------------------------------------------------------------
# PIL font (Helvetica 73 Extended Bold)
# ---------------------------------------------------------------------------
_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "font", "Helvetica73-Extended Bold.ttf")
_pil_font_cache: dict = {}

def _get_font(scale: float) -> ImageFont.FreeTypeFont:
    px = max(9, int(scale * 40))
    if px not in _pil_font_cache:
        try:
            _pil_font_cache[px] = ImageFont.truetype(_FONT_PATH, px)
        except Exception:
            _pil_font_cache[px] = ImageFont.load_default()
    return _pil_font_cache[px]


def get_text_width(text: str, scale: float) -> int:
    bbox = _get_font(scale).getbbox(text)
    return bbox[2] - bbox[0]


# ---------------------------------------------------------------------------
# PIL font – Modern Vision (terminálový typewriter)
# ---------------------------------------------------------------------------
_FONT_MV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "font", "modern-vision.ttf")
_FONT_MV_SIZE = 100
_font_mv: ImageFont.FreeTypeFont | None = None

def _get_font_mv() -> ImageFont.FreeTypeFont:
    global _font_mv
    if _font_mv is None:
        try:
            _font_mv = ImageFont.truetype(_FONT_MV_PATH, _FONT_MV_SIZE)
        except Exception:
            _font_mv = _get_font(0.65)
    return _font_mv


# ---------------------------------------------------------------------------
# Terminálové zprávy – T2 HUD typewriter (střed dolní třetiny)
# ---------------------------------------------------------------------------
TERMINAL_MESSAGES: list[str] = [
    "SCANNING",
    "MATCH FOUND",
    "SEARCHING",
    "TARGET ACQUIRED",
    "ANALYZING THREAT",
    "IDENTIFY",
    "ACCESSING",
    "PATTERN MATCH",
    "THREAT ASSESSMENT",
    "VOICE PRINT MATCH",
    "POSITIVE ID",
    "ACQUIRING TARGET",
    "CPU SCAN COMPLETE",
    "ESTIMATED RISK: HIGH",
    "SUBJECT IDENTIFIED",
    "INITIATING SEQUENCE",
    "TRACKING ACTIVE",
    "WEAPONS CHECK",
    "DECISION: ACQUIRE",
    "FILE FOUND",
    "ANALYSIS COMPLETE",
    "WANT AND WARRANT: NONE",
    "DATABANK ACCESSED",
    "POSSIBLE MATCH",
    "ALERT: MULTIPLE TARGETS",
]

_TW_CHAR_DELAY  = 0.055   # sekund na jeden znak
_TW_ERASE_WAIT  = 2.0     # sekund čekání po dopsání před smazáním

# Stav typewriteru – persistentní mezi framy
# state: "idle" | "typing" | "pause"
_tw: dict = {
    "state":        "idle",
    "msg":           0,
    "chars":         0,
    "last_tick":     0.0,
    "pause_until":   0.0,
    "cooldown_until": 0.0,   # random 5–10 s prodleva před dalším textem
}
_tw_triggered: bool = False   # True = nová detekce, spustit další zprávu


def terminal_trigger() -> None:
    """Zavolej když je detekován nový objekt – spustí další zprávu."""
    global _tw_triggered
    _tw_triggered = True


# ---------------------------------------------------------------------------
# Fronta textových příkazů
# ---------------------------------------------------------------------------

# Hlavní fronta: (array_id, text, x, y, scale, bgr_color)
_text_queue: list = []
# MV fronta:    (array_id, typed_text, x, y, cursor_visible)
_mv_queue:    list = []


def flush_text(img: np.ndarray) -> None:
    """Vykreslí všechny čekající texty (obě fronty) jedinou PIL konverzí."""
    global _text_queue, _mv_queue
    tid = id(img)
    items    = [(t, x, y, s, c) for (i, t, x, y, s, c) in _text_queue if i == tid]
    mv_items = [(t, x, y, v)    for (i, t, x, y, v)    in _mv_queue    if i == tid]
    if not items and not mv_items:
        return
    pil  = Image.fromarray(img[:, :, ::-1])
    draw = ImageDraw.Draw(pil)

    # Helvetica texty
    for text, x, y, scale, bgr in items:
        font   = _get_font(scale)
        ascent = font.getmetrics()[0]
        py     = y - ascent
        rgb    = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            draw.text((x + dx, py + dy), text, font=font, fill=(0, 0, 0))
        draw.text((x, py), text, font=font, fill=rgb)

    # Modern Vision typewriter texty + blokový kurzor
    if mv_items:
        fmv             = _get_font_mv()
        ascent, descent = fmv.getmetrics()
        ch  = ascent + descent                 # výška kurzoru = plná výška textu
        cw  = fmv.getbbox("M")[2]             # šířka kurzoru = šířka "M"
        for text, x, y, cursor_visible in mv_items:
            py = y - ascent
            # shadow
            for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                draw.text((x + dx, py + dy), text, font=fmv, fill=(0, 0, 0))
            draw.text((x, py), text, font=fmv, fill=(255, 255, 255))
            # blokový kurzor – 2 znaky vpředu, bliká
            if cursor_visible:
                tw = fmv.getbbox(text)[2] if text else 0
                cx0 = x + tw + cw          # přeskočí jedno "prázdné" místo
                draw.rectangle([cx0, py, cx0 + cw, py + ch], fill=(255, 255, 255))

    img[:, :, :] = np.array(pil)[:, :, ::-1]
    _text_queue = [(i, t, x, y, s, c) for (i, t, x, y, s, c) in _text_queue if i != tid]
    _mv_queue   = [(i, t, x, y, v)    for (i, t, x, y, v)    in _mv_queue    if i != tid]


def retarget_text(old_img: np.ndarray, new_img: np.ndarray) -> None:
    """Přesměruje čekající texty z old_img na new_img (bez flush)."""
    global _text_queue, _mv_queue
    oid, nid = id(old_img), id(new_img)
    _text_queue = [(nid if i == oid else i, t, x, y, s, c)
                   for (i, t, x, y, s, c) in _text_queue]
    _mv_queue   = [(nid if i == oid else i, t, x, y, v)
                   for (i, t, x, y, v) in _mv_queue]


def draw_terminal_text(img: np.ndarray, tick: float) -> None:
    """Typewriter zprávy ve středu dolní třetiny obrazu s blokovým kurzorem."""
    global _tw, _tw_triggered
    h, w = img.shape[:2]

    # ---- state machine ------------------------------------------------
    if _tw["state"] == "idle":
        if _tw_triggered and tick >= _tw["cooldown_until"]:
            _tw["msg"]        = (_tw["msg"] + 1) % len(TERMINAL_MESSAGES)
            _tw["chars"]      = 0
            _tw["last_tick"]  = tick
            _tw["state"]      = "typing"
            _tw_triggered     = False
        else:
            return   # nic nezobrazujeme

    elif _tw["state"] == "typing":
        msg = TERMINAL_MESSAGES[_tw["msg"]]
        if _tw["chars"] < len(msg):
            if tick - _tw["last_tick"] >= _TW_CHAR_DELAY:
                _tw["chars"]    += 1
                _tw["last_tick"] = tick
        else:
            # text dopsán → čekej 2 s pak smaž
            _tw["state"]      = "pause"
            _tw["pause_until"] = tick + _TW_ERASE_WAIT

    elif _tw["state"] == "pause":
        if tick >= _tw["pause_until"]:
            _tw["state"]          = "idle"
            _tw["chars"]          = 0
            _tw["cooldown_until"] = tick + random.uniform(5.0, 10.0)
            return   # text smazán, nic nevykreslujeme

    # ---- vykreslení ---------------------------------------------------
    msg   = TERMINAL_MESSAGES[_tw["msg"]]
    typed = msg[:_tw["chars"]]

    fmv = _get_font_mv()
    tw  = fmv.getbbox(typed)[2] if typed else 0
    cw  = fmv.getbbox("M")[2]
    total_w = tw + 2 * cw               # text + mezera + kurzor
    x = (w - total_w) // 2
    y = h * 5 // 6                      # střed dolní třetiny (baseline)

    # kurzor bliká 2× za sekundu; v pause fázi nezobrazujeme
    cursor_visible = (_tw["state"] == "typing") and (int(tick * 2) % 2 == 0)

    _mv_queue.append((id(img), typed, x, y, cursor_visible))

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

# LUT tabulky pro červený filtr – předpočítány jednou při importu
_lut_b = np.clip(np.arange(256, dtype=np.float32) * 0.08,       0, 255).astype(np.uint8)
_lut_g = np.clip(np.arange(256, dtype=np.float32) * 0.25,       0, 255).astype(np.uint8)
_lut_r = np.clip(np.arange(256, dtype=np.float32) * 1.3 + 20,   0, 255).astype(np.uint8)


def apply_red_filter(frame: np.ndarray) -> np.ndarray:
    """Boost R, potlač G a B – LUT (bez float32 konverzí)."""
    out = frame.copy()
    out[:, :, 0] = cv2.LUT(frame[:, :, 0], _lut_b)
    out[:, :, 1] = cv2.LUT(frame[:, :, 1], _lut_g)
    out[:, :, 2] = cv2.LUT(frame[:, :, 2], _lut_r)
    return out


def apply_scanlines(frame: np.ndarray) -> np.ndarray:
    """Každý 3. řádek ztmavit na 55 % – CRT scanline efekt."""
    out = frame.copy()
    out[::3] = (out[::3] * 0.55).astype(np.uint8)
    return out


_vignette_cache: dict = {}

def apply_vignette(frame: np.ndarray) -> np.ndarray:
    """Gaussovský vignette – maska se počítá jednou a cachuje."""
    h, w = frame.shape[:2]
    if (h, w) not in _vignette_cache:
        gx = cv2.getGaussianKernel(w, w * 0.55)
        gy = cv2.getGaussianKernel(h, h * 0.55)
        kernel = gy * gx.T
        mask = kernel / kernel.max()
        _vignette_cache[(h, w)] = np.clip(mask + 0.45, 0.0, 1.0).astype(np.float32)
    v = _vignette_cache[(h, w)]
    return np.clip(frame.astype(np.float32) * v[:, :, np.newaxis], 0, 255).astype(np.uint8)


# Pool předgenerovaných šumových polí – inicializuje se jednou v main()
_noise_pool: list = []


def _init_noise_pool(h: int, w: int, n: int = 30) -> None:
    """Předgeneruje n šumových polí pro add_noise."""
    global _noise_pool
    _noise_pool = [
        np.random.randint(-6, 7, (h, w, 3), dtype=np.int8)
        for _ in range(n)
    ]


def add_noise(frame: np.ndarray, frame_no: int) -> np.ndarray:
    """Per-pixel šum ±6 – z předgenerovaného poolu (bez alokace za frame)."""
    noise = _noise_pool[frame_no % len(_noise_pool)]
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pomocné kreslení
# ---------------------------------------------------------------------------

def put_text_outlined(img, text, pos, font_scale=FONT_SM, color=RED_TEXT, thickness=1):
    """Zařadí text do fronty – vykreslí se hromadně přes flush_text()."""
    _text_queue.append((id(img), str(text), int(pos[0]), int(pos[1]), font_scale, color))


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
    """Skenovací mřížka a sweep linka – kreslí přímo bez alokace extra bufferu."""
    h, w = img.shape[:2]
    spacing = 32
    gc = (38, 38, 38)
    for x in range(0, w + spacing, spacing):
        cv2.line(img, (x, 0), (x, h), gc, 1)
    for y in range(0, h + spacing, spacing):
        cv2.line(img, (0, y), (w, y), gc, 1)

    sweep_y = int((tick % 3.5) / 3.5 * h)
    cv2.line(img, (0, sweep_y), (w, sweep_y), (130, 130, 130), 1)
    for off in range(1, 10):
        sy = sweep_y - off
        if sy >= 0:
            v = int(90 * (1 - off / 10))
            cv2.line(img, (0, sy), (w, sy), (v, v, v), 1)


_GRID_COLS   = 12
_GRID_ROWS   = 12
_GRID_CELL_W = 22   # px na buňku (šířka)
_GRID_CELL_H = 15   # px na buňku (výška)
_GRID_W      = _GRID_COLS * _GRID_CELL_W   # 264 px
_GRID_H      = _GRID_ROWS * _GRID_CELL_H   # 180 px


def _draw_grid_hud(img: np.ndarray, tick: float, px: int, py: int) -> None:
    """12×12 mřížka s plynule pohyblivými vlasovými osami X a Y."""
    cols, rows = _GRID_COLS, _GRID_ROWS
    cw, ch     = _GRID_CELL_W, _GRID_CELL_H
    gw = cols * cw
    gh = rows * ch

    dim    = (255, 255, 255)
    bright = (255, 255, 255)

    # dim mřížka
    for c in range(cols + 1):
        x = px + c * cw
        cv2.line(img, (x, py), (x, py + gh), dim, 1, cv2.LINE_AA)
    for r in range(rows + 1):
        y = py + r * ch
        cv2.line(img, (px, y), (px + gw, y), dim, 1, cv2.LINE_AA)

    # plynulý pohyb os – dvě nezávislé sinusoidy s nesouměřitelnými frekvencemi
    # osa Y (svislá linka) se pohybuje po ose X
    ax = px + int((gw - 1) * (0.5 + 0.5 * np.sin(tick * 0.53)))
    # osa X (vodorovná linka) se pohybuje po ose Y
    ay = py + int((gh - 1) * (0.5 + 0.5 * np.sin(tick * 0.37 + 1.2)))

    cv2.line(img, (ax, py), (ax, py + gh), bright, 1, cv2.LINE_AA)
    cv2.line(img, (px, ay), (px + gw, ay), bright, 1, cv2.LINE_AA)


def draw_right_hud(img: np.ndarray, tick: float) -> None:
    """Pravý HUD panel – střídá 6 módů každých 6 s."""
    h, w = img.shape[:2]
    font_scale = 0.65
    line_h     = 22

    mode = int(tick / _RIGHT_HUD_SWITCH) % 6  # 0=scan mode+grid, 1=acquire, 2=priority, 3=data dump, 4=env scan, 5=compass

    # ------------------------------------------------------------------ #
    # Mód 5 – kompasová růžice + TARGET FIELD                            #
    # ------------------------------------------------------------------ #
    if mode == 5:
        panel_w = 200
        panel_h = 220
        pos_seed = int(tick / 10)
        pos_rng  = random.Random(pos_seed + 999)
        x_min = w * 2 // 3
        x_max = max(x_min + 1, w - panel_w - 10)
        y_min = 35
        y_max = max(y_min + 1, h - panel_h - 35)
        px = pos_rng.randint(x_min, x_max)
        py = pos_rng.randint(y_min, y_max)

        r  = 58
        cx = px + panel_w // 2
        cy = py + r + 12
        _draw_compass_rose_at(img, cx, cy, r)

        # TARGET FIELD text pod růžicí
        rng = random.Random(int(tick / 4.0))
        tf_y = cy + r + 28
        put_text_outlined(img, "TARGET FIELD:", (px, tf_y), 0.40, RED_TEXT)
        for i in range(3):
            row = "".join(str(rng.randint(0, 9)) for _ in range(12))
            put_text_outlined(img, row, (px, tf_y + (i + 1) * 18), 0.38, RED_TEXT)
        return

    if mode == 0:
        scan_num = (int(tick / 2.3) * 137 + 3958) % 99999
        rng = random.Random(int(tick / 1.8))
        lines = [f"SCAN MODE {scan_num:05d}", "." * 16]
        for tag in _RIGHT_HUD_TAGS[:5]:
            a = rng.randint(10000, 99999)
            b = rng.randint(1000,  9999)
            c = rng.randint(0,     99)
            lines.append(f"{tag}  {a} {b} {c:02d}")

    elif mode == 1:
        acq_idx = int(tick / 4.5) % len(SCAN_ACQUIRES)
        rng = random.Random(int(tick / 2.5))
        lines = [SCAN_ACQUIRES[acq_idx], "*" * 14]
        for tag in _RIGHT_HUD_TAGS:
            n1 = rng.randint(100, 999)
            n2 = rng.randint(10000, 99999)
            lines.append(f"{n1} {tag}  {n2}")

    elif mode == 2:
        priority = f"PRIORITY {int(tick * 1237) % 9999999:07d}D"
        rng = random.Random(int(tick / 3.3))
        lines = [priority, "-" * 17]
        for _ in range(6):
            a = rng.randint(100000, 999999)
            b = rng.randint(100, 999)
            c = rng.randint(10, 99)
            lines.append(f"{a} {b} {c}")
        assess = "STATUS: CONFIRMED" if rng.random() > 0.4 else "STATUS: PENDING"
        lines.append(assess)

    elif mode == 3:
        rng = random.Random(int(tick / 1.1))
        lines = ["DATA DUMP:", "*" * 10]
        for _ in range(4):
            a = rng.randint(100000, 999999)
            b = rng.randint(100, 999)
            c = rng.randint(10, 99)
            lines.append(f"{a} {b} {c}")
        lines.append("")
        for _ in range(3):
            a = rng.randint(100000, 999999)
            b = rng.randint(100, 999)
            lines.append(f"{a} {b}")

    else:  # mode == 4
        rng = random.Random(int(tick / 2.7))
        temp = round(rng.uniform(36.1, 37.9), 1)
        dist = rng.randint(1, 999)
        lines = [
            "ENVIRONMENT SCAN:",
            f"TEMP: {temp}C",
            f"RANGE: {dist:03d}m",
            "MOTION: DETECTED",
            "*" * 14,
        ]
        for _ in range(5):
            a = rng.randint(100000, 999999)
            b = rng.randint(100, 999)
            c = rng.randint(10, 99)
            lines.append(f"{a} {b} {c}")

    panel_h = len(lines) * line_h + 10
    panel_w = 280

    # pozice se mění každých 10 s v rámci pravé třetiny obrazu
    pos_seed = int(tick / 10)
    pos_rng  = random.Random(pos_seed + 999)   # jiný seed než levý panel
    x_min = w * 2 // 3
    x_max = max(x_min + 1, w - panel_w - 10)
    y_min = 35
    y_max = max(y_min + 1, h - panel_h - 35)
    px = pos_rng.randint(x_min, x_max)
    py = pos_rng.randint(y_min, y_max)

    # typewriter efekt – řádky se zobrazují postupně od začátku každého módu
    mode_start = int(tick / _RIGHT_HUD_SWITCH) * _RIGHT_HUD_SWITCH
    elapsed    = tick - mode_start
    visible    = max(1, int(elapsed / 0.18))   # nový řádek každých 0.18 s

    for i, line in enumerate(lines[:visible]):
        put_text_outlined(img, line, (px, py + (i + 1) * line_h), font_scale, RED_TEXT)

    # mřížka se zobrazí pouze v SCAN MODE (mode 0) – pod textovými řádky
    if mode == 0:
        grid_top = py + (len(lines) + 1) * line_h + 6
        _draw_grid_hud(img, tick, px, grid_top)


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


def draw_camera_viewfinder(img: np.ndarray, tick: float) -> None:
    """Split-image rangefinder hledáček pohybující se po obrazovce – T2 styl."""
    h, w = img.shape[:2]

    # Lissajousův pohyb – dvě různé frekvence → neperiodická dráha
    margin_x = int(w * 0.28)
    margin_y = int(h * 0.22)
    cx = int(w / 2 + margin_x * np.sin(tick * 0.31))
    cy = int(h / 2 + margin_y * np.sin(tick * 0.47 + 1.1))

    r_outer = int(min(w, h) * 0.080)  # ~8 % kratší strany
    r_inner = int(r_outer * 0.38)
    col     = RED_TEXT
    col_dim = RED_DIM

    # ---- pulzující rádius (potřebujeme ho brzy) ----------------------
    pulse = int(r_outer * 1.30 + r_outer * 0.06 * np.sin(tick * 3.2))

    # ---- ROI pro alpha blend – jen oblast kolem kruhu ----------------
    rx1 = max(0, cx - pulse - 2)
    ry1 = max(0, cy - pulse - 2)
    rx2 = min(w,  cx + pulse + 2)
    ry2 = min(h,  cy + pulse + 2)
    roi       = img[ry1:ry2, rx1:rx2]
    layer_roi = roi.copy()

    # bílý fill a pulzující kruh – jen do ROI (souřadnice relativní)
    cv2.circle(layer_roi, (cx - rx1, cy - ry1), r_outer, (255, 255, 255), -1)
    cv2.circle(layer_roi, (cx - rx1, cy - ry1), pulse,   col_dim, 1, cv2.LINE_AA)
    cv2.addWeighted(layer_roi, 0.25, roi, 0.75, 0, roi)

    # ---- vnitřní kruh -------------------------------------------------
    cv2.circle(img, (cx, cy), r_inner, (60, 60, 60), 1, cv2.LINE_AA)

    # ---- tenký vertikální kříž (jen uvnitř vnitřního kruhu) -----------
    cv2.line(img, (cx, cy - r_inner), (cx, cy + r_inner), (80, 80, 80), 1, cv2.LINE_AA)

    # ---- centrální tečka ----------------------------------------------
    cv2.circle(img, (cx, cy), 2, (40, 40, 40), -1, cv2.LINE_AA)

    # ---- vnější kruh s tick marks ------------------------------------
    cv2.circle(img, (cx, cy), r_outer, col, 1, cv2.LINE_AA)
    for deg in range(0, 360, 45):
        a    = np.radians(deg)
        tlen = 9 if deg % 90 == 0 else 5
        xo = int(cx + r_outer          * np.cos(a))
        yo = int(cy + r_outer          * np.sin(a))
        xi = int(cx + (r_outer - tlen) * np.cos(a))
        yi = int(cy + (r_outer - tlen) * np.sin(a))
        cv2.line(img, (xi, yi), (xo, yo), col, 1, cv2.LINE_AA)


def _draw_compass_rose_at(img: np.ndarray, cx: int, cy: int, r: int) -> None:
    """Kompasová růžice T2 stylu – všechny paprsky stejně dlouhé, texty vně."""
    col = RED_TEXT

    # 8 paprsků ze středu na okraj – všechny stejné délky, bez kruhů
    for deg, label, loff in [
        (  0, "N",  (-4, -10)),
        ( 45, "NE", (  3,  -6)),
        ( 90, "E",  ( 10,   4)),
        (135, "SE", (  3,   12)),
        (180, "S",  (-4,   14)),
        (225, "SW", (-18,  12)),
        (270, "W",  (-18,   4)),
        (315, "NW", (-18,  -6)),
    ]:
        a  = np.radians(deg - 90)
        xo = int(cx + r * np.cos(a))
        yo = int(cy + r * np.sin(a))
        cv2.line(img, (cx, cy), (xo, yo), col, 1, cv2.LINE_AA)
        xl = int(cx + (r + 11) * np.cos(a)) + loff[0]
        yl = int(cy + (r + 11) * np.sin(a)) + loff[1]
        put_text_outlined(img, label, (xl, yl), 0.32, col)


_LEFT_HUD_TAGS     = ["VEHI", "SIZE", "TSPD", "HPWR", "CODE", "RNGE", "CAPC", "MAXI", "TORQ"]
_LEFT_HUD_TAGS_EXT = ["VEHI", "SIZE", "TSPD", "HPWR", "CODE", "RNGE", "MAXI", "SUSP", "TORQ", "WGHT", "TANK"]
_LEFT_HUD_SWITCH   = 5.0   # sekundy mezi přepnutím módu

_RIGHT_HUD_TAGS    = ["VEHI", "MTRC", "TRCT", "LOCO", "DRVR", "FUEL", "SPNS"]
_RIGHT_HUD_SWITCH  = 6.0   # sekundy mezi přepnutím módu (jiný rytmus než levý)


def draw_left_hud(img: np.ndarray, tick: float) -> None:
    """Levý HUD panel – střídá SCAN LEVELS / CRITERIA / ANALYSIS:MATCH každých 5 s."""
    h, w = img.shape[:2]
    font_scale = 0.65
    line_h     = 22

    mode = int(tick / _LEFT_HUD_SWITCH) % 5   # 0=scan levels, 1=criteria, 2=analysis, 3=data dump, 4=visual assessment

    if mode == 0:
        rng = random.Random(int(tick / 0.9))
        lines = ["SCAN LEVELS:", "." * 16]
        for _ in range(7):
            a = rng.randint(100000, 999999)
            b = rng.randint(100, 999)
            c = rng.randint(10, 99)
            lines.append(f"{a} {b} {c}")
        header = "SCAN"

    elif mode == 1:
        rng = random.Random(int(tick / 3.0))
        lines = ["CRITERIA:", "*" * 9]
        for tag in _LEFT_HUD_TAGS:
            n1 = rng.randint(100, 999)
            n2 = rng.randint(10000, 99999)
            lines.append(f"{n1} {tag}  {n2}")
        header = "CRIT"

    elif mode == 2:
        rng = random.Random(int(tick / 3.0))
        lines = ["ANALYSIS:  MATCH:", "*" * 15]
        for tag in _LEFT_HUD_TAGS_EXT:
            n1 = rng.randint(100, 999)
            n2 = rng.randint(10000, 99999)
            lines.append(f"{n1} {tag}  {n2}")
        assess = "ASSESS: SUITABLE" if rng.random() > 0.3 else "ASSESS: POSSIBLE"
        lines.append(assess)
        header = "ANAL"

    elif mode == 3:
        rng = random.Random(int(tick / 1.1))
        lines = ["*" * 15]
        # skupina 1
        for _ in range(5):
            a = rng.randint(100000, 999999)
            b = rng.randint(100, 999)
            c = rng.randint(10, 99)
            lines.append(f"{a} {b} {c}")
        lines.append("")  # prázdný řádek
        # skupina 2
        for _ in range(3):
            a = rng.randint(100000, 999999)
            b = rng.randint(100, 999)
            c = rng.randint(10, 99)
            lines.append(f"{a} {b} {c}")
        header = "****"

    else:
        rng = random.Random(int(tick / 2.0))
        prob = round(rng.uniform(0.85, 0.99), 2)
        level = rng.randint(10000, 99999)
        lines = [
            "VISUAL ASSESSMENT:",
            f"LEVEL {level} SCAN",
            "PROBABILITY OF",
            "CAMOUFLAGE",
            f"ACQUISITION: {prob}",
            "BEARING",
            "*" * 13,
        ]
        for _ in range(5):
            a = rng.randint(100000, 999999)
            b = rng.randint(100, 999)
            c = rng.randint(10, 99)
            lines.append(f"{a} {b} {c}")
        header = "VISU"

    panel_h = len(lines) * line_h + 10
    panel_w = 270

    pos_seed = int(tick / 10)
    pos_rng  = random.Random(pos_seed)
    x_min = 10
    x_max = max(x_min + 1, w // 3 - panel_w)
    y_min = 35
    y_max = max(y_min + 1, h - panel_h - 35)
    px = pos_rng.randint(x_min, x_max)
    py = pos_rng.randint(y_min, y_max)

    # typewriter efekt – řádky se zobrazují postupně od začátku každého módu
    mode_start = int(tick / _LEFT_HUD_SWITCH) * _LEFT_HUD_SWITCH
    elapsed    = tick - mode_start
    visible    = max(1, int(elapsed / 0.18))   # nový řádek každých 0.18 s

    for i, line in enumerate(lines[:visible]):
        put_text_outlined(img, line, (px, py + (i + 1) * line_h), font_scale, RED_TEXT)


def draw_search_criteria(img: np.ndarray, tick: float) -> None:
    """SEARCH CRITERIA / MATCH MODE / ALL LEVELS OPERATIVE – vlevo dole."""
    h, w = img.shape[:2]
    match_num = (int(tick / 3.1) * 47 + 5498) % 99999
    lines = [
        "SEARCH CRITERIA",
        f"MATCH MODE {match_num:05d}",
        "ALL LEVELS OPERATIVE",
    ]
    y = h - len(lines) * 13 - 28
    for line in lines:
        color = RED_TEXT if "SEARCH" in line else RED_DIM
        put_text_outlined(img, line, (10, y), 0.32, color)
        y += 13


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
    tw = get_text_width(now_str, FONT_SM)
    put_text_outlined(img, now_str, (w - tw - 10, 22), FONT_SM, RED_TEXT)

    # dolní řádek
    fps_str     = f"FPS: {fps:4.1f}"
    center_str  = "NEURAL NET PROCESSOR :: ACTIVE"
    targets_str = f"TARGETS: {det_count}"

    put_text_outlined(img, fps_str,     (10, h - 10), FONT_SM, RED_TEXT)
    ctw = get_text_width(center_str, FONT_SM)
    put_text_outlined(img, center_str,  ((w - ctw) // 2, h - 10), FONT_SM, RED_TEXT)
    ttw = get_text_width(targets_str, FONT_SM)
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

    # čárkovaná linie do středu obrazu (jen pro osoby)
    if is_person:
        _dashed_line(img, (cx, cy), (w // 2, h // 2), (80, 80, 80), gap=10)

    # blikající kroužek ve středu pro osoby
    if is_person and int(tick * 2) % 2 == 0:
        cv2.circle(img, (cx, cy), 6, RED_TEXT, 2, cv2.LINE_AA)

    # info box (nad/pod bboxem)
    display_name = "HUMAN" if is_person else label.upper()
    conf_str     = f"CONF: {int(conf * 100)}%"

    phrase = ""
    height_str = ""
    if is_person:
        phrase_idx = int(tick / 2) % len(PERSON_PHRASES)
        phrase     = PERSON_PHRASES[(phrase_idx + idx) % len(PERSON_PHRASES)]
        est_h      = int((y2 - y1) / h * 175)
        height_str = f"HEIGHT EST: {est_h}cm"

    lines = [display_name, conf_str]
    if is_person:
        lines.append(phrase)
        lines.append(height_str)

    line_h = 16
    box_h  = len(lines) * line_h + 6
    box_w  = max(get_text_width(l, FONT_SM) for l in lines) + 10

    if y1 - box_h - 4 >= 0:
        bx1, by1 = x1, y1 - box_h - 4
    else:
        bx1, by1 = x1, y2 + 4

    cv2.rectangle(img, (bx1, by1), (bx1 + box_w, by1 + box_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        put_text_outlined(img, line, (bx1 + 4, by1 + (i + 1) * line_h), FONT_SM, RED_TEXT)

    # ------------------------------------------------------------------
    # CRITERIA panel – tělesné míry osoby (vpravo od bboxu nebo na pravé straně)
    # ------------------------------------------------------------------
    if is_person:
        rng2 = random.Random(idx * 1337 + int(tick / 6))
        meas = [
            ("HGHT", rng2.randint(100000, 999999), rng2.randint(100, 999)),
            ("WGHT", rng2.randint(100000, 999999), rng2.randint(100, 999)),
            ("SHLD", rng2.randint(100000, 999999), rng2.randint(100, 999)),
            ("BACK", rng2.randint(100000, 999999), rng2.randint(100, 999)),
            ("INSM", rng2.randint(100000, 999999), rng2.randint(100, 999)),
            ("SLEV", rng2.randint(100000, 999999), rng2.randint(100, 999)),
            ("CHST", rng2.randint(100000, 999999), rng2.randint(100, 999)),
            ("COLR", rng2.randint(100000, 999999), rng2.randint(100, 999)),
            ("BICP", rng2.randint(100000, 999999), rng2.randint(100, 999)),
            ("TRIC", rng2.randint(100000, 999999), rng2.randint(100, 999)),
        ]
        panel_w = 168
        # umístění: vlevo od bboxu pokud je místo, jinak vpravo, pokud tam vychází mimo obraz – vlevo
        if x2 + panel_w + 6 <= w:
            px = x2 + 6
        elif x1 - panel_w - 6 >= 0:
            px = x1 - panel_w - 6
        else:
            px = max(w - panel_w - 6, 0)
        py = max(y1, 35)

        put_text_outlined(img, "CRITERIA:", (px, py),      0.33, RED_TEXT)
        put_text_outlined(img, "." * 14,    (px, py + 12), 0.33, RED_DIM)
        for j, (lbl, v1, v2) in enumerate(meas):
            put_text_outlined(img, f"{lbl}  {v1} {v2}",
                              (px, py + 24 + j * 12), 0.30, RED_DIM)
        # ASSESS
        assess = "ASSESS: SUITABLE" if conf >= 0.70 else "ASSESS: POSSIBLE"
        put_text_outlined(img, assess,
                          (px, py + 24 + len(meas) * 12 + 5), 0.30, RED_TEXT)

        # PROBABILITY (vlevo od/pod bboxem)
        prob_str = f"PROBABILITY .{int(conf * 100):02d}"
        put_text_outlined(img, prob_str, (x1, y2 + 14), 0.32, RED_DIM)

    # ------------------------------------------------------------------
    # VISUAL: MODEL XXX – pro ne-osoby
    # ------------------------------------------------------------------
    else:
        model_num = random.Random(idx * 777).randint(100, 999)
        put_text_outlined(img, "VISUAL:",          (x1, y2 + 14), 0.32, RED_TEXT)
        put_text_outlined(img, f"MODEL {model_num}", (x1, y2 + 27), 0.32, RED_DIM)

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

    # inicializace zvuku – defaultně vypnutý
    sound = SoundEngine()
    sound.enabled = False
    sound_on = False

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

    fullscreen = True
    debug_mode = False
    WIN_NAME   = "Terminator Vision"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


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

    # inicializuj noise pool po prvním přečtení rozlišení
    ret0, frame0 = cap.read()
    if not ret0:
        print("[ERR] Nelze číst první frame.", file=sys.stderr)
        sys.exit(1)
    _init_noise_pool(frame0.shape[0], frame0.shape[1])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # vrať se na začátek (pro video i kameru)

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

            # resample mesh + Delaunay triangulace každých 0.5 s
            if tick - last_mesh_tick > 0.5:
                last_mesh_tick = tick
                new_cache = {}
                for i, (det, mbin) in enumerate(zip(cached_detections, cached_masks_bin)):
                    if mbin is None:
                        continue
                    lbl      = det["label"]
                    bbox_det = (det["x1"], det["y1"], det["x2"], det["y2"])
                    if lbl == "person":
                        zone_idx, zone_start = mesh_zone_state.get(i, (0, tick))
                        if tick - zone_start > ZONE_DURATION:
                            zone_idx   = (zone_idx + 1) % len(BODY_ZONES)
                            zone_start = tick
                        mesh_zone_state[i] = (zone_idx, zone_start)
                        zbbox = get_zone_bbox(bbox_det, BODY_ZONES[zone_idx])
                        pts   = sample_points_in_zone(mbin, zbbox, n_points=40)
                        tris  = _compute_triangles(pts, mbin, bbox_det, zbbox)
                        new_cache[i] = (tris, zbbox, BODY_ZONES[zone_idx][0])
                    else:
                        pts  = sample_points_in_mask(mbin, 25)
                        tris = _compute_triangles(pts, mbin, bbox_det)
                        new_cache[i] = (tris, None, "")
                mesh_points_cache = new_cache

        # ---------------------------------------------------------------
        # Aplikuj obrazové filtry
        # ---------------------------------------------------------------
        filtered = apply_red_filter(frame)
        filtered = apply_vignette(filtered)
        filtered = add_noise(filtered, frame_no)

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

            # mesh overlay – trojúhelníky jsou předpočítané
            entry = mesh_points_cache.get(i)
            if entry:
                tris, zbbox, zone_name = entry
                color = (200, 200, 200) if lbl == "person" else (140, 140, 140)
                _draw_triangles(overlay, tris, color)
                if zone_name and zbbox:
                    put_text_outlined(overlay, f"SCAN: {zone_name}",
                                      (x2 + 4, zbbox[1] + 10), 0.30, RED_DIM)

        # HUD per-detekce (na overlay)
        for det in cached_detections:
            draw_detection(overlay, det, frame.shape, tick, debug=debug_mode)

        # zkombinuj overlay s filtrovaným framem (85/15)
        out = cv2.addWeighted(overlay, 0.85, filtered, 0.15, 0)

        # přesměruj čekající texty z overlay na out (bez další konverze)
        retarget_text(overlay, out)

        # globální HUD přímo na výsledný obraz
        fps_frame_count += 1
        if tick - fps_timer >= 0.5:
            fps = fps_frame_count / (tick - fps_timer)
            fps_frame_count = 0
            fps_timer = tick

        draw_global_hud(out, fps, len(cached_detections), frame_no)
        draw_left_hud(out, tick)
        draw_right_hud(out, tick)
        draw_search_criteria(out, tick)

        draw_camera_viewfinder(out, tick)
        draw_terminal_text(out, tick)

        # jediný flush_text za frame – PIL konverze proběhne jen jednou
        flush_text(out)

        # ---------------------------------------------------------------
        # Zvukové události
        # ---------------------------------------------------------------
        det_count = len(cached_detections)
        if det_count > prev_det_count:
            # nový cíl přibyl
            sound.play_target()
            terminal_trigger()
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
            sound_on = not sound_on
            sound.enabled = sound_on
            if sound_on:
                sound.start_ambient()
                print("[INFO] Zvuk: ZAP")
            else:
                sound.stop()
                print("[INFO] Zvuk: VYP")
        elif key == ord('p'):
            ts  = datetime.now().strftime("%H%M%S")
            fn  = f"terminator_{ts}.png"
            cv2.imwrite(fn, out)
            print(f"[INFO] Screenshot uložen: {fn}")
        elif key == ord('f'):
            fullscreen = not fullscreen
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

def _compute_triangles(points: list, mask_bin: np.ndarray,
                       bbox: tuple, zone_bbox: tuple = None) -> list:
    """Spočítá a filtruje Delaunay trojúhelníky – voláno jen při resample, ne každý frame."""
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = zone_bbox if zone_bbox else (x1, y1, x2, y2)
    rw = max(rx2 - rx1, 4)
    rh = max(ry2 - ry1, 4)
    if len(points) < 4:
        return []

    corners = [
        (rx1, ry1), (rx2, ry1), (rx1, ry2), (rx2, ry2),
        ((rx1 + rx2) // 2, ry1), ((rx1 + rx2) // 2, ry2),
        (rx1, (ry1 + ry2) // 2), (rx2, (ry1 + ry2) // 2),
    ]
    subdiv = cv2.Subdiv2D((rx1, ry1, rw, rh))
    for p in list(points) + corners:
        px, py = int(p[0]), int(p[1])
        if rx1 <= px <= rx2 and ry1 <= py <= ry2:
            try:
                subdiv.insert((float(px), float(py)))
            except cv2.error:
                pass
    try:
        raw = subdiv.getTriangleList().astype(np.int32)
    except cv2.error:
        return []

    mh, mw = mask_bin.shape[:2]
    result = []
    for t in raw:
        pt1, pt2, pt3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])
        tcx = (pt1[0] + pt2[0] + pt3[0]) // 3
        tcy = (pt1[1] + pt2[1] + pt3[1]) // 3
        if not (0 <= tcy < mh and 0 <= tcx < mw and mask_bin[tcy, tcx] > 0):
            continue
        if zone_bbox and not (rx1 <= tcx <= rx2 and ry1 <= tcy <= ry2):
            continue
        result.append((pt1, pt2, pt3))
    return result


def _draw_triangles(frame: np.ndarray, triangles: list, color: tuple) -> None:
    """Nakreslí předpočítané trojúhelníky – žádná triangulace, jen cv2.line."""
    for pt1, pt2, pt3 in triangles:
        cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.line(frame, pt2, pt3, color, 1, cv2.LINE_AA)
        cv2.line(frame, pt3, pt1, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
