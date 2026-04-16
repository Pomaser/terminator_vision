# Terminator Vision – dokumentace projektu

## Cíl
Python aplikace `terminator_vision.py` simulující červené HUD vidění Terminátora T-800 z filmu T2. Detekuje osoby a objekty přes YOLOv8-seg a zobrazuje je s T2-style overlay.

---

## Závislosti

```
pip install opencv-python ultralytics numpy pillow pygame
```

- Model `yolov8n-seg.pt` se stáhne automaticky při prvním spuštění.
- Font HUD: `font/Helvetica73-Extended Bold.ttf`
- Font terminál: `font/modern-vision.ttf` (100 px, typewriter dolní třetina)
- Zvuky (volitelné): `sounds/{startup,ambient,target,new_target,scan,alert}.wav`

---

## Spuštění

```bash
python terminator_vision.py                                  # výchozí kamera
python terminator_vision.py --camera 1                       # jiný index
python terminator_vision.py --width 1280 --height 720
python terminator_vision.py --source sample/Terminator.mkv  # video soubor
```

---

## Ovládání

| Klávesa | Akce |
|---------|------|
| `q` | Ukončit aplikaci |
| `s` | Screenshot → `terminator_HHMMSS.png` |
| `f` | Přepnout fullscreen |
| `d` | Debug info (raw YOLO boxy) |
| `S` | Zapnout/vypnout zvuk (default: vypnuto) |

---

## Barevné schéma

Veškerý overlay je **bílý** (ne červený):
- `RED_TEXT = (255, 255, 255)` — hlavní bílá
- `RED_DIM  = (160, 160, 160)` — tlumená šedá

---

## Obrazové filtry (pořadí na každý frame)

1. **Červený filtr** – LUT tabulky (`_lut_b/g/r`), bez float32 konverzí
2. **Vignette** – Gaussovský gradient, cachováno per rozlišení
3. **Šum** – pool 30 předgenerovaných polí, `_init_noise_pool()` při startu

> Scanlines a globální mřížka přes celý obraz byly odstraněny.

---

## Detekce objektů

- `YOLO("yolov8n-seg.pt")`, `conf=0.40`
- Každý **2. frame** se spouští inference, výsledky se kešují
- Extrahuje: `boxes.xyxy`, `boxes.conf`, `boxes.cls`, `masks.xy`, `masks.data`

---

## PIL textový systém

Všechny texty se renderují přes Pillow, **jedinou PIL konverzí za frame**.

### Dvě fronty:
- `_text_queue` – Helvetica 73 Extended Bold (veškerý HUD text)
- `_mv_queue`   – Modern Vision 100 px (terminálový typewriter)

### Klíčové funkce:
```python
put_text_outlined(img, text, pos, font_scale, color)  # přidá do _text_queue
flush_text(img)       # jediný call za frame – flushuje obě fronty najednou
retarget_text(overlay, out)  # přesměruje fronty po addWeighted blend
```

> `flush_text(out)` se volá **jednou** na úplném konci framové smyčky.
> Před tím se volá `retarget_text(overlay, out)` aby texty z overlay fáze
> skončily ve správném cílovém bufferu.

---

## Segmentační obrys (`draw_segmentation_outline`)

- Polygon z `masks.xy[i]` → `cv2.polylines`
- `person`: bílý obrys (255,255,255), tloušťka 2, glow halo (80,80,80) tloušťka 5
- ostatní: šedý obrys (180,180,180), glow (60,60,60)

---

## Mesh overlay – Delaunay triangulace (`draw_mesh_overlay`)

- Binární maska z `masks.data[i]` → vzorkování bodů uvnitř masky
- `cv2.Subdiv2D` triangulace, filtr středem trojúhelníku uvnitř masky
- **Částečné skenování těla** – `BODY_ZONES` cykluje po 0.7 s:
  `HEAD, TORSO, LEFT ARM, RIGHT ARM, LEFT LEG, RIGHT LEG`
- Trojúhelníky se **cachují** (přepočítávají jen každých 0.5 s)
- `person`: 150 bodů, (200,200,200); ostatní: 80 bodů, (140,140,140)

---

## HUD overlay – globální (`draw_global_hud`)

- Rohové závorky ve 4 rozích obrazu
- Nahoře vlevo: `CYBERDYNE SYSTEMS MODEL 101`
- Nahoře vpravo: čas `HH:MM:SS`
- Dole vlevo: `FPS: XX.X`
- Dole uprostřed: `NEURAL NET PROCESSOR :: ACTIVE`
- Dole vpravo: `TARGETS: N`

---

## HUD overlay – detekce (`draw_detection`)

1. Rohové závorky kolem bounding boxu
2. Čárkovaná linie do středu obrazu (pouze osoby, ~30 % alpha)
3. Info box: třída, `CONF: XX%`, fráze pro osoby, `HEIGHT EST: XXXcm`
4. Blikající kroužek ve středu bbox (osoby, každých 0.5 s)

---

## Levý HUD panel (`draw_left_hud`)

- Pozice: levá třetina, mění se každých 10 s
- Přepíná **5 módů** po **5 sekundách**
- Typewriter efekt: nový řádek každých 0.18 s

| Mód | Obsah |
|-----|-------|
| 0 | `SCAN LEVELS:` + skupiny čísel |
| 1 | `CRITERIA:` + tabulka tagů s čísly |
| 2 | `ANALYSIS: MATCH:` + rozšířená tabulka + `ASSESS: SUITABLE/POSSIBLE` |
| 3 | Data dump – dvě skupiny čísel |
| 4 | `VISUAL ASSESSMENT:` + pravděpodobnost, data |

---

## Pravý HUD panel (`draw_right_hud`)

- Pozice: pravá třetina, mění se každých 10 s
- Přepíná **6 módů** po **6 sekundách**
- Typewriter efekt: nový řádek každých 0.18 s

| Mód | Obsah |
|-----|-------|
| 0 | `SCAN MODE` + tabulka tagů + **12×12 mřížka s pohyblivými osami** |
| 1 | `ACQUIRE ...` + tabulka tagů |
| 2 | `PRIORITY XXXXXXXD` + data + `STATUS` |
| 3 | `DATA DUMP:` + skupiny čísel |
| 4 | `ENVIRONMENT SCAN:` + teplota, vzdálenost, motion |
| 5 | Kompasová růžice + `TARGET FIELD:` + 3 × 12 číslic |

### 12×12 mřížka – `_draw_grid_hud` (mód 0)
- 12 × 12 buněk, každá 22×15 px, celkem 264×180 px
- Všechny linky bílé
- Osa X (vodorovná) a Y (svislá) – vlasové bílé čáry, plynulý pohyb po sinusoidách
  s nesouměřitelnými frekvencemi (0.37 Hz a 0.53 Hz)

### Kompasová růžice – `_draw_compass_rose_at` (mód 5)
- 8 paprsků ze středu, popisky N/NE/E/SE/S/SW/W/NW vně, bez kruhů
- Pod růžicí: `TARGET FIELD:` + 3 řady 12místných čísel (mění se každé 4 s)

---

## Terminálový typewriter – `draw_terminal_text`

Font: **Modern Vision 100 px**. Pozice: střed dolní třetiny obrazu (baseline `y = h * 5 // 6`).

### State machine:
| Stav | Popis |
|------|-------|
| `idle` | Nic se nezobrazuje; čeká na `terminal_trigger()` + cooldown |
| `typing` | Vypisuje znak po znaku (0.055 s/znak); kurzor bliká 2 Hz |
| `pause` | 2 s zobrazuje hotový text bez kurzoru; pak přechází do `idle` |

### Cooldown:
Po smazání textu náhodná prodleva **5–10 s** před dalším spuštěním.

### Trigger:
```python
terminal_trigger()   # volá se z main() při det_count > prev_det_count
```

### Blokový kurzor:
- Bílý obdélník, šířka `M`, výška = `ascent + descent`
- Pozice = text width + 2× šířka znaku (jeden prázdný meziprostor)
- Bliká jen ve stavu `typing`

### Seznam zpráv:
```python
TERMINAL_MESSAGES: list[str] = [
    "SCANNING", "MATCH FOUND", "SEARCHING", "TARGET ACQUIRED", ...
]
```
Uživatel může přidávat vlastní položky.

---

## Pohyblivý hledáček (`draw_camera_viewfinder`)

- Lissajousův pohyb (frekvence 0.31 a 0.47 Hz)
- Průměr ~8 % kratší strany obrazu
- Alpha blend pouze přes ROI oblast kruhu (ne celý frame) – výkon

---

## Zvukový engine (`SoundEngine`)

- `pygame.mixer`, zvuky: `startup`, `ambient`, `target`/`new_target`, `scan`, `alert`
- Default: **vypnuto**; klávesa `S` přepíná

---

## Výkonnostní optimalizace

| Oblast | Řešení |
|--------|--------|
| PIL konverze | Jedna za frame (`retarget_text` + jediný `flush_text`) |
| Šum | Pool 30 předgenerovaných polí místo `np.random` za frame |
| Červený filtr | LUT tabulky místo float32 konverzí |
| Viewfinder copy | ROI kopie ~64× menší než celý frame |
| Mesh triangulace | Cache, přepočet jen každých 0.5 s |
| Vignette | Cache per rozlišení |

---

## Struktura funkcí

```
apply_red_filter(frame)            → frame   (LUT)
apply_vignette(frame)              → frame   (cachováno)
add_noise(frame, frame_no)         → frame   (pool)
_init_noise_pool(h, w)             → None    (volat jednou v main)
draw_corner_bracket(img, ...)      → None
draw_segmentation_outline(...)     → None
sample_points_in_mask(...)         → list
sample_points_in_zone(...)         → list
get_zone_bbox(...)                 → tuple
draw_mesh_overlay(...)             → None
_compute_triangles(...)            → list
_draw_triangles(...)               → None
put_text_outlined(img, ...)        → None    (fronta)
flush_text(img)                    → None    (PIL render, 1×/frame)
retarget_text(old, new)            → None
_get_font_mv()                     → FreeTypeFont
terminal_trigger()                 → None    (spustí další zprávu)
draw_terminal_text(img, tick)      → None
draw_global_grid(img, tick)        → None    (sweep linka)
_draw_compass_rose_at(img, ...)    → None
_draw_grid_hud(img, tick, px, py)  → None    (12×12 mřížka)
draw_left_hud(img, tick)           → None
draw_right_hud(img, tick)          → None
draw_center_reticle(img, tick)     → None
draw_camera_viewfinder(img, tick)  → None
draw_search_criteria(img, tick)    → None
draw_detection(img, det, ...)      → None
draw_global_hud(img, fps, ...)     → None
main()                             → None
```
