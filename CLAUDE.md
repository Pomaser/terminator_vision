# Terminator Vision – dokumentace projektu

## Cíl
Python aplikace `terminator_vision.py` simulující červené HUD vidění Terminátora T-800 z filmu T2. Detekuje osoby a objekty přes YOLOv8-seg a zobrazuje je s T2-style overlay.

---

## Závislosti

```
pip install opencv-python ultralytics numpy pillow pygame
```

- Model `yolov8n-seg.pt` se stáhne automaticky při prvním spuštění.
- Font: `font/Helvetica73-Extended Bold.ttf` – musí být přítomen.
- Zvuky (volitelné): `sounds/{startup,ambient,target,new_target,scan,alert}.wav`

---

## Spuštění

```bash
python terminator_vision.py             # výchozí kamera (index 0)
python terminator_vision.py --camera 1  # jiný index kamery
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

## Obrazové filtry (pořadí na každý frame)

1. **Červený filtr** – R `* 1.3 + 20`, G `* 0.25`, B `* 0.08`, clamp 0–255
2. **Vignette** – Gaussovský gradient (cachováno per rozlišení, síla 0.55)
3. **Šum** – per-pixel ±6 na všech kanálech

> Scanlines a globální mřížka přes celý obraz byly odstraněny.

---

## Detekce objektů

- `YOLO("yolov8n-seg.pt")`, `conf=0.40`
- Každý **2. frame** se spouští inference, výsledky se kešují
- Extrahuje: `boxes.xyxy`, `boxes.conf`, `boxes.cls`, `masks.xy`, `masks.data`

---

## Segmentační obrys (`draw_segmentation_outline`)

- Polygon z `masks.xy[i]` → `cv2.polylines`
- `person`: bílý obrys (255,255,255), tloušťka 2, glow halo (80,80,80) tloušťka 5
- ostatní: šedý obrys (180,180,180), glow (60,60,60)
- Bez fill překryvu (odstraněno pro výkon)

---

## Mesh overlay – Delaunay triangulace (`draw_mesh_overlay`)

- Binární maska z `masks.data[i]` → vzorkování bodů uvnitř masky
- `cv2.Subdiv2D` triangulace, filtr středem trojúhelníku uvnitř masky
- **Částečné skenování těla** – `BODY_ZONES` (HEAD, TORSO, LEFT/RIGHT ARM, LEFT/RIGHT LEG), cykluje po 0.7 s
- Trojúhelníky se **cachují** (přepočítávají jen při resample, ne každý frame)
- `person`: 150 bodů, barva (200,200,200); ostatní: 80 bodů, (140,140,140)

---

## PIL textový systém

Všechny texty se renderují fontem **Helvetica73-Extended Bold** přes Pillow (cv2.freetype není dostupné).

```python
put_text_outlined(img, text, pos, font_scale, color)  # zařadí do fronty
flush_text(img)   # jednou PIL konverzí vykreslí všechny texty pro daný frame
```

Barvy: `RED_TEXT = (255,255,255)` (bílá), `RED_DIM = (160,160,160)` (šedá).

---

## HUD overlay – globální (`draw_global_hud`)

- **Rohové závorky** ve 4 rozích obrazu
- **Nahoře vlevo**: `CYBERDYNE SYSTEMS MODEL 101`
- **Nahoře vpravo**: čas `HH:MM:SS`
- **Dole vlevo**: `FPS: XX.X`
- **Dole uprostřed**: `NEURAL NET PROCESSOR :: ACTIVE`
- **Dole vpravo**: `TARGETS: N`

---

## HUD overlay – detekce (`draw_detection`)

Pro každou detekci:
1. Rohové závorky kolem bounding boxu
2. Čárkovaná linie ze středu bbox do středu obrazu (pouze osoby, alpha ~30 %)
3. Info box (třída, `CONF: XX%`, phrase pro osoby, `HEIGHT EST: XXXcm`)
4. Blikající kroužek ve středu bbox (osoby, každých 0.5 s)

---

## Levý HUD panel (`draw_left_hud`)

Umístění: levá třetina obrazu, pozice se mění každých 10 s.
Přepíná **5 módů** po **5 sekundách** s **typewriter efektem** (nový řádek každých 0.18 s):

| Mód | Obsah |
|-----|-------|
| 0 | `SCAN LEVELS:` + skupiny čísel |
| 1 | `CRITERIA:` + tabulka tagů s čísly |
| 2 | `ANALYSIS: MATCH:` + rozšířená tabulka + `ASSESS: SUITABLE/POSSIBLE` |
| 3 | Data dump – dvě skupiny čísel oddělené prázdným řádkem |
| 4 | `VISUAL ASSESSMENT:` + pravděpodobnost, data |

---

## Pravý HUD panel (`draw_right_hud`)

Umístění: pravá třetina obrazu, pozice se mění každých 10 s.
Přepíná **6 módů** po **6 sekundách** s **typewriter efektem**:

| Mód | Obsah |
|-----|-------|
| 0 | `SCAN MODE XXXXX` + tabulka tagů + **12×12 mřížka s pohyblivými osami** |
| 1 | `ACQUIRE ...` + tabulka tagů |
| 2 | `PRIORITY XXXXXXXD` + data + `STATUS: CONFIRMED/PENDING` |
| 3 | `DATA DUMP:` + dvě skupiny čísel |
| 4 | `ENVIRONMENT SCAN:` + teplota, vzdálenost, motion, data |
| 5 | Kompasová růžice (N/NE/E/SE/S/SW/W/NW) + `TARGET FIELD:` + 3 řady čísel |

### 12×12 mřížka (mód 0)

- 12 sloupců × 12 řádků, buňka 22×15 px, celkem 264×180 px
- Všechny linky bílé (255,255,255)
- **Osa X** (vodorovná) a **osa Y** (svislá) jsou vlasové bílé čáry pohybující se plynule po sinusoidách s nesouměřitelnými frekvencemi (0.37 a 0.53 Hz)

### Kompasová růžice (mód 5)

- 8 stejně dlouhých paprsků ze středu, bez kruhů
- Popisky N, NE, E, SE, S, SW, W, NW vně paprsků
- Pod růžicí: `TARGET FIELD:` + 3 × 12místné číslo (mění se každé 4 s)

---

## Pohyblivý hledáček (`draw_camera_viewfinder`)

- Split-image rangefinder pohybující se po obrazovce Lissajousovým pohybem
- Průměr ~8 % kratší strany obrazu
- Poloprůhledný bílý fill (alpha 0.25), vnější kruh s tick marky, pulzující outer ring

---

## Zvukový engine (`SoundEngine`)

- Používá `pygame.mixer`
- Zvuky: `startup`, `ambient` (smyčka), `target`/`new_target`, `scan` (každé ~4 s), `alert` (≥3 cíle)
- Defaultně **vypnuto** – klávesa `S` přepíná

---

## Struktura funkcí

```
apply_red_filter(frame)           → frame
apply_vignette(frame)             → frame  (výsledek cachován)
add_noise(frame)                  → frame
draw_corner_bracket(img, ...)     → None
draw_segmentation_outline(...)    → None
sample_points_in_mask(...)        → list
sample_points_in_zone(...)        → list
get_zone_bbox(...)                → tuple
draw_mesh_overlay(...)            → None
_compute_triangles(...)           → list   (cache triangulace)
_draw_triangles(...)              → None
put_text_outlined(img, ...)       → None   (fronta)
flush_text(img)                   → None   (PIL render)
draw_global_grid(img, tick)       → None   (sweep linka)
_draw_compass_rose_at(img, ...)   → None   (pomocná)
_draw_grid_hud(img, tick, px, py) → None   (12×12 mřížka)
draw_left_hud(img, tick)          → None
draw_right_hud(img, tick)         → None
draw_center_reticle(img, tick)    → None
draw_camera_viewfinder(img, tick) → None
draw_search_criteria(img, tick)   → None
draw_detection(img, det, ...)     → None
draw_global_hud(img, fps, ...)    → None
main()                            → None
```
