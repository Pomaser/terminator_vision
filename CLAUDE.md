# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Projekt

Python aplikace `terminator_vision.py` – T-800 HUD simulace z T2 přes webcam/video. YOLOv8-seg detekce osob a objektů, bílý overlay na filtrovaném obrazu.

---

## Spuštění

```bash
python3.14 terminator_vision.py --source sample/sample.mp4   # video soubor
python3.14 terminator_vision.py                               # výchozí kamera
python3.14 terminator_vision.py --camera 1 --width 1280 --height 720
```

**Syntaktická kontrola** (bez spuštění):
```bash
python3.14 -c "import ast; ast.parse(open('terminator_vision.py').read()); print('OK')"
```

**Závislosti:**
```bash
python3.14 -m pip install opencv-python ultralytics numpy pillow pygame
```

Model `yolov8n-seg.pt` se stáhne automaticky při prvním spuštění (pokud není přítomen lokálně).

---

## Build pro Windows (PyInstaller)

Sestavit exe lze pouze na Windows (PyInstaller je platform-native):

```bat
build_windows.bat
```

Výstup: `dist\TerminatorVision\TerminatorVision.exe`

Cesty k assetům jsou řešeny přes `_asset(rel)` – funguje jak při normálním spuštění, tak jako frozen exe (`sys._MEIPASS`). Vždy používej `_asset()` místo `__file__`-relativních cest.

---

## Ovládání

| Klávesa | Akce |
|---------|------|
| `q` | Ukončit |
| `s` | Screenshot → `terminator_HHMMSS.png` |
| `f` | Fullscreen |
| `d` | Debug (raw YOLO boxy) |
| `S` | Zapnout/vypnout zvuk (default: vypnuto) |

---

## Soubory projektu

| Soubor | Účel |
|--------|------|
| `terminator_vision.py` | Celá aplikace (jediný zdrojový soubor) |
| `terminal_messages.txt` | Zprávy pro terminálový typewriter, jeden řádek = jedna zpráva, `#` = komentář |
| `terminator_vision.spec` | PyInstaller spec pro Windows build |
| `build_windows.bat` | Build skript – spustit na Windows |
| `font/Helvetica73-Extended Bold.ttf` | Font HUD panelů |
| `font/modern-vision.ttf` | Font terminálového typewriteru (100 px) |
| `sounds/*.wav` | Zvuky: startup, ambient, target, new_target, scan, alert |
| `sample/sample.mp4` | Referenční video pro vývoj bez kamery |

---

## Barevné schéma

Veškerý overlay je **bílý**:
- `RED_TEXT = (255, 255, 255)` — hlavní bílá
- `RED_DIM  = (160, 160, 160)` — tlumená šedá

Při přidávání nových prvků používej výhradně tyto konstanty, ne hardcoded barvy.

---

## Architektura hlavní smyčky

```
cap.read() → apply_red_filter → apply_vignette → add_noise
    ↓
overlay = filtered.copy()
    ↓
[YOLO inference každý 2. frame, výsledky v cached_detections]
    ↓
Filtr velikosti bbox (MAX_BBOX_AREA_RATIO) + výběr max. 3 viditelných detekcí
(priority: osoby, rotace non-person objektů každých OBJ_CYCLE_SECS)
    ↓
draw_segmentation_outline + _draw_triangles + put_text_outlined  → overlay
    ↓
out = cv2.addWeighted(overlay, 0.85, filtered, 0.15, 0)
retarget_text(overlay, out)   ← přesměruje textovou frontu
    ↓
draw_global_hud + draw_left_hud + draw_right_hud + draw_search_criteria
draw_camera_viewfinder + draw_terminal_text  → out
    ↓
flush_text(out)   ← jediná PIL konverze za frame (flushuje obě fronty)
    ↓
cv2.imshow
```

---

## PIL textový systém – kritická pravidla

Všechny texty jdou přes fronty, PIL konverze proběhne **jednou za frame**:

```python
put_text_outlined(img, text, pos, font_scale, color)  # → _text_queue (Helvetica)
# MV font se přidává přes _mv_queue v draw_terminal_text (interně)
flush_text(out)      # volat JEDNOU, na konci framové smyčky
retarget_text(overlay, out)  # volat po addWeighted, před flush_text
```

**Nesmí** se volat `flush_text` vícekrát za frame – způsobí duplicitní PIL konverzi.

---

## Detekce a výběr objektů

- YOLO inference každý **2. frame**, výsledky cachované v `cached_detections`
- **Filtr velikosti:** `MAX_BBOX_AREA_RATIO = 0.40` – bbox větší než 40 % plochy framu se ignoruje (nastavit v `main()`)
- Max. **3 objekty** současně: priority mají osoby, non-person objekty rotují po `_OBJ_CYCLE_SECS` sekundách
- Stav rotace: `_obj_state = [last_cycle_tick, offset]` v `main()`
- Mesh trojúhelníky se přepočítávají každých **0.5 s** a cachují v `mesh_points_cache`

---

## HUD panely – pozice

Obě panely se pohybují každých 10 s (seeded random, různé seedy):

| Panel | Horizontálně | Vertikálně |
|-------|-------------|------------|
| `draw_left_hud` | `x` ∈ `[10, w//6]` (max polovina levé třetiny) | střední třetina `[h//3, 2h//3]` |
| `draw_right_hud` | pravý okraj max na `5w//6` | střední třetina `[h//3, 2h//3]` |

---

## Terminálový typewriter

Zprávy se načítají ze souboru `terminal_messages.txt` při startu. State machine:

```
idle ──(terminal_trigger() + cooldown 5–10 s)──► typing ──(celý text)──► pause (2 s) ──► idle
```

`terminal_trigger()` se volá v `main()` při `det_count > prev_det_count`.

---

## Levý HUD panel – 5 módů (každých 5 s)

| Mód | Obsah |
|-----|-------|
| 0 | `SCAN LEVELS:` + čísla |
| 1 | `CRITERIA:` + tabulka tagů |
| 2 | `ANALYSIS: MATCH:` + tabulka + `ASSESS` |
| 3 | Data dump |
| 4 | `VISUAL ASSESSMENT:` |

## Pravý HUD panel – 6 módů (každých 6 s)

| Mód | Obsah |
|-----|-------|
| 0 | `SCAN MODE` + tabulka + 12×12 mřížka s plynulými osami |
| 1 | `ACQUIRE` + tabulka |
| 2 | `PRIORITY` + data |
| 3 | `DATA DUMP` |
| 4 | `ENVIRONMENT SCAN` |
| 5 | Kompasová růžice + `TARGET FIELD` |

---

## Výkonnostní optimalizace (již implementováno)

| Oblast | Řešení |
|--------|--------|
| PIL konverze | `retarget_text` + jediný `flush_text` |
| Šum | Pool 30 předgenerovaných polí (`_init_noise_pool` při startu) |
| Červený filtr | LUT tabulky (`_lut_b/g/r`) |
| Viewfinder | ROI kopie místo celého framu |
| Mesh | Cache trojúhelníků, přepočet každých 0.5 s |
| Vignette | Cache masky per rozlišení |

---

## Pohyblivý hledáček (`draw_camera_viewfinder`)

- Lissajousův pohyb, frekvence 0.31 a 0.47 Hz
- Čáry uvnitř kruhu jsou **100% průhledné** – implementováno jako výřez z bílého fillu (obnovení originálních pixelů přes `line_mask`)
- Blend bílého fillu: alpha 0.45
