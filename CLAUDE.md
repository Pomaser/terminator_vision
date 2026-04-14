# Terminator Vision – zadání pro Claude Code

## Cíl
Vytvoř Python aplikaci `terminator_vision.py`, která přes webcam simuluje červené HUD vidění Terminátora z filmu T2. Aplikace detekuje osoby a objekty v obraze a zobrazuje k nim informace ve stylu T-800.

---

## Závislosti
```
pip install opencv-python ultralytics numpy
```
Model YOLOv8-seg (`yolov8n-seg.pt`) se stáhne automaticky při prvním spuštění přes knihovnu `ultralytics`.

---

## Obrazové filtry (v tomto pořadí na každý frame)

1. **Červený filtr** – boost R kanálu (`* 1.3 + 20`), potlačení G (`* 0.25`) a B (`* 0.08`), hodnoty clampovat na 0–255
2. **Scanlines** – každý 3. řádek ztmavit na 55 % původní hodnoty
3. **Vignette** – Gaussovský gradient ztmavující okraje obrazu (síla ~0.55)
4. **Šum** – náhodný per-pixel šum ±6 na všech kanálech

---

## Detekce objektů

- Použij `ultralytics YOLO("yolov8n-seg.pt")` s `conf=0.40`  
  *(segmentační model – vrací polygon masku každé instance, ne jen bounding box)*
- Zpracovávej každý 2. frame pro výkon, výsledky kešuj pro mezilehlé framy
- Z výsledků extrahuj: `boxes.xyxy`, `boxes.conf`, `boxes.cls`, **`masks.xy`** (seznam polygonů bodů masky)

---

## Obrysové linky kolem objektů (hlavní efekt – T2 styl)

Toto je klíčový vizuální efekt – místo (nebo doplňkově vedle) bounding boxu se kreslí **přesný obrys tvaru objektu** sledující jeho siluetu.

### Postup pro každou detekci:

1. **Získej polygon masky** z `masks.xy[i]` – pole bodů `[[x,y], [x,y], ...]` ve float32
2. **Převeď na int32** a reshape na `(-1, 1, 2)` pro OpenCV
3. **Nakresli obrys** přes `cv2.polylines(frame, [pts], isClosed=True, color, thickness)` – **ne** `fillPoly`
4. **Barvy dle třídy:**
   - `person` → `(0, 40, 255)` (jasná červenooranžová), tloušťka 2
   - ostatní objekty → `(0, 0, 180)` (tlumená červená), tloušťka 1
5. **Vnitřní plocha (volitelné):** semitransparentní fill masky přes `cv2.addWeighted`:
   - vytvoř kopii framu, vyplň masku barvou `(0, 0, 120)`, smíchej s alpha `0.12`
   - efekt: jemné červené "thermal glow" uvnitř objektu
6. **Glow efekt obrysu** (simuluje záři z filmu):
   - nakresli obrys ještě jednou, ale tloušťka +4, barva `(0, 0, 100)`, před hlavním obrysem
   - výsledek: tenčí světlá linka na širším tmavším halo

### Funkce:
```python
def draw_segmentation_outline(frame: np.ndarray, mask_pts, label: str) -> None:
    """Nakreslí T2-style obrys segmentační masky na frame (in-place)."""
    pts = np.array(mask_pts, dtype=np.int32).reshape((-1, 1, 2))
    is_person = (label == "person")
    bright = (0, 50, 255) if is_person else (0, 0, 180)
    glow   = (0, 0,  90) if is_person else (0, 0,  60)
    # glow halo
    cv2.polylines(frame, [pts], True, glow,   thickness=5, lineType=cv2.LINE_AA)
    # hlavní obrys
    cv2.polylines(frame, [pts], True, bright, thickness=2, lineType=cv2.LINE_AA)
    # vnitřní fill
    fill = frame.copy()
    cv2.fillPoly(fill, [pts], bright)
    cv2.addWeighted(fill, 0.10, frame, 0.90, 0, frame)
```

### Poznámka k výkonu:
`masks.xy` vrací polygon s redukovaným počtem bodů – pro webcam výkon je to dostatečné. Nepoužívej `masks.data` (binární maska plného rozlišení) – je pomalejší.

---

## Mesh overlay – Delaunay triangulace (T2 wireframe efekt)

Přes plochu detekovaného objektu vykresli trojúhelníkový drátěný model (wireframe mesh) pomocí Delaunay triangulace. Efekt odpovídá scéně z T2 kde Terminátorova vizuální analýza "projíždí" objekt sítí trojúhelníků.

### Postup:

1. **Získej binární masku** objektu z `masks.data[i]` (H×W tensor) – převeď na `uint8` numpy array (0/255)
2. **Vzorkuj body uvnitř masky:**
   ```python
   def sample_points_in_mask(mask_bin, n_points=120):
       ys, xs = np.where(mask_bin > 0)
       if len(xs) < n_points:
           return list(zip(xs.tolist(), ys.tolist()))
       idx = np.random.choice(len(xs), n_points, replace=False)
       return list(zip(xs[idx].tolist(), ys[idx].tolist()))
   ```
3. **Přidej rohové body bounding boxu** jako kotevní body triangulace (zabraňuje "utíkání" triangulace za okraj masky)
4. **Delaunay triangulace** přes `cv2.Subdiv2D`:
   ```python
   def delaunay_mesh(frame, points, bbox, color=(0,0,140), thickness=1):
       x1,y1,x2,y2 = bbox
       rect = (x1, y1, x2-x1, y2-y1)
       subdiv = cv2.Subdiv2D(rect)
       for p in points:
           if x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
               subdiv.insert((float(p[0]), float(p[1])))
       triangles = subdiv.getTriangleList().astype(np.int32)
       for t in triangles:
           pt1,pt2,pt3 = (t[0],t[1]),(t[2],t[3]),(t[4],t[5])
           # kresli jen trojúhelníky jejichž všechny vrcholy leží uvnitř masky
           cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
           cv2.line(frame, pt2, pt3, color, thickness, cv2.LINE_AA)
           cv2.line(frame, pt3, pt1, color, thickness, cv2.LINE_AA)
   ```
5. **Filtruj trojúhelníky mimo masku** – před kreslením ověř, že střed trojúhelníku leží uvnitř masky:
   ```python
   cx = (pt1[0]+pt2[0]+pt3[0])//3
   cy = (pt1[1]+pt2[1]+pt3[1])//3
   if 0<=cy<mask_bin.shape[0] and 0<=cx<mask_bin.shape[1] and mask_bin[cy,cx] > 0:
       # kresli
   ```

### Parametry stylu:

| Situace | Barva (`BGR`) | Tloušťka | Počet bodů |
|---------|--------------|----------|------------|
| `person` | `(0, 0, 160)` | 1 | 150 |
| ostatní objekty | `(0, 0, 100)` | 1 | 80 |

### Animace (volitelné – vypadá skvěle):
- Každých 0.4 s **resample** nové náhodné body → mesh se "přepočítává" a mění tvar
- Ukládej poslední mesh body per-objekt do slovníku klíčovaného `track_id` nebo indexem detekce

### Výkon:
- `cv2.Subdiv2D` je rychlý, ale `masks.data` (binární maska) je pomalejší než `masks.xy`
- Pro výkon: spočítej binární masku jednou per inference frame, ne na každém display framu
- Pokud FPS klesne pod 15, sniž `n_points` na 60

### Funkce:
```python
def draw_mesh_overlay(frame: np.ndarray, mask_bin: np.ndarray,
                      bbox: tuple, label: str, tick: float) -> None:
    """Delaunay triangulace mesh přes segmentační masku."""
    ...
```

---

## HUD overlay – globální prvky

- **Rohové závorky** ve všech 4 rozích celého obrazu (velikost 20 px, tloušťka 2, barva červená)
- **Stavový řádek nahoře** (y ≈ 20 px):
  - vlevo: `CYBERDYNE SYSTEMS MODEL 101`
  - vpravo: aktuální čas `HH:MM:SS`
- **Stavový řádek dole** (y ≈ výška – 10 px):
  - vlevo: `FPS: XX.X`
  - uprostřed: `NEURAL NET PROCESSOR :: ACTIVE`
  - vpravo: `TARGETS: N` (počet aktuálních detekcí)
- Veškerý text: `cv2.FONT_HERSHEY_SIMPLEX`, červená barva, černý outline pro čitelnost

---

## HUD overlay – detekce

Pro každou detekci nakresli:

1. **Rohové závorky** kolem bounding boxu (ne celý obdélník) – velikost rohů 14 px
2. **Tenká čárkovaná linie** ze středu bounding boxu do středu obrazu (pro osoby; `cv2.LINE_4`, alpha blend ~30 %)
3. **Info box** nad bounding boxem (nebo pod ním, pokud je bbox u horního okraje):
   - Řádek 1: název třídy velkými písmeny, např. `HUMAN` (pro `person`) nebo původní název (`CAR`, `BOTTLE` …)
   - Řádek 2: `CONF: XX%`
   - Řádek 3 (pouze pro `person`): náhodně vybraná fráze z tohoto seznamu (rotovat per-objekt každé 2 s):
     ```
     TARGET ACQUIRED
     THREAT ASSESSMENT: HOSTILE
     INITIATING TARGETING
     SUBJECT IDENTIFIED
     TRACKING ACTIVE
     ```
   - Řádek 4 (pouze pro `person`): `HEIGHT EST: XXXcm` – odhadni výšku jako `(bbox_height / frame_height) * 175` zaokrouhleno na celé cm
4. **Blikající kroužek** (`cv2.circle`) ve středu bboxu pro osoby – střídá viditelnost každých 0.5 s

---

## Ovládání (klávesy)

| Klávesa | Akce |
|---------|------|
| `q` | Ukončit aplikaci |
| `s` | Screenshot uložit jako `terminator_HHMMSS.png` |
| `f` | Přepnout fullscreen |
| `d` | Zobrazit/skrýt debug info (raw YOLO boxy bez stylizace) |

---

## Spuštění

```bash
python terminator_vision.py             # výchozí kamera (index 0)
python terminator_vision.py --camera 1  # jiný index kamery
python terminator_vision.py --width 1280 --height 720  # rozlišení
```

---

## Struktura kódu

Rozděl do funkcí:
- `apply_red_filter(frame)` → frame
- `apply_scanlines(frame)` → frame
- `apply_vignette(frame)` → frame
- `add_noise(frame)` → frame
- `draw_corner_bracket(img, x1, y1, x2, y2, ...)` → None
- `draw_segmentation_outline(frame, mask_pts, label)` → None
- `draw_mesh_overlay(frame, mask_bin, bbox, label, tick)` → None
- `draw_detection(img, det, frame_shape, tick)` → None  (`tick` = čas pro blikání/rotaci frází)
- `draw_global_hud(img, fps, det_count, frame_no)` → None
- `main()` – argparse + hlavní smyčka

---

## Poznámky

- Veškerý overlay kresli na **kopii** filtrovaného framu (ne in-place), pak zkombinuj přes `cv2.addWeighted` pro průhlednost HUD prvků (~85 % overlay, 15 % original) – pouze pro detekční boxy, globální HUD kresli přímo
- Pokud kamera není dostupná, vypiš chybovou hlášku a ukonči s `sys.exit(1)`
- Otestuj na rozlišení 640×480 i 1280×720
