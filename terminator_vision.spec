# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec soubor pro Terminator Vision
# Spustit na Windows: pyinstaller terminator_vision.spec

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['terminator_vision.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Fonty
        ('font/Helvetica73-Extended Bold.ttf', 'font'),
        ('font/modern-vision.ttf',             'font'),
        # Zvuky (volitelné – pokud složka existuje)
        ('sounds/*.wav',  'sounds'),
        # Konfigurační soubor zpráv
        ('terminal_messages.txt', '.'),
        # YOLO model
        ('yolov8n-seg.pt', '.'),
    ],
    hiddenimports=[
        # OpenCV
        'cv2',
        # Ultralytics / YOLOv8
        'ultralytics',
        'ultralytics.nn.tasks',
        'ultralytics.nn.modules',
        'ultralytics.utils',
        'ultralytics.utils.checks',
        'ultralytics.utils.ops',
        'ultralytics.models.yolo',
        'ultralytics.models.yolo.segment',
        'ultralytics.models.yolo.detect',
        'ultralytics.data',
        'ultralytics.data.loaders',
        # PyTorch
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torchvision',
        # Pillow
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        # Pygame
        'pygame',
        'pygame.mixer',
        # Ostatní
        'numpy',
        'scipy',
        'scipy.spatial',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # zmenší exe – vynech to co nepotřebujeme
        'matplotlib',
        'tkinter',
        'notebook',
        'IPython',
        'pandas',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TerminatorVision',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,       # False = bez cmd okna, ale pak nevidíš chyby
    icon=None,          # 'icon.ico' pokud máš ikonu
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TerminatorVision',
)
