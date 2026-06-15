#!/usr/bin/env python3
"""
sizing.py — Estimasi ukuran nyata kerusakan jalan dari bounding box (lokal).

Untuk kamera dashcam yang dipasang tetap di depan mobil, dengan asumsi permukaan
jalan datar, kita bisa memetakan piksel gambar -> koordinat tanah (meter) lewat
satu matriks HOMOGRAPHY 3x3 hasil kalibrasi sekali (scripts/calibrate_camera.py).

Kerusakan jalan (lubang/retak) menempel di permukaan jalan, jadi keempat sudut
bbox-nya dianggap berada di bidang tanah. Memetakan 4 sudut itu lewat homography
memberi perkiraan lebar (melintang), panjang (searah jalan), dan luas (m²).

⚠️ Keterbatasan jujur:
  - Akurasi tergantung kalibrasi & asumsi jalan datar. Guncangan, tanjakan/turunan,
    dan kemiringan kamera menurunkan akurasi.
  - Anggap hasilnya PERKIRAAN untuk membanding-bandingkan & estimasi kasar, bukan
    pengukuran survei presisi. Verifikasi manual untuk klaim formal.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


def load_calibration(path: str | Path) -> Optional[dict]:
    """Muat file kalibrasi (configs/camera_calib.json). None bila tidak ada."""
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if "homography" not in data:
        raise ValueError(f"File kalibrasi tidak valid (tak ada 'homography'): {p}")
    return data


def _apply_homography(H, pts):
    """Petakan list titik piksel [(x,y),...] -> titik tanah [(X,Y),...] (meter)."""
    out = []
    for x, y in pts:
        denom = H[2][0] * x + H[2][1] * y + H[2][2]
        if abs(denom) < 1e-12:
            denom = 1e-12
        X = (H[0][0] * x + H[0][1] * y + H[0][2]) / denom
        Y = (H[1][0] * x + H[1][1] * y + H[1][2]) / denom
        out.append((X, Y))
    return out


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _shoelace(poly):
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def bbox_ground_size(H, x1, y1, x2, y2) -> dict:
    """Perkiraan ukuran tanah dari bbox piksel.

    Return dict: width_m (melintang), length_m (searah jalan), area_m2.
    """
    # Sudut bbox: TL, TR, BR, BL (searah jarum jam).
    corners_px = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    g = _apply_homography(H, corners_px)
    tl, tr, br, bl = g
    width_top = _dist(tl, tr)
    width_bot = _dist(bl, br)
    len_left = _dist(tl, bl)
    len_right = _dist(tr, br)
    width_m = (width_top + width_bot) / 2.0
    length_m = (len_left + len_right) / 2.0
    area_m2 = _shoelace([tl, tr, br, bl])
    return {
        "width_m": round(width_m, 3),
        "length_m": round(length_m, 3),
        "area_m2": round(area_m2, 4),
    }


if __name__ == "__main__":
    # Uji cepat: homography skala murni 1px = 0.01 m -> bbox 100x50 px = 0.5 m².
    H = [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]]
    print(bbox_ground_size(H, 0, 0, 100, 50))
