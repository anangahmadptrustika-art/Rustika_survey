#!/usr/bin/env python3
"""
calibrate_camera.py — Kalibrasi kamera dashcam untuk estimasi ukuran (sekali jalan).

Menghasilkan configs/camera_calib.json berisi matriks HOMOGRAPHY yang memetakan
piksel gambar -> koordinat tanah (meter), dipakai sizing.py untuk menaksir ukuran
nyata kerusakan jalan.

CARA KALIBRASI (lakukan SEKALI setelah kamera terpasang di mobil):
  1. Parkir di permukaan datar. Letakkan tanda di jalan pada 4+ titik yang
     terlihat kamera, dan UKUR posisinya (meter) relatif satu titik acuan.
     Contoh mudah: pakai 4 sudut sebuah objek persegi berukuran diketahui
     (mis. lakban membentuk 2m x 1m), atau marka jalan dengan jarak terukur.
  2. Ambil 1 frame dari kamera, catat koordinat PIKSEL tiap titik (buka di
     image viewer / pakai mode --click di laptop).
  3. Buat file CSV: img_x,img_y,ground_x,ground_y  (satu baris per titik, >=4).
  4. Jalankan:
       python scripts/calibrate_camera.py --points-file kalibrasi.csv --image frame.jpg

Konvensi tanah yang disarankan: X = melintang jalan (kanan +), Y = searah jalan
menjauh dari mobil (depan +), dalam meter.

⚠️ Estimasi ukuran = perkiraan (asumsi jalan datar). Lihat catatan di sizing.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np


def compute_homography(src, dst):
    """Homography (3x3) src(piksel) -> dst(meter) via DLT. src/dst: list (x,y)."""
    if len(src) < 4:
        raise ValueError("Butuh minimal 4 titik korespondensi.")
    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, x * X, y * X, X])
        A.append([0, 0, 0, -x, -y, -1, x * Y, y * Y, Y])
    A = np.asarray(A, dtype=float)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def reproject_error(H, src, dst):
    """Rata-rata galat reproyeksi (meter) untuk menilai kualitas kalibrasi."""
    errs = []
    for (x, y), (X, Y) in zip(src, dst):
        d = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        px = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / d
        py = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / d
        errs.append(((px - X) ** 2 + (py - Y) ** 2) ** 0.5)
    return sum(errs) / len(errs)


def read_points(path: Path):
    src, dst = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().lower().startswith(("#", "img")):
                continue  # lewati komentar / header
            ix, iy, gx, gy = (float(v) for v in row[:4])
            src.append((ix, iy)); dst.append((gx, gy))
    return src, dst


def click_points(image_path: Path):
    """Mode interaktif (butuh display): klik titik di gambar, ketik koordinat tanah."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Tidak bisa baca gambar: {image_path}")
    pts = []
    print("Klik titik kalibrasi di jendela. Tekan 'q' bila selesai (>=4 titik).")

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("kalibrasi", img)
            print(f"  titik {len(pts)}: piksel ({x},{y})")

    cv2.imshow("kalibrasi", img)
    cv2.setMouseCallback("kalibrasi", on_click)
    while True:
        if (cv2.waitKey(20) & 0xFF) == ord("q") and len(pts) >= 4:
            break
    cv2.destroyAllWindows()
    src, dst = [], []
    for i, (x, y) in enumerate(pts):
        gx = float(input(f"titik {i+1} piksel ({x},{y}) -> ground_x (m): "))
        gy = float(input(f"titik {i+1} piksel ({x},{y}) -> ground_y (m): "))
        src.append((x, y)); dst.append((gx, gy))
    return src, dst


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Kalibrasi kamera untuk estimasi ukuran")
    p.add_argument("--points-file", help="CSV: img_x,img_y,ground_x,ground_y (>=4)")
    p.add_argument("--image", help="Frame kalibrasi (untuk metadata / mode --click)")
    p.add_argument("--click", action="store_true", help="Mode klik interaktif (butuh display)")
    p.add_argument("--out", default="configs/camera_calib.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.click:
        if not args.image:
            raise SystemExit("Mode --click butuh --image.")
        src, dst = click_points(Path(args.image))
    elif args.points_file:
        src, dst = read_points(Path(args.points_file))
    else:
        raise SystemExit("Pilih --points-file CSV atau --click --image.")

    H = compute_homography(src, dst)
    err = reproject_error(H, src, dst)

    img_size = None
    if args.image and Path(args.image).exists():
        try:
            import cv2
            im = cv2.imread(args.image)
            if im is not None:
                img_size = [int(im.shape[1]), int(im.shape[0])]
        except Exception:
            pass

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "homography": H.tolist(),
        "points_px": src,
        "points_ground_m": dst,
        "image_size": img_size,
        "reprojection_error_m": round(err, 4),
        "created": date.today().isoformat(),
        "note": "piksel->meter; asumsi jalan datar. Estimasi, bukan presisi survei.",
    }, indent=2), encoding="utf-8")

    print(f"Kalibrasi tersimpan -> {out}")
    print(f"  Titik dipakai        : {len(src)}")
    print(f"  Galat reproyeksi rata2: {err:.3f} m  "
          f"({'BAGUS' if err < 0.1 else 'cek ulang titik' if err > 0.3 else 'cukup'})")
    print("Pakai saat survey: python road_survey.py --source 0 --calib", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
