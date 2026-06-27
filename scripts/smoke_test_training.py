#!/usr/bin/env python3
"""
smoke_test_training.py — Verifikasi environment training TANPA RDD2022.

Membuat dataset SINTETIS kecil (gambar 'aspal' dengan bentuk yang mewakili
retak & lubang) lengkap dengan anotasi PASCAL VOC XML — sama formatnya dengan
RDD2022. Gunanya: cek bahwa rantai voc_to_yolo -> train.py -> road_survey.py
berjalan di mesin lo SEBELUM menunggu unduhan RDD2022 yang besar.

⚠️ Ini BUKAN data nyata. Model hasil smoke-test TIDAK untuk produksi — hanya
membuktikan pipa training jalan.

Alur lengkap (lihat juga docs/FASE2.md):
    python scripts/smoke_test_training.py --out datasets/smoke --n 24
    python scripts/voc_to_yolo.py --input datasets/smoke --out datasets/smoke_yolo
    python train.py --data datasets/smoke_yolo/data.yaml --model yolov8n.pt \
        --epochs 3 --imgsz 320 --device cpu --name smoke
    python road_survey.py --source datasets/smoke_yolo/images/val \
        --model runs/detect/smoke/weights/best.pt --no-display
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np

# (nama_kelas RDD, fungsi gambar). Kode RDD dipakai supaya voc_to_yolo memetakan.
RDD_CODES = ["D00", "D10", "D20", "D40"]   # long, trans, alligator, pothole


def asphalt(w, h):
    """Latar mirip aspal: abu gelap + noise."""
    base = np.full((h, w, 3), (60, 60, 62), np.uint8)
    noise = np.random.randint(-18, 18, (h, w, 3), np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def draw_damage(img, code):
    """Gambar satu kerusakan, kembalikan bbox (xmin,ymin,xmax,ymax)."""
    h, w = img.shape[:2]
    if code == "D40":  # pothole: blob gelap
        cx, cy = random.randint(60, w - 60), random.randint(60, h - 60)
        rx, ry = random.randint(20, 45), random.randint(18, 40)
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (15, 15, 18), -1)
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (5, 5, 6), 3)
        return cx - rx, cy - ry, cx + rx, cy + ry
    if code in ("D00", "D10"):  # crack: garis tipis (vertikal/horizontal)
        if code == "D00":  # longitudinal (memanjang ~ vertikal)
            x = random.randint(40, w - 40)
            y1, y2 = random.randint(20, h // 2), random.randint(h // 2, h - 20)
            pts = [(x + random.randint(-6, 6), y) for y in range(y1, y2, 12)]
        else:              # transverse (melintang ~ horizontal)
            y = random.randint(40, h - 40)
            x1, x2 = random.randint(20, w // 2), random.randint(w // 2, w - 20)
            pts = [(x, y + random.randint(-6, 6)) for x in range(x1, x2, 12)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (20, 20, 22), 2)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return min(xs) - 4, min(ys) - 4, max(xs) + 4, max(ys) + 4
    # D20 alligator: jaring retak
    cx, cy = random.randint(70, w - 70), random.randint(70, h - 70)
    s = random.randint(40, 70)
    x0, y0, x1, y1 = cx - s, cy - s, cx + s, cy + s
    for _ in range(14):
        p = (random.randint(x0, x1), random.randint(y0, y1))
        q = (random.randint(x0, x1), random.randint(y0, y1))
        cv2.line(img, p, q, (22, 22, 24), 1)
    return x0, y0, x1, y1


def write_voc(xml_path: Path, fname: str, w, h, objs):
    obj_xml = "".join(
        f"""  <object>
    <name>{escape(code)}</name>
    <bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>"""
        f"""<xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox>
  </object>
"""
        for code, (xmin, ymin, xmax, ymax) in objs
    )
    xml_path.write_text(
        f"""<annotation>
  <filename>{escape(fname)}</filename>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
{obj_xml}</annotation>
""",
        encoding="utf-8",
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Buat dataset sintetis untuk smoke-test")
    p.add_argument("--out", default="datasets/smoke")
    p.add_argument("--n", type=int, default=24, help="Jumlah gambar")
    p.add_argument("--size", type=int, default=416)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    random.seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out)
    img_dir = out / "train" / "images"
    xml_dir = out / "train" / "annotations" / "xmls"
    img_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        w = h = args.size
        img = asphalt(w, h)
        objs = []
        for _ in range(random.randint(1, 3)):
            code = random.choice(RDD_CODES)
            objs.append((code, draw_damage(img, code)))
        fname = f"smoke_{i:04d}.jpg"
        cv2.imwrite(str(img_dir / fname), img)
        write_voc(xml_dir / f"smoke_{i:04d}.xml", fname, w, h, objs)

    print(f"Dataset sintetis dibuat: {args.n} gambar di {out}")
    print("Lanjut:")
    print(f"  python scripts/voc_to_yolo.py --input {out} --out {out}_yolo")
    print(f"  python train.py --data {out}_yolo/data.yaml --model yolov8n.pt "
          f"--epochs 3 --imgsz 320 --device cpu --name smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
