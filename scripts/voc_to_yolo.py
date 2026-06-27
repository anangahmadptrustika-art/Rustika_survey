#!/usr/bin/env python3
"""
voc_to_yolo.py — Konversi anotasi PASCAL VOC (XML) RDD2022 -> format YOLO.

RDD2022 (Road Damage Dataset 2022) memakai anotasi PASCAL VOC (.xml). YOLO/
ultralytics butuh satu file .txt per gambar berisi baris:
    <class_id> <x_center> <y_center> <width> <height>   (semua ternormalisasi 0..1)

Skema kelas target proyek (4 kelas — sesuai brief: minimal lubang + retak):
    0 longitudinal_crack   (RDD: D00, D01)
    1 transverse_crack     (RDD: D10, D11)
    2 alligator_crack      (RDD: D20)
    3 pothole              (RDD: D40)

Kode RDD lain (D43 crosswalk blur, D44 white-line blur, D50 manhole, dll) di-SKIP
secara default. Ubah CLASS_MAP di bawah kalau mau menyertakan/menggabungkan kelas.

Struktur input RDD2022 per negara (contoh):
    RDD2022/Japan/train/images/*.jpg
    RDD2022/Japan/train/annotations/xmls/*.xml

Struktur output (siap dilatih YOLO):
    <out>/images/train/*.jpg   <out>/labels/train/*.txt
    <out>/images/val/*.jpg     <out>/labels/val/*.txt
    <out>/data.yaml

Contoh:
    python scripts/voc_to_yolo.py \
        --input datasets/RDD2022/Japan datasets/RDD2022/India \
        --out datasets/rdd_yolo --val-split 0.2

Konverter ini dirancang aman untuk diuji: jalankan pada folder XML kecil dan
periksa hasil .txt-nya.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# Nama kelas target (urutan = class_id YOLO).
CLASS_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]

# Pemetaan kode kerusakan RDD2022 -> class_id target. Kode tak terdaftar di-skip.
CLASS_MAP = {
    "D00": 0, "D01": 0,        # retak memanjang (longitudinal)
    "D10": 1, "D11": 1,        # retak melintang (transverse)
    "D20": 2,                  # retak buaya (alligator)
    "D40": 3,                  # lubang (pothole)
    # Sinonim/ejaan yang kadang muncul di varian dataset:
    "longitudinal_crack": 0, "transverse_crack": 1,
    "alligator_crack": 2, "pothole": 3,
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def find_image_for_xml(xml_path: Path, images_dir: Path) -> Path | None:
    """Cari file gambar yang berpasangan dengan satu XML."""
    stem = xml_path.stem
    for ext in IMG_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    # Fallback: cari rekursif (struktur dataset kadang beda).
    for ext in IMG_EXTS:
        hits = list(images_dir.rglob(f"{stem}{ext}"))
        if hits:
            return hits[0]
    return None


def convert_one(xml_path: Path, stats: Counter) -> tuple[list[str], int]:
    """Parse satu XML -> daftar baris YOLO. Return (lines, jumlah_objek_valid)."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as e:
        stats["xml_parse_error"] += 1
        print(f"  ! XML rusak dilewati: {xml_path.name} ({e})")
        return [], 0

    size = root.find("size")
    if size is None:
        stats["no_size"] += 1
        return [], 0
    W = float(size.findtext("width") or 0)
    H = float(size.findtext("height") or 0)
    if W <= 0 or H <= 0:
        stats["bad_size"] += 1
        return [], 0

    lines: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASS_MAP:
            stats[f"skip_{name or 'EMPTY'}"] += 1
            continue
        cls_id = CLASS_MAP[name]
        bb = obj.find("bndbox")
        if bb is None:
            stats["no_bndbox"] += 1
            continue
        try:
            xmin = float(bb.findtext("xmin"))
            ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax"))
            ymax = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            stats["bad_bndbox"] += 1
            continue

        # Clamp ke dalam batas gambar, lalu normalisasi ke format YOLO.
        xmin, xmax = max(0.0, min(xmin, xmax)), min(W, max(xmin, xmax))
        ymin, ymax = max(0.0, min(ymin, ymax)), min(H, max(ymin, ymax))
        bw, bh = xmax - xmin, ymax - ymin
        if bw <= 1 or bh <= 1:           # bbox degenerate
            stats["degenerate_bbox"] += 1
            continue
        xc = (xmin + xmax) / 2.0 / W
        yc = (ymin + ymax) / 2.0 / H
        lines.append(
            f"{cls_id} {xc:.6f} {yc:.6f} {bw / W:.6f} {bh / H:.6f}"
        )
        stats[f"obj_{CLASS_NAMES[cls_id]}"] += 1

    return lines, len(lines)


def gather_xmls(input_dirs: list[Path]) -> list[tuple[Path, Path]]:
    """Kumpulkan (xml, images_dir) dari beberapa folder negara/dataset."""
    pairs = []
    for d in input_dirs:
        if not d.exists():
            print(f"PERINGATAN: folder input tidak ada: {d}")
            continue
        # Cari semua XML di bawah folder ini.
        xmls = list(d.rglob("*.xml"))
        if not xmls:
            print(f"PERINGATAN: tidak ada .xml di {d}")
            continue
        for x in xmls:
            # Tebak folder gambar: '.../annotations/xmls/a.xml' -> '.../images'
            images_dir = None
            for parent in x.parents:
                cand = parent / "images"
                if cand.is_dir():
                    images_dir = cand
                    break
            if images_dir is None:
                images_dir = d  # fallback: cari rekursif dari root input
            pairs.append((x, images_dir))
    return pairs


def write_data_yaml(out: Path) -> Path:
    yaml_path = out / "data.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        f"# data.yaml — dibuat oleh voc_to_yolo.py\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    return yaml_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Konversi RDD2022 VOC XML -> YOLO")
    p.add_argument("--input", nargs="+", required=True,
                   help="Satu/lebih folder dataset (mis. RDD2022/Japan RDD2022/India)")
    p.add_argument("--out", required=True, help="Folder output dataset YOLO")
    p.add_argument("--val-split", type=float, default=0.2, help="Porsi data validasi")
    p.add_argument("--seed", type=int, default=42, help="Seed acak untuk split")
    p.add_argument("--copy", action="store_true",
                   help="Salin gambar (default: symlink, hemat disk; pakai --copy di Windows bila symlink bermasalah)")
    p.add_argument("--keep-negatives", action="store_true",
                   help="Sertakan gambar tanpa objek valid (background). Default: dilewati.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    input_dirs = [Path(d) for d in args.input]
    out = Path(args.out)

    pairs = gather_xmls(input_dirs)
    if not pairs:
        sys.exit("ERROR: tidak ada anotasi XML ditemukan di folder input.")
    print(f"Ditemukan {len(pairs)} file anotasi XML.")

    # Buat struktur folder.
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    converted = []  # (img_path, yolo_lines)
    for xml_path, images_dir in pairs:
        lines, n = convert_one(xml_path, stats)
        if n == 0 and not args.keep_negatives:
            stats["images_no_valid_obj"] += 1
            continue
        img = find_image_for_xml(xml_path, images_dir)
        if img is None:
            stats["image_missing"] += 1
            continue
        converted.append((img, lines))

    if not converted:
        sys.exit("ERROR: tidak ada pasangan gambar+anotasi valid. "
                 "Cek struktur folder (images/ vs annotations/xmls/).")

    random.shuffle(converted)
    n_val = int(len(converted) * args.val_split)
    splits = {"val": converted[:n_val], "train": converted[n_val:]}

    for split, items in splits.items():
        for img, lines in items:
            dst_img = out / f"images/{split}" / img.name
            dst_lbl = out / f"labels/{split}" / f"{img.stem}.txt"
            # Gambar: symlink (hemat) atau copy.
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            if args.copy:
                shutil.copy2(img, dst_img)
            else:
                try:
                    dst_img.symlink_to(img.resolve())
                except OSError:
                    shutil.copy2(img, dst_img)  # fallback (mis. Windows tanpa izin)
            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""),
                               encoding="utf-8")

    yaml_path = write_data_yaml(out)

    # Ringkasan.
    print("-" * 60)
    print("KONVERSI SELESAI")
    print(f"  Train : {len(splits['train'])} gambar")
    print(f"  Val   : {len(splits['val'])} gambar")
    print("  Objek per kelas:")
    for i, n in enumerate(CLASS_NAMES):
        print(f"     {i} {n:<20}: {stats.get(f'obj_{n}', 0)}")
    skipped = {k: v for k, v in stats.items() if k.startswith("skip_")}
    if skipped:
        print("  Kelas RDD di-skip (tidak dipetakan):")
        for k, v in sorted(skipped.items()):
            print(f"     {k[5:]:<20}: {v}")
    for key in ("images_no_valid_obj", "image_missing", "xml_parse_error",
                "degenerate_bbox", "bad_bndbox"):
        if stats.get(key):
            print(f"  {key}: {stats[key]}")
    print(f"  data.yaml -> {yaml_path}")
    print("-" * 60)
    print("Lanjut latih:  python train.py --data", yaml_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
