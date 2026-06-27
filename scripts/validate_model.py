#!/usr/bin/env python3
"""
validate_model.py — Validasi akurasi model SEBELUM dipakai produksi.

Brief menegaskan: jangan anggap model generik/RDD2022 pasti akurat di jalan
Luwu Timur. Skrip ini dua mode:

  1. METRIK (butuh label):  hitung mAP/precision/recall per kelas pada val set
     berlabel (data.yaml). Cara objektif mengukur akurasi.
        python scripts/validate_model.py --model best.pt --data datasets/rdd_yolo/data.yaml

  2. REVIEW VISUAL (tanpa label): jalankan model pada folder footage lokal lo,
     simpan gambar beranotasi + ringkasan jumlah deteksi per kelas, untuk
     diperiksa mata sebelum produksi.
        python scripts/validate_model.py --model best.pt --source footage_luwu/

Pakai mode 2 pada footage drone DJI / video HP dari jalan Luwu Timur untuk
melihat apakah model benar-benar menangkap lubang/retak SETEMPAT.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
VID_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def run_metrics(model, data: str, imgsz: int, device: str) -> None:
    print("== MODE METRIK (val set berlabel) ==")
    metrics = model.val(data=data, imgsz=imgsz, device=device, verbose=False)
    # ultralytics DetMetrics
    box = metrics.box
    names = model.names
    print(f"\n  mAP@0.5      : {box.map50:.3f}")
    print(f"  mAP@0.5:0.95 : {box.map:.3f}")
    print(f"  Precision    : {box.mp:.3f}")
    print(f"  Recall       : {box.mr:.3f}")
    print("\n  Per kelas (AP@0.5):")
    try:
        for i, ap in zip(box.ap_class_index, box.ap50):
            print(f"     {names.get(int(i), i):<22}: {ap:.3f}")
    except Exception:
        pass
    print("\n  Interpretasi kasar: mAP@0.5 > 0.5 = lumayan; > 0.7 = bagus untuk")
    print("  kondisi jalan. Bila rendah di kelas tertentu, tambah data lokal kelas itu.")


def run_visual(model, source: Path, out: Path, conf: float,
               imgsz: int, device: str, stride: int) -> None:
    import cv2
    print("== MODE REVIEW VISUAL (footage lokal, tanpa label) ==")
    out.mkdir(parents=True, exist_ok=True)
    annotated_dir = out / "annotated"
    annotated_dir.mkdir(exist_ok=True)
    names = model.names

    # Kumpulkan sumber: file gambar/video atau folder.
    sources: list[Path] = []
    if source.is_dir():
        for ext in IMG_EXTS + VID_EXTS:
            sources += sorted(source.rglob(f"*{ext}"))
    else:
        sources = [source]
    if not sources:
        sys.exit(f"Tidak ada gambar/video di {source}")

    counts = Counter()
    n_frames = 0
    csv_path = out / "validation_detections.csv"
    cf = open(csv_path, "w", newline="", encoding="utf-8")
    w = csv.writer(cf)
    w.writerow(["source", "frame", "class", "confidence", "x1", "y1", "x2", "y2"])

    for src in sources:
        is_video = src.suffix.lower() in VID_EXTS
        if is_video:
            cap = cv2.VideoCapture(str(src))
            fi = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                fi += 1
                if stride > 1 and fi % stride:
                    continue
                _infer_draw(model, frame, src.name, fi, conf, imgsz, device,
                            names, counts, w, annotated_dir, cv2)
                n_frames += 1
            cap.release()
        else:
            frame = cv2.imread(str(src))
            if frame is None:
                continue
            _infer_draw(model, frame, src.name, 0, conf, imgsz, device,
                        names, counts, w, annotated_dir, cv2)
            n_frames += 1
    cf.close()

    print(f"\n  Diproses          : {n_frames} frame/gambar dari {len(sources)} sumber")
    print(f"  Total deteksi     : {sum(counts.values())}")
    print("  Deteksi per kelas :")
    for cls, n in counts.most_common():
        print(f"     {cls:<22}: {n}")
    print(f"\n  Gambar beranotasi : {annotated_dir}")
    print(f"  CSV               : {csv_path}")
    print("\n  >> Buka folder 'annotated' dan periksa MATA: apakah kotak benar di")
    print("     lubang/retak asli? Banyak false positive/negative = perlu fine-tune.")


def _infer_draw(model, frame, src_name, fi, conf, imgsz, device, names,
                counts, writer, annotated_dir, cv2):
    res = model.predict(frame, conf=conf, imgsz=imgsz, device=device,
                        verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return
    for b in boxes:
        cid = int(b.cls[0]); cf_ = float(b.conf[0])
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        cname = names.get(cid, str(cid))
        counts[cname] += 1
        writer.writerow([src_name, fi, cname, f"{cf_:.3f}", x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{cname} {cf_:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    fn = f"{Path(src_name).stem}_f{fi}.jpg"
    cv2.imwrite(str(annotated_dir / fn), frame)


def resolve_device(req: str) -> str:
    if req != "auto":
        return req
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Validasi akurasi model kerusakan jalan")
    p.add_argument("--model", required=True, help="Path bobot .pt")
    p.add_argument("--data", help="data.yaml dengan val set berlabel (mode metrik)")
    p.add_argument("--source", help="Folder/gambar/video footage lokal (mode visual)")
    p.add_argument("--out", default="output/validation", help="Folder hasil review visual")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--stride", type=int, default=10, help="Proses tiap N frame video.")
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.data and not args.source:
        sys.exit("Pilih salah satu: --data (metrik) atau --source (review visual).")
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ERROR: ultralytics belum terpasang.")
    device = resolve_device(args.device)
    model = YOLO(args.model)
    print(f"Model: {args.model} | kelas: {list(model.names.values())} | device: {device}\n")

    if args.data:
        run_metrics(model, args.data, args.imgsz, device)
    if args.source:
        run_visual(model, Path(args.source), Path(args.out), args.conf,
                   args.imgsz, device, args.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
