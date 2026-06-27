#!/usr/bin/env python3
"""
train.py — Latih / fine-tune YOLOv8 untuk deteksi kerusakan jalan (Fase 2B).

Melatih model dari `data.yaml` (hasil voc_to_yolo.py untuk RDD2022, dan/atau
data lokal Luwu Timur). Sadar lingkungan: mendeteksi GPU (CUDA) dan memberi tahu
apakah training lokal realistis atau sebaiknya pakai Google Colab.

Alur yang disarankan:
  1. (opsional) Mulai dari bobot pre-trained jalur A sebagai titik awal lebih baik.
  2. Latih di RDD2022 dulu (model dasar kerusakan jalan).
  3. Fine-tune di data lokal Luwu Timur (akurasi untuk jalan lo).

Contoh:
  # Training penuh di RDD2022 (GPU lokal)
  python train.py --data datasets/rdd_yolo/data.yaml --model yolov8s.pt --epochs 100

  # Fine-tune model RDD dengan data lokal (lebih sedikit epoch)
  python train.py --data datasets/luwu_yolo/data.yaml \
        --model runs/detect/rdd/weights/best.pt --epochs 50 --name luwu_finetune

  # Cek kelayakan lingkungan saja (tanpa training)
  python train.py --data datasets/rdd_yolo/data.yaml --check-only
"""
from __future__ import annotations

import argparse
import sys


def check_environment() -> dict:
    info = {"device": "cpu", "device_name": "CPU", "cuda": False, "vram_gb": 0.0}
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch belum terpasang. Lihat README (install CUDA).")
        return info
    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        info["cuda"] = True
        info["device"] = "0"
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        info["device"] = "mps"
        info["device_name"] = "Apple Silicon (MPS)"
    return info


def advise(info: dict, epochs: int) -> None:
    print("=" * 64)
    print(" CEK KELAYAKAN TRAINING")
    print("=" * 64)
    print(f"  PyTorch : {info.get('torch', '-')}")
    print(f"  Device  : {info['device_name']}")
    if info["cuda"]:
        print(f"  VRAM    : {info['vram_gb']} GB")
        print("  -> GPU NVIDIA terdeteksi. Training LOKAL realistis. ✅")
        if info["vram_gb"] < 6:
            print("  CATATAN : VRAM < 6 GB -> pakai --model yolov8n/s, --batch kecil"
                  " (mis. 8/16) atau --imgsz 512.")
        print(f"  Estimasi: {epochs} epoch RDD2022 di GPU desktop ~ beberapa jam.")
    elif info["device"] == "mps":
        print("  -> Apple Silicon (MPS). Bisa, tapi lambat untuk training penuh.")
        print("     Saran: untuk training berat pakai Google Colab (lihat notebooks/).")
    else:
        print("  -> TIDAK ada GPU. Training penuh di CPU TIDAK realistis")
        print("     (bisa berhari-hari). REKOMENDASI:")
        print("       • Pakai Google Colab gratis (GPU T4): notebooks/train_colab.ipynb")
        print("       • Atau latih di mesin ber-GPU NVIDIA.")
    print("=" * 64)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Latih YOLOv8 deteksi kerusakan jalan")
    p.add_argument("--data", required=True, help="Path data.yaml")
    p.add_argument("--model", default="yolov8s.pt",
                   help="Bobot awal (yolov8n/s/m.pt) atau best.pt untuk fine-tune.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=-1,
                   help="Ukuran batch (-1 = auto sesuai VRAM).")
    p.add_argument("--device", default="auto", help="auto|0|cpu|mps")
    p.add_argument("--project", default=None,
                   help="Folder project khusus. Default: runs/detect (konvensi ultralytics).")
    p.add_argument("--name", default="road_damage")
    p.add_argument("--patience", type=int, default=30,
                   help="Early stopping: berhenti bila tak membaik N epoch.")
    p.add_argument("--check-only", action="store_true",
                   help="Hanya cek kelayakan lingkungan, jangan latih.")
    p.add_argument("--resume", action="store_true", help="Lanjut training terakhir.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    info = check_environment()
    advise(info, args.epochs)

    if args.check_only:
        print("(--check-only) Selesai tanpa training.")
        return 0

    if not info["cuda"] and info["device"] != "mps" and args.device == "auto":
        print("\nPERINGATAN: training akan jalan di CPU (sangat lambat).")
        print("Tambahkan --device cpu secara eksplisit kalau memang sengaja "
              "(mis. smoke-test kecil), atau pakai Colab untuk training nyata.")
        # Tetap lanjut hanya bila device dipaksa.
        if args.device == "auto":
            print("Dibatalkan. Jalankan ulang dengan --device cpu bila yakin.")
            return 1

    device = info["device"] if args.device == "auto" else args.device

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ERROR: ultralytics belum terpasang (pip install ultralytics).")

    print(f"\nMulai training: model={args.model} data={args.data} device={device}")
    model = YOLO(args.model)
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        # Augmentasi ringan; kondisi jalan tropis Indonesia beda dgn RDD2022.
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, mosaic=1.0,
    )
    if args.project:                       # hanya kirim bila user set (hindari path dobel)
        train_kwargs["project"] = args.project
    results = model.train(**train_kwargs)
    save_dir = getattr(results, "save_dir", f"runs/detect/{args.name}")
    print("-" * 60)
    print("TRAINING SELESAI.")
    print(f"  Bobot terbaik : {save_dir}/weights/best.pt")
    print(f"  Pakai di app  : python road_survey.py --source <video> "
          f"--model {save_dir}/weights/best.pt")
    print(f"  Validasi      : python scripts/validate_model.py "
          f"--model {save_dir}/weights/best.pt --data {args.data}")
    print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
