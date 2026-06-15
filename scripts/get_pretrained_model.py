#!/usr/bin/env python3
"""
get_pretrained_model.py — Jalur A: ambil model pre-trained pothole/road-damage.

Tujuan: validasi pipeline CEPAT dengan model jadi, sebelum training serius
(jalur B). Setelah dapat .pt, langsung:
    python road_survey.py --source <video> --model models/road_damage.pt --conf 0.35

⚠️ Jalankan di mesin ber-internet NORMAL. Banyak sumber model (HuggingFace,
Roboflow) DIBLOKIR di lingkungan dengan egress allowlist (mis. container CI) —
jadi unduh di laptop lo, lalu copy .pt-nya ke folder models/.

Tiga sumber didukung:

  1) HuggingFace Hub (butuh `pip install huggingface_hub`):
       python scripts/get_pretrained_model.py hf --repo <user/model> --file best.pt

  2) URL langsung (.pt dari GitHub release / mana saja):
       python scripts/get_pretrained_model.py url --url https://.../best.pt

  3) Roboflow (butuh `pip install roboflow` + API key gratis):
       python scripts/get_pretrained_model.py roboflow --api-key KEY \
            --workspace WS --project PROJ --version 1
     (Roboflow mengekspor DATASET; latih cepat dgn train.py, atau pakai bobot
      yang mereka sediakan bila ada.)

Saran tempat cari model (buka di browser, cek lisensi):
  • HuggingFace: cari "pothole yolov8", "road damage RDD yolov8"
  • Roboflow Universe: universe.roboflow.com -> "pothole" / "road damage"
  • GitHub: repo RDD2022 yang melampirkan best.pt di Releases
PENTING: model pre-trained ini DILATIH DI JALAN LUAR — validasi dulu pakai
footage Luwu Timur (scripts/validate_model.py) sebelum percaya.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


def save_dir() -> Path:
    d = Path("models")
    d.mkdir(exist_ok=True)
    return d


def from_url(url: str, out: Path) -> Path:
    print(f"Unduh: {url}")
    try:
        urllib.request.urlretrieve(url, out)
    except Exception as e:
        sys.exit(f"GAGAL: {e}\n(Cek koneksi / egress allowlist. Jalankan di laptop "
                 "ber-internet normal.)")
    print(f"Tersimpan -> {out}")
    return out


def from_hf(repo: str, file: str, out: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("Butuh: pip install huggingface_hub")
    print(f"HuggingFace: {repo} :: {file}")
    try:
        path = hf_hub_download(repo_id=repo, filename=file)
    except Exception as e:
        sys.exit(f"GAGAL: {e}")
    import shutil
    shutil.copy2(path, out)
    print(f"Tersimpan -> {out}")
    return out


def from_roboflow(api_key, workspace, project, version, out_dir: Path) -> None:
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("Butuh: pip install roboflow")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)
    print("Mengekspor dataset Roboflow (format yolov8) ...")
    ds = ver.download("yolov8", location=str(out_dir / "roboflow_dataset"))
    print(f"Dataset -> {ds.location}")
    print("Roboflow Universe umumnya memberi DATASET, bukan bobot .pt langsung.")
    print("Latih cepat:")
    print(f"  python train.py --data {ds.location}/data.yaml --model yolov8s.pt "
          f"--epochs 50 --device 0")


def verify(pt: Path) -> None:
    """Muat model & tampilkan kelasnya supaya yakin formatnya benar."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("(ultralytics belum ada — lewati verifikasi muat model.)")
        return
    try:
        m = YOLO(str(pt))
        print(f"OK. Model termuat. Kelas: {list(m.names.values())}")
    except Exception as e:
        print(f"PERINGATAN: gagal memuat {pt}: {e}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ambil model pre-trained road-damage")
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("url", help="Unduh .pt dari URL langsung")
    u.add_argument("--url", required=True)
    u.add_argument("--name", default="road_damage.pt")

    h = sub.add_parser("hf", help="Unduh dari HuggingFace Hub")
    h.add_argument("--repo", required=True)
    h.add_argument("--file", default="best.pt")
    h.add_argument("--name", default="road_damage.pt")

    r = sub.add_parser("roboflow", help="Ekspor dataset/bobot dari Roboflow")
    r.add_argument("--api-key", required=True)
    r.add_argument("--workspace", required=True)
    r.add_argument("--project", required=True)
    r.add_argument("--version", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    d = save_dir()
    if args.cmd == "url":
        pt = from_url(args.url, d / args.name); verify(pt)
    elif args.cmd == "hf":
        pt = from_hf(args.repo, args.file, d / args.name); verify(pt)
    elif args.cmd == "roboflow":
        from_roboflow(args.api_key, args.workspace, args.project, args.version, d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
