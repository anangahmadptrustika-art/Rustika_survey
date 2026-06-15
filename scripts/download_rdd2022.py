#!/usr/bin/env python3
"""
download_rdd2022.py — Unduh dataset RDD2022 (Road Damage Dataset 2022).

RDD2022 dirilis oleh sekilab (University of Tokyo) — berisi >47k gambar jalan
beranotasi dari beberapa negara. Anotasi: PASCAL VOC XML (pakai voc_to_yolo.py
untuk konversi ke YOLO).

⚠️ PENTING:
  • Dataset BESAR (per negara ratusan MB; total beberapa GB). Unduh di tempat
    ber-internet stabil (BUKAN di lapangan).
  • Isinya jalan LUAR NEGERI (Jepang, India, Ceko, dll) — BUKAN Indonesia.
    Ini titik awal yang bagus, tapi WAJIB di-fine-tune dgn data lokal Luwu Timur
    (lihat docs/LABELING.md) sebelum dipakai produksi.
  • URL mengikuti rilis publik sekilab. Bila berubah, cek repo resmi:
    https://github.com/sekilab/RoadDamageDetector

Contoh:
  python scripts/download_rdd2022.py --country Japan India --out datasets/RDD2022
  python scripts/download_rdd2022.py --all --out datasets/RDD2022
  python scripts/download_rdd2022.py --list
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = ("https://mycityreport.s3-ap-northeast-1.amazonaws.com/"
        "02_RoadDamageDataset/public_data/RDD2022/Country_specific_data")

# Nama negara -> nama file zip RDD2022 resmi.
COUNTRIES = {
    "Japan": "RDD2022_Japan.zip",
    "India": "RDD2022_India.zip",
    "Czech": "RDD2022_Czech.zip",
    "Norway": "RDD2022_Norway.zip",
    "United_States": "RDD2022_United_States.zip",
    "China_Drone": "RDD2022_China_Drone.zip",
    "China_MotorBike": "RDD2022_China_MotorBike.zip",
}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(url: str, dst: Path) -> bool:
    """Unduh dengan progress + resume sederhana."""
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  sudah ada, lewati: {dst.name} ({human(dst.stat().st_size)})")
        return True
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  unduh: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        print(f"\r    {pct:5.1f}%  {human(done)}/{human(total)}",
                              end="", flush=True)
            print()
        tmp.rename(dst)
        return True
    except Exception as e:
        print(f"\n  GAGAL unduh {dst.name}: {e}")
        print("  (Jika di container/host dengan egress allowlist, jalankan ini di "
              "laptop lo yang internetnya normal.)")
        if tmp.exists():
            tmp.unlink()
        return False


def extract(zip_path: Path, out_dir: Path) -> None:
    print(f"  ekstrak: {zip_path.name} -> {out_dir}")
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
    except zipfile.BadZipFile:
        print(f"  ! {zip_path.name} bukan zip valid (unduhan rusak?). Hapus & ulangi.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Unduh dataset RDD2022")
    p.add_argument("--country", nargs="+", choices=list(COUNTRIES),
                   help="Negara yang diunduh (mis. Japan India).")
    p.add_argument("--all", action="store_true", help="Unduh semua negara.")
    p.add_argument("--out", default="datasets/RDD2022", help="Folder tujuan.")
    p.add_argument("--no-extract", action="store_true", help="Jangan auto-ekstrak.")
    p.add_argument("--list", action="store_true", help="Tampilkan daftar negara.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list:
        print("Negara RDD2022 yang tersedia:")
        for c, f in COUNTRIES.items():
            print(f"  {c:<16} {BASE}/{f}")
        return 0

    countries = list(COUNTRIES) if args.all else (args.country or [])
    if not countries:
        sys.exit("Pilih --country <Negara...> atau --all. Lihat --list.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Tujuan: {out.resolve()}")
    print("CATATAN: RDD2022 = jalan luar negeri. Fine-tune dgn data lokal "
          "(docs/LABELING.md) sebelum produksi.\n")

    ok = 0
    for c in countries:
        print(f"== {c} ==")
        zip_path = out / COUNTRIES[c]
        if download(f"{BASE}/{COUNTRIES[c]}", zip_path):
            ok += 1
            if not args.no_extract:
                extract(zip_path, out)
    print("-" * 60)
    print(f"Selesai: {ok}/{len(countries)} negara berhasil.")
    if ok:
        print("Lanjut konversi ke YOLO:")
        print(f"  python scripts/voc_to_yolo.py --input {out}/<Negara> "
              f"--out datasets/rdd_yolo")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
