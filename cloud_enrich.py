#!/usr/bin/env python3
"""
cloud_enrich.py — (Opsional, Hybrid) perkaya laporan dgn Claude vision.

PENTING — batasan desain:
  Deteksi real-time TETAP 100% LOKAL (road_survey.py). Skrip ini TIDAK ikut di
  loop per-frame. Ia hanya dijalankan SETELAH survey, saat ada internet, pada
  FOTO kerusakan yang sudah ter-capture (puluhan, bukan ribuan frame) — untuk:
    - second-opinion tingkat keparahan (Ringan/Sedang/Berat)
    - deskripsi singkat tiap kerusakan untuk narasi laporan
  Jadi hemat biaya & tetap offline-first di lapangan.

Output: <sesi>/cloud_assessment.csv (screenshot, severity, jenis, deskripsi).
Bisa dipakai melengkapi laporan PDF/Excel.

Prasyarat (di laptop ber-internet):
  pip install anthropic
  set ANTHROPIC_API_KEY=...        (Windows: setx ANTHROPIC_API_KEY "...")

Contoh:
  python cloud_enrich.py output/survey_20260615_120000 --max-images 30
  python cloud_enrich.py output/survey_xxx --dry-run     # cek tanpa panggil API

Catatan biaya: default model claude-opus-4-8. Untuk lebih murah, pakai
--model claude-haiku-4-5 (lebih hemat untuk klasifikasi gambar).
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path

# Skema keluaran terstruktur (dipaksa lewat output_config.format).
RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_road_damage": {"type": "boolean"},
        "damage_type": {
            "type": "string",
            "enum": ["lubang", "retak_memanjang", "retak_melintang",
                     "retak_buaya", "lainnya", "tidak_ada"],
        },
        "severity": {"type": "string", "enum": ["Ringan", "Sedang", "Berat"]},
        "description": {"type": "string"},
        "confidence": {"type": "string", "enum": ["rendah", "sedang", "tinggi"]},
    },
    "required": ["is_road_damage", "damage_type", "severity",
                 "description", "confidence"],
    "additionalProperties": False,
}

PROMPT = (
    "Foto ini diambil saat survey kondisi jalan oleh konsultan teknik sipil di "
    "Kabupaten Luwu Timur. Kotak merah menandai dugaan kerusakan dari model AI "
    "lokal. Tugasmu memberi penilaian KEDUA (verifikasi manusia-pakar):\n"
    "- Apakah benar ada kerusakan jalan di kotak itu?\n"
    "- Jenis kerusakan (lubang / retak memanjang / melintang / buaya / lainnya).\n"
    "- Tingkat keparahan: Ringan / Sedang / Berat.\n"
    "- Deskripsi singkat (1 kalimat, Bahasa Indonesia) untuk dimasukkan ke laporan.\n"
    "Jika sebenarnya bukan kerusakan (mis. bayangan, tambalan, marka), tandai "
    "is_road_damage=false dan jelaskan."
)

MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp", ".gif": "image/gif"}


def collect_images(session_dir: Path, limit: int) -> list[Path]:
    shots = session_dir / "screenshots"
    if not shots.is_dir():
        sys.exit(f"Folder screenshots tidak ada: {shots}")
    imgs = sorted(p for p in shots.iterdir()
                  if p.suffix.lower() in MEDIA and not p.name.startswith("manual_"))
    return imgs[:limit] if limit else imgs


def encode_image(path: Path) -> tuple[str, str]:
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return MEDIA[path.suffix.lower()], data


def assess_one(client, model: str, path: Path) -> dict:
    media_type, data = encode_image(path)
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": media_type, "data": data}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        output_config={"format": {"type": "json_schema", "schema": RESULT_SCHEMA}},
    )
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    return json.loads(text)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Perkaya laporan dgn Claude vision (opsional)")
    p.add_argument("session_dir", help="Folder sesi survey (berisi screenshots/)")
    p.add_argument("--model", default="claude-opus-4-8",
                   help="Model Claude (default opus; haiku lebih hemat).")
    p.add_argument("--max-images", type=int, default=40,
                   help="Batas jumlah foto yang dinilai (hemat biaya). 0 = semua.")
    p.add_argument("--dry-run", action="store_true",
                   help="Tampilkan rencana tanpa memanggil API (uji tanpa internet/key).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    session_dir = Path(args.session_dir)
    imgs = collect_images(session_dir, args.max_images)
    print(f"Foto kerusakan yang akan dinilai: {len(imgs)} (model: {args.model})")

    if args.dry_run:
        for p in imgs:
            print(f"  [dry-run] {p.name}")
        print("Dry-run: tidak ada panggilan API. Hapus --dry-run untuk menjalankan.")
        return 0

    try:
        import anthropic
    except ImportError:
        sys.exit("Butuh: pip install anthropic")

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY belum di-set. Lihat docstring (jalankan di "
                 "laptop ber-internet).")

    client = anthropic.Anthropic()
    out_csv = session_dir / "cloud_assessment.csv"
    rows = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["screenshot", "is_road_damage", "damage_type",
                    "severity", "confidence", "description"])
        for p in imgs:
            try:
                r = assess_one(client, args.model, p)
                w.writerow([p.name, r.get("is_road_damage"), r.get("damage_type"),
                            r.get("severity"), r.get("confidence"),
                            r.get("description")])
                rows += 1
                print(f"  ✓ {p.name}: {r.get('severity')} — {r.get('damage_type')}")
            except anthropic.APIStatusError as e:
                print(f"  ✗ {p.name}: API error {e.status_code} ({e.type})")
            except anthropic.APIConnectionError:
                print(f"  ✗ {p.name}: koneksi gagal (offline?) — hentikan.")
                break
            except Exception as e:
                print(f"  ✗ {p.name}: {e}")
    print(f"Selesai. {rows} foto dinilai -> {out_csv}")
    print("Hasil ini bisa dilampirkan/ditinjau bersama laporan PDF/Excel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
