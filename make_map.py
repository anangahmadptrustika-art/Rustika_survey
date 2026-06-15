#!/usr/bin/env python3
"""
make_map.py — Peta titik kerusakan jalan dari detections.csv (Fase 3).

Menghasilkan:
  • <sesi>/map.html        peta interaktif (Folium/Leaflet): marker per kerusakan,
                           warna per jenis, popup (jenis, confidence, waktu, foto),
                           plus garis rute survei.
  • <sesi>/detections.geojson  ekspor GeoJSON untuk QGIS / Google Earth (offline).

Catatan offline: file map.html berdiri sendiri (buka dgn klik dua kali, tanpa
server). Marker/popup/rute SEMUA tertanam → tampil offline. LATAR peta (tile
jalan) butuh internet untuk render; tanpa internet, marker tetap muncul di latar
abu. Untuk peta offline penuh, pakai GeoJSON di QGIS dengan basemap lokal.

Pakai mandiri:
  python make_map.py output/survey_20260615_120000/detections.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Warna per kelas (hex) — konsisten untuk laporan.
CLASS_COLORS = {
    "longitudinal_crack": "#1f78b4",
    "transverse_crack": "#33a02c",
    "alligator_crack": "#ff7f00",
    "pothole": "#e31a1c",
}
DEFAULT_COLOR = "#6a3d9a"


def read_rows(csv_path: Path) -> list[dict]:
    """Baca baris CSV yang punya koordinat valid."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            lat, lon = d.get("lat", ""), d.get("lon", "")
            if lat in ("", None) or lon in ("", None):
                continue
            try:
                d["_lat"] = float(lat); d["_lon"] = float(lon)
            except ValueError:
                continue
            rows.append(d)
    return rows


def write_geojson(rows: list[dict], out_path: Path) -> None:
    features = []
    for d in rows:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [d["_lon"], d["_lat"]]},
            "properties": {
                "class": d.get("class_name", ""),
                "confidence": d.get("confidence", ""),
                "timestamp": d.get("timestamp_iso", ""),
                "frame": d.get("frame_idx", ""),
                "screenshot": d.get("screenshot", ""),
                "bbox_area_px": d.get("area_px", ""),
            },
        })
    fc = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(fc, indent=1), encoding="utf-8")


def _legend_html(counts: dict) -> str:
    items = "".join(
        f'<div><span style="display:inline-block;width:12px;height:12px;'
        f'background:{CLASS_COLORS.get(c, DEFAULT_COLOR)};margin-right:6px;'
        f'border-radius:50%"></span>{c} ({n})</div>'
        for c, n in sorted(counts.items())
    )
    return (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;'
        'background:white;padding:10px 12px;border:1px solid #999;'
        'border-radius:6px;font:13px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)">'
        '<b>Kerusakan Jalan</b>' + items + '</div>'
    )


def build_from_csv(csv_path, out_dir, screenshots: bool = True) -> Path | None:
    """Bangun map.html + geojson. Return path map.html (atau None bila folium absen)."""
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    rows = read_rows(csv_path)
    if not rows:
        raise SystemExit("tidak ada baris ber-koordinat di CSV.")

    # GeoJSON selalu dibuat (tanpa dependensi).
    geojson_path = out_dir / "detections.geojson"
    write_geojson(rows, geojson_path)

    counts: dict[str, int] = {}
    for d in rows:
        counts[d.get("class_name", "?")] = counts.get(d.get("class_name", "?"), 0) + 1

    try:
        import folium
    except ImportError:
        print("  Peta  : folium belum terpasang (pip install folium). "
              f"GeoJSON tetap dibuat -> {geojson_path}")
        return None

    lats = [d["_lat"] for d in rows]; lons = [d["_lon"] for d in rows]
    center = [sum(lats) / len(lats), sum(lons) / len(lons)]
    fmap = folium.Map(location=center, zoom_start=16, control_scale=True,
                      tiles="OpenStreetMap")

    # Layer rute survei (urut waktu).
    try:
        ordered = sorted(rows, key=lambda d: float(d.get("epoch_s", 0) or 0))
        track = [[d["_lat"], d["_lon"]] for d in ordered]
        if len(track) > 1:
            fg_track = folium.FeatureGroup(name="Rute survei", show=True)
            folium.PolyLine(track, color="#555", weight=2, opacity=0.6).add_to(fg_track)
            fg_track.add_to(fmap)
    except Exception:
        pass

    # Layer marker per kelas.
    groups: dict[str, folium.FeatureGroup] = {}
    for d in rows:
        cls = d.get("class_name", "?")
        if cls not in groups:
            groups[cls] = folium.FeatureGroup(name=f"{cls} ({counts[cls]})", show=True)
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        shot = d.get("screenshot", "")
        popup = (f"<b>{cls}</b><br>conf: {d.get('confidence','')}<br>"
                 f"waktu: {d.get('timestamp_iso','')}<br>frame: {d.get('frame_idx','')}<br>"
                 f"{d['_lat']:.6f}, {d['_lon']:.6f}")
        if screenshots and shot:
            popup += f'<br><img src="screenshots/{shot}" width="220">'
        folium.CircleMarker(
            location=[d["_lat"], d["_lon"]], radius=6, color=color,
            fill=True, fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(popup, max_width=260),
            tooltip=cls,
        ).add_to(groups[cls])
    for g in groups.values():
        g.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(_legend_html(counts)))
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    map_path = out_dir / "map.html"
    fmap.save(str(map_path))
    print(f"  Peta  : {map_path}  ({len(rows)} titik)")
    print(f"  GeoJSON: {geojson_path}")
    print("  Ringkasan titik per kelas:")
    for c, n in sorted(counts.items()):
        print(f"     - {c:<22}: {n}")
    return map_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Peta titik kerusakan dari detections.csv")
    p.add_argument("csv", help="Path detections.csv")
    p.add_argument("--out", default=None, help="Folder output (default: folder CSV)")
    p.add_argument("--no-screenshots", action="store_true",
                   help="Jangan sematkan foto di popup.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV tidak ditemukan: {csv_path}")
    out_dir = Path(args.out) if args.out else csv_path.parent
    build_from_csv(csv_path, out_dir, screenshots=not args.no_screenshots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
