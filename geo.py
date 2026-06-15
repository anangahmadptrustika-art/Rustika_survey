#!/usr/bin/env python3
"""
geo.py — Sumber lokasi GPS untuk Road Survey AI (Fase 3).

Menyediakan beberapa "provider" GPS dengan antarmuka seragam, dipakai
road_survey.py untuk memberi koordinat lat/lon pada tiap deteksi kerusakan.

Sumber yang didukung (spec string di --gps):
  srt:FILE.SRT          telemetry drone DJI (lat/lon per-frame, dari footage)
  gpx:TRACK.GPX         track GPS dari HP (rekam saat survey), cocokkan via waktu
  nmea:PORT@BAUD        USB GPS dongle (NMEA via serial), mis. nmea:COM3@4800
  tcp:HOST:PORT         HP yang broadcast NMEA via TCP (mis. app "GPS 2 IP")
  fixed:LAT,LON         satu titik tetap (uji / lokasi statis)
  auto                  bila source file, cari FILE.SRT di sebelahnya
  none                  tanpa GPS

Provider file-based (srt/gpx) di-query dgn waktu video (detik); provider live
(nmea/tcp) mengembalikan fix terkini. Antarmuka tunggal: get_fix(video_time).
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GpsFix:
    lat: float
    lon: float
    alt: Optional[float] = None


# --------------------------------------------------------------------------- #
# Util konversi NMEA (ddmm.mmmm -> derajat desimal)
# --------------------------------------------------------------------------- #
def _nmea_to_deg(val: str, hemi: str) -> Optional[float]:
    if not val:
        return None
    try:
        v = float(val)
    except ValueError:
        return None
    deg = int(v // 100)
    minutes = v - deg * 100
    dec = deg + minutes / 60.0
    if hemi in ("S", "W"):
        dec = -dec
    return dec


def parse_nmea_line(line: str) -> Optional[GpsFix]:
    """Parse satu kalimat NMEA (GGA/RMC) -> GpsFix. None bila tak valid/no-fix."""
    line = line.strip()
    if not line.startswith("$"):
        return None
    parts = line.split(",")
    sentence = parts[0][-3:]  # GGA / RMC / GLL
    try:
        if sentence == "GGA" and len(parts) >= 6:
            # $..GGA,time,lat,N,lon,E,fixqual,...
            if parts[6] in ("", "0"):
                return None  # no fix
            lat = _nmea_to_deg(parts[2], parts[3])
            lon = _nmea_to_deg(parts[4], parts[5])
            alt = float(parts[9]) if len(parts) > 9 and parts[9] else None
            if lat is not None and lon is not None:
                return GpsFix(lat, lon, alt)
        elif sentence == "RMC" and len(parts) >= 7:
            # $..RMC,time,status,lat,N,lon,E,...
            if parts[2] != "A":
                return None  # void
            lat = _nmea_to_deg(parts[3], parts[4])
            lon = _nmea_to_deg(parts[5], parts[6])
            if lat is not None and lon is not None:
                return GpsFix(lat, lon)
        elif sentence == "GLL" and len(parts) >= 6:
            if len(parts) > 6 and parts[6] == "V":
                return None
            lat = _nmea_to_deg(parts[1], parts[2])
            lon = _nmea_to_deg(parts[3], parts[4])
            if lat is not None and lon is not None:
                return GpsFix(lat, lon)
    except (ValueError, IndexError):
        return None
    return None


# --------------------------------------------------------------------------- #
# Provider dasar
# --------------------------------------------------------------------------- #
class GpsProvider:
    kind = "base"

    def get_fix(self, video_time: float | None = None) -> Optional[GpsFix]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FixedProvider(GpsProvider):
    kind = "fixed"

    def __init__(self, lat: float, lon: float):
        self._fix = GpsFix(lat, lon)

    def get_fix(self, video_time=None) -> Optional[GpsFix]:
        return self._fix


# --------------------------------------------------------------------------- #
# DJI .SRT (footage drone)
# --------------------------------------------------------------------------- #
_TS = re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s*-->")
_LAT = re.compile(r"latitude\s*[:=]\s*([-\d.]+)", re.I)
_LON = re.compile(r"longitude\s*[:=]\s*([-\d.]+)", re.I)
_ALT = re.compile(r"abs_alt\s*[:=]\s*([-\d.]+)", re.I)
# Format DJI lama: GPS(lon,lat,sats)
_GPS_OLD = re.compile(r"GPS\(\s*([-\d.]+)\s*,\s*([-\d.]+)", re.I)


class SrtProvider(GpsProvider):
    """Parse telemetry GPS dari file .SRT drone DJI, indeks per waktu video."""
    kind = "srt"

    def __init__(self, srt_path: str | Path):
        self.entries: list[tuple[float, GpsFix]] = []  # (start_seconds, fix)
        self._load(Path(srt_path))

    def _load(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Pisahkan per blok subtitle (dipisah baris kosong ganda).
        blocks = re.split(r"\n\s*\n", text)
        for blk in blocks:
            mt = _TS.search(blk)
            if not mt:
                continue
            h, m, s, ms = (int(x) for x in mt.groups())
            t = h * 3600 + m * 60 + s + ms / 1000.0
            fix = self._extract_fix(blk)
            if fix:
                self.entries.append((t, fix))
        self.entries.sort(key=lambda e: e[0])

    @staticmethod
    def _extract_fix(blk: str) -> Optional[GpsFix]:
        mla, mlo = _LAT.search(blk), _LON.search(blk)
        if mla and mlo:
            alt = _ALT.search(blk)
            return GpsFix(float(mla.group(1)), float(mlo.group(1)),
                          float(alt.group(1)) if alt else None)
        mo = _GPS_OLD.search(blk)
        if mo:  # GPS(lon,lat,sats)
            return GpsFix(float(mo.group(2)), float(mo.group(1)))
        return None

    def get_fix(self, video_time: float | None = None) -> Optional[GpsFix]:
        if not self.entries:
            return None
        if video_time is None:
            return self.entries[0][1]
        # Cari entri dengan start_time <= video_time terdekat (binary-ish).
        best = self.entries[0][1]
        for t, fix in self.entries:
            if t <= video_time:
                best = fix
            else:
                break
        return best


# --------------------------------------------------------------------------- #
# GPX (track HP), dicocokkan via waktu
# --------------------------------------------------------------------------- #
class GpxProvider(GpsProvider):
    """Track GPS dari HP. Cocokkan waktu video (relatif start) ke trackpoint."""
    kind = "gpx"

    def __init__(self, gpx_path: str | Path, video_start_epoch: float | None = None):
        self.points: list[tuple[float, GpsFix]] = []  # (epoch_seconds, fix)
        self._load(Path(gpx_path))
        # Titik awal video pada skala waktu GPX (epoch). Bila None, pakai titik
        # pertama GPX (asumsi rekaman mulai bersamaan).
        self.start = video_start_epoch if video_start_epoch is not None else (
            self.points[0][0] if self.points else 0.0)

    def _load(self, path: Path) -> None:
        import xml.etree.ElementTree as ET
        from datetime import datetime, timezone
        ns = {"g": "http://www.topografix.com/GPX/1/1"}
        root = ET.parse(path).getroot()
        # Dukung dgn/atau tanpa namespace.
        pts = root.findall(".//g:trkpt", ns) or root.findall(".//trkpt")
        for p in pts:
            lat = float(p.get("lat")); lon = float(p.get("lon"))
            # NB: Element tanpa anak bersifat "falsy" -> jangan pakai 'or', cek None.
            tnode = p.find("g:time", ns)
            if tnode is None:
                tnode = p.find("time")
            ts = 0.0
            if tnode is not None and tnode.text:
                txt = tnode.text.strip().replace("Z", "+00:00")
                try:
                    ts = datetime.fromisoformat(txt).replace(
                        tzinfo=timezone.utc).timestamp() if "+" not in txt else \
                        datetime.fromisoformat(txt).timestamp()
                except ValueError:
                    ts = 0.0
            ele = p.find("g:ele", ns) if ns else p.find("ele")
            alt = float(ele.text) if ele is not None and ele.text else None
            self.points.append((ts, GpsFix(lat, lon, alt)))
        self.points.sort(key=lambda e: e[0])

    def get_fix(self, video_time: float | None = None) -> Optional[GpsFix]:
        if not self.points:
            return None
        if video_time is None:
            return self.points[0][1]
        target = self.start + video_time
        # Trackpoint terdekat ke target.
        best = min(self.points, key=lambda e: abs(e[0] - target))
        return best[1]


# --------------------------------------------------------------------------- #
# NMEA live (USB dongle via serial, atau HP via TCP)
# --------------------------------------------------------------------------- #
class _LiveNmeaProvider(GpsProvider):
    """Dasar provider live: thread membaca baris, simpan fix terkini."""
    kind = "nmea"

    def __init__(self):
        self._latest: Optional[GpsFix] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _start(self):
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _lines(self):
        raise NotImplementedError

    def _reader(self):
        for line in self._lines():
            if self._stop.is_set():
                break
            fix = parse_nmea_line(line)
            if fix:
                with self._lock:
                    self._latest = fix

    def get_fix(self, video_time=None) -> Optional[GpsFix]:
        with self._lock:
            return self._latest

    def close(self):
        self._stop.set()


class SerialNmeaProvider(_LiveNmeaProvider):
    def __init__(self, port: str, baud: int = 4800):
        super().__init__()
        try:
            import serial  # pyserial
        except ImportError:
            raise SystemExit("Butuh pyserial untuk GPS USB: pip install pyserial")
        self._ser = serial.Serial(port, baud, timeout=1)
        self._start()

    def _lines(self):
        while not self._stop.is_set():
            try:
                raw = self._ser.readline()
                if raw:
                    yield raw.decode("ascii", errors="ignore")
            except Exception:
                break

    def close(self):
        super().close()
        try:
            self._ser.close()
        except Exception:
            pass


class TcpNmeaProvider(_LiveNmeaProvider):
    def __init__(self, host: str, port: int):
        super().__init__()
        import socket
        self._sock = socket.create_connection((host, port), timeout=5)
        self._start()

    def _lines(self):
        buf = b""
        while not self._stop.is_set():
            try:
                data = self._sock.recv(1024)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    yield line.decode("ascii", errors="ignore")
            except Exception:
                break

    def close(self):
        super().close()
        try:
            self._sock.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def find_sidecar_srt(source: str) -> Optional[Path]:
    """Cari file .SRT bersebelahan dgn video (DJI menulisnya otomatis)."""
    p = Path(source)
    if not p.exists():
        return None
    for cand in (p.with_suffix(".SRT"), p.with_suffix(".srt")):
        if cand.exists():
            return cand
    return None


def make_provider(spec: str, source: str | None = None,
                  video_start_epoch: float | None = None) -> Optional[GpsProvider]:
    """Bangun provider dari spec string. Return None bila tanpa GPS."""
    if not spec or spec == "none":
        return None
    if spec == "auto":
        if source:
            srt = find_sidecar_srt(source)
            if srt:
                print(f"[GPS] Auto: ditemukan telemetry drone {srt.name}")
                return SrtProvider(srt)
        print("[GPS] Auto: tidak ada .SRT di sebelah video; berjalan tanpa GPS.")
        return None

    kind, _, arg = spec.partition(":")
    kind = kind.lower()
    if kind == "srt":
        return SrtProvider(arg)
    if kind == "gpx":
        return GpxProvider(arg, video_start_epoch)
    if kind == "fixed":
        lat, lon = (float(x) for x in arg.split(","))
        return FixedProvider(lat, lon)
    if kind == "nmea":
        port, _, baud = arg.partition("@")
        return SerialNmeaProvider(port, int(baud) if baud else 4800)
    if kind == "tcp":
        host, _, port = arg.partition(":")
        return TcpNmeaProvider(host, int(port))
    raise SystemExit(f"Spec GPS tidak dikenal: {spec}")


if __name__ == "__main__":
    # Uji cepat parser dari argumen.
    import sys
    if len(sys.argv) > 1:
        prov = make_provider(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        if prov:
            print("kind:", prov.kind, "fix@0s:", prov.get_fix(0.0))
