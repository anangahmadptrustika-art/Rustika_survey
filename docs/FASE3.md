# Fase 3 — Lokasi GPS per Deteksi + Peta

Inti survey jalan: tiap kerusakan harus punya **koordinat**. Fase 3 menambah kolom
`lat`/`lon` ke CSV (slot sudah disiapkan sejak Fase 1) dan membuat **peta** titik
kerusakan.

## Sumber GPS yang didukung (`--gps`)

| Spec | Sumber | Kapan dipakai |
|------|--------|---------------|
| `auto` (default) | cari `FILE.SRT` di sebelah video | footage drone DJI |
| `srt:FILE.SRT` | telemetry drone DJI (lat/lon per-frame) | footage drone DJI Air 3S |
| `gpx:TRACK.GPX` | track GPS dari HP (rekam saat survey) | survey jalan kaki/mobil + HP |
| `nmea:PORT@BAUD` | USB GPS dongle (NMEA via serial) | survey real-time dgn dongle |
| `tcp:HOST:PORT` | HP broadcast NMEA via TCP | HP sebagai GPS live |
| `fixed:LAT,LON` | satu titik tetap | uji / lokasi statis |
| `none` | tanpa GPS | — |

### 1. Drone DJI Air 3S (paling mudah)

DJI menulis file telemetry `.SRT` otomatis di sebelah video (mis. `DJI_0001.MP4`
+ `DJI_0001.SRT`). Pastikan **"Video Subtitles / Caption"** aktif di setelan DJI
Fly sebelum terbang.

```powershell
# auto-detect .SRT di sebelah video:
python road_survey.py --source DJI_0001.MP4 --model models\road_damage.pt --map
# atau eksplisit:
python road_survey.py --source DJI_0001.MP4 --gps srt:DJI_0001.SRT --map
```

### 2. GPS HP (track GPX)

Rekam track saat survey pakai app HP (mis. *GPX Logger*, *OsmAnd*, *Strava*),
export `.gpx`. Cocokkan ke video lewat waktu:

```powershell
python road_survey.py --source clip.mp4 --gps gpx:track.gpx ^
    --gps-start "2026-06-15T09:00:00" --map
```
`--gps-start` = waktu mulai video (supaya frame dicocokkan ke trackpoint yang benar).
Bila video & track mulai bersamaan, boleh dikosongkan (pakai titik awal GPX).

### 3. USB GPS dongle (real-time)

Dongle GPS USB umumnya muncul sebagai COM port dan mengirim NMEA.
```powershell
pip install pyserial
python road_survey.py --source 0 --gps nmea:COM3@4800 --map
```
Cek nomor COM di Device Manager. Baud umum: 4800 atau 9600.

### 4. HP sebagai GPS live (TCP NMEA)

App seperti *GPS 2 IP* (Android) bisa broadcast NMEA via TCP. HP & laptop harus
satu jaringan (mis. hotspot HP).
```powershell
python road_survey.py --source 0 --gps tcp:192.168.43.1:11123 --map
```

## Output peta

Dengan `--map`, setelah selesai dibuat di folder sesi:
- **`map.html`** — peta interaktif (Leaflet): marker per kerusakan (warna per jenis),
  popup berisi jenis/confidence/waktu/foto, plus garis rute survei. Buka dgn klik
  dua kali (tanpa server).
- **`detections.geojson`** — untuk QGIS / Google Earth.

Atau buat ulang peta kapan saja dari CSV:
```powershell
python make_map.py output\survey_YYYYMMDD_HHMMSS\detections.csv
```

> **Soal "offline":** `map.html` berdiri sendiri dan semua titik/rute/popup
> tertanam → tampil **offline**. Namun **latar peta (tile jalan)** butuh internet
> untuk render; tanpa internet, marker tetap muncul di latar abu (posisi tetap
> benar). Untuk peta jalan offline penuh di lapangan, buka `detections.geojson`
> di **QGIS** dengan basemap lokal/satelit yang sudah diunduh.

## Akurasi GPS — jujur

- **Drone DJI**: GPS drone biasanya ~beberapa meter; cukup untuk lokasi titik
  kerusakan di laporan. Ketinggian (`abs_alt`) ikut terbaca.
- **HP/dongle consumer**: ~3–10 m, lebih buruk di bawah pohon/gedung.
- Koordinat diambil **per-frame** (drone) atau **fix terkini** (live). Untuk video,
  pastikan timeline .SRT cocok dengan video (DJI: otomatis cocok).
