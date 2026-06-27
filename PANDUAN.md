# PANDUAN PEMAKAIAN — Road Survey AI (SOP Lapangan)

Panduan praktis pakai aplikasi survey jalan dari nol sampai laporan jadi.
Untuk konsultan Rustika Citra Group (Luwu Timur).

Ada **2 tahap**:
- **Tahap 1 — Persiapan** (sekali saja, butuh internet, lakukan di kantor/rumah)
- **Tahap 2 — Tiap survey** (boleh 100% offline, di lapangan)

---

## TAHAP 1 — Persiapan (sekali, perlu internet)

### 1.1 Install Python & dependensi
1. Install **Python 3.11** dari https://python.org — centang **"Add Python to PATH"**.
2. Buka folder project ini di PowerShell, lalu:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   # PyTorch versi CUDA DULU (sesuaikan: https://pytorch.org/get-started/locally/)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
3. Cek GPU terbaca:
   ```powershell
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```
   Harus `CUDA: True`. Kalau `False`, ulangi install PyTorch CUDA.

### 1.2 Siapkan model deteksi
> Tanpa model kerusakan jalan, aplikasi hanya mengenali objek umum (orang/mobil),
> **bukan** lubang/retak.

**Pilihan cepat (validasi pipeline):**
```powershell
python scripts\get_pretrained_model.py hf --repo <user/model-pothole> --file best.pt
# hasil -> models\road_damage.pt
```

**Pilihan akurat (untuk produksi) — lihat docs\FASE2.md & docs\LABELING.md:**
```powershell
python scripts\download_rdd2022.py --country Japan India
python scripts\voc_to_yolo.py --input datasets\RDD2022\Japan datasets\RDD2022\India --out datasets\rdd_yolo
python train.py --data datasets\rdd_yolo\data.yaml --model yolov8s.pt --epochs 100
# lalu FINE-TUNE dgn foto jalan Luwu Timur (kunci akurasi):
python train.py --data datasets\luwu_yolo\data.yaml --model runs\detect\road_damage\weights\best.pt --epochs 50 --name luwu_finetune
# validasi sebelum dipakai:
python scripts\validate_model.py --model runs\detect\luwu_finetune\weights\best.pt --data datasets\luwu_yolo\data.yaml
```
Salin bobot final ke `models\road_damage.pt`.

### 1.3 Cek setup tanpa download besar (opsional)
```powershell
python scripts\smoke_test_training.py --out datasets\smoke --n 120
python scripts\voc_to_yolo.py --input datasets\smoke --out datasets\smoke_yolo
python train.py --data datasets\smoke_yolo\data.yaml --model yolov8n.pt --epochs 40 --imgsz 416 --device cpu --name smoke
```

---

## TAHAP 2 — Survey di lapangan (boleh offline)

### Skenario A — Footage drone DJI Air 3S (utama)

1. **Sekali set di DJI Fly:** aktifkan **Video Caption / Subtitles = ON**
   (Settings → Camera). Ini membuat file `.SRT` berisi GPS per-frame.
2. Terbang **rendah & pelan** di atas jalan. Tips:
   - Ketinggian **konsisten** (mis. 12–15 m) → estimasi keparahan lebih stabil.
   - Pelan & stabil → tidak blur.
   - Cahaya cukup (hindari kontras ekstrem).
3. Copy **DUA file** dari SD card ke laptop: `DJI_0001.MP4` **dan** `DJI_0001.SRT`
   (taruh di folder yang sama, mis. `footage\`).
4. Jalankan (satu perintah → deteksi + GPS + peta + laporan):
   ```powershell
   python road_survey.py --source footage\DJI_0001.MP4 --model models\road_damage.pt ^
       --conf 0.35 --map --report --road "Ruas Malili-Wawondula" --surveyor "Anang A."
   ```
   - Video panjang & ingin cepat? tambah `--stride 3` (proses tiap 3 frame).
   - Ingin video hasil anotasi? tambah `--save-video`.

### Skenario B — Kamera / HP live + GPS

```powershell
# Webcam laptop + GPS USB dongle (cek nomor COM di Device Manager)
python road_survey.py --source 0 --model models\road_damage.pt --gps nmea:COM3@4800 --map --report

# HP sebagai kamera (RTSP, app "IP Webcam") + HP lain sebagai GPS (app "GPS 2 IP")
python road_survey.py --source "rtsp://192.168.43.10:8554/live" --model models\road_damage.pt --gps tcp:192.168.43.1:11123 --map --report
```
**Kontrol window:** `q` keluar · `p` pause/lanjut · `s` screenshot manual.

### Skenario C — Video HP/dashcam + track GPS (GPX)

Rekam track GPS pakai app HP (GPX Logger/OsmAnd), export `.gpx`:
```powershell
python road_survey.py --source footage\jalan.mp4 --model models\road_damage.pt ^
    --gps gpx:track.gpx --gps-start "2026-06-15T09:00:00" --map --report
```

---

## HASIL (folder `output\survey_TANGGAL_JAM\`)

| File | Isi | Untuk |
|------|-----|-------|
| `laporan.pdf` | laporan A4 lengkap | **dilampirkan ke klien** |
| `laporan.xlsx` | data terstruktur | analisis/olah data |
| `map.html` | peta interaktif | lihat sebaran (buka di browser) |
| `detections.geojson` | titik GPS | QGIS / Google Earth (offline) |
| `detections.csv` | log mentah semua deteksi | arsip/audit |
| `screenshots\` | foto tiap kerusakan | bukti |
| `annotated.mp4` | video ber-kotak (jika `--save-video`) | dokumentasi |

Buat ulang peta / laporan kapan saja dari CSV:
```powershell
python make_map.py output\survey_xxx\detections.csv
python make_report.py output\survey_xxx\detections.csv --road "..." --surveyor "..."
```

---

## TIPS & CATATAN JUJUR

- **Validasi dulu sebelum produksi.** Jangan percaya model mentah; uji di footage
  Luwu Timur (`scripts\validate_model.py --source footage\`). Lihat folder
  `annotated\`, pastikan kotak benar di kerusakan asli.
- **Keparahan (Ringan/Sedang/Berat)** = perkiraan dari ukuran kotak, **bukan**
  ukuran teknik. Dipengaruhi ketinggian drone. Kalibrasi `--sev-low/--sev-high`
  dan verifikasi manual untuk klaim formal.
- **Atur confidence** dgn `--conf` (mis. 0.35–0.5). Terlalu rendah = banyak salah;
  terlalu tinggi = banyak terlewat.
- **Offline:** deteksi 100% offline. Peta `map.html` titiknya offline; latar peta
  jalan butuh internet — untuk lapangan pakai GeoJSON di QGIS dgn basemap terunduh.
- **GPS drone** ~beberapa meter (cukup untuk laporan). Pastikan `.SRT` ikut tercopy.

Detail per fitur: `docs\FASE2.md` (model), `docs\FASE3.md` (GPS/peta),
`docs\FASE4.md` (laporan), `docs\LABELING.md` (data lokal).
