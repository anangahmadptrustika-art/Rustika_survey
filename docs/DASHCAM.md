# Mode Dashcam — Scan Jalan Real-Time dari Mobil

Skenario: webcam USB dipasang di depan mobil → laptop di dalam mobil → aplikasi
scan jalan **real-time** sambil jalan, **auto-tandai** + **auto-capture** tiap
kerusakan, dengan **GPS**, **estimasi ukuran**, dan opsi second-opinion cloud.

> Deteksi tetap **100% lokal/offline** — aman di jalan tanpa sinyal. Cloud (opsional)
> hanya dipakai belakangan untuk memperkaya laporan saat sudah ada internet.

---

## 1. Pasang & jalankan (real-time)

1. Pasang webcam USB di depan mobil (dashboard/grille), arah ke jalan di depan.
   Pasang **stabil** dan **jangan diubah-ubah** posisinya (penting untuk estimasi ukuran).
2. Colok ke laptop. Cek nomor kamera (`0`, atau `1` bila ada kamera bawaan).
3. Idealnya colok juga **GPS USB dongle** (survey butuh koordinat).
4. Jalankan satu perintah:

```powershell
python road_survey.py --source 0 --model models\road_damage.pt --conf 0.4 ^
    --gps nmea:COM3@4800 --calib configs\camera_calib.json ^
    --map --report --road "Ruas X" --surveyor "Nama"
```

- Window live menampilkan kotak deteksi + FPS. Tiap kerusakan **otomatis ter-capture**
  ke `screenshots\` + tercatat ke CSV (waktu, kelas, confidence, GPS, ukuran).
- Kontrol: `q` keluar · `p` pause · `s` screenshot manual.
- GPU CUDA-mu bikin YOLOv8n/s ~30+ FPS — cukup untuk kecepatan mobil. Kalau lambat,
  pakai model lebih kecil atau turunkan `--imgsz`.

---

## 2. Estimasi UKURAN kerusakan (kalibrasi sekali)

Supaya tahu ukuran nyata (cm/m²), kamera perlu dikalibrasi **sekali** setelah dipasang.

### Kalibrasi (sekali)
1. Parkir di jalan datar. Letakkan tanda di 4+ titik yang terlihat kamera dan
   **ukur posisinya** (meter) — cara mudah: lakban membentuk persegi panjang
   berukuran diketahui (mis. 2m × 1m), atau marka jalan dengan jarak terukur.
2. Ambil 1 frame dari kamera, catat koordinat **piksel** tiap titik.
3. Buat CSV `kalibrasi.csv`:
   ```
   img_x,img_y,ground_x,ground_y
   120,300,0,0
   520,300,2,0
   540,460,2,1
   100,460,0,1
   ```
4. Hitung kalibrasi:
   ```powershell
   python scripts\calibrate_camera.py --points-file kalibrasi.csv --image frame.jpg
   ```
   Hasil → `configs\camera_calib.json`. Cek "galat reproyeksi" kecil (idealnya < 0.1 m).
   (Di laptop dengan layar, bisa pakai `--click --image frame.jpg` untuk klik titik.)

### Pakai
Tambahkan `--calib configs\camera_calib.json` saat menjalankan `road_survey.py`
(lihat perintah di atas). Kolom `width_m`, `length_m`, `area_m2` akan terisi di CSV,
dan laporan memakai **luas m²** untuk tingkat keparahan.

> ⚠️ **Jujur soal akurasi:** estimasi ukuran mengasumsikan **jalan datar** dan
> posisi kamera tetap. Guncangan, tanjakan, dan kemiringan menurunkan akurasi.
> Anggap PERKIRAAN untuk membanding-bandingkan, bukan ukuran survei presisi.
> Verifikasi manual untuk klaim formal ke klien.

---

## 3. Tingkat keparahan (Parah/Sedang/Ringan)

- **Tanpa kalibrasi** → keparahan dari rasio ukuran kotak terhadap frame
  (ambang `--sev-low`/`--sev-high` di `make_report.py`).
- **Dengan kalibrasi** → keparahan dari **luas nyata m²** (lebih bermakna):
  ambang `--sev-low-m2` (default 0.05 m²) dan `--sev-high-m2` (default 0.25 m²).
  Kalibrasi ambang ini sesuai standar/penilaianmu.

---

## 4. (Opsional) Second-opinion AI cloud — HYBRID

Ini menjawab "kalau bisa AI tahu parah/sedang" tanpa melanggar aturan offline:
deteksi real-time tetap lokal; cloud hanya menilai **foto yang sudah ter-capture**,
dijalankan **belakangan saat ada internet**.

```powershell
pip install anthropic
setx ANTHROPIC_API_KEY "kunci-api-anthropic-mu"
# (tutup & buka ulang terminal setelah setx)

python cloud_enrich.py output\survey_YYYYMMDD_HHMMSS --max-images 30
```
Hasil → `output\survey_xxx\cloud_assessment.csv` (per foto: benar kerusakan?,
jenis, keparahan, deskripsi 1 kalimat untuk laporan).

- Cek dulu tanpa biaya: tambah `--dry-run` (tidak memanggil API).
- Hemat biaya: `--model claude-haiku-4-5` (lebih murah dari opus untuk klasifikasi),
  dan batasi `--max-images`.
- **Kenapa bukan per-frame?** Mengirim tiap frame ke cloud = butuh internet terus,
  lambat (delay per frame), dan mahal (ribuan frame). Di jalan rural Luwu Timur
  sering tanpa sinyal — deteksi wajib lokal. Cloud pada puluhan foto = murah,
  offline-first, dan tetap dapat "sentuhan AI cloud".

---

## Ringkasan output sesi

```
output\survey_xxx\
├── detections.csv         # + kolom width_m, length_m, area_m2 (bila --calib)
├── screenshots\           # auto-capture tiap kerusakan
├── map.html               # peta titik (Fase 3)
├── detections.geojson
├── laporan.pdf / .xlsx    # keparahan dari m² bila terkalibrasi (Fase 4)
└── cloud_assessment.csv   # bila menjalankan cloud_enrich.py (opsional)
```
