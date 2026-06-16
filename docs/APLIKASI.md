# Aplikasi Desktop (GUI) — Pakai Tanpa Ketik Perintah

`gui.py` membungkus seluruh fitur jadi satu jendela aplikasi. Dibuka dengan
**dobel-klik** — cocok untuk dipakai sehari-hari di lapangan.

## Pasang sekali (di laptop)

PowerShell di folder project (venv aktif):
```powershell
pip install pyside6
```
(Dependensi lain sudah dari `requirements.txt`.)

## Buka aplikasinya

**Dobel-klik file `Road Survey AI.bat`** di folder project.
Jendela "Road Survey AI — Rustika Citra Group" akan terbuka.

> Mau lebih praktis? Klik kanan `Road Survey AI.bat` → **Send to → Desktop (create shortcut)**.
> Bisa juga ganti ikon shortcut-nya. Sekarang tinggal dobel-klik dari Desktop.

## Cara pakai jendelanya

1. **Sumber** — pilih: Webcam / File video / Folder gambar / RTSP-HP.
   - Webcam: isi `0` (atau `1` kalau webcam USB).
   - File/Folder: klik **Pilih…** untuk browse.
2. **Model** — default `models/road_damage.pt` (model hasil training-mu). Bisa ganti via **Pilih…**.
3. **Confidence** — ambang keyakinan (0.30 default; turunkan kalau banyak terlewat).
4. **GPS** — "Auto" akan membaca file `.SRT` di sebelah video drone DJI.
5. **Nama ruas** & **Surveyor** — untuk identitas di laporan.
6. Klik **▶ MULAI** — video tampil, dengan kotak deteksi + counter per jenis (live).
7. Klik **■ STOP** untuk berhenti (webcam). File video/folder berhenti sendiri.
8. Klik **📄 Export Laporan + Peta** → membuat `laporan.pdf`, `laporan.xlsx`, `map.html`, `detections.geojson`.
9. Klik **📂 Buka Folder Hasil** → buka folder sesi (foto, CSV, laporan).

## Catatan

- **Deteksi tetap 100% lokal/offline** — GUI ini cuma pembungkus loop yang sama.
- Tiap kali **MULAI** = satu sesi baru di `output/survey_YYYYMMDD_HHMMSS/`.
- Estimasi ukuran (m²) otomatis aktif kalau ada `configs/camera_calib.json`
  (lihat `docs/DASHCAM.md` soal kalibrasi).
- Jendela video di GUI ini menampilkan frame sendiri (lewat Qt), jadi **tidak
  terpengaruh** masalah `opencv-headless`. Mode CLI `--source 0` (yang pakai
  jendela OpenCV) butuh `opencv-python` versi penuh — kalau perlu itu lagi:
  `pip install --force-reinstall opencv-python`.

## (Lanjutan) Jadikan satu file .exe

Biar bisa dibagikan tanpa install Python, nanti bisa dibungkus jadi `.exe`
dengan PyInstaller:
```powershell
pip install pyinstaller
pyinstaller --noconfirm --windowed --name "RoadSurveyAI" gui.py
```
Hasilnya di `dist\RoadSurveyAI\`. Catatan: bundling PyTorch/ultralytics bikin
file besar dan kadang perlu penyesuaian — ini opsi lanjutan, bukan keharusan.
