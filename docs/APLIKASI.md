# Aplikasi Road Survey AI — Pakai Tanpa Ketik Perintah

Ada **dua bentuk** aplikasi (deteksi sama-sama 100% lokal/offline):

| Bentuk | Buka dengan | Cocok untuk |
|---|---|---|
| 🌐 **Web App** (disarankan) | dobel-klik `Road Survey AI (Web).bat` → browser `http://localhost:5000` | UI dashboard rapi, ada tombol **Cek Kamera**, gampang dikembangkan |
| 🖥️ **Desktop (PySide6)** | dobel-klik `Road Survey AI.bat` | jendela native tanpa browser |

---

## 🌐 Web App (localhost)

Pasang sekali: `pip install flask`
Lalu **dobel-klik `Road Survey AI (Web).bat`** → server jalan, browser kebuka di
`http://localhost:5000` (otomatis). Tutup jendela hitam (cmd) untuk berhenti.

Di halaman web:
1. **Sumber** = Webcam → isi **Nomor kamera** (`0` laptop, `1`/`2` USB external).
   Klik **Cek Kamera** untuk tahu nomor webcam yang nyambung.
2. **Model** = pilih `models/road_damage.pt` (model 4-kelas-mu).
3. Atur **Confidence**, isi **Nama ruas** & **Surveyor**.
4. **▶ MULAI** → video live + kotak deteksi + kartu statistik (FPS, total, per jenis).
5. **■ STOP** → **📄 Export Laporan + Peta** → muncul link unduh `laporan.pdf`,
   `laporan.xlsx`, `map.html`.

> Server cuma di `127.0.0.1` (localhost) — tidak terbuka ke jaringan luar.

---

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
