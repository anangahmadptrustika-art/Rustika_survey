# Road Survey AI — Deteksi Jalan Rusak (Rustika Citra Group)

Aplikasi desktop untuk survey kondisi jalan di lapangan (Kabupaten Luwu Timur).
Menjalankan deteksi kerusakan jalan (lubang/retak) dengan model AI **lokal
(YOLO/ultralytics)**, mencatat tiap deteksi, dan menghasilkan laporan survey.

> **100% offline.** Tidak ada frame yang dikirim ke cloud AI apapun. Cocok untuk
> lokasi tanpa sinyal internet.

---

## Status & Roadmap

| Fase | Isi | Status |
|------|-----|--------|
| **1** | Core detection loop: webcam/video/RTSP → YOLO → bbox → CSV + screenshot + FPS | ✅ selesai (`road_survey.py`) |
| **2** | Model jalan rusak: (A) pre-trained pothole, lalu (B) fine-tune RDD2022 + data lokal | ✅ pipeline siap — lihat [`docs/FASE2.md`](docs/FASE2.md) |
| **3** | GPS per deteksi (HP / USB dongle / telemetry .SRT DJI) + peta HTML offline | ⏳ |
| **4** | Laporan survey PDF/Excel A4 (ringkasan, tabel, peta, foto) | ⏳ |
| **5** | (Opsional) Manajemen sesi + GUI | ⏳ |

---

## Lingkungan target

- **OS:** Windows 10/11
- **GPU:** NVIDIA (CUDA) — inferensi real-time + training/fine-tune lokal
- **Python:** 3.9–3.11 disarankan

> Kode **sadar lingkungan**: saat dijalankan ia mendeteksi OS, device
> (CUDA/MPS/CPU), dan backend kamera secara otomatis, lalu menampilkannya di banner.

---

## Instalasi (Windows + NVIDIA CUDA)

```powershell
# 1. Buat virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 2. Install PyTorch versi CUDA DULU (contoh CUDA 12.1)
#    Sesuaikan dengan driver/CUDA kamu: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install sisanya
pip install -r requirements.txt

# 4. Cek GPU kebaca
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

Kalau `CUDA: True` muncul, GPU siap dipakai.

---

## Cara pakai (Fase 1)

```powershell
# Webcam laptop
python road_survey.py --source 0

# File video / footage drone DJI Air 3S
python road_survey.py --source "DJI_0001.MP4"

# RTSP dari HP (mis. app "IP Webcam" Android)
python road_survey.py --source "rtsp://192.168.1.10:8554/live"

# Proses video offline tanpa window + simpan video hasil anotasi
python road_survey.py --source clip.mp4 --no-display --save-video

# Pakai model jalan rusak (setelah Fase 2)
python road_survey.py --source clip.mp4 --model models\road_damage.pt --conf 0.35
```

**Kontrol window:** `q` keluar · `p` pause/lanjut · `s` screenshot manual.

### Argumen penting

| Argumen | Default | Fungsi |
|---|---|---|
| `--source` | `0` | webcam index / path video / URL RTSP |
| `--model` | `yolov8n.pt` | path bobot YOLO (bisa diganti — batasan desain #2) |
| `--conf` | `0.25` | ambang confidence |
| `--device` | `auto` | `auto`/`cpu`/`cuda`/`0`/`mps` |
| `--stride` | `1` | proses tiap N frame (percepat video panjang) |
| `--max-shots-per-class` | `50` | batas screenshot per kelas (anti-banjir) |
| `--no-display` | off | mode headless / batch |
| `--save-video` | off | simpan video hasil anotasi |

### Output (per sesi)

```
output/survey_YYYYMMDD_HHMMSS/
├── detections.csv         # log tiap deteksi (waktu, frame, kelas, conf, bbox, ...)
├── screenshots/           # screenshot anotasi (dibatasi per kelas)
└── annotated.mp4          # (jika --save-video)
```

Kolom CSV sudah menyertakan `lat`/`lon` (kosong sekarang, diisi di **Fase 3**).

---

## ⚠️ Catatan akurasi (WAJIB dibaca)

Default `yolov8n.pt` adalah bobot **COCO** yang hanya mengenali objek umum
(orang, mobil, dll). **Ia TIDAK bisa mendeteksi lubang/retak jalan.** Default ini
ada **hanya untuk menguji bahwa pipeline berjalan** (kamera → inferensi → CSV →
screenshot).

Deteksi kerusakan jalan yang sebenarnya datang di **Fase 2** dengan model khusus
road-damage. Model generik **tidak otomatis akurat** di jalan Luwu Timur —
akurasi akhir **wajib divalidasi** dengan footage lokal kamu sendiri sebelum
dipakai produksi.

---

## Fase 2 — Model kerusakan jalan (ringkas)

Panduan lengkap: **[`docs/FASE2.md`](docs/FASE2.md)** · Label data lokal: **[`docs/LABELING.md`](docs/LABELING.md)**

| Komponen | File |
|---|---|
| Jalur A: ambil model pre-trained | `scripts/get_pretrained_model.py` |
| Unduh RDD2022 | `scripts/download_rdd2022.py` |
| Konversi VOC XML → YOLO | `scripts/voc_to_yolo.py` |
| Training / fine-tune (sadar GPU) | `train.py` |
| Validasi akurasi (metrik + visual) | `scripts/validate_model.py` |
| Smoke-test setup tanpa RDD2022 | `scripts/smoke_test_training.py` |
| Training di Colab (GPU gratis) | `notebooks/train_colab.ipynb` |
| Kelas & data.yaml template | `configs/classes.txt`, `configs/road_damage.yaml` |

Verifikasi cepat seluruh rantai (dataset sintetis):
```powershell
python scripts\smoke_test_training.py --out datasets\smoke --n 120
python scripts\voc_to_yolo.py --input datasets\smoke --out datasets\smoke_yolo
python train.py --data datasets\smoke_yolo\data.yaml --model yolov8n.pt --epochs 40 --imgsz 416 --device cpu --name smoke
python road_survey.py --source <video> --model runs\detect\smoke\weights\best.pt --conf 0.25
```
