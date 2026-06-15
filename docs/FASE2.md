# Fase 2 — Model Deteksi Kerusakan Jalan

Bagian TERPENTING. Bobot default `yolov8n.pt` (COCO) hanya kenal orang/mobil/dll —
**tidak bisa** deteksi lubang/retak. Di sini kita pasang model kerusakan jalan
sungguhan lewat dua jalur.

**Kelas target (4):**

| id | nama | RDD2022 |
|----|------|---------|
| 0 | `longitudinal_crack` (retak memanjang) | D00, D01 |
| 1 | `transverse_crack` (retak melintang) | D10, D11 |
| 2 | `alligator_crack` (retak buaya) | D20 |
| 3 | `pothole` (lubang) | D40 |

---

## Jalur A — Model pre-trained (validasi cepat)

Tujuan: cepat lihat pipeline bekerja dengan model jadi, **sebelum** training serius.

```powershell
# Dari URL .pt langsung (mis. GitHub release)
python scripts\get_pretrained_model.py url --url https://.../best.pt

# Dari HuggingFace Hub
python scripts\get_pretrained_model.py hf --repo <user/model> --file best.pt

# Lalu jalankan app dengan model itu
python road_survey.py --source clip.mp4 --model models\road_damage.pt --conf 0.35
```

**Cari model di:** HuggingFace (cari "pothole yolov8"), Roboflow Universe
(universe.roboflow.com → "road damage"), atau repo GitHub RDD2022. **Cek lisensi.**

**Trade-off jalur A:**
- ✅ Cepat (menit), bagus untuk uji pipa & demo.
- ❌ Dilatih di jalan luar / kondisi lain → sering **meleset di jalan Luwu Timur**
  (aspal, pencahayaan tropis, jenis kerusakan beda). Kelas model belum tentu sama
  dengan 4 kelas kita.
- ➡️ **Wajib divalidasi** dgn footage lokal (`scripts/validate_model.py`) sebelum
  dipercaya. Anggap ini "model sementara".

> ⚠️ Catatan lingkungan: HuggingFace/Roboflow bisa **terblokir** di jaringan
> ber-egress-allowlist. Unduh model di laptop ber-internet normal, lalu copy
> `.pt`-nya ke folder `models/`.

---

## Jalur B — Training/fine-tune RDD2022 (akurat)

### B1. Unduh RDD2022

```powershell
python scripts\download_rdd2022.py --list                 # lihat negara
python scripts\download_rdd2022.py --country Japan India   # unduh + ekstrak
```
Dataset besar (GB) → unduh di tempat ber-internet stabil. Isinya jalan **luar
negeri**, jadi ini baru fondasi.

### B2. Konversi VOC XML → YOLO

```powershell
python scripts\voc_to_yolo.py --input datasets\RDD2022\Japan datasets\RDD2022\India ^
    --out datasets\rdd_yolo --val-split 0.2
```
Menghasilkan `datasets/rdd_yolo/{images,labels}/{train,val}` + `data.yaml`, plus
ringkasan jumlah objek per kelas. (Pakai `--copy` bila symlink bermasalah di Windows.)

### B3. Latih

```powershell
# Cek dulu kelayakan GPU
python train.py --data datasets\rdd_yolo\data.yaml --check-only

# Training penuh (GPU NVIDIA lo)
python train.py --data datasets\rdd_yolo\data.yaml --model yolov8s.pt --epochs 100
```
Hasil: `runs/detect/road_damage/weights/best.pt`.

- **Punya GPU NVIDIA** (kasus lo) → training lokal realistis (beberapa jam).
- **Tanpa GPU** → pakai `notebooks/train_colab.ipynb` (GPU T4 gratis di Colab).
- `train.py` otomatis mendeteksi & memberi tahu yang mana.

### B4. Fine-tune dengan data lokal Luwu Timur (KUNCI akurasi)

Lihat **[LABELING.md](LABELING.md)** untuk mengumpulkan + melabeli foto jalan lokal.
Setelah ada dataset lokal:

```powershell
# Mulai dari model RDD, lanjut belajar ciri jalan lokal
python train.py --data datasets\luwu_yolo\data.yaml ^
    --model runs\detect\road_damage\weights\best.pt ^
    --epochs 50 --name luwu_finetune
```

### B5. Validasi SEBELUM produksi

```powershell
# Metrik objektif di val set berlabel
python scripts\validate_model.py --model runs\detect\luwu_finetune\weights\best.pt ^
    --data datasets\luwu_yolo\data.yaml

# Review visual di footage drone DJI / video HP lo
python scripts\validate_model.py --model runs\detect\luwu_finetune\weights\best.pt ^
    --source footage_luwu\
```
Buka folder `output/validation/annotated/` dan **periksa mata**: kotak benar di
lubang/retak asli? Banyak salah → tambah data lokal kelas yang lemah, ulangi.

---

## Smoke-test (verifikasi setup tanpa RDD2022)

Sebelum menunggu unduhan RDD2022 yang besar, pastikan rantai training jalan di
mesin lo pakai data **sintetis**:

```powershell
python scripts\smoke_test_training.py --out datasets\smoke --n 120
python scripts\voc_to_yolo.py --input datasets\smoke --out datasets\smoke_yolo
python train.py --data datasets\smoke_yolo\data.yaml --model yolov8n.pt ^
    --epochs 40 --imgsz 416 --device cpu --name smoke
python road_survey.py --source <video> --model runs\detect\smoke\weights\best.pt --conf 0.25
```
Pipeline ini sudah diuji: 120 gambar sintetis → mAP@0.5 ≈ 0.98 → app mendeteksi 4
kelas. (Model sintetis **bukan** untuk produksi — hanya bukti pipa.)

---

## Alur yang disarankan (ringkas)

```
Jalur A (cepat)  ──► validasi pipa ──┐
                                     ▼
RDD2022 ─► voc_to_yolo ─► train ─► best.pt (dasar) ─► fine-tune data LOKAL ─► validasi ─► PRODUKSI
```
Akurasi nyata di jalan Luwu Timur datang dari **fine-tune data lokal + validasi**,
bukan dari model generik.
