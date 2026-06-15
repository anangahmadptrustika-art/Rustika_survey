# Mengumpulkan & Melabeli Data Jalan Lokal (Luwu Timur)

RDD2022 = jalan luar negeri (Jepang, India, Ceko, dll). Aspal, cuaca, pencahayaan,
dan jenis kerusakan di Luwu Timur **berbeda**. Supaya model akurat di lapangan lo,
kita kumpulkan + label foto jalan lokal sendiri, lalu fine-tune.

Target: **minimal 200–300 gambar berlabel per kelas** untuk fine-tune yang berarti
(makin banyak makin baik). Mulai dari yang ada, tambah bertahap.

---

## 1. Kumpulkan footage

Sumber yang lo punya:
- **Drone DJI Air 3S** — terbang rendah/pelan di atas jalan, rekam video. Ekstrak
  frame jadi gambar.
- **HP / dashcam** — video sambil jalan.

Ekstrak frame dari video jadi gambar (tiap 1 detik):

```powershell
# Pakai ffmpeg (gratis). 1 frame per detik:
ffmpeg -i DJI_0001.MP4 -vf fps=1 frames\luwu_%05d.jpg
```

Tips ambil data yang berguna:
- Variasikan: pagi/siang/sore, kering/basah, aspal/beton, bayangan/terang.
- Sertakan jalan **mulus** juga (gambar tanpa kerusakan = "negatif", bantu kurangi
  false positive).
- Jangan blur; usahakan kerusakan terlihat jelas.

---

## 2. Pilih tool pelabelan

### Opsi A — LabelImg (offline, gratis, ringan)
```powershell
pip install labelImg
labelImg frames\ configs\classes.txt
```
- Set format ke **YOLO** (tombol kiri toggle "PascalVOC"/"YOLO"). Pilih **YOLO**
  supaya langsung keluar `.txt`.
- `configs/classes.txt` sudah berisi 4 kelas kita (urutan = class id).
- Shortcut: `w` buat kotak, `d` gambar berikutnya, `a` sebelumnya.

> Kalau menyimpan sebagai **PascalVOC (XML)**, konversi dgn:
> `python scripts/voc_to_yolo.py --input frames_voc --out datasets/luwu_yolo`

### Opsi B — Roboflow (web, kolaboratif, butuh internet)
- Buat project "Object Detection", upload gambar, label, set 4 kelas sama persis:
  `longitudinal_crack, transverse_crack, alligator_crack, pothole`.
- Export **YOLOv8** → dapat `data.yaml` + folder train/val.
- Atau pakai `scripts/get_pretrained_model.py roboflow ...` untuk menarik dataset.

---

## 3. Konsistensi label (PENTING)

Pakai definisi yang sama tiap kali:
- **longitudinal_crack** — retak searah arah jalan (memanjang).
- **transverse_crack** — retak memotong jalan (melintang).
- **alligator_crack** — retak saling menyambung seperti sisik buaya / jaring.
- **pothole** — lubang (kehilangan material, cekung).

Kotak rapat mengelilingi kerusakan. Satu kerusakan = satu kotak. Kalau ragu antara
retak buaya vs retak biasa, konsisten pilih satu aturan dan catat.

---

## 4. Susun struktur dataset

Hasil akhir harus:
```
datasets/luwu_yolo/
├── images/train/  images/val/
├── labels/train/  labels/val/
└── data.yaml          # salin dari configs/road_damage.yaml, sesuaikan 'path'
```
LabelImg menaruh `.txt` bersebelahan dgn gambar — pindahkan ke struktur di atas
(80% train / 20% val), atau kumpulkan sebagai VOC lalu pakai `voc_to_yolo.py` yang
otomatis membuat split + `data.yaml`.

---

## 5. Fine-tune & validasi

```powershell
# Fine-tune dari model RDD2022 (atau dari pre-trained jalur A)
python train.py --data datasets\luwu_yolo\data.yaml ^
    --model runs\detect\road_damage\weights\best.pt --epochs 50 --name luwu_finetune

# Validasi metrik + visual SEBELUM dipakai produksi
python scripts\validate_model.py --model runs\detect\luwu_finetune\weights\best.pt ^
    --data datasets\luwu_yolo\data.yaml
python scripts\validate_model.py --model runs\detect\luwu_finetune\weights\best.pt ^
    --source footage_baru_luwu\
```

**Kriteria siap produksi (saran):**
- mAP@0.5 per kelas masuk akal (mis. > 0.5–0.6; lubang biasanya lebih mudah).
- Review visual: mayoritas kerusakan asli terdeteksi, false positive sedikit.
- Uji di footage yang **tidak dipakai training**.

Belum memuaskan? Tambah data lokal pada kelas/kondisi yang lemah, lalu ulangi.
Ini proses iteratif — model membaik seiring data lokal bertambah.
