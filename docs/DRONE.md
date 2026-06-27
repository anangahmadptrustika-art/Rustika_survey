# Pakai Drone DJI Air 3S (Remote RC2)

Caranya: **rekam → upload video + .SRT → proses**. GPS presisi dari drone otomatis
masuk ke tiap kerusakan. (Live streaming yang ribet sengaja tidak dipakai.)

---

## 1. Sekali: aktifkan subtitle GPS di DJI Fly
Di **DJI Fly** (RC2) → setting kamera → cari **"Video Caption" / "Subtitle"** →
**AKTIFKAN**. Ini bikin file **`.SRT`** (berisi GPS tiap detik) otomatis tersimpan
bareng video. **Tanpa ini, hasil tidak ada koordinat.**

## 2. Terbang & rekam
- Terbang **rendah & pelan** mengikuti jalur jalan (tinggi **15–30 m**, konsisten).
- Kamera **menunduk** ke jalan (gimbal ke bawah ~45–90°).
- Pencahayaan cukup, jangan terlalu cepat (biar tidak blur).

## 3. Copy file ke laptop
Dari SD card drone, copy **DUA file** (nama sama):
- `DJI_0001.MP4`
- `DJI_0001.SRT`

## 4. Upload & proses di aplikasi web
Buka **`Road Survey AI (Web).bat`**, lalu di halaman web:
1. **Sumber** = **`Upload Video Drone (DJI)`**
2. **Video drone (.MP4)** → klik, pilih `DJI_0001.MP4`
3. **File GPS (.SRT)** → klik, pilih `DJI_0001.SRT`  *(ini yang bikin ada koordinat)*
4. **Model** = `models/road_damage.pt`
5. **▶ MULAI** → video di-upload ke server lokal, lalu diproses. Tunggu selesai.
6. **📄 Export** → **peta + laporan dengan titik lokasi**. 🗺️

> Video besar (GB)? Upload ke server lokal butuh beberapa menit — sabar, itu cuma
> menyalin file di laptopmu sendiri (tetap offline).

---

## Catatan
- **Wajib ada .SRT** kalau mau lokasi. Tanpa .SRT, deteksi tetap jalan tapi
  laporan/peta tidak ada koordinat.
- Format .SRT DJI dibaca otomatis (`[latitude: ...] [longitude: ...]`). Kalau
  drone-mu pakai format .SRT lain dan koordinat tidak kebaca, kirim contoh
  isi .SRT-nya — parser-nya gampang disesuaikan.
- **Live streaming (RTMP)** sengaja dihilangkan dari aplikasi karena ribet & tidak
  memberi GPS drone yang akurat. Mode upload ini lebih simpel dan justru lebih
  tepat untuk survey resmi.
