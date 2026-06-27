# Pakai Drone DJI Air 3S (Remote RC2)

Dua cara: **footage** (utama, akurat) dan **live** (pemantauan cepat). RC2 tidak
punya HDMI-out, jadi live harus lewat streaming RTMP.

---

## A. FOOTAGE — rekam dulu, proses setelah mendarat ⭐ (disarankan)

Ini cara survey yang akurat: GPS presisi dari drone masuk ke tiap kerusakan.

### 1. Sekali: aktifkan subtitle GPS di DJI Fly
Di **DJI Fly** (RC2) → masuk setting kamera → cari **"Video Caption" / "Subtitle"**
→ **AKTIFKAN**. Ini bikin file **`.SRT`** (berisi GPS tiap detik) otomatis
tersimpan bareng video. **Tanpa ini, footage tidak ada koordinat.**

### 2. Terbang & rekam
- Terbang **rendah & pelan** mengikuti jalur jalan (mis. tinggi **15–30 m**, konsisten).
- Kamera **menunduk** ke jalan (gimbal ke bawah ~45–90°).
- Pencahayaan cukup, hindari terlalu cepat (biar tidak blur).

### 3. Copy file ke laptop
Dari SD card / penyimpanan drone, copy **DUA file** ke satu folder di laptop:
- `DJI_0001.MP4`
- `DJI_0001.SRT`  ← **harus ikut, nama sama**

### 4. Proses di aplikasi web
Buka aplikasi (`Road Survey AI (Web).bat`), lalu:
- **Sumber** = `File video` → **Pilih** `DJI_0001.MP4`
- **GPS** = **`Auto — .SRT drone DJI`**  (otomatis baca DJI_0001.SRT di sebelahnya)
- **Model** = `models/road_damage.pt`
- **▶ MULAI** → setelah selesai, **📄 Export** → peta + laporan **dengan titik lokasi**. 🗺️

> Video panjang? Bisa juga ekstrak jadi frame dulu pakai ffmpeg lalu pakai
> "Folder gambar" — tapi cara File video + Auto SRT paling simpel.

---

## B. LIVE — deteksi real-time saat drone terbang (lanjutan)

⚠️ **Jujur dulu:**
- Live cocok buat **pemantauan cepat**, **bukan** survey presisi.
- **GPS-nya bukan posisi drone.** Stream live tidak membawa .SRT, jadi kerusakan
  yang ketangkap live **tidak dapat koordinat drone yang akurat**. Untuk peta &
  lokasi presisi, tetap pakai **cara A (footage)**.
- Ada **delay** (1–3 detik) dan tergantung kualitas sinyal.

### 1. Pasang server streaming lokal (MediaMTX)
- Download **MediaMTX** (1 file .exe): https://github.com/bluenviron/mediamtx/releases
  (pilih `mediamtx_..._windows_amd64.zip`, extract).
- Jalankan **`mediamtx.exe`** (dobel-klik). Dia jadi server: RTMP di port `1935`,
  RTSP di `8554`. Biarkan terbuka.

### 2. Samakan jaringan RC2 ↔ laptop
Paling gampang: **laptop bikin hotspot**.
- Windows: **Settings → Network → Mobile hotspot → ON**.
- Di **RC2**, sambungkan ke WiFi hotspot laptop itu.
- Cari **IP laptop**: PowerShell → `ipconfig` → lihat **IPv4 Address** adapter
  hotspot (mis. `192.168.137.1`).

### 3. Stream dari DJI Fly (RC2)
- DJI Fly → menu **Transmission / Live Streaming** → **RTMP Custom**.
- URL: **`rtmp://<IP-laptop>:1935/live/drone`** (mis. `rtmp://192.168.137.1:1935/live/drone`).
- **Start Live Streaming.**

### 4. Baca di aplikasi
- **Sumber** = `Stream (RTSP / RTMP / HP / drone live)`
- **Lokasi** = **`rtsp://localhost:8554/live/drone`**  *(MediaMTX ubah RTMP→RTSP, lebih stabil buat OpenCV)*
  - kalau itu tidak jalan, coba `rtmp://localhost:1935/live/drone`
- **▶ MULAI** → deteksi jalan di feed drone secara live (dengan delay).

> Kalau gambar tidak muncul: cek mediamtx.exe jalan, IP benar, RC2 sudah konek
> hotspot, dan DJI Fly statusnya "streaming". Firewall Windows mungkin minta izin
> port 1935/8554 → Allow.

---

## Ringkas
| Kebutuhan | Pakai |
|---|---|
| Survey resmi + lokasi presisi + laporan | **A. Footage (.SRT)** |
| Lihat-lihat cepat sambil terbang | **B. Live (RTMP)** |
