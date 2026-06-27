# Fase 4 — Laporan Survey (PDF + Excel A4)

Dari `detections.csv` satu sesi → laporan profesional siap lampir ke laporan klien.

## Sekali perintah

```powershell
python make_report.py output\survey_20260615_120000\detections.csv ^
    --road "Ruas Malili - Wawondula" --surveyor "Anang A." --date 2026-06-15
```
Menghasilkan di folder sesi:
- **`laporan.pdf`** (A4) — judul + identitas survey, ringkasan (tabel jenis × keparahan + grafik), sebaran lokasi, tabel detail, lampiran foto.
- **`laporan.xlsx`** — sheet *Ringkasan* + *Detail Kerusakan* (semua kerusakan, bisa difilter/diolah).

Atau langsung saat survey (end-to-end dalam satu proses):
```powershell
python road_survey.py --source DJI_0001.MP4 --model models\road_damage.pt ^
    --map --report --road "Ruas Malili" --surveyor "Anang A."
```

## Penghitungan jujur (klaster)

CSV mencatat deteksi **per-frame** — satu lubang muncul di puluhan frame. Laporan
**meng-klaster** deteksi jadi **kerusakan unik** sebelum menghitung:
- **Ada GPS** → gabung deteksi sejenis dalam radius `--merge-dist` meter (default 8 m).
- **Tanpa GPS** → gabung deteksi sejenis dalam jeda `--merge-frames` frame (default 20).

Contoh demo: 180 deteksi mentah → **29 kerusakan unik**. Tanpa klaster, laporan
akan melebih-lebihkan jumlah.

## Estimasi keparahan

Ringan / Sedang / Berat ditaksir dari **rasio luas bbox terhadap frame**:
- `< --sev-low` (default 0.3%) → Ringan
- `< --sev-high` (default 1.5%) → Sedang
- selebihnya → Berat

> ⚠️ **Ini proksi kasar, bukan ukuran teknik.** Ukuran bbox dipengaruhi jarak
> kamera / ketinggian drone, dan retak tipis-memanjang bisa "terlihat kecil"
> walau parah. **Kalibrasi** `--sev-low/--sev-high` dengan footage sendiri pada
> ketinggian terbang yang konsisten, dan **verifikasi manual** sebelum dipakai
> sebagai klaim formal ke klien. Untuk ukuran nyata (m²/panjang), perlu kalibrasi
> GSD dari ketinggian drone — bisa ditambahkan menyusul bila diperlukan.

## Opsi berguna

| Argumen | Default | Fungsi |
|---|---|---|
| `--road` | nama folder sesi | nama ruas di laporan |
| `--surveyor` | `-` | nama surveyor |
| `--date` | hari ini | tanggal survey |
| `--pdf-only` / `--excel-only` | keduanya | batasi format output |
| `--merge-dist` | `8` (m) | radius klaster (mode GPS) |
| `--merge-frames` | `20` | jeda klaster (tanpa GPS) |
| `--sev-low` / `--sev-high` | `0.003` / `0.015` | ambang keparahan |
| `--max-photos` | `9` | jumlah foto di lampiran PDF |
