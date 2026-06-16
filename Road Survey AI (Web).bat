@echo off
REM ==== Road Survey AI - Web App (localhost) ====
REM Dobel-klik untuk menjalankan server lokal & membuka browser otomatis.
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment .venv belum ada. Lihat PANDUAN.md Langkah 1.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
echo Menjalankan server... browser akan terbuka otomatis di http://localhost:5000
echo (Biarkan jendela ini terbuka selama aplikasi dipakai. Tutup jendela untuk berhenti.)
python webapp.py

if errorlevel 1 (
    echo.
    echo [Server berhenti dengan error di atas.]
    pause
)
