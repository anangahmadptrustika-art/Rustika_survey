@echo off
REM ==== Road Survey AI - launcher dobel-klik (Windows) ====
REM Dobel-klik file ini untuk membuka aplikasi (tanpa ngetik perintah).
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment .venv belum ada.
    echo Jalankan setup dulu ^(lihat PANDUAN.md Langkah 1^).
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python gui.py

REM Kalau aplikasi keluar dengan error, jendela tetap terbuka biar pesannya kebaca.
if errorlevel 1 (
    echo.
    echo [Aplikasi berhenti dengan error di atas.]
    pause
)
