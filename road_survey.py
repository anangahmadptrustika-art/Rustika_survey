#!/usr/bin/env python3
"""
Road Survey AI — Fase 1: Core detection loop.

Aplikasi desktop survey kondisi jalan untuk Rustika Citra Group (Luwu Timur).
Menjalankan inferensi YOLO LOKAL (offline, tanpa cloud) pada webcam / file video /
RTSP / footage drone, menggambar bounding box, mencatat tiap deteksi ke CSV, dan
menyimpan screenshot (dibatasi per kelas biar tidak banjir).

Batasan desain (lihat brief):
  1. Inferensi 100% lokal — tidak ada frame yang dikirim ke cloud AI apapun.
  2. Model bisa diganti — path model = argumen (`--model`).
  3. Sadar lingkungan — auto-deteksi OS, device (CUDA/MPS/CPU), backend kamera.

Catatan penting Fase 1:
  Bobot default YOLO (COCO) HANYA mengenali orang/mobil/dll — TIDAK bisa deteksi
  lubang/retak. Default `yolov8n.pt` di sini cuma untuk menguji bahwa pipeline
  (kamera -> inferensi -> gambar -> CSV -> screenshot) berjalan. Model jalan rusak
  yang sebenarnya dipasang di Fase 2.

Contoh pakai:
  # Webcam (Windows)
  python road_survey.py --source 0
  # File video / footage drone DJI
  python road_survey.py --source "DJI_0001.MP4"
  # RTSP dari HP (mis. IP Webcam)
  python road_survey.py --source "rtsp://192.168.1.10:8554/live"
  # Proses video offline tanpa window, simpan video hasil (untuk mesin headless)
  python road_survey.py --source clip.mp4 --no-display --save-video
  # Pakai model jalan rusak (Fase 2)
  python road_survey.py --source clip.mp4 --model models/road_damage.pt --conf 0.35
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit(
        "ERROR: OpenCV belum terpasang.\n"
        "  pip install opencv-python\n"
        "Lihat README.md untuk instruksi install lengkap (Windows + CUDA)."
    )


# --------------------------------------------------------------------------- #
# Deteksi lingkungan (batasan #3: sadar lingkungan)
# --------------------------------------------------------------------------- #
def detect_environment() -> dict:
    """Deteksi OS, versi Python, dan device akselerasi yang tersedia."""
    info = {
        "os": platform.system(),          # 'Windows' | 'Linux' | 'Darwin'
        "os_release": platform.release(),
        "python": platform.python_version(),
        "torch": None,
        "device": "cpu",
        "device_name": "CPU",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["device_name"] = torch.cuda.get_device_name(0)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["device"] = "mps"
            info["device_name"] = "Apple Silicon (MPS)"
    except ImportError:
        # torch dibawa otomatis oleh ultralytics; kalau belum ada, biarkan CPU.
        pass
    return info


def pick_camera_backend(os_name: str):
    """Backend OpenCV terbaik per-OS untuk webcam lokal."""
    if os_name == "Windows":
        return cv2.CAP_DSHOW          # DirectShow: paling stabil di Windows
    if os_name == "Linux":
        return cv2.CAP_V4L2
    if os_name == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def print_banner(env: dict, args, model_names) -> None:
    line = "=" * 64
    print(line)
    print(" ROAD SURVEY AI — Fase 1: Core Detection Loop")
    print(" Rustika Citra Group · inferensi LOKAL (offline)")
    print(line)
    print(f"  OS          : {env['os']} {env['os_release']}")
    print(f"  Python      : {env['python']}")
    print(f"  PyTorch     : {env['torch'] or 'belum terpasang (CPU)'}")
    print(f"  Device      : {env['device'].upper()}  ({env['device_name']})")
    print(f"  Model       : {args.model}")
    if model_names:
        preview = ", ".join(list(model_names.values())[:6])
        more = " ..." if len(model_names) > 6 else ""
        print(f"  Kelas model : {len(model_names)} kelas -> {preview}{more}")
    print(f"  Sumber      : {args.source}")
    print(f"  Confidence  : {args.conf}")
    print(f"  Output      : {args.output}")
    if env["device"] == "cpu":
        print("  CATATAN     : Jalan di CPU -> inferensi lambat. Pakai model 'n'/'s'")
        print("                atau proses video offline (--no-display).")
    print(line)


# --------------------------------------------------------------------------- #
# Helpers sumber input
# --------------------------------------------------------------------------- #
def is_webcam(source: str) -> bool:
    """True jika source berupa indeks kamera (mis. '0', '1')."""
    return str(source).isdigit()


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ImageFolderCapture:
    """Pengganti cv2.VideoCapture untuk folder gambar / pola glob / satu gambar.

    Membuat road_survey.py bisa memproses kumpulan foto (mis. frame hasil
    ekstrak footage drone, atau val set) seolah-olah video. Meniru antarmuka
    cv2.VideoCapture yang dipakai loop utama (isOpened/get/read/release).
    """

    def __init__(self, paths, fps: float = 2.0):
        self.paths = [str(p) for p in paths]
        self.idx = 0
        self.fps = fps if fps and fps > 0 else 2.0
        self._w = self._h = 0
        if self.paths:
            first = cv2.imread(self.paths[0])
            if first is not None:
                self._h, self._w = first.shape[:2]

    def isOpened(self):
        return len(self.paths) > 0 and self._w > 0

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return (max(0, self.idx - 1) / self.fps) * 1000.0
        return 0.0

    def read(self):
        while self.idx < len(self.paths):
            frame = cv2.imread(self.paths[self.idx])
            self.idx += 1
            if frame is not None:
                return True, frame
        return False, None

    def release(self):
        pass


def open_capture(source: str, env: dict):
    """Buka sumber: webcam / RTSP / file video / folder gambar / glob / 1 gambar."""
    src_str = str(source)
    if is_webcam(src_str):
        backend = pick_camera_backend(env["os"])
        return cv2.VideoCapture(int(src_str), backend)
    if src_str.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://")):
        return cv2.VideoCapture(src_str, cv2.CAP_FFMPEG)

    # Pola glob (mis. frames/*.jpg)
    if any(c in src_str for c in "*?["):
        import glob as _glob
        imgs = sorted(x for x in _glob.glob(src_str)
                      if Path(x).suffix.lower() in IMAGE_EXTS)
        if not imgs:
            sys.exit(f"ERROR: tidak ada gambar cocok pola: {src_str}")
        print(f"[sumber] {len(imgs)} gambar (pola glob)")
        return ImageFolderCapture(imgs)

    p = Path(src_str)
    if not p.exists():
        sys.exit(f"ERROR: sumber tidak ditemukan: {src_str}")
    if p.is_dir():
        imgs = sorted(x for x in p.iterdir()
                      if x.suffix.lower() in IMAGE_EXTS)
        if not imgs:
            sys.exit(f"ERROR: folder tidak berisi gambar: {src_str}")
        print(f"[sumber] {len(imgs)} gambar dari folder {p}")
        return ImageFolderCapture(imgs)
    if p.suffix.lower() in IMAGE_EXTS:
        return ImageFolderCapture([p])
    # File video / footage drone
    return cv2.VideoCapture(src_str)


# --------------------------------------------------------------------------- #
# Visualisasi
# --------------------------------------------------------------------------- #
def color_for_class(class_id: int) -> tuple:
    """Warna BGR deterministik per class id."""
    palette = [
        (56, 56, 255), (50, 205, 50), (255, 178, 29), (207, 210, 49),
        (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52),
        (0, 212, 187), (44, 153, 168), (255, 91, 0), (133, 56, 255),
    ]
    return palette[class_id % len(palette)]


def draw_detection(frame, x1, y1, x2, y2, label, color) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ytop = max(0, y1 - th - baseline - 4)
    cv2.rectangle(frame, (x1, ytop), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_hud(frame, fps: float, total_dets: int, frame_idx: int) -> None:
    """Heads-up display: FPS + counter."""
    h = frame.shape[0]
    txt = f"FPS: {fps:5.1f} | Frame: {frame_idx} | Deteksi: {total_dets}"
    cv2.rectangle(frame, (0, h - 28), (360, h), (0, 0, 0), -1)
    cv2.putText(
        frame, txt, (8, h - 9),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
    )


# --------------------------------------------------------------------------- #
# CSV logger
# --------------------------------------------------------------------------- #
CSV_HEADER = [
    "detection_id", "timestamp_iso", "epoch_s", "frame_idx", "video_time_s",
    "class_id", "class_name", "confidence",
    "x1", "y1", "x2", "y2", "bbox_w", "bbox_h", "area_px",
    "width_m", "length_m", "area_m2",   # diisi bila --calib (estimasi ukuran)
    "frame_w", "frame_h",
    "lat", "lon",            # diisi di Fase 3 (GPS); sekarang kosong
    "screenshot", "source",
]


def open_csv(path: Path):
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(CSV_HEADER)
    return f, writer


# --------------------------------------------------------------------------- #
# Argumen
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Road Survey AI — Fase 1 (deteksi jalan rusak, inferensi lokal)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default="0",
                   help="Sumber: indeks webcam (0), path file video, atau URL RTSP.")
    p.add_argument("--model", default="yolov8n.pt",
                   help="Path bobot YOLO. Default COCO HANYA untuk uji pipeline "
                        "(bukan deteksi lubang/retak). Ganti di Fase 2.")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Ambang confidence minimal.")
    p.add_argument("--iou", type=float, default=0.45, help="Ambang IoU NMS.")
    p.add_argument("--imgsz", type=int, default=640, help="Ukuran input inferensi.")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | 0 | mps. 'auto' = deteksi otomatis.")
    p.add_argument("--classes", default=None,
                   help="Filter id kelas, dipisah koma (mis. '0,3'). Kosong = semua.")
    p.add_argument("--output", default="output",
                   help="Folder output (sesi dibuat di dalamnya).")
    p.add_argument("--session-name", default=None,
                   help="Nama sesi survey (default: timestamp).")
    p.add_argument("--max-shots-per-class", type=int, default=50,
                   help="Batas screenshot per kelas biar tidak banjir.")
    p.add_argument("--shot-cooldown", type=int, default=15,
                   help="Jeda minimal (frame) antar screenshot untuk kelas sama.")
    p.add_argument("--stride", type=int, default=1,
                   help="Proses tiap N frame (>1 mempercepat video panjang di CPU).")
    p.add_argument("--no-display", action="store_true",
                   help="Tanpa window (mesin headless / proses batch).")
    p.add_argument("--save-video", action="store_true",
                   help="Simpan video hasil anotasi ke folder sesi.")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Berhenti setelah N frame (0 = tanpa batas). Berguna untuk uji.")
    # --- Fase 3: GPS + peta ---
    p.add_argument("--gps", default="auto",
                   help="Sumber GPS: auto | none | srt:FILE.SRT | gpx:TRACK.GPX | "
                        "nmea:PORT@BAUD | tcp:HOST:PORT | fixed:LAT,LON. "
                        "'auto' mencari .SRT di sebelah video (footage drone DJI).")
    p.add_argument("--gps-start", default=None,
                   help="Waktu mulai video (ISO/epoch) untuk pencocokan GPX HP.")
    p.add_argument("--map", action="store_true",
                   help="Generate peta HTML + GeoJSON dari deteksi setelah selesai.")
    p.add_argument("--calib", default=None,
                   help="File kalibrasi kamera (configs/camera_calib.json) untuk "
                        "estimasi ukuran nyata kerusakan (m). Lihat calibrate_camera.py.")
    # --- Fase 4: laporan ---
    p.add_argument("--report", action="store_true",
                   help="Generate laporan PDF + Excel setelah selesai (sekali jalan).")
    p.add_argument("--road", default=None, help="Nama ruas jalan (untuk laporan).")
    p.add_argument("--surveyor", default="-", help="Nama surveyor (untuk laporan).")
    return p.parse_args(argv)


def resolve_device(requested: str, env: dict) -> str:
    if requested == "auto":
        return env["device"]
    return requested


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    env = detect_environment()
    device = resolve_device(args.device, env)

    # Import ultralytics setelah cek env supaya pesan error lebih jelas.
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ERROR: ultralytics belum terpasang.\n"
            "  pip install ultralytics\n"
            "Lihat README.md untuk install + PyTorch CUDA (Windows)."
        )

    # Muat model (batasan #2: path model = argumen, mudah diganti).
    if not Path(args.model).exists() and "/" in args.model.replace("\\", "/"):
        print(f"PERINGATAN: file model '{args.model}' tidak ada di disk; "
              f"ultralytics akan coba mengunduh bila itu nama bobot resmi.")
    model = YOLO(args.model)
    class_names = model.names if hasattr(model, "names") else {}

    print_banner(env, args, class_names)

    class_filter = None
    if args.classes:
        class_filter = [int(c) for c in args.classes.split(",") if c.strip() != ""]

    # Siapkan folder sesi: output/<session>/{screenshots, detections.csv}
    session = args.session_name or datetime.now().strftime("survey_%Y%m%d_%H%M%S")
    session_dir = Path(args.output) / session
    shots_dir = session_dir / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = session_dir / "detections.csv"
    csv_file, csv_writer = open_csv(csv_path)
    print(f"  Sesi        : {session}")
    print(f"  CSV         : {csv_path}")
    print(f"  Screenshot  : {shots_dir}")
    print("-" * 64)

    # Buka sumber.
    cap = open_capture(args.source, env)
    if not cap.isOpened():
        csv_file.close()
        sys.exit(f"ERROR: tidak bisa membuka sumber '{args.source}'. "
                 "Cek indeks webcam / path file / URL RTSP.")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    # --- Fase 3: provider GPS ---
    gps = None
    try:
        import geo
        start_epoch = None
        if args.gps_start:
            try:
                start_epoch = float(args.gps_start)
            except ValueError:
                start_epoch = datetime.fromisoformat(args.gps_start).timestamp()
        gps = geo.make_provider(args.gps, source=str(args.source),
                                video_start_epoch=start_epoch)
        if gps:
            print(f"  GPS         : {gps.kind} aktif")
    except Exception as e:
        print(f"  GPS         : gagal inisialisasi ({e}); lanjut tanpa GPS.")
        gps = None
    n_geo = 0  # jumlah deteksi dgn koordinat

    # --- estimasi ukuran (kalibrasi kamera) ---
    calib_H = None
    if args.calib:
        try:
            import sizing
            cal = sizing.load_calibration(args.calib)
            if cal:
                calib_H = cal["homography"]
                print(f"  Ukuran      : kalibrasi aktif ({args.calib})")
            else:
                print(f"  Ukuran      : file kalibrasi tidak ada ({args.calib})")
        except Exception as e:
            print(f"  Ukuran      : gagal muat kalibrasi ({e})")

    writer_out = None
    if args.save_video:
        out_path = session_dir / "annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = src_fps if src_fps and src_fps > 1 else 20.0
        # ukuran ditentukan saat frame pertama bila fw/fh 0 (mis. beberapa stream)

    # State.
    shots_per_class = defaultdict(int)
    last_shot_frame = defaultdict(lambda: -10_000)
    detection_id = 0
    total_dets = 0
    frame_idx = 0
    fps_ema = 0.0          # exponential moving average FPS
    alpha = 0.1
    window = "Road Survey AI — Fase 1  (q: keluar, p: pause, s: screenshot)"
    paused = False

    print("Mulai loop deteksi. Tekan 'q' untuk keluar.\n")
    try:
        while True:
            if not paused:
                t0 = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Sumber selesai / tidak ada frame lagi.")
                    break
                frame_idx += 1

                # stride: lewati frame untuk video panjang di CPU.
                if args.stride > 1 and (frame_idx % args.stride) != 0:
                    continue

                if fw == 0 or fh == 0:
                    fh, fw = frame.shape[:2]

                # Inferensi LOKAL (tidak ada frame ke cloud).
                results = model.predict(
                    frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                    device=device, classes=class_filter, verbose=False,
                )
                r = results[0]
                ts = datetime.now()
                video_time = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0

                # Fase 3: koordinat GPS untuk frame ini.
                fix = gps.get_fix(video_time) if gps else None
                lat_s = f"{fix.lat:.7f}" if fix else ""
                lon_s = f"{fix.lon:.7f}" if fix else ""

                frame_class_saved = set()
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                        cls_name = class_names.get(cls_id, str(cls_id))
                        color = color_for_class(cls_id)
                        label = f"{cls_name} {conf:.2f}"
                        draw_detection(frame, x1, y1, x2, y2, label, color)

                        detection_id += 1
                        total_dets += 1
                        bw, bh = x2 - x1, y2 - y1

                        # Estimasi ukuran nyata (m) bila kamera terkalibrasi.
                        w_m = l_m = a_m2 = ""
                        if calib_H is not None:
                            import sizing
                            sz = sizing.bbox_ground_size(calib_H, x1, y1, x2, y2)
                            w_m, l_m, a_m2 = sz["width_m"], sz["length_m"], sz["area_m2"]

                        # Screenshot: dibatasi per kelas + cooldown agar variatif.
                        shot_name = ""
                        if (shots_per_class[cls_name] < args.max_shots_per_class
                                and cls_name not in frame_class_saved
                                and frame_idx - last_shot_frame[cls_name]
                                >= args.shot_cooldown):
                            shots_per_class[cls_name] += 1
                            last_shot_frame[cls_name] = frame_idx
                            frame_class_saved.add(cls_name)
                            safe = cls_name.replace(" ", "_").replace("/", "-")
                            shot_name = (f"{safe}_{shots_per_class[cls_name]:04d}"
                                         f"_f{frame_idx}.jpg")

                        if fix:
                            n_geo += 1
                        csv_writer.writerow([
                            detection_id, ts.isoformat(timespec="milliseconds"),
                            f"{ts.timestamp():.3f}", frame_idx, f"{video_time:.3f}",
                            cls_id, cls_name, f"{conf:.4f}",
                            x1, y1, x2, y2, bw, bh, bw * bh,
                            w_m, l_m, a_m2,
                            fw, fh, lat_s, lon_s,
                            shot_name, args.source,
                        ])

                # FPS (EMA).
                dt = time.time() - t0
                inst_fps = 1.0 / dt if dt > 0 else 0.0
                fps_ema = inst_fps if fps_ema == 0 else (
                    alpha * inst_fps + (1 - alpha) * fps_ema
                )
                draw_hud(frame, fps_ema, total_dets, frame_idx)

                # Simpan screenshot anotasi untuk kelas yang baru ter-capture
                # di frame ini (sudah lolos batas per-kelas + cooldown di atas).
                if frame_class_saved:
                    # Simpan satu file annotated per kelas yang baru ke-capture.
                    for cls_name in frame_class_saved:
                        safe = cls_name.replace(" ", "_").replace("/", "-")
                        fn = (f"{safe}_{shots_per_class[cls_name]:04d}"
                              f"_f{frame_idx}.jpg")
                        cv2.imwrite(str(shots_dir / fn), frame)

                # Video output.
                if args.save_video:
                    if writer_out is None:
                        out_path = session_dir / "annotated.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out_fps = src_fps if src_fps and src_fps > 1 else 20.0
                        writer_out = cv2.VideoWriter(
                            str(out_path), fourcc, out_fps,
                            (frame.shape[1], frame.shape[0]))
                    writer_out.write(frame)

                if args.max_frames and frame_idx >= args.max_frames:
                    print(f"Mencapai --max-frames={args.max_frames}, berhenti.")
                    break

            # Tampilkan window (kalau ada display).
            if not args.no_display:
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Keluar (q).")
                    break
                if key == ord("p"):
                    paused = not paused
                    print("Pause." if paused else "Lanjut.")
                if key == ord("s"):
                    man = shots_dir / f"manual_f{frame_idx}.jpg"
                    cv2.imwrite(str(man), frame)
                    print(f"Screenshot manual -> {man}")
            elif paused:
                break  # tidak ada cara unpause tanpa display
    except KeyboardInterrupt:
        print("\nDihentikan (Ctrl-C).")
    finally:
        cap.release()
        if writer_out is not None:
            writer_out.release()
        csv_file.close()
        if gps is not None:
            gps.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    # Ringkasan.
    print("-" * 64)
    print("SELESAI. Ringkasan sesi:")
    print(f"  Total frame diproses : {frame_idx}")
    print(f"  Total deteksi        : {total_dets}")
    print(f"  Deteksi ber-GPS      : {n_geo}")
    if shots_per_class:
        print("  Screenshot tersimpan :")
        for cls_name, n in sorted(shots_per_class.items()):
            print(f"     - {cls_name:<22}: {n}")
    print(f"  CSV   : {csv_path}")
    print(f"  Folder: {session_dir}")

    # Fase 3: generate peta otomatis.
    if args.map:
        if n_geo == 0:
            print("  Peta  : dilewati (tidak ada deteksi ber-koordinat). "
                  "Pastikan --gps benar.")
        else:
            try:
                import make_map
                make_map.build_from_csv(csv_path, session_dir, screenshots=True)
            except SystemExit as e:
                print(f"  Peta  : {e}")
            except Exception as e:
                print(f"  Peta  : gagal ({e})")

    # Fase 4: laporan PDF + Excel.
    if args.report:
        if total_dets == 0:
            print("  Laporan: dilewati (tidak ada deteksi).")
        else:
            try:
                import make_report
                meta = {
                    "Ruas jalan": args.road or session,
                    "Tanggal survey": datetime.now().strftime("%Y-%m-%d"),
                    "Surveyor": args.surveyor,
                    "Konsultan": "Rustika Citra Group",
                    "Wilayah": "Kabupaten Luwu Timur",
                }
                make_report.build_report(csv_path, session_dir, meta)
            except Exception as e:
                print(f"  Laporan: gagal ({e})")
    print("-" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
