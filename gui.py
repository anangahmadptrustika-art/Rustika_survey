#!/usr/bin/env python3
"""
gui.py — Road Survey AI: aplikasi desktop (Fase 5).

Jendela aplikasi supaya dipakai tanpa ngetik perintah:
  - Pilih sumber (webcam / file video / folder gambar / RTSP) lewat klik
  - Pilih model (.pt)
  - Start / Stop
  - Lihat video live + counter deteksi per jenis kerusakan + FPS
  - Export laporan (PDF + Excel) & peta sekali klik
  - Buka folder hasil

Deteksi tetap 100% LOKAL (offline). GUI ini cuma "pembungkus" loop deteksi yang
sama dengan road_survey.py (helper-nya dipakai ulang).

Jalankan: dobel-klik "Road Survey AI.bat", atau:  python gui.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2

try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QImage, QPixmap, QFont
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
        QLineEdit, QFileDialog, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox,
        QDoubleSpinBox, QMessageBox, QSizePolicy,
    )
except ImportError:
    sys.exit("PySide6 belum terpasang. Jalankan dulu:  pip install pyside6")

import road_survey as rs


# --------------------------------------------------------------------------- #
# Worker thread: loop deteksi (biar UI tidak macet)
# --------------------------------------------------------------------------- #
class DetectionWorker(QThread):
    frame_ready = Signal(object)   # numpy BGR frame ber-anotasi
    stats = Signal(dict)
    log = Signal(str)
    done = Signal(str)             # path folder sesi

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        cfg = self.cfg
        try:
            from ultralytics import YOLO
        except ImportError:
            self.log.emit("ERROR: ultralytics belum terpasang."); return

        env = rs.detect_environment()
        device = env["device"]
        try:
            model = YOLO(cfg["model"])
        except Exception as e:
            self.log.emit(f"ERROR muat model: {e}"); return
        names = model.names

        session = datetime.now().strftime("survey_%Y%m%d_%H%M%S")
        session_dir = Path(cfg["output"]) / session
        shots_dir = session_dir / "screenshots"
        shots_dir.mkdir(parents=True, exist_ok=True)
        csv_path = session_dir / "detections.csv"
        cf = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(cf); writer.writerow(rs.CSV_HEADER)

        gps = None
        try:
            import geo
            gps = geo.make_provider(cfg.get("gps", "none"), source=str(cfg["source"]))
        except Exception as e:
            self.log.emit(f"GPS nonaktif: {e}")
        calib_H = None
        if cfg.get("calib") and Path(cfg["calib"]).exists():
            try:
                import sizing
                cal = sizing.load_calibration(cfg["calib"])
                calib_H = cal["homography"] if cal else None
            except Exception:
                calib_H = None

        cap = rs.open_capture(cfg["source"], env)
        if not cap.isOpened():
            self.log.emit(f"ERROR: tidak bisa membuka sumber '{cfg['source']}'.")
            cf.close(); return

        self.log.emit(f"Mulai · {device.upper()} · model {Path(cfg['model']).name} "
                      f"· GPS {gps.kind if gps else 'tidak ada'}"
                      f"{' · ukuran ON' if calib_H is not None else ''}")
        shots_per_class = defaultdict(int)
        last_shot = defaultdict(lambda: -10_000)
        per_class = defaultdict(int)
        det_id = total = frame_idx = n_geo = 0
        fps_ema = 0.0
        while not self._stop:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            res = model.predict(frame, conf=cfg["conf"], iou=0.45,
                                imgsz=cfg.get("imgsz", 640), device=device,
                                verbose=False)[0]
            ts = datetime.now()
            vt = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            fix = gps.get_fix(vt) if gps else None
            lat_s = f"{fix.lat:.7f}" if fix else ""
            lon_s = f"{fix.lon:.7f}" if fix else ""
            fh, fw = frame.shape[:2]
            frame_saved = set()
            if res.boxes is not None:
                for b in res.boxes:
                    cid = int(b.cls[0]); conf = float(b.conf[0])
                    x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                    cname = names.get(cid, str(cid))
                    rs.draw_detection(frame, x1, y1, x2, y2,
                                      f"{cname} {conf:.2f}", rs.color_for_class(cid))
                    det_id += 1; total += 1; per_class[cname] += 1
                    bw, bh = x2 - x1, y2 - y1
                    w_m = l_m = a_m2 = ""
                    if calib_H is not None:
                        import sizing
                        sz = sizing.bbox_ground_size(calib_H, x1, y1, x2, y2)
                        w_m, l_m, a_m2 = sz["width_m"], sz["length_m"], sz["area_m2"]
                    shot = ""
                    if (shots_per_class[cname] < cfg.get("max_shots", 50)
                            and cname not in frame_saved
                            and frame_idx - last_shot[cname] >= cfg.get("cooldown", 15)):
                        shots_per_class[cname] += 1
                        last_shot[cname] = frame_idx
                        frame_saved.add(cname)
                        safe = cname.replace(" ", "_").replace("/", "-")
                        shot = f"{safe}_{shots_per_class[cname]:04d}_f{frame_idx}.jpg"
                    if fix:
                        n_geo += 1
                    writer.writerow([
                        det_id, ts.isoformat(timespec="milliseconds"),
                        f"{ts.timestamp():.3f}", frame_idx, f"{vt:.3f}",
                        cid, cname, f"{conf:.4f}", x1, y1, x2, y2, bw, bh, bw * bh,
                        w_m, l_m, a_m2, fw, fh, lat_s, lon_s, shot, str(cfg["source"]),
                    ])
            dt = time.time() - t0
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_ema = inst if fps_ema == 0 else 0.1 * inst + 0.9 * fps_ema
            rs.draw_hud(frame, fps_ema, total, frame_idx)
            for cname in frame_saved:
                safe = cname.replace(" ", "_").replace("/", "-")
                fn = f"{safe}_{shots_per_class[cname]:04d}_f{frame_idx}.jpg"
                cv2.imwrite(str(shots_dir / fn), frame)
            self.frame_ready.emit(frame.copy())
            self.stats.emit({"fps": fps_ema, "frame": frame_idx, "total": total,
                             "per_class": dict(per_class), "geo": n_geo})

        cap.release()
        if gps:
            gps.close()
        cf.close()
        self.done.emit(str(session_dir))


# --------------------------------------------------------------------------- #
# Jendela utama
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Road Survey AI — Rustika Citra Group")
        self.resize(1180, 720)
        self.worker = None
        self.last_session = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---- Panel kiri: kontrol ----
        panel = QVBoxLayout()
        form_box = QGroupBox("Pengaturan Survei")
        form = QFormLayout(form_box)

        self.source_type = QComboBox()
        self.source_type.addItems(["Webcam", "File video", "Folder gambar", "RTSP / HP"])
        self.source_type.currentIndexChanged.connect(self._on_source_type)
        form.addRow("Sumber:", self.source_type)

        src_row = QHBoxLayout()
        self.source_edit = QLineEdit("0")
        self.browse_btn = QPushButton("Pilih…")
        self.browse_btn.clicked.connect(self._browse_source)
        src_row.addWidget(self.source_edit); src_row.addWidget(self.browse_btn)
        form.addRow("Lokasi:", self._wrap(src_row))

        model_row = QHBoxLayout()
        self.model_edit = QLineEdit("models/road_damage.pt")
        mbtn = QPushButton("Pilih…"); mbtn.clicked.connect(self._browse_model)
        model_row.addWidget(self.model_edit); model_row.addWidget(mbtn)
        form.addRow("Model:", self._wrap(model_row))

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95); self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.30)
        form.addRow("Confidence:", self.conf_spin)

        self.gps_combo = QComboBox()
        self.gps_combo.addItems(["Tidak ada", "Auto (.SRT drone di sebelah video)"])
        form.addRow("GPS:", self.gps_combo)

        self.road_edit = QLineEdit(); self.road_edit.setPlaceholderText("mis. Ruas Malili–Wawondula")
        form.addRow("Nama ruas:", self.road_edit)
        self.surveyor_edit = QLineEdit(); self.surveyor_edit.setPlaceholderText("nama surveyor")
        form.addRow("Surveyor:", self.surveyor_edit)

        panel.addWidget(form_box)

        # Tombol Start/Stop
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  MULAI")
        self.start_btn.setStyleSheet("background:#2ca02c;color:white;font-weight:bold;padding:10px;")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("■  STOP")
        self.stop_btn.setStyleSheet("background:#d62728;color:white;font-weight:bold;padding:10px;")
        self.stop_btn.clicked.connect(self.stop); self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn); btn_row.addWidget(self.stop_btn)
        panel.addLayout(btn_row)

        # Counter
        cbox = QGroupBox("Hasil Deteksi (live)")
        cl = QVBoxLayout(cbox)
        self.counter_label = QLabel("Belum mulai.")
        self.counter_label.setFont(QFont("Consolas", 11))
        self.counter_label.setAlignment(Qt.AlignTop)
        self.counter_label.setWordWrap(True)
        cl.addWidget(self.counter_label)
        panel.addWidget(cbox, 1)

        # Export
        ex_row = QHBoxLayout()
        self.export_btn = QPushButton("📄  Export Laporan + Peta")
        self.export_btn.clicked.connect(self.export); self.export_btn.setEnabled(False)
        self.open_btn = QPushButton("📂  Buka Folder Hasil")
        self.open_btn.clicked.connect(self.open_folder); self.open_btn.setEnabled(False)
        ex_row.addWidget(self.export_btn); ex_row.addWidget(self.open_btn)
        panel.addLayout(ex_row)

        left = QWidget(); left.setLayout(panel); left.setFixedWidth(380)
        root.addWidget(left)

        # ---- Panel kanan: video ----
        self.video = QLabel("Tekan MULAI untuk menjalankan deteksi.")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#111;color:#888;")
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setMinimumSize(640, 480)
        root.addWidget(self.video, 1)

        self.statusBar().showMessage("Siap. Inferensi lokal (offline).")

    @staticmethod
    def _wrap(layout):
        w = QWidget(); w.setLayout(layout); return w

    # ---- handler sumber ----
    def _on_source_type(self):
        t = self.source_type.currentText()
        defaults = {"Webcam": "0", "File video": "", "Folder gambar": "",
                    "RTSP / HP": "rtsp://"}
        self.source_edit.setText(defaults.get(t, ""))
        self.browse_btn.setEnabled(t in ("File video", "Folder gambar"))

    def _browse_source(self):
        t = self.source_type.currentText()
        if t == "File video":
            f, _ = QFileDialog.getOpenFileName(self, "Pilih video", "",
                "Video (*.mp4 *.mov *.avi *.mkv *.MP4);;Semua (*.*)")
            if f:
                self.source_edit.setText(f)
        elif t == "Folder gambar":
            d = QFileDialog.getExistingDirectory(self, "Pilih folder gambar")
            if d:
                self.source_edit.setText(d)

    def _browse_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Pilih model", "models",
                                           "Model YOLO (*.pt)")
        if f:
            self.model_edit.setText(f)

    # ---- start / stop ----
    def start(self):
        src = self.source_edit.text().strip()
        if not src:
            QMessageBox.warning(self, "Sumber kosong", "Isi/Pilih sumber dulu."); return
        if not Path(self.model_edit.text().strip()).exists():
            QMessageBox.warning(self, "Model tidak ada",
                                f"File model tidak ditemukan:\n{self.model_edit.text()}"); return
        cfg = {
            "source": src,
            "model": self.model_edit.text().strip(),
            "conf": float(self.conf_spin.value()),
            "output": "output",
            "gps": "auto" if self.gps_combo.currentIndex() == 1 else "none",
            "calib": "configs/camera_calib.json",
            "road": self.road_edit.text().strip(),
            "surveyor": self.surveyor_edit.text().strip() or "-",
        }
        self.worker = DetectionWorker(cfg)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.stats.connect(self.on_stats)
        self.worker.log.connect(lambda m: self.statusBar().showMessage(m))
        self.worker.done.connect(self.on_done)
        self.worker.start()
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.statusBar().showMessage("Menghentikan…")

    def on_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(self.video.width(), self.video.height(),
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video.setPixmap(pix)

    def on_stats(self, s):
        lines = [f"FPS        : {s['fps']:5.1f}",
                 f"Frame      : {s['frame']}",
                 f"Total      : {s['total']}",
                 f"Ber-GPS    : {s['geo']}", "", "Per jenis:"]
        pc = s["per_class"]
        if pc:
            for cls, n in sorted(pc.items(), key=lambda x: -x[1]):
                lines.append(f"  • {cls:<20}: {n}")
        else:
            lines.append("  (belum ada)")
        self.counter_label.setText("\n".join(lines))

    def on_done(self, session_dir):
        self.last_session = session_dir
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(True); self.open_btn.setEnabled(True)
        self.statusBar().showMessage(f"Selesai. Hasil di: {session_dir}")

    # ---- export ----
    def export(self):
        if not self.last_session:
            return
        csv_path = Path(self.last_session) / "detections.csv"
        QApplication.setOverrideCursor(Qt.WaitCursor)
        msgs = []
        try:
            import make_map
            try:
                make_map.build_from_csv(csv_path, self.last_session, screenshots=True)
                msgs.append("Peta + GeoJSON dibuat.")
            except SystemExit as e:
                msgs.append(f"Peta dilewati: {e}")
        except Exception as e:
            msgs.append(f"Peta gagal: {e}")
        try:
            import make_report
            meta = {
                "Ruas jalan": self.road_edit.text().strip() or Path(self.last_session).name,
                "Tanggal survey": datetime.now().strftime("%Y-%m-%d"),
                "Surveyor": self.surveyor_edit.text().strip() or "-",
                "Konsultan": "Rustika Citra Group",
                "Wilayah": "Kabupaten Luwu Timur",
            }
            make_report.build_report(csv_path, self.last_session, meta)
            msgs.append("Laporan PDF + Excel dibuat.")
        except Exception as e:
            msgs.append(f"Laporan gagal: {e}")
        QApplication.restoreOverrideCursor()
        QMessageBox.information(self, "Export selesai",
                               "\n".join(msgs) + f"\n\nFolder:\n{self.last_session}")

    def open_folder(self):
        target = self.last_session or "output"
        try:
            os.startfile(os.path.abspath(target))   # Windows
        except AttributeError:
            import subprocess
            subprocess.Popen(["xdg-open", os.path.abspath(target)])

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop(); self.worker.wait(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
