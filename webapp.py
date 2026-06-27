#!/usr/bin/env python3
"""
webapp.py — Road Survey AI sebagai WEB APP lokal (Fase 5b).

Jalan di localhost laptop (offline). Backend Flask membuka webcam/video via
OpenCV, menjalankan YOLO LOKAL, menggambar kotak, dan men-stream frame ke
browser (MJPEG). Browser hanya menampilkan + kontrol — deteksi tetap 100% lokal.

Jalankan:  dobel-klik "Road Survey AI (Web).bat"  atau  python webapp.py
Lalu buka: http://localhost:5000  (otomatis kebuka).

Dipakai ulang: helper road_survey + geo + sizing + make_map + make_report.
"""
from __future__ import annotations

import csv
import threading
import time
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from flask import (Flask, Response, request, jsonify, render_template_string,
                       send_from_directory, abort)
except ImportError:
    raise SystemExit("Flask belum terpasang. Jalankan:  pip install flask")

import road_survey as rs

OUTPUT_DIR = Path("output")
UPLOAD_DIR = Path("uploads")


# --------------------------------------------------------------------------- #
# Util
# --------------------------------------------------------------------------- #
def list_models():
    """Kumpulkan model .pt yang ada (models/ + hasil training runs/)."""
    found = []
    for p in sorted(Path("models").glob("*.pt")) if Path("models").exists() else []:
        found.append(str(p).replace("\\", "/"))
    for p in sorted(Path("runs").rglob("weights/best.pt")) if Path("runs").exists() else []:
        found.append(str(p).replace("\\", "/"))
    # default fallback
    if not found:
        found = ["yolov8n.pt"]
    return found


def probe_cameras(max_idx=4):
    """Cek indeks webcam yang nyambung (0..max_idx-1)."""
    backend = rs.pick_camera_backend(rs.detect_environment()["os"])
    avail = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, backend)
        ok = cap.isOpened()
        if ok:
            r, _ = cap.read()
            ok = r
        cap.release()
        if ok:
            avail.append(i)
    return avail


def placeholder_jpeg(text="Tekan MULAI"):
    img = np.full((480, 640, 3), 28, np.uint8)
    cv2.putText(img, text, (130, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (140, 140, 140), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Engine deteksi (thread latar)
# --------------------------------------------------------------------------- #
class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.error = ""
        self._jpeg = placeholder_jpeg()
        self.stats = self._empty_stats()
        self.session_dir = None
        self._model = None
        self._model_path = None

    def _empty_stats(self):
        return {"fps": 0.0, "frame": 0, "total": 0, "geo": 0,
                "per_class": {}, "session": "", "device": "", "model": ""}

    def get_jpeg(self):
        with self.lock:
            return self._jpeg

    def snapshot_stats(self):
        with self.lock:
            s = dict(self.stats)
            s["running"] = self.running
            s["error"] = self.error
            return s

    def start(self, cfg):
        if self.running:
            return False, "Sudah berjalan."
        self.error = ""
        self.stats = self._empty_stats()
        self.running = True
        self.thread = threading.Thread(target=self._loop, args=(cfg,), daemon=True)
        self.thread.start()
        return True, "ok"

    def stop(self):
        self.running = False

    def _load_model(self, path):
        if self._model is not None and self._model_path == path:
            return self._model
        from ultralytics import YOLO
        self._model = YOLO(path)
        self._model_path = path
        return self._model

    def _loop(self, cfg):
        try:
            env = rs.detect_environment()
            device = env["device"]
            model = self._load_model(cfg["model"])
            names = model.names
        except Exception as e:
            self.error = f"Gagal muat model: {e}"; self.running = False; return

        session = datetime.now().strftime("survey_%Y%m%d_%H%M%S")
        session_dir = OUTPUT_DIR / session
        shots_dir = session_dir / "screenshots"
        shots_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = session_dir
        cf = open(session_dir / "detections.csv", "w", newline="", encoding="utf-8")
        writer = csv.writer(cf); writer.writerow(rs.CSV_HEADER)

        gps = None
        try:
            import geo
            gps = geo.make_provider(cfg.get("gps", "none"), source=str(cfg["source"]))
        except Exception as e:
            self.error = f"GPS nonaktif: {e}"
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
            self.error = f"Tidak bisa membuka sumber '{cfg['source']}'."
            cf.close(); self.running = False; return

        with self.lock:
            self.stats.update({"session": session, "device": device.upper(),
                               "model": Path(cfg["model"]).name})

        shots_per_class = defaultdict(int)
        last_shot = defaultdict(lambda: -10_000)
        per_class = defaultdict(int)
        det_id = total = frame_idx = n_geo = 0
        fps_ema = 0.0
        try:
            while self.running:
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
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self.lock:
                    if ok2:
                        self._jpeg = buf.tobytes()
                    self.stats.update({"fps": round(fps_ema, 1), "frame": frame_idx,
                                       "total": total, "geo": n_geo,
                                       "per_class": dict(per_class)})
        finally:
            cap.release()
            if gps:
                gps.close()
            cf.close()
            self.running = False
            with self.lock:
                self._jpeg = placeholder_jpeg("Berhenti — tekan MULAI")


engine = Engine()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 ** 3   # izinkan video drone besar (≤16 GB)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template_string(PAGE, models=list_models())


@app.route("/video_feed")
def video_feed():
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            jpg = engine.get_jpeg()
            yield boundary + jpg + b"\r\n"
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/cameras")
def cameras():
    if engine.running:
        return jsonify(error="Hentikan deteksi dulu sebelum cek kamera.", cameras=[])
    return jsonify(cameras=probe_cameras())


@app.route("/start", methods=["POST"])
def start():
    d = request.get_json(force=True)
    cfg = {
        "source": str(d.get("source", "0")).strip(),
        "model": d.get("model", "models/road_damage.pt"),
        "conf": float(d.get("conf", 0.3)),
        "gps": d.get("gps", "none"),
        "calib": "configs/camera_calib.json",
        "road": d.get("road", ""),
        "surveyor": d.get("surveyor", "-"),
    }
    if not Path(cfg["model"]).exists():
        return jsonify(ok=False, error=f"Model tidak ada: {cfg['model']}")
    ok, msg = engine.start(cfg)
    return jsonify(ok=ok, error="" if ok else msg)


@app.route("/stop", methods=["POST"])
def stop():
    engine.stop()
    return jsonify(ok=True)


@app.route("/upload", methods=["POST"])
def upload():
    """Terima video drone (+ .SRT opsional) dari browser, simpan ke uploads/."""
    from werkzeug.utils import secure_filename
    v = request.files.get("video")
    if not v or not v.filename:
        return jsonify(ok=False, error="Tidak ada file video.")
    UPLOAD_DIR.mkdir(exist_ok=True)
    vname = secure_filename(v.filename) or "drone.mp4"
    stem = Path(vname).stem
    vpath = UPLOAD_DIR / vname
    v.save(str(vpath))
    srt_saved = False
    s = request.files.get("srt")
    if s and s.filename:
        # simpan .SRT dengan nama dasar yang sama -> dideteksi otomatis (GPS auto)
        s.save(str(UPLOAD_DIR / (stem + ".SRT")))
        srt_saved = True
    return jsonify(ok=True, video_path=str(vpath).replace("\\", "/"), srt=srt_saved)


@app.route("/stats")
def stats():
    return jsonify(engine.snapshot_stats())


@app.route("/export", methods=["POST"])
def export():
    d = request.get_json(force=True)
    if not engine.session_dir:
        return jsonify(ok=False, error="Belum ada sesi. Jalankan deteksi dulu.")
    sd = Path(engine.session_dir)
    csv_path = sd / "detections.csv"
    out = {"ok": True, "files": [], "messages": []}
    try:
        import make_map
        try:
            make_map.build_from_csv(csv_path, sd, screenshots=True)
            out["messages"].append("Peta + GeoJSON dibuat.")
        except SystemExit as e:
            out["messages"].append(f"Peta dilewati: {e}")
    except Exception as e:
        out["messages"].append(f"Peta gagal: {e}")
    try:
        import make_report
        meta = {
            "Ruas jalan": d.get("road") or sd.name,
            "Tanggal survey": datetime.now().strftime("%Y-%m-%d"),
            "Surveyor": d.get("surveyor") or "-",
            "Konsultan": "Rustika Citra Group",
            "Wilayah": "Kabupaten Luwu Timur",
        }
        make_report.build_report(csv_path, sd, meta)
        out["messages"].append("Laporan PDF + Excel dibuat.")
    except Exception as e:
        out["ok"] = False
        out["messages"].append(f"Laporan gagal: {e}")
    sess = sd.name
    for fn in ("laporan.pdf", "laporan.xlsx", "map.html"):
        if (sd / fn).exists():
            out["files"].append({"name": fn, "url": f"/file/{sess}/{fn}"})
    out["folder"] = str(sd.resolve())
    return jsonify(out)


@app.route("/file/<path:relpath>")
def serve_file(relpath):
    # Hanya layani dari dalam folder output (cegah path traversal).
    base = OUTPUT_DIR.resolve()
    target = (base / relpath).resolve()
    if base not in target.parents and target != base:
        abort(403)
    if not target.exists():
        abort(404)
    return send_from_directory(target.parent, target.name)


# --------------------------------------------------------------------------- #
# Halaman (HTML + CSS + JS, self-contained / offline)
# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Road Survey AI — Rustika Citra Group</title>
<style>
  :root{
    --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --line:#2a2f3a;
    --txt:#e7eaf0; --muted:#9aa3b2; --brand:#2f80ed; --green:#27ae60;
    --red:#e74c3c; --amber:#f39c12;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
    font-family:Segoe UI,system-ui,Arial,sans-serif;}
  header{display:flex;align-items:center;gap:12px;padding:12px 20px;
    background:linear-gradient(90deg,#10243f,#0f1115);border-bottom:1px solid var(--line)}
  header .logo{width:34px;height:34px;border-radius:8px;background:var(--brand);
    display:flex;align-items:center;justify-content:center;font-weight:700}
  header h1{font-size:17px;margin:0;font-weight:600}
  header .sub{color:var(--muted);font-size:12px}
  .pill{margin-left:auto;font-size:12px;padding:5px 10px;border-radius:999px;
    background:#10331f;color:#5bd98a;border:1px solid #1c5634}
  .wrap{display:grid;grid-template-columns:340px 1fr;gap:16px;padding:16px;
    max-width:1400px;margin:0 auto}
  @media(max-width:900px){.wrap{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
  .card h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;
    color:var(--muted);margin:0 0 12px}
  label{display:block;font-size:13px;color:var(--muted);margin:10px 0 4px}
  input,select{width:100%;padding:9px 10px;background:var(--panel2);
    border:1px solid var(--line);border-radius:8px;color:var(--txt);font-size:14px}
  .row{display:flex;gap:8px}
  .row>*{flex:1}
  .btn{border:0;border-radius:9px;padding:11px;font-weight:700;cursor:pointer;
    font-size:14px;color:#fff}
  .btn.start{background:var(--green)} .btn.stop{background:var(--red)}
  .btn.ghost{background:var(--panel2);color:var(--txt);border:1px solid var(--line);font-weight:600}
  .btn:disabled{opacity:.45;cursor:not-allowed}
  .btns{display:flex;gap:8px;margin-top:14px}
  .small{font-size:12px;color:var(--muted)}
  #video{width:100%;border-radius:12px;background:#000;display:block;aspect-ratio:4/3;object-fit:contain}
  .statgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:12px}
  .stat{background:var(--panel2);border:1px solid var(--line);border-radius:10px;
    padding:12px;text-align:center}
  .stat .n{font-size:24px;font-weight:700} .stat .l{font-size:11px;color:var(--muted)}
  .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
  .chip{display:flex;align-items:center;gap:8px;background:var(--panel2);
    border:1px solid var(--line);border-radius:999px;padding:6px 12px;font-size:13px}
  .dot{width:10px;height:10px;border-radius:50%}
  .status{margin-top:10px;font-size:13px;color:var(--muted);min-height:18px}
  a.dl{color:#7bb8ff;font-size:13px;display:inline-block;margin-right:14px}
  .badge{font-size:11px;padding:2px 8px;border-radius:6px;background:#222a38;color:var(--muted)}
</style>
</head>
<body>
<header>
  <div class="logo">R</div>
  <div>
    <h1>Road Survey AI</h1>
    <div class="sub">Rustika Citra Group · Deteksi Kerusakan Jalan</div>
  </div>
  <div class="pill">● Lokal / Offline</div>
</header>

<div class="wrap">
  <!-- Kontrol -->
  <div class="card">
    <h2>Pengaturan Survei</h2>
    <label>Sumber</label>
    <select id="srcType">
      <option value="drone">Upload Video Drone (DJI)</option>
      <option value="webcam">Webcam (kamera mobil / live)</option>
      <option value="folder">Folder gambar</option>
    </select>

    <div id="droneRow">
      <label>Video drone (.MP4)</label>
      <input type="file" id="videoFile" accept="video/*,.mp4,.mov,.avi,.mkv">
      <label>File GPS (.SRT) <span class="small">— buat titik koordinat</span></label>
      <input type="file" id="srtFile" accept=".srt,.SRT">
      <div class="small">Pilih <b>DJI_xxxx.MP4</b> + <b>DJI_xxxx.SRT</b> dari SD card drone
        (dua-duanya). Tanpa .SRT, deteksi tetap jalan tapi tanpa lokasi.</div>
    </div>
    <div id="camRow" style="display:none">
      <label>Nomor kamera <span class="small">(0=laptop, 1/2=USB external)</span></label>
      <div class="row">
        <input id="camIdx" value="0">
        <button class="btn ghost" id="checkCam" style="flex:0 0 auto;padding:9px 12px">Cek Kamera</button>
      </div>
      <div class="small" id="camResult"></div>
    </div>
    <div id="pathRow" style="display:none">
      <label>Path folder gambar (di laptop)</label>
      <input id="srcPath" placeholder="mis. C:\survey\frames">
    </div>

    <label>Model</label>
    <select id="model">
      {% for m in models %}<option value="{{m}}">{{m}}</option>{% endfor %}
    </select>

    <label>Confidence: <b id="confVal">0.30</b></label>
    <input type="range" id="conf" min="0.05" max="0.9" step="0.05" value="0.30">

    <div id="gpsRow" style="display:none">
      <label>GPS</label>
      <select id="gpsType">
        <option value="none">Tidak ada</option>
        <option value="nmea">USB GPS dongle (COM)</option>
        <option value="tcp">HP via TCP (NMEA)</option>
      </select>
      <input id="gpsDetail" style="display:none;margin-top:6px" placeholder="">
    </div>

    <div class="row">
      <div><label>Nama ruas</label><input id="road" placeholder="Ruas Malili–Wawondula"></div>
      <div><label>Surveyor</label><input id="surveyor" placeholder="nama"></div>
    </div>

    <div class="btns">
      <button class="btn start" id="startBtn" style="flex:1">▶ MULAI</button>
      <button class="btn stop" id="stopBtn" style="flex:1" disabled>■ STOP</button>
    </div>
    <div class="btns">
      <button class="btn ghost" id="exportBtn" style="flex:1" disabled>📄 Export Laporan + Peta</button>
    </div>
    <div class="status" id="status">Siap.</div>
    <div class="status" id="files"></div>
  </div>

  <!-- Video + statistik -->
  <div class="card">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
      <h2 style="margin:0">Pemantauan Langsung</h2>
      <span class="badge" id="devBadge">device: -</span>
      <span class="badge" id="modelBadge">model: -</span>
    </div>
    <img id="video" src="/video_feed" alt="video">
    <div class="statgrid">
      <div class="stat"><div class="n" id="sFps">0</div><div class="l">FPS</div></div>
      <div class="stat"><div class="n" id="sTotal">0</div><div class="l">Total Deteksi</div></div>
      <div class="stat"><div class="n" id="sGeo">0</div><div class="l">Ber-GPS</div></div>
    </div>
    <h2 style="margin-top:16px">Per Jenis Kerusakan</h2>
    <div class="chips" id="chips"><span class="small">Belum ada deteksi.</span></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const COLORS = {"Pothole":"#e31a1c","Lubang":"#e31a1c","Alligator Crack":"#ff7f00",
  "Longitudinal Crack":"#1f78b4","Transverse Crack":"#33a02c"};
const colorFor = n => COLORS[n] || "#9b59b6";

$("conf").oninput = e => $("confVal").textContent = (+e.target.value).toFixed(2);

$("srcType").onchange = e => {
  const t = e.target.value;
  $("droneRow").style.display = t==="drone"  ? "block":"none";
  $("camRow").style.display   = t==="webcam" ? "block":"none";
  $("pathRow").style.display  = t==="folder" ? "block":"none";
  // GPS dropdown hanya untuk webcam/folder; drone pakai .SRT otomatis.
  $("gpsRow").style.display   = t==="drone"  ? "none":"block";
};

$("gpsType").onchange = e => {
  const t = e.target.value, d = $("gpsDetail");
  if(t==="nmea"){ d.style.display="block"; d.value="COM3@4800"; }
  else if(t==="tcp"){ d.style.display="block"; d.value="192.168.43.1:11123"; }
  else { d.style.display="none"; d.value=""; }
};
function gpsSpec(){
  const t = $("gpsType").value, d = $("gpsDetail").value.trim();
  if(t==="nmea") return "nmea:"+d;
  if(t==="tcp") return "tcp:"+d;
  return t;  // "none" atau "auto"
}

$("checkCam").onclick = async () => {
  $("camResult").textContent = "Mengecek...";
  const r = await fetch("/cameras"); const j = await r.json();
  if(j.error){ $("camResult").textContent = j.error; return; }
  $("camResult").innerHTML = j.cameras.length ?
     "Kamera terdeteksi di nomor: <b>"+j.cameras.join(", ")+"</b>" :
     "Tidak ada kamera terdeteksi.";
};

async function uploadDrone(){
  const v = $("videoFile").files[0];
  if(!v){ alert("Pilih file video drone dulu."); return {ok:false,error:"video kosong"}; }
  const fd = new FormData();
  fd.append("video", v);
  const s = $("srtFile").files[0];
  if(s) fd.append("srt", s);
  $("status").textContent = "Mengupload video drone... (file besar butuh beberapa menit, sabar)";
  const r = await fetch("/upload", {method:"POST", body:fd});
  return await r.json();
}

$("startBtn").onclick = async () => {
  const t = $("srcType").value;
  let source, gps;
  if(t==="drone"){
    const up = await uploadDrone();
    if(!up.ok){ $("status").textContent = "Upload gagal: "+(up.error||""); return; }
    source = up.video_path;
    gps = up.srt ? "auto" : "none";
    if(!up.srt) $("status").textContent = "Catatan: tanpa .SRT, hasil tidak ada koordinat. ";
  } else {
    source = (t==="webcam") ? $("camIdx").value.trim() : $("srcPath").value.trim();
    gps = gpsSpec();
  }
  const body = {source, model:$("model").value, conf:$("conf").value,
    road:$("road").value, surveyor:$("surveyor").value, gps};
  $("status").textContent += "Memulai...";
  const r = await fetch("/start",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)});
  const j = await r.json();
  if(!j.ok){ $("status").textContent = "Gagal: "+j.error; return; }
  $("startBtn").disabled=true; $("stopBtn").disabled=false; $("exportBtn").disabled=true;
  $("files").textContent="";
  $("status").textContent = "Berjalan...";
};

$("stopBtn").onclick = async () => {
  await fetch("/stop",{method:"POST"});
  $("status").textContent = "Dihentikan.";
};

$("exportBtn").onclick = async () => {
  $("status").textContent = "Membuat laporan...";
  const r = await fetch("/export",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({road:$("road").value, surveyor:$("surveyor").value})});
  const j = await r.json();
  $("status").textContent = (j.messages||[]).join(" ");
  $("files").innerHTML = (j.files||[]).map(f =>
     `<a class="dl" href="${f.url}" target="_blank">⬇ ${f.name}</a>`).join("")
     + (j.folder? `<div class="small">Folder: ${j.folder}</div>`:"");
};

async function poll(){
  try{
    const r = await fetch("/stats"); const s = await r.json();
    $("sFps").textContent = (s.fps||0).toFixed ? (s.fps||0).toFixed(0) : s.fps;
    $("sTotal").textContent = s.total||0;
    $("sGeo").textContent = s.geo||0;
    $("devBadge").textContent = "device: "+(s.device||"-");
    $("modelBadge").textContent = "model: "+(s.model||"-");
    const pc = s.per_class||{};
    const keys = Object.keys(pc).sort((a,b)=>pc[b]-pc[a]);
    $("chips").innerHTML = keys.length ? keys.map(k =>
      `<span class="chip"><span class="dot" style="background:${colorFor(k)}"></span>${k}: <b>${pc[k]}</b></span>`).join("")
      : '<span class="small">Belum ada deteksi.</span>';
    const running = s.running;
    $("startBtn").disabled = running;
    $("stopBtn").disabled = !running;
    if(!running && (s.total>0)) $("exportBtn").disabled = false;
    if(s.error) $("status").textContent = s.error;
  }catch(e){}
}
setInterval(poll, 600);
poll();
</script>
</body>
</html>
"""


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    url = f"http://localhost:{args.port}"
    print("=" * 56)
    print(" ROAD SURVEY AI — Web App (lokal / offline)")
    print(f" Buka di browser: {url}")
    print(" Tutup: tutup jendela ini (atau Ctrl+C)")
    print("=" * 56)
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
