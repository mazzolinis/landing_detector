#!/usr/bin/env python3
"""Analizza un video sul PC con la stessa rete del telefono e salva le rilevazioni in un file JSON.

Il file viene poi letto da tool/replay_test.dart, che lo fa passare nella stessa LandingLogic
dell'app (calibrazione della zona, piedi, punteggio, allarmi). Così si possono provare video e
costanti senza telefono e ripetere lo stesso test quante volte si vuole.

Uso (dalla cartella "tchoukball landing project", con il venv attivo):
    unset PYTHONPATH
    source .venv/bin/activate
    python landing_detector/tool/replay_video.py test_videos/*.mp4     # uno o più video

Opzioni utili:
    --model models/landing_n640_v6.tflite   rete da usare (default: la stessa dell'app)
    --cycle-ms 45                            un'analisi ogni 45 ms, come il telefono dal vivo
    --all-frames                             analizza tutti i frame (il punteggio scatta prima!)
    --start 10 --end 40                      solo un tratto del video (secondi)

Cosa imita del telefono:
- stessa rete (.tflite), confidenza 0.25, IoU 0.5, al massimo 30 rilevamenti, NMS per classe;
- maschere sulla stessa griglia: 1/4 della risoluzione della rete, senza il bordo del letterbox
  (es. video 16:9 → 160x90 celle; la camera del telefono, 4:3, dà 160x120);
- ritmo: si analizza un frame ogni --cycle-ms, prendendo l'ultimo frame disponibile in quel
  momento, come fa la camera dal vivo;
- maschere solo per portatori e zona (la palla, come nell'app, arriva senza maschera).
Differenze: qui la rete gira sulla CPU del PC, non sulla GPU del telefono (numeri quasi uguali),
e il campo inquadrato è quello del video (16:9), non quello della camera dell'app (4:3).
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[1]  # cartella landing_detector
DEFAULT_MODEL = PROJECT.parent / "models" / "landing_n640_v6.tflite"
IMGSZ = 640
CONF = 0.25  # come confidenceThreshold dell'app
IOU = 0.5  # come defaultIouThreshold del plugin
MAX_DET = 30  # come numItemsThreshold del plugin
MASK_CLASSES = {"carrier_ground", "carrier_jump", "forbidden_area"}


def rle(mask: np.ndarray) -> str:
    """Maschera 0/1 → lunghezze alternate di 0 e di 1 (riga per riga, partendo dagli 0).

    Esempio: 0 0 1 1 1 0 → "2,3,1". Molto più compatto di una lista di celle.
    """
    flat = mask.reshape(-1).astype(np.uint8)
    change = np.flatnonzero(np.diff(flat)) + 1  # posizioni in cui il valore cambia
    runs = np.diff(np.concatenate(([0], change, [flat.size]))).tolist()
    if flat.size and flat[0] == 1:
        runs = [0] + runs  # si parte sempre da un tratto di 0 (anche lungo 0)
    return ",".join(map(str, runs))


def content_rect(h0: int, w0: int):
    """Parte dell'ingresso 640x640 occupata dall'immagine (il resto è il bordo del letterbox).

    Stessa formula di Ultralytics (LetterBox): scala per far stare l'immagine, bordo diviso a metà.
    """
    gain = min(IMGSZ / h0, IMGSZ / w0)
    nw, nh = round(w0 * gain), round(h0 * gain)
    left = round((IMGSZ - nw) / 2 - 0.1)
    top = round((IMGSZ - nh) / 2 - 0.1)
    return top, left, nh, nw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="uno o più video")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--cycle-ms", type=float, default=45.0, help="ms tra due analisi (telefono: ~45)")
    ap.add_argument("--all-frames", action="store_true", help="analizza tutti i frame")
    ap.add_argument("--start", type=float, default=0.0, help="secondo di inizio")
    ap.add_argument("--end", type=float, default=None, help="secondo di fine")
    ap.add_argument("--out", default=None, help="file JSON (solo con un video; default: data_logs/replay/<video>__<rete>.json)")
    args = ap.parse_args()
    if args.out and len(args.videos) > 1:
        raise SystemExit("--out si può usare solo con un video")
    model = YOLO(args.model, task="segment")
    for v in args.videos:
        analyze(Path(v), model, args)
    print("Ora, dalla cartella landing_detector:\n  flutter test tool/replay_test.dart --dart-define=REPLAY=data_logs/replay")


def analyze(video: Path, model, args):
    """Analizza un video e salva il JSON."""
    print(f"== {video}")
    cap = cv2.VideoCapture(str(video))  # OpenCV raddrizza da solo i video verticali
    if not cap.isOpened():
        raise SystemExit(f"Impossibile aprire il video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cycle = 1000.0 / fps if args.all_frames else args.cycle_ms
    end_s = args.end if args.end is not None else total / fps

    out = Path(args.out) if args.out else PROJECT / "data_logs" / "replay" / f"{video.stem}__{Path(args.model).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    grid = None
    frame_size = None
    next_t = args.start  # istante (s) della prossima analisi
    index = -1
    t_start = time.time()
    while True:
        ok = cap.grab()  # legge il frame successivo senza ancora convertirlo
        if not ok:
            break
        index += 1
        t = index / fps
        if t > end_s:
            break
        # Dal vivo si analizza l'ultimo frame arrivato quando la rete si libera: qui si prende
        # il frame che copre l'istante next_t (quello in corso tra t e t + 1/fps).
        if t + 1.0 / fps <= next_t:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        next_t = max(next_t + cycle / 1000.0, t)

        h0, w0 = img.shape[:2]
        frame_size = [w0, h0]
        top, left, nh, nw = content_rect(h0, w0)
        r = model.predict(img, imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET, verbose=False)[0]

        dets = []
        masks = None if r.masks is None else r.masks.data.cpu().numpy()  # (n, 640, 640), 0/1
        for k in range(len(r.boxes)):
            cls = r.names[int(r.boxes.cls[k])]
            conf = round(float(r.boxes.conf[k]), 3)
            x1, y1, x2, y2 = (float(v) for v in r.boxes.xyxyn[k])
            m = None
            if masks is not None and cls in MASK_CLASSES:
                # Si toglie il bordo del letterbox e si prende il centro di ogni blocco 4x4:
                # è la stessa griglia (1/4 della rete) che il plugin manda all'app.
                cell = masks[k, top:top + nh, left:left + nw][2::4, 2::4]
                grid = [cell.shape[1], cell.shape[0]]
                m = rle(cell > 0)
            dets.append([cls, conf, round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4), m])
        frames.append({"t": round(t - args.start, 3), "i": index, "d": dets})

        if len(frames) % 50 == 0:
            spent = time.time() - t_start
            print(f"  {t:6.1f} s di video, {len(frames)} frame analizzati ({spent:.0f} s)", flush=True)

    cap.release()
    if grid is None and frame_size:  # nessuna maschera nel video: griglia calcolata dalla forma
        top, left, nh, nw = content_rect(frame_size[1], frame_size[0])
        grid = [len(range(2, nw, 4)), len(range(2, nh, 4))]
    data = {
        "video": str(video),
        "model": str(args.model),
        "fps": fps,
        "frame_size": frame_size,
        "grid": grid,
        "cycle_ms": cycle,
        "start_s": args.start,
        "frames": frames,
    }
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Fatto: {len(frames)} frame analizzati su {index + 1} in {time.time() - t_start:.0f} s, "
          f"griglia {grid}, video {frame_size} a {fps:.1f} fps")
    print(f"Salvato in {out}")


if __name__ == "__main__":
    main()
