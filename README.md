# landing_detector

A Flutter application for Android (iOS support planned) that records a tchoukball court
from the sideline and emits an audible alert when a **landing** fault occurs, that is,
when a player lands inside the forbidden area while still holding the ball.

Detection relies on a **YOLO26-seg** instance segmentation model, trained separately, with
four classes: `ball`, `carrier_ground` (ball carrier with at least one foot on the ground),
`carrier_jump` (ball carrier in the air) and `forbidden_area`. Inference runs entirely on the
device.

A frame counts as a landing when the **feet** (bottom part of the mask) of a `carrier_ground`
overlap the forbidden area. The area is a fixed reference: at startup (and on demand) the app
calibrates it for 3 s, taking the per-cell median of the network's `forbidden_area` masks, and
the user confirms it. After that the network no longer changes it. The area can also be drawn or
corrected by hand with a brush (trace the outline, add, erase). The area is calibrated again
when the phone is moved (gyroscope and accelerometer) and whenever the app returns from the
background. A rolling score (+1 landing frame, −1 otherwise) triggers the alarm when it reaches
the threshold.

## Project Status

The project is under active development. Planned milestones:

- [✓] Sound playback triggered by a button
- [✓] Camera preview
- [✓] Model inference on camera frames
- [✓] Fault logic (feet/zone overlap and rolling score) with audible alert and debug overlay
- [✓] Replay of recorded videos on the PC (same network and fault logic, no phone)
- [ ] Analysis of recorded videos inside the app

## Requirements

- Flutter SDK (stable channel)
- Android SDK (installed through Android Studio)
- An Android device with USB debugging enabled

## Getting Started

```bash
flutter pub get
flutter test              # unit tests of the fault logic (no phone needed)
flutter run --profile     # on the phone; profile mode is faster than debug
```

The app opens on a start screen whose choices are saved on the phone:

- **Mode**: *Match* (clean screen, no log file) or *Test* (data collection: the log is always
  written to a file on the phone, one file per session).
- **Log** (Test only): *reduced* (one summary line every 10 s with analysis rate, timings,
  detections, battery temperature and thermal state, plus the main events) or *full* (also three
  lines per analysed frame). It can be changed during the session from the settings button.
- **Clip quality** (for the future micro-clips) and **orientation** (landscape, portrait or
  automatic; landscape and portrait stay locked for the whole session).

Log files: `adb pull /sdcard/Android/data/com.example.landing_detector/files/logs data_logs/phone`.

`flutter run` also prints hundreds of Android and camera messages, mostly at startup. To keep only
the app's lines:

```bash
flutter run --profile 2>&1 | grep --line-buffered -E '^(I/flutter|I/LandingTiming)|^[^A-Z]|^[A-Z][^/]|FAILURE|rror:|Exception' | tee data_logs/test.txt
```

## Replay of recorded videos on the PC

Two steps, from the "tchoukball landing project" folder with the Ultralytics venv active:

```bash
python landing_detector/tool/replay_video.py test_videos/*.mp4      # network → data_logs/replay/*.json
cd landing_detector
flutter test tool/replay_test.dart --dart-define=REPLAY=data_logs/replay
```

The first step runs the same `.tflite` model as the app on the PC, with the same thresholds and
mask grid, analysing one frame every 45 ms like the phone (`--all-frames` for every frame). The
second step feeds the detections to the app's `LandingLogic`: the zone is calibrated on the first
3 s of the video, then the whole video is analysed from the start. It prints the alarms with the
video time, saves a report next to each JSON and a table in `data_logs/replay/RIEPILOGO.txt`.

Annotations: for `uno.mp4`, the file `data_logs/replay/uno_landing.txt`, one time per line
(`#` starts a comment): `00:05.0` is a true landing, `? 00:03.7` a difficult moment that is not a
landing. The report lists detected landings with their delay, alarms on difficult moments and
other false alarms.

## Model

The model file is **not included** in this repository. Export it from the trained
`best.pt` weights using Ultralytics:

```bash
yolo export model=best.pt format=litert nms=None imgsz=640   # ultralytics>=8.4.188
```

Copy the resulting `.tflite` file into `assets/models/` following the naming convention
`landing_<arch><imgsz>_v<version>.tflite` (for example `landing_n640_v2.tflite` for a
YOLO26n-seg model exported at 640), then set `kModelName` (file name without extension) and
`kModelSize` (the export `imgsz`) in `lib/main.dart`.

The plugin copies the model into the app storage on first launch and reuses that copy as
long as the file name does not change, so every new model needs a new file name. Older
versions can stay in `assets/models/` as a fallback, but every file there is bundled into
the APK: keep only the active model and one backup.

## Local plugin copy

`packages/ultralytics_yolo` is a modified copy of the `ultralytics_yolo` 0.7.2 plugin, used as a
path dependency. The changes (mask class filter, frame-skipping before the bitmap conversion,
timing logs) are listed in `packages/ultralytics_yolo/LANDING_CHANGES.md`.

## License

This project is intended for personal use. The `ultralytics_yolo` plugin and the
Ultralytics library are distributed under the AGPL-3.0 license.
