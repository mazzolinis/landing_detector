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
- [ ] Test mode on pre-recorded videos

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

By default the log has three lines per analysed frame. For long tests, use the grouped log: one
line every 10 s (analysis rate, timings, detections, battery temperature, thermal state) plus
the main events:

```bash
flutter run --profile --dart-define=LOG=summary
```

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
