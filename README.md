# landing_detector

A Flutter application for Android (iOS support planned) that records a tchoukball court
from the sideline and emits an audible alert when a **landing** fault occurs, that is,
when a player lands inside the forbidden area while still holding the ball.

Detection relies on a **YOLO26-seg** instance segmentation model, trained separately, with
four classes: `ball`, `carrier_ground` (ball carrier with at least one foot on the ground),
`carrier_jump` (ball carrier in the air) and `forbidden_area`. Inference runs entirely on the
device.

A frame counts as a landing when the **feet** (bottom part of the mask) of a `carrier_ground`
overlap the forbidden area. The area is kept in a per-cell low-pass filtered memory, so short
detection gaps do not matter. A rolling score (+1 landing frame, −1 otherwise) triggers the
alarm when it reaches the threshold.

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

## Model

The model file is **not included** in this repository. Export it from the trained
`best.pt` weights using Ultralytics:

```bash
yolo export model=best.pt format=litert nms=None imgsz=640   # ultralytics>=8.4.188
```

Copy the resulting `.tflite` file into `assets/models/` as `landing_<imgsz>.tflite` and set
`kModelSize` in `lib/main.dart` accordingly.

The plugin copies the model into the app storage on first launch and reuses that copy as
long as the file name does not change: give every new model a new name (for example
`landing_640_v2.tflite`, updating `kModelPath`) or clear the app data on the phone.

## License

This project is intended for personal use. The `ultralytics_yolo` plugin and the
Ultralytics library are distributed under the AGPL-3.0 license.
