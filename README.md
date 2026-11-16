landing_detector

A Flutter application for Android (iOS support planned) that records a tchoukball court from the sideline and emits an audible alert when a landing fault occurs, that is, when a player lands inside the forbidden area while still holding the ball.

Detection relies on a YOLO26-seg instance segmentation model, trained separately, with three classes: ball carrier, ball, and forbidden area. Inference runs entirely on the device.

Project Status

The project is under active development. Planned milestones:

[ ] Sound playback triggered by a button
[ ] Camera preview
[ ] Model inference on camera frames
[ ] Fault logic (mask overlap and rolling score) with audible alert
[ ] Test mode on pre-recorded videos
Requirements
Flutter SDK (stable channel)
Android SDK (installed through Android Studio)
An Android device with USB debugging enabled
Getting Started
bash
flutter pub get
flutter run
Model

The model file is not included in this repository. Export it from the trained best.pt weights using Ultralytics:

python
from ultralytics import YOLO

YOLO("best.pt").export(format="litert", imgsz=640)  # use the same imgsz as in training

Copy the resulting .tflite file into assets/models/, where it is declared in pubspec.yaml.

License

This project is intended for personal use. The ultralytics_yolo plugin and the Ultralytics library are distributed under the AGPL-3.0 license.