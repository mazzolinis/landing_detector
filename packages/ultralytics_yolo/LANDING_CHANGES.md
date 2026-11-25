# Local changes to ultralytics_yolo 0.7.2

This folder is a copy of the [ultralytics_yolo](https://pub.dev/packages/ultralytics_yolo) Flutter
plugin, version 0.7.2 (AGPL-3.0), modified for the landing detector app. The `example/`, `test/`,
`doc/` and `scripts/` folders of the original package were not copied.

Every change is marked with a `[landing_detector]` comment in the sources.

## Changes

1. **`maskClasses` streaming option**: names of the classes whose instance masks are built and
   streamed to Dart (null = all, the original behaviour). For the other classes the mask is not
   computed at all (no instance mask, no overlay paint) and the detection is sent without `mask`.
   - Dart: `YOLOStreamingConfig.maskClasses`, sent by `YOLOViewController.setStreamingConfig` and
     by the `YOLOView` creation parameters.
   - Kotlin: `YOLOStreamConfig.maskClasses`, parsed in `YOLOPlatformView`, passed to
     `BasePredictor.maskClasses`, applied in `Segmenter.generateCombinedMaskImage`; empty masks are
     skipped in `YOLOView.convertResultToStreamData`.
   - Reason: each 160x120 mask is converted to nested lists of boxed doubles and serialized value
     by value, which cost about 20 ms per detection and frame in the original code
     (`data_logs/speed_1.txt`). With changes 3-5 a mask costs about 1-2 ms (`data_logs/speed_3.txt`).

2. **Frequency check before the bitmap conversion** (`YOLOView.onFrame`): the original code
   converted every camera frame (about 30 per second) from YUV to a bitmap and only then discarded
   the frames it did not analyse. Now `shouldRunInference()` runs first (except while scanning a
   QR code). Measured afterwards: the conversion takes only about 1 ms, so the saving is small.

3. **Non-blocking hand-off to the main thread** (`YOLOPlatformView.sendStreamData`): the original
   code posted the results to the main thread and then waited (`CountDownLatch`, up to 100 ms) for
   the serialization to finish, so the analysis thread was idle about 20 ms per mask
   (`data_logs/speed_2.txt`). Now the analysis thread continues at once. If the previous hand-off
   is still pending, its data is replaced by the newest (latest wins); dropped results are logged
   every 10 as `LandingTiming: risultati scartati`.

4. **Mask rows as typed arrays** (`YOLOView.convertResultToStreamData`): each row is a `DoubleArray`,
   which the Flutter codec copies in one block (`Float64List` in Dart), instead of 19,200 boxed
   doubles encoded one by one. `MapConverter.convertMaskData` uses `Float64List` rows without copying.

5. **No colour overlay image when overlays are hidden** (`Segmenter.generateCombinedMaskImage`,
   `BasePredictor.buildOverlayImage` = `YOLOView.showOverlays`): the 640x480 colour image of the
   masks is only painted when the native overlay is visible.

6. **Timing logs** (Android only, for measurements):
   - Kotlin, tag `LandingTiming`, averages every 10 s: analysed frames, cycle time (action mode
     only), bitmap conversion, `predict`, conversion of the results for Dart, total `onFrame` time,
     masks per frame.
   - Dart (`YOLOView._handleDetectionResults`), printed as `LandingTiming dart`: latency from the
     native send to the Dart side, parsing of the results, time of the app's `onResult`.

7. **Thermal info** (Android only): `YOLOViewController.getThermalInfo()` returns the battery
   temperature, the Android thermal status (0-6) and the thermal headroom (1.0 = "severe").
   CPU/GPU temperatures are not accessible to apps.

## Known caveats

- The original throttling (`inferenceFrequency`) counts the interval from the **end** of the
  previous analysis. The app therefore passes `inferenceFrequency: null` in action mode.
- After moving the plugin into this folder, a stale Kotlin incremental cache in
  `build/ultralytics_yolo` caused "Unresolved reference" errors; deleting that folder fixed it.
- iOS sources are unchanged (the `maskClasses` option is ignored there).
