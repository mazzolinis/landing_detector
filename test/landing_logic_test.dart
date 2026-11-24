// Test della logica del fallo con maschere sintetiche (niente telefono, niente rete).
// Si lanciano con:  flutter test
//
// Griglia di prova 20x20 celle. Disegniamo rettangoli "pieni" come maschere:
//   - zona proibita: righe 10..19 (metà bassa dell'immagine)
//   - giocatore: un rettangolo alto 10 righe; i piedi sono le sue righe più basse.

import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Canvas, PictureRecorder, Rect, Size;

import 'package:flutter_test/flutter_test.dart';
import 'package:landing_detector/main.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

const int w = 20;
const int h = 20;

/// Maschera w x h con valore 1 nel rettangolo [x0, x1] x [y0, y1] (estremi inclusi), 0 fuori.
List<List<double>> rectMask(int x0, int y0, int x1, int y1) => [
      for (var y = 0; y < h; y++)
        [for (var x = 0; x < w; x++) (x >= x0 && x <= x1 && y >= y0 && y <= y1) ? 1.0 : 0.0],
    ];

/// Come rectMask, ma con un "buco" rettangolare (es. un giocatore che copre la zona).
List<List<double>> rectWithHole(List<int> outer, List<int> hole) {
  final m = rectMask(outer[0], outer[1], outer[2], outer[3]);
  for (var y = hole[1]; y <= hole[3]; y++) {
    for (var x = hole[0]; x <= hole[2]; x++) {
      m[y][x] = 0;
    }
  }
  return m;
}

YOLOResult det(String cls, double conf, List<List<double>> mask) => YOLOResult(
      classIndex: 0,
      className: cls,
      confidence: conf,
      boundingBox: Rect.zero,
      normalizedBox: Rect.zero,
      mask: mask,
    );

final zone = det('forbidden_area', 0.7, rectMask(0, 10, 19, 19));

/// Giocatore a terra con i piedi DENTRO la zona (righe 5..14, piedi alle righe 13..14).
final groundInZone = det('carrier_ground', 0.4, rectMask(8, 5, 11, 14));

/// Logica con una zona già attiva (salta la calibrazione): per i test sui piedi.
LandingLogic activeLogic([YOLOResult? zoneDet]) {
  final mask = (zoneDet ?? zone).mask!;
  final cells = Uint8List(w * h);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      if (mask[y][x] > 0) cells[y * w + x] = 1;
    }
  }
  return LandingLogic()..setZone(cells, w, h); // ".." = chiama setZone e restituisce l'oggetto
}

/// Simula la calibrazione: un frame ogni 125 ms (8 analisi/s) per 3 s + 1 frame finale.
/// [frameAt] dà i rilevamenti del frame i-esimo.
List<FrameReport> calibrate(LandingLogic logic, List<YOLOResult> Function(int i) frameAt) {
  final t0 = DateTime(2026, 1, 1, 12);
  final n = kCalibDuration.inMilliseconds ~/ 125; // "~/" = divisione intera
  return [
    for (var i = 0; i <= n; i++)
      logic.process(frameAt(i), now: t0.add(Duration(milliseconds: 125 * i))),
  ];
}

/// Manda al MotionMonitor campioni a 50 Hz per [dur], a partire da [t0]; [w] dà ω (rad/s)
/// in funzione del tempo in secondi dall'inizio. Restituisce l'istante dopo l'ultimo campione.
DateTime feedGyro(MotionMonitor m, DateTime t0, Duration dur,
    (double, double, double) Function(double s) w) {
  const step = Duration(milliseconds: 20);
  var t = t0;
  for (var i = 0; i * 20 < dur.inMilliseconds; i++) {
    final (x, y, z) = w(i * 0.02);
    m.addGyro(x, y, z, t);
    t = t.add(step);
  }
  return t;
}

void main() {
  // ------------------------------------------------------------------------------------------
  // Calibrazione della zona
  // ------------------------------------------------------------------------------------------

  test('all\'avvio la logica è in calibrazione', () {
    expect(LandingLogic().zoneState, ZoneState.calibrating);
  });

  test('calibrazione: la mediana ignora un giocatore che passa sopra la zona', () {
    final logic = LandingLogic();
    // In 1 frame su 3 un giocatore "buca" la zona (colonne 8..11, righe 10..14).
    final holed = det('forbidden_area', 0.7, rectWithHole([0, 10, 19, 19], [8, 10, 11, 14]));
    final reports = calibrate(logic, (i) => [i % 3 == 0 ? holed : zone]);

    expect(reports.last.zoneState, ZoneState.proposal);
    logic.confirmZone();
    expect(logic.zoneState, ZoneState.active);
    expect(logic.displayZoneCells, 10 * w); // zona intera, senza buco
    expect(logic.isZoneCell(9, 12), isTrue);
  });

  test('calibrazione: i frame senza zona non la cancellano (non votano)', () {
    final logic = LandingLogic();
    final reports = calibrate(logic, (i) => i % 3 != 0 ? [zone] : []); // zona vista in 2 frame su 3
    expect(reports.last.zoneState, ZoneState.proposal);
    expect(logic.displayZoneCells, 10 * w);
  });

  test('calibrazione: zona vista in troppo pochi frame → nuovo tentativo automatico', () {
    final logic = LandingLogic();
    final reports = calibrate(logic, (i) => i < kCalibMinFrames - 1 ? [zone] : [groundInZone]);
    expect(reports.last.zoneState, ZoneState.calibrating);
    expect(logic.failedAttempts, 1);
    expect(logic.zoneText, contains('non trovata'));
  });

  test('camera coperta dopo pochi frame → nessuna proposta (meno della metà dei frame)', () {
    final logic = LandingLogic();
    // Zona vista nei primi 8 frame (≥ kCalibMinFrames), poi camera coperta: 8/25 < 50%.
    final reports = calibrate(logic, (i) => i < 8 ? [zone] : [groundInZone]);
    expect(reports.any((r) => r.zoneState == ZoneState.proposal), isFalse);
    expect(logic.failedAttempts, 1);
  });

  test('dopo un tentativo fallito, la calibrazione riesce quando la zona torna visibile', () {
    final logic = LandingLogic();
    calibrate(logic, (_) => [groundInZone]); // 3 s senza zona: tentativo fallito
    expect(logic.failedAttempts, 1);
    final t1 = DateTime(2026, 1, 1, 12, 0, 10);
    for (var i = 0; i <= 24; i++) {
      logic.process([zone], now: t1.add(Duration(milliseconds: 125 * i)));
    }
    expect(logic.zoneState, ZoneState.proposal);
  });

  test('zona mai trovata per kMaxCalibAttempts tentativi → analisi ferma', () {
    final logic = LandingLogic();
    final t0 = DateTime(2026, 1, 1, 12);
    // Camera coperta (nessun rilevamento), telefono fermo: 8 frame/s per molto tempo.
    var i = 0;
    while (logic.zoneState == ZoneState.calibrating && i < 1000) {
      logic.process([], now: t0.add(Duration(milliseconds: 125 * i++)));
    }
    expect(logic.zoneState, ZoneState.gaveUp);
    expect(logic.failedAttempts, kMaxCalibAttempts);
    // Circa kMaxCalibAttempts × kCalibDuration (15 s), non oltre.
    expect(i * 125, lessThan(kMaxCalibAttempts * kCalibDuration.inMilliseconds + 1000));
    expect(logic.zoneText, contains('ANALISI FERMA'));
    // Ferma: anche se ora arrivasse la zona, non succede nulla finché non si riprova.
    expect(logic.process([zone, groundInZone], now: t0.add(const Duration(minutes: 1))).landing, isFalse);
    expect(logic.zoneState, ZoneState.gaveUp);
    // "Riprova" = startCalibration: si riparte da zero.
    logic.startCalibration();
    expect(logic.zoneState, ZoneState.calibrating);
    expect(logic.failedAttempts, 0);
  });

  test('telefono mai fermo (in tasca mentre si cammina) → analisi ferma dopo kMaxCalibTime', () {
    final logic = LandingLogic();
    final t0 = DateTime(2026, 1, 1, 12);
    final steps = kMaxCalibTime.inMilliseconds ~/ 125;
    for (var i = 0; i < steps; i++) {
      logic.process([], now: t0.add(Duration(milliseconds: 125 * i)), still: false);
    }
    expect(logic.zoneState, ZoneState.calibrating); // appena prima del limite
    expect(logic.failedAttempts, 0); // in attesa: i tentativi non vengono contati
    logic.process([], now: t0.add(kMaxCalibTime), still: false);
    expect(logic.zoneState, ZoneState.gaveUp);
  });

  test('ricalibrazione senza nessun rilevamento → nessuna proposta (non riusa la zona vecchia)', () {
    final logic = LandingLogic();
    calibrate(logic, (_) => [zone]);
    logic.confirmZone();
    logic.startCalibration();
    final reports = calibrate(logic, (_) => []); // la rete non vede niente per 3 s
    expect(reports.last.zoneState, ZoneState.calibrating);
    expect(logic.failedAttempts, 1);
    expect(logic.displayZoneCells, 0);
  });

  test('calibrazione: finché il telefono si muove non parte', () {
    final logic = LandingLogic();
    final t0 = DateTime(2026, 1, 1, 12);
    // 4 s di frame con la zona ma telefono in movimento: niente proposta.
    for (var i = 0; i <= 32; i++) {
      logic.process([zone], now: t0.add(Duration(milliseconds: 125 * i)), still: false);
    }
    expect(logic.zoneState, ZoneState.calibrating);
    expect(logic.zoneText, contains('attesa'));
    // Telefono fermo: dopo kCalibDuration arriva la proposta.
    final t1 = t0.add(const Duration(seconds: 5));
    for (var i = 0; i <= 24; i++) {
      logic.process([zone], now: t1.add(Duration(milliseconds: 125 * i)));
    }
    expect(logic.zoneState, ZoneState.proposal);
  });

  test('movimento durante la calibrazione → i voti raccolti prima vengono buttati', () {
    final logic = LandingLogic();
    final t0 = DateTime(2026, 1, 1, 12);
    final zoneA = zone; // righe 10..19 (posizione vecchia della camera)
    final zoneB = det('forbidden_area', 0.7, rectMask(0, 0, 19, 9)); // righe 0..9 (posizione nuova)
    var t = t0;
    void frame(YOLOResult z, {bool still = true}) {
      logic.process([z], now: t, still: still);
      t = t.add(const Duration(milliseconds: 125));
    }

    for (var i = 0; i < 16; i++) {
      frame(zoneA); // 2 s con la zona A
    }
    frame(zoneB, still: false); // il telefono viene spostato
    for (var i = 0; i <= 24; i++) {
      frame(zoneB); // 3 s fermo nella posizione nuova
    }
    expect(logic.zoneState, ZoneState.proposal);
    expect(logic.isDisplayZoneCell(0, 5), isTrue); // zona B
    expect(logic.isDisplayZoneCell(0, 15), isFalse); // niente residui della zona A
  });

  test('durante calibrazione e proposta non si rilevano landing', () {
    final logic = LandingLogic();
    final reports = calibrate(logic, (_) => [zone, groundInZone]);
    expect(reports.any((r) => r.landing || r.alarm), isFalse);
    expect(logic.zoneState, ZoneState.proposal);
    // Zona proposta ma non ancora confermata: ancora niente landing.
    expect(logic.process([groundInZone]).landing, isFalse);
    logic.confirmZone();
    expect(logic.process([groundInZone]).landing, isTrue);
  });

  test('zona attiva: la rete non la modifica più', () {
    final logic = activeLogic();
    final other = det('forbidden_area', 0.9, rectMask(0, 0, 19, 5)); // zona "diversa" vista dalla rete
    for (var i = 0; i < 10; i++) {
      logic.process([other]);
    }
    expect(logic.displayZoneCells, 10 * w);
    expect(logic.isZoneCell(5, 2), isFalse);
  });

  test('griglia diversa (telefono ruotato) a zona attiva → ricalibrazione', () {
    final logic = activeLogic();
    final rotated = det('carrier_ground', 0.5, [for (var y = 0; y < 30; y++) List.filled(15, 1.0)]);
    logic.process([rotated]);
    expect(logic.zoneState, ZoneState.calibrating);
  });

  test('ricalibra: la zona precedente viene scartata', () {
    final logic = activeLogic();
    logic.startCalibration();
    expect(logic.zoneState, ZoneState.calibrating);
    expect(logic.process([groundInZone]).landing, isFalse);
  });

  // ------------------------------------------------------------------------------------------
  // Piedi e punteggio (con zona già attiva)
  // ------------------------------------------------------------------------------------------

  test('piedi a terra dentro la zona → landing e allarme al 3° frame', () {
    final logic = activeLogic();
    final r1 = logic.process([groundInZone]);
    final r2 = logic.process([groundInZone]);
    final r3 = logic.process([groundInZone]);

    expect(r1.landing, isTrue);
    expect(r1.feetInZone, greaterThanOrEqualTo(kFeetMinCells));
    expect([r1.alarm, r2.alarm, r3.alarm], [false, false, true]);
    expect(r3.score, 3);
  });

  test('l\'allarme non si ripete finché il punteggio non torna a 0', () {
    final logic = activeLogic();
    final alarms = <bool>[];
    for (var i = 0; i < 8; i++) {
      alarms.add(logic.process([groundInZone]).alarm);
    }
    expect(alarms.where((a) => a).length, 1);
    expect(logic.score, kScoreMax); // il punteggio non supera il tetto

    // Svuota il punteggio (frame senza landing), poi un nuovo landing fa suonare di nuovo.
    for (var i = 0; i < kScoreMax; i++) {
      logic.process([]);
    }
    expect(logic.score, 0);
    final again = [for (var i = 0; i < 3; i++) logic.process([groundInZone]).alarm];
    expect(again.last, isTrue);
  });

  test('prospettiva: busto sopra la zona ma piedi fuori → NON è landing', () {
    // Zona in alto nell'immagine (righe 0..9, lontana), giocatore davanti (righe 5..19):
    // il busto copre la zona, i piedi (righe 17..19) no.
    final logic = activeLogic(det('forbidden_area', 0.7, rectMask(0, 0, 19, 9)));
    final inFront = det('carrier_ground', 0.6, rectMask(8, 5, 11, 19));
    final r = logic.process([inFront]);
    expect(r.landing, isFalse);
    expect(r.feetCells, greaterThan(0));
    expect(r.feetInZone, 0);
  });

  // Zona solo nella metà sinistra in basso (colonne 0..9, righe 10..19), per i test sulla frazione.
  final leftZone = det('forbidden_area', 0.7, rectMask(0, 10, 9, 19));

  test('un piede dentro e uno fuori (~50%) → landing', () {
    // Giocatore colonne 8..11: piedi (righe 13..14) metà dentro (colonne 8, 9), metà fuori.
    final r = activeLogic(leftZone).process([groundInZone]);
    expect(r.feetInZone, 4);
    expect(r.feetCells, 8);
    expect(r.landing, isTrue);
  });

  test('pochi piedi nella zona rispetto al totale → NON è landing', () {
    // Giocatore largo (colonne 8..19): 4 celle nella zona su 24 (17%): sopra kFeetMinCells
    // ma sotto kFeetMinFraction (il caso dei falsi positivi visti in fault_logic_1).
    final wide = det('carrier_ground', 0.7, rectMask(8, 5, 19, 14));
    final r = activeLogic(leftZone).process([wide]);
    expect(r.feetInZone, greaterThanOrEqualTo(kFeetMinCells));
    expect(r.feetInZone, lessThan(kFeetMinFraction * r.feetCells));
    expect(r.landing, isFalse);
  });

  test('zona attiva ma telefono in movimento → rilevamento in pausa', () {
    final logic = activeLogic();
    final r = logic.process([groundInZone], still: false);
    expect(r.landing, isFalse);
    expect(r.still, isFalse);
    expect(logic.process([groundInZone]).landing, isTrue); // di nuovo fermo
  });

  test('carrier_jump sopra la zona → NON è landing', () {
    final jump = det('carrier_jump', 0.9, rectMask(8, 5, 11, 14));
    expect(activeLogic().process([jump]).landing, isFalse);
  });

  test('un solo portatore: vince la confidenza più alta', () {
    // jump 0.6 batte ground 0.35 (il caso visto nel test a 960) → nessun landing.
    final ground = det('carrier_ground', 0.35, rectMask(8, 5, 11, 14));
    final jump = det('carrier_jump', 0.6, rectMask(2, 5, 5, 14));
    final r = activeLogic().process([ground, jump]);
    expect(r.carrier?.className, 'carrier_jump');
    expect(r.landing, isFalse);
  });

  test('portatore sotto la confidenza minima → ignorato', () {
    final weak = det('carrier_ground', kCarrierMinConf - 0.01, rectMask(8, 5, 11, 14));
    final r = activeLogic().process([weak]);
    expect(r.carrier, isNull);
    expect(r.landing, isFalse);
  });

  // ------------------------------------------------------------------------------------------
  // Giroscopio
  // ------------------------------------------------------------------------------------------

  test('MotionMonitor: senza campioni (niente giroscopio) il telefono è fermo', () {
    expect(MotionMonitor().isStill(DateTime.now()), isTrue);
  });

  test('MotionMonitor: tremolio della mano → fermo', () {
    final m = MotionMonitor();
    final t0 = DateTime(2026, 1, 1, 12);
    // Oscillazione ±0.05 rad/s a 8 Hz (sotto kStillGyroRate): avanti e indietro, si annulla.
    final t = feedGyro(m, t0, const Duration(seconds: 3), (s) => (0.05 * math.sin(2 * math.pi * 8 * s), 0, 0));
    expect(m.isStill(t), isTrue);
    expect(m.windowAngleDeg, lessThan(0.2));
  });

  test('MotionMonitor: movimento lento sotto la soglia di velocità → NON fermo', () {
    final m = MotionMonitor();
    final t0 = DateTime(2026, 1, 1, 12);
    // 0.03 rad/s costanti (≈1.7°/s): nessun campione supera kStillGyroRate, ma l'angolo sì.
    final t = feedGyro(m, t0, const Duration(seconds: 3), (_) => (0, 0.03, 0));
    expect(m.isStill(t), isFalse);
    expect(m.windowAngleDeg, greaterThan(kStillAngleDeg));
  });

  test('MotionMonitor: urto (avanti e indietro) → vibrazione, camera non ruotata', () {
    final m = MotionMonitor();
    final t0 = DateTime(2026, 1, 1, 12);
    var t = feedGyro(m, t0, const Duration(seconds: 2), (_) => (0, 0, 0));
    // +0.5 rad/s per 100 ms poi −0.5 rad/s per 100 ms: torna dov'era.
    t = feedGyro(m, t, const Duration(milliseconds: 200), (s) => (s < 0.1 ? 0.5 : -0.5, 0, 0));
    expect(m.isStill(t), isFalse);
    expect(m.takeFinishedMovement(t), isNull); // movimento non ancora finito
    t = feedGyro(m, t, const Duration(seconds: 3), (_) => (0, 0, 0));
    final mov = m.takeFinishedMovement(t);
    expect(mov, isNotNull);
    expect(mov!.angleDeg, lessThan(kMovedAngleDeg));
    expect(mov.cameraMoved, isFalse);
    expect(m.takeFinishedMovement(t), isNull); // l'episodio viene riportato una volta sola
  });

  test('MotionMonitor: scossone forte che torna al punto di partenza → ricalibrare', () {
    final m = MotionMonitor();
    final t0 = DateTime(2026, 1, 1, 12);
    var t = feedGyro(m, t0, const Duration(seconds: 2), (_) => (0, 0, 0));
    // ±4 rad/s per 320 ms (come in zone_calib_4): angolo netto quasi nullo ma movimento forte.
    // 8 campioni a +4 e 8 a −4 (50 Hz): il telefono torna esattamente al punto di partenza.
    t = feedGyro(m, t, const Duration(milliseconds: 320), (s) => (s < 0.155 ? 4.0 : -4.0, 0, 0));
    t = feedGyro(m, t, const Duration(seconds: 3), (_) => (0, 0, 0));
    final mov = m.takeFinishedMovement(t)!; // "!" = sono sicuro che non è null
    expect(mov.angleDeg, lessThan(kMovedAngleDeg));
    expect(mov.peakGyro, greaterThanOrEqualTo(kStrongGyroRate));
    expect(mov.cameraMoved, isTrue);
  });

  test('MotionMonitor: accelerazione forte senza rotazione (traslazione) → ricalibrare', () {
    final m = MotionMonitor();
    final t0 = DateTime(2026, 1, 1, 12);
    var t = feedGyro(m, t0, const Duration(seconds: 2), (_) => (0, 0, 0));
    m.addAccel(5.0, 0, 0, t); // spinta laterale: 5 m/s², il giroscopio non vede nulla
    expect(m.isStill(t), isFalse);
    t = feedGyro(m, t, const Duration(seconds: 2), (_) => (0, 0, 0));
    final mov = m.takeFinishedMovement(t)!;
    expect(mov.angleDeg, lessThan(0.1));
    expect(mov.peakAccel, 5.0);
    expect(mov.cameraMoved, isTrue);
  });

  test('MotionMonitor: spostamento vero → camera ruotata', () {
    final m = MotionMonitor();
    final t0 = DateTime(2026, 1, 1, 12);
    var t = feedGyro(m, t0, const Duration(seconds: 2), (_) => (0, 0, 0));
    // Panoramica lenta: 0.1 rad/s per 2 s ≈ 11.5° attorno all'asse y.
    t = feedGyro(m, t, const Duration(seconds: 2), (_) => (0, 0.1, 0));
    t = feedGyro(m, t, const Duration(seconds: 3), (_) => (0, 0, 0));
    expect(m.isStill(t), isTrue);
    final mov = m.takeFinishedMovement(t)!;
    expect(mov.angleDeg, closeTo(11.5, 0.5));
    expect(mov.cameraMoved, isTrue);
    expect(m.takePeaks().gyro, closeTo(0.1, 1e-9));
  });

  test('ricalibrazione quando la camera è stata spostata (moved)', () {
    final logic = activeLogic();
    logic.process([groundInZone], moved: true);
    expect(logic.zoneState, ZoneState.calibrating);
  });

  // ------------------------------------------------------------------------------------------
  // Overlay
  // ------------------------------------------------------------------------------------------

  test('overlay: si disegna senza errori in ogni stato della zona', () {
    final calibrating = LandingLogic();
    final calibReport = calibrating.process([zone, groundInZone]);
    final proposal = LandingLogic();
    final proposalReport = calibrate(proposal, (_) => [zone, groundInZone]).last;
    final active = activeLogic();
    final activeReport = active.process([groundInZone]);
    final rotated = det('carrier_ground', 0.5, [for (var y = 0; y < 30; y++) List.filled(15, 1.0)]);

    final cases = [
      (calibrating, calibReport),
      (proposal, proposalReport),
      (active, activeReport),
      (active, active.process([rotated])), // griglia cambiata → ricalibrazione
      (LandingLogic(), null),
    ];
    for (final (logic, report) in cases) {
      LogicOverlayPainter(logic, report).paint(Canvas(PictureRecorder()), const Size(400, 800));
    }
  });
}
