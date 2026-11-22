// Test della logica del fallo con maschere sintetiche (niente telefono, niente rete).
// Si lanciano con:  flutter test
//
// Griglia di prova 20x20 celle. Disegniamo rettangoli "pieni" come maschere:
//   - zona proibita: righe 10..19 (metà bassa dell'immagine)
//   - giocatore: un rettangolo alto 10 righe; i piedi sono la sua riga più bassa.

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

YOLOResult det(String cls, double conf, List<List<double>> mask) => YOLOResult(
      classIndex: 0,
      className: cls,
      confidence: conf,
      boundingBox: Rect.zero,
      normalizedBox: Rect.zero,
      mask: mask,
    );

final zone = det('forbidden_area', 0.7, rectMask(0, 10, 19, 19));

/// Giocatore a terra con i piedi DENTRO la zona (righe 5..14, piedi alla riga 14).
final groundInZone = det('carrier_ground', 0.4, rectMask(8, 5, 11, 14));

void main() {
  test('piedi a terra dentro la zona → landing e allarme al 3° frame', () {
    final logic = LandingLogic();
    final r1 = logic.process([zone, groundInZone]);
    final r2 = logic.process([zone, groundInZone]);
    final r3 = logic.process([zone, groundInZone]);

    expect(r1.landing, isTrue);
    expect(r1.feetInZone, greaterThanOrEqualTo(kFeetMinCells));
    expect([r1.alarm, r2.alarm, r3.alarm], [false, false, true]);
    expect(r3.score, 3);
  });

  test('l\'allarme non si ripete finché il punteggio non torna a 0', () {
    final logic = LandingLogic();
    final alarms = <bool>[];
    for (var i = 0; i < 8; i++) {
      alarms.add(logic.process([zone, groundInZone]).alarm);
    }
    expect(alarms.where((a) => a).length, 1);
    expect(logic.score, kScoreMax); // il punteggio non supera il tetto

    // Svuota il punteggio (frame senza landing), poi un nuovo landing fa suonare di nuovo.
    for (var i = 0; i < kScoreMax; i++) {
      logic.process([zone]);
    }
    expect(logic.score, 0);
    final again = [for (var i = 0; i < 3; i++) logic.process([zone, groundInZone]).alarm];
    expect(again.last, isTrue);
  });

  test('prospettiva: busto sopra la zona ma piedi fuori → NON è landing', () {
    // Zona in alto nell'immagine (righe 0..9, lontana), giocatore davanti (righe 5..19):
    // il busto copre la zona, i piedi (riga 19) no.
    final farZone = det('forbidden_area', 0.7, rectMask(0, 0, 19, 9));
    final inFront = det('carrier_ground', 0.6, rectMask(8, 5, 11, 19));
    final logic = LandingLogic();
    final r = logic.process([farZone, inFront]);
    expect(r.landing, isFalse);
    expect(r.feetCells, greaterThan(0));
    expect(r.feetInZone, 0);
  });

  // Zona solo nella metà sinistra in basso (colonne 0..9, righe 10..19), per i test sulla frazione.
  final leftZone = det('forbidden_area', 0.7, rectMask(0, 10, 9, 19));

  test('un piede dentro e uno fuori (~50%) → landing', () {
    // Giocatore colonne 8..11: piedi (righe 13..14) metà dentro (colonne 8, 9), metà fuori.
    final r = LandingLogic().process([leftZone, groundInZone]);
    expect(r.feetInZone, 4);
    expect(r.feetCells, 8);
    expect(r.landing, isTrue);
  });

  test('pochi piedi nella zona rispetto al totale → NON è landing', () {
    // Giocatore largo (colonne 8..19): 4 celle nella zona su 24 (17%): sopra kFeetMinCells
    // ma sotto kFeetMinFraction (il caso dei falsi positivi visti in fault_logic_1).
    final wide = det('carrier_ground', 0.7, rectMask(8, 5, 19, 14));
    final r = LandingLogic().process([leftZone, wide]);
    expect(r.feetInZone, greaterThanOrEqualTo(kFeetMinCells));
    expect(r.feetInZone, lessThan(kFeetMinFraction * r.feetCells));
    expect(r.landing, isFalse);
  });

  test('reset della zona → la memoria è vuota finché la zona non torna', () {
    final logic = LandingLogic();
    logic.process([zone]);
    logic.resetZone();
    final r1 = logic.process([groundInZone]); // zona non vista: niente memoria → niente landing
    expect(r1.zoneCells, 0);
    expect(r1.landing, isFalse);
    final r2 = logic.process([zone, groundInZone]); // la zona torna: memoria ricostruita subito
    expect(r2.zoneCells, 10 * w);
    expect(r2.landing, isTrue);
  });

  test('overlay: si disegna senza errori (anche con griglia diversa, es. dopo rotazione)', () {
    final logic = LandingLogic();
    final report = logic.process([zone, groundInZone]);
    final rotated = det('carrier_ground', 0.5, [for (var y = 0; y < 30; y++) List.filled(15, 1.0)]);
    final rotatedReport = logic.process([rotated]);
    for (final r in [null, report, rotatedReport]) {
      final canvas = Canvas(PictureRecorder());
      LogicOverlayPainter(logic, r).paint(canvas, const Size(400, 800));
    }
  });

  test('carrier_jump sopra la zona → NON è landing', () {
    final jump = det('carrier_jump', 0.9, rectMask(8, 5, 11, 14));
    final r = LandingLogic().process([zone, jump]);
    expect(r.landing, isFalse);
  });

  test('un solo portatore: vince la confidenza più alta', () {
    // jump 0.6 batte ground 0.35 (il caso visto nel test a 960) → nessun landing.
    final ground = det('carrier_ground', 0.35, rectMask(8, 5, 11, 14));
    final jump = det('carrier_jump', 0.6, rectMask(2, 5, 5, 14));
    final r = LandingLogic().process([zone, ground, jump]);
    expect(r.carrier?.className, 'carrier_jump');
    expect(r.landing, isFalse);
  });

  test('portatore sotto la confidenza minima → ignorato', () {
    final weak = det('carrier_ground', kCarrierMinConf - 0.01, rectMask(8, 5, 11, 14));
    final r = LandingLogic().process([zone, weak]);
    expect(r.carrier, isNull);
    expect(r.landing, isFalse);
  });

  test('zona persa per qualche frame → resta in memoria', () {
    final logic = LandingLogic();
    logic.process([zone]); // la zona entra in memoria
    // Due frame in cui la rete non vede la zona ma vede il portatore a terra.
    final r1 = logic.process([groundInZone]);
    final r2 = logic.process([groundInZone]);
    expect(r1.zoneSeen, isFalse);
    expect(r1.landing, isTrue);
    expect(r2.landing, isTrue);
  });

  test('zona non vista per più di kZoneForget → dimenticata', () {
    final logic = LandingLogic();
    final t0 = DateTime(2026, 1, 1, 12);
    logic.process([zone], now: t0);
    final r = logic.process([groundInZone], now: t0.add(kZoneForget + const Duration(seconds: 1)));
    expect(r.zoneCells, 0);
    expect(r.landing, isFalse);
  });

  test('filtro della zona: una cella nuova entra dopo alcuni frame, non subito', () {
    final logic = LandingLogic();
    logic.process([zone]); // memoria iniziale = zona righe 10..19
    // Ora la zona "cresce" di una riga (riga 9): con α = 0.8 serve qualche frame per accettarla.
    final bigger = det('forbidden_area', 0.7, rectMask(0, 9, 19, 19));
    final counts = [for (var i = 0; i < 5; i++) logic.process([bigger]).zoneCells];
    expect(counts.first, 10 * w); // al primo frame la riga 9 non è ancora zona
    expect(counts.last, 11 * w); // dopo alcuni frame sì
  });
}
