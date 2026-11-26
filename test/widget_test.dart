// Test dell'interfaccia (senza telefono: niente camera, sensori e audio nativi).

import 'package:material_ui/material_ui.dart' show MaterialApp;
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import 'package:landing_detector/main.dart';

/// Schermo da telefono: 2400x1080 pixel fisici, in verticale o in orizzontale.
void phoneScreen(WidgetTester tester, {required bool landscape}) {
  tester.view.physicalSize = landscape ? const Size(2400, 1080) : const Size(1080, 2400);
  tester.view.devicePixelRatio = 2.6;
  addTearDown(tester.view.reset);
}

/// Avvia l'app e preme "AVVIA" (con le impostazioni salvate in [prefs]).
Future<void> openLive(WidgetTester tester, {Map<String, Object> prefs = const {}}) async {
  SharedPreferences.setMockInitialValues(prefs); // archivio delle impostazioni finto e vuoto
  await tester.pumpWidget(const LandingApp(useSensors: false));
  await tester.pumpAndSettle(); // aspetta la lettura delle impostazioni
  await tester.tap(find.text('AVVIA'));
  await settle(tester);
}

/// Fa avanzare il tempo di 1 s (animazioni di pagine e pannelli). Nella schermata della camera
/// non si può usare pumpAndSettle: il riepilogo ogni 10 s aggiorna sempre lo schermo, quindi
/// "aspettare che tutto si fermi" non finirebbe mai.
Future<void> settle(WidgetTester tester) async {
  await tester.pump();
  await tester.pump(const Duration(seconds: 1));
}

/// Logica portata nello stato voluto senza camera: zona proposta (3 s di frame con la zona)
/// oppure analisi ferma (nessuna zona per molto tempo). Griglia 120x160 (verticale).
LandingLogic logicIn(ZoneState target) {
  final logic = LandingLogic();
  final zone = YOLOResult(
    classIndex: 0,
    className: 'forbidden_area',
    confidence: 0.9,
    boundingBox: Rect.zero,
    normalizedBox: Rect.zero,
    mask: [for (var y = 0; y < 160; y++) List<double>.filled(120, y > 100 ? 1 : 0)],
  );
  final t0 = DateTime(2026, 1, 1, 12);
  for (var i = 0; logic.zoneState != target && i < 1000; i++) {
    logic.process(target == ZoneState.proposal ? [zone] : [], now: t0.add(Duration(milliseconds: 125 * i)));
  }
  expect(logic.zoneState, target);
  return logic;
}

void main() {
  for (final landscape in [false, true]) {
    for (final mode in ['test', 'match']) {
      testWidgets('Schermata iniziale ${landscape ? 'orizzontale' : 'verticale'}, modalità $mode', (tester) async {
        phoneScreen(tester, landscape: landscape);
        SharedPreferences.setMockInitialValues({'mode': mode});
        await tester.pumpWidget(const LandingApp(useSensors: false));
        await tester.pumpAndSettle();
        // Se qualcosa non ci stesse, Flutter segnalerebbe un "overflow" e il test fallirebbe.
        expect(find.text('AVVIA'), findsOneWidget);
        expect(find.text('Orientamento'), findsOneWidget);
        expect(find.text('Log'), mode == 'test' ? findsOneWidget : findsNothing); // log solo in Test
      });
    }
  }

  testWidgets('Le scelte vengono salvate e ritrovate al prossimo avvio', (tester) async {
    SharedPreferences.setMockInitialValues({});
    await tester.pumpWidget(const LandingApp(useSensors: false));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Partita'));
    await tester.tap(find.text('Verticale'));
    await tester.pump();
    await tester.tap(find.text('AVVIA'));
    await settle(tester);
    final saved = await AppSettings.load();
    expect(saved.mode, AppMode.match);
    expect(saved.orientation, OrientationChoice.portrait);
  });

  for (final landscape in [false, true]) {
    testWidgets('Camera ${landscape ? 'orizzontale' : 'verticale'}: comandi di Test; Indietro torna all\'inizio',
        (tester) async {
      phoneScreen(tester, landscape: landscape);
      await openLive(tester, prefs: {'mode': 'test'});
      // Se qualcosa non ci stesse, il test fallirebbe con un "overflow".
      for (final t in ['Impostazioni', 'Disegna zona', 'Ricalibra zona', 'Segna landing', 'Mostra rilevamenti grezzi']) {
        expect(find.byTooltip(t), findsOneWidget, reason: t);
      }
      expect(find.text('allarmi 0 · segnati 0'), findsOneWidget);
      expect(find.textContaining('CALIBRAZIONE'), findsOneWidget); // pillola di stato
      await tester.binding.handlePopRoute(); // il gesto "indietro" di Android
      await settle(tester);
      expect(find.text('AVVIA'), findsOneWidget);
    });
  }

  for (final landscape in [false, true]) {
    for (final state in [ZoneState.proposal, ZoneState.gaveUp]) {
      testWidgets('${state.name} ${landscape ? 'orizzontale' : 'verticale'}: pulsanti tutti sullo schermo',
          (tester) async {
        phoneScreen(tester, landscape: landscape);
        SharedPreferences.setMockInitialValues({});
        await tester.pumpWidget(
          MaterialApp(home: DetectionPage(useSensors: false, logic: logicIn(state))),
        );
        await tester.pump();
        final tips = state == ZoneState.proposal
            ? ['Conferma zona', 'Ricalibra', 'Correggi', 'Impostazioni']
            : ['Riprova', 'Disegna a mano', 'Impostazioni'];
        final screen = tester.view.physicalSize / tester.view.devicePixelRatio;
        for (final t in tips) {
          final r = tester.getRect(find.byTooltip(t));
          // Ogni pulsante deve stare interamente dentro lo schermo.
          expect(r.left >= 0 && r.top >= 0 && r.right <= screen.width && r.bottom <= screen.height, isTrue,
              reason: '$t fuori dallo schermo: $r');
        }
        await tester.pumpWidget(const SizedBox()); // chiude la pagina (timer, sensori)
      });
    }
  }

  for (final landscape in [false, true]) {
    testWidgets('Disegno ${landscape ? 'orizzontale' : 'verticale'}: strumenti tutti sullo schermo', (tester) async {
      phoneScreen(tester, landscape: landscape);
      SharedPreferences.setMockInitialValues({});
      final logic = logicIn(ZoneState.proposal)..startEditing(fallbackW: 120, fallbackH: 160);
      await tester.pumpWidget(MaterialApp(home: DetectionPage(useSensors: false, logic: logic)));
      await tester.pump();
      final screen = tester.view.physicalSize / tester.view.devicePixelRatio;
      for (final t in ['Traccia bordo', 'Aggiungi', 'Cancella', 'Pennello medio', 'Conferma disegno', 'Annulla disegno']) {
        final r = tester.getRect(find.byTooltip(t));
        expect(r.left >= 0 && r.top >= 0 && r.right <= screen.width && r.bottom <= screen.height, isTrue,
            reason: '$t fuori dallo schermo: $r');
      }
      await tester.pumpWidget(const SizedBox());
    });
  }

  testWidgets('Pennello: dimensione a giro; il piccolo cancella meno del grande', (tester) async {
    phoneScreen(tester, landscape: false);
    SharedPreferences.setMockInitialValues({});
    final logic = logicIn(ZoneState.proposal)..startEditing(fallbackW: 120, fallbackH: 160);
    await tester.pumpWidget(MaterialApp(home: DetectionPage(useSensors: false, logic: logic)));
    await tester.pump();
    await tester.tap(find.byTooltip('Cancella'));
    await tester.pump();

    // Una passata di Cancella sulla zona (parte bassa dello schermo), con un dato pennello.
    // [y] diversa per ogni passata: una passata sola sulla stessa riga dell'altra non troverebbe
    // più niente da cancellare.
    Future<int> erased(double y) async {
      final before = logic.displayZoneCells;
      final g = await tester.startGesture(Offset(60, y));
      await g.moveTo(Offset(300, y));
      await g.up();
      await tester.pump();
      return before - logic.displayZoneCells;
    }

    await tester.tap(find.byTooltip('Pennello medio'));
    await tester.pump();
    expect(find.byTooltip('Pennello grande'), findsOneWidget);
    final big = await erased(650);
    await tester.tap(find.byTooltip('Pennello grande'));
    await tester.pump();
    expect(find.byTooltip('Pennello piccolo'), findsOneWidget);
    final small = await erased(760);
    expect(small, greaterThan(0));
    expect(small, lessThan(big));
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('Camera in Partita: niente pulsanti di Test, niente contatori, niente dettagli', (tester) async {
    phoneScreen(tester, landscape: true);
    await openLive(tester, prefs: {'mode': 'match'});
    expect(find.byTooltip('Segna landing'), findsNothing);
    expect(find.byTooltip('Mostra rilevamenti grezzi'), findsNothing);
    expect(find.textContaining('segnati'), findsNothing);
    await tester.tap(find.textContaining('CALIBRAZIONE')); // in Partita la pillola non apre i dettagli
    await settle(tester);
    expect(find.textContaining('punteggio'), findsNothing);
    expect(find.byTooltip('Impostazioni'), findsOneWidget);
  });

  testWidgets('Test: la pillola apre e chiude i dettagli; Termina sessione torna all\'inizio', (tester) async {
    phoneScreen(tester, landscape: true);
    await openLive(tester, prefs: {'mode': 'test'});
    await tester.tap(find.textContaining('CALIBRAZIONE'));
    await settle(tester);
    expect(find.textContaining('analisi reali'), findsOneWidget);
    await tester.tap(find.textContaining('CALIBRAZIONE'));
    await settle(tester);
    expect(find.textContaining('analisi reali'), findsNothing);
    await tester.tap(find.byTooltip('Impostazioni'));
    await settle(tester);
    await tester.tap(find.text('Termina sessione'));
    await settle(tester);
    expect(find.text('AVVIA'), findsOneWidget);
  });

  testWidgets('Log completo acceso e spento a metà sessione (solo in Test)', (tester) async {
    await openLive(tester, prefs: {'mode': 'test', 'fullLog': false});
    expect(LogConfig.full, isFalse);
    await tester.tap(find.byTooltip('Impostazioni'));
    await settle(tester);
    await tester.tap(find.text('Log completo'));
    await settle(tester);
    expect(LogConfig.full, isTrue);
    await tester.tap(find.text('Log completo'));
    await settle(tester);
    expect(LogConfig.full, isFalse);
  });

  testWidgets('Disegno a mano: strumenti, Annulla', (tester) async {
    phoneScreen(tester, landscape: false);
    await openLive(tester);
    await tester.tap(find.byTooltip('Disegna zona'));
    await tester.pump();
    expect(find.byTooltip('Traccia bordo'), findsOneWidget); // strumenti del disegno
    expect(find.byTooltip('Segna landing'), findsNothing); // comandi abituali nascosti mentre si disegna
    // Un bordo tracciato col dito sull'immagine.
    final gesture = await tester.startGesture(const Offset(100, 300));
    for (final p in const [Offset(300, 300), Offset(300, 500), Offset(100, 500), Offset(100, 310)]) {
      await gesture.moveTo(p);
    }
    await gesture.up();
    await tester.pump();
    expect(find.textContaining(' 0 celle'), findsNothing); // il bordo ha riempito delle celle
    await tester.tap(find.byTooltip('Annulla disegno'));
    await tester.pump();
    expect(find.byTooltip('Segna landing'), findsOneWidget);
  });

  testWidgets('Background: ricalibra al ritorno; la tendina notifiche non conta', (tester) async {
    // Sostituiamo debugPrint per leggere il log dell'app.
    final original = debugPrint;
    final log = <String>[];
    debugPrint = (String? message, {int? wrapWidth}) => log.add(message ?? '');
    int count(String text) => log.where((l) => l.contains(text)).length;
    void go(List<AppLifecycleState> states) => states.forEach(tester.binding.handleAppLifecycleStateChanged);

    try {
      await openLive(tester);
      go([AppLifecycleState.resumed, AppLifecycleState.inactive, AppLifecycleState.resumed]); // tendina
      expect(count('BACKGROUND'), 0);

      go([AppLifecycleState.inactive, AppLifecycleState.hidden, AppLifecycleState.paused]); // blocco
      expect(count('APP IN BACKGROUND'), 1);
      expect(count('RIEPILOGO'), 0);
      await tester.pump(kSummaryPeriod * 3); // in background il timer del riepilogo è fermo
      expect(count('RIEPILOGO'), 0);

      go([AppLifecycleState.hidden, AppLifecycleState.inactive, AppLifecycleState.resumed]); // sblocco
      expect(count('RITORNO DAL BACKGROUND: ricalibrazione'), 1);
      await tester.pump(kSummaryPeriod + const Duration(seconds: 1));
      expect(count('RIEPILOGO'), 1); // ripartito, una sola volta
    } finally {
      debugPrint = original;
    }
  });
}
