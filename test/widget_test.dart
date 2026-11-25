// Test minimi dell'interfaccia.
// (Non premiamo il pulsante: nei test non c'è un lettore audio nativo.)

import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:landing_detector/main.dart';

void main() {
  testWidgets('Mostra pulsante e contatori', (WidgetTester tester) async {
    await tester.pumpWidget(const LandingApp(useSensors: false));

    expect(find.text('LANDING!'), findsOneWidget);
    expect(find.text('Allarmi automatici: 0   manuali: 0'), findsOneWidget);
  });

  testWidgets('Disegno a mano: strumenti, Annulla', (tester) async {
    tester.view.physicalSize = const Size(1080, 2400); // schermo verticale come un telefono
    tester.view.devicePixelRatio = 2.6;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(const LandingApp(useSensors: false));
    await tester.tap(find.byTooltip('Disegna zona'));
    await tester.pump();
    expect(find.text('Traccia bordo'), findsOneWidget); // barra del disegno
    expect(find.text('LANDING!'), findsNothing); // nascosto mentre si disegna
    // Un bordo tracciato col dito sull'immagine.
    final gesture = await tester.startGesture(const Offset(100, 300));
    for (final p in const [Offset(300, 300), Offset(300, 500), Offset(100, 500), Offset(100, 310)]) {
      await gesture.moveTo(p);
    }
    await gesture.up();
    await tester.pump();
    expect(find.textContaining('celle'), findsWidgets);
    expect(find.textContaining(' 0 celle'), findsNothing); // il bordo ha riempito delle celle
    await tester.tap(find.text('Annulla'));
    await tester.pump();
    expect(find.text('LANDING!'), findsOneWidget);
  });

  testWidgets('Background: ricalibra al ritorno; la tendina notifiche non conta', (tester) async {
    // Sostituiamo debugPrint per leggere il log dell'app.
    final original = debugPrint;
    final log = <String>[];
    debugPrint = (String? message, {int? wrapWidth}) => log.add(message ?? '');
    int count(String text) => log.where((l) => l.contains(text)).length;
    void go(List<AppLifecycleState> states) => states.forEach(tester.binding.handleAppLifecycleStateChanged);

    try {
      await tester.pumpWidget(const LandingApp(useSensors: false));
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
