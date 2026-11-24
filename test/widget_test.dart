// Test minimo dell'interfaccia: la schermata mostra il pulsante e i contatori a zero.
// (Non premiamo il pulsante: nei test non c'è un lettore audio nativo.)

import 'package:flutter_test/flutter_test.dart';

import 'package:landing_detector/main.dart';

void main() {
  testWidgets('Mostra pulsante e contatori', (WidgetTester tester) async {
    await tester.pumpWidget(const LandingApp(useSensors: false));

    expect(find.text('LANDING!'), findsOneWidget);
    expect(find.text('Allarmi automatici: 0   manuali: 0'), findsOneWidget);
  });
}
