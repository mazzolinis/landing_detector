// Test minimo: verifica che la schermata mostri il pulsante e il contatore a zero.
// (Non premiamo il pulsante: nei test non c'è un lettore audio nativo.)

import 'package:flutter_test/flutter_test.dart';

import 'package:landing_detector/main.dart';

void main() {
  testWidgets('Mostra pulsante e contatore', (WidgetTester tester) async {
    await tester.pumpWidget(const LandingApp());

    expect(find.text('LANDING!'), findsOneWidget);
    expect(find.text('Allarmi suonati: 0'), findsOneWidget);
  });
}
