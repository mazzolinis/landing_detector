// Passo 1: un pulsante che riproduce un suono di allarme.
//
// Concetti chiave:
// - In Flutter TUTTO è un "widget": un oggetto che descrive un pezzo di interfaccia.
//   Il metodo build() restituisce un albero di widget (un po' come comporre oggetti in C++).
// - StatelessWidget: non ha stato modificabile (tipo una classe con soli membri const).
// - StatefulWidget: ha uno "State" separato che sopravvive ai ridisegni; quando chiami
//   setState(), Flutter richiama build() e ridisegna lo schermo.

import 'package:audioplayers/audioplayers.dart'; // pacchetto esterno (dichiarato in pubspec.yaml)
import 'package:material_ui/material_ui.dart'; // widget grafici in stile Material

// Punto di ingresso, come main() in C++.
void main() {
  runApp(const LandingApp());
}

/// Widget radice dell'app: imposta tema e schermata iniziale. Non ha stato.
class LandingApp extends StatelessWidget {
  const LandingApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Landing Detector',
      theme: ThemeData(colorScheme: .fromSeed(seedColor: Colors.orange)),
      home: const SoundTestPage(),
    );
  }
}

/// Schermata con il pulsante. È "stateful" perché ricorda quante volte è suonato l'allarme.
class SoundTestPage extends StatefulWidget {
  const SoundTestPage({super.key});

  @override
  State<SoundTestPage> createState() => _SoundTestPageState();
}

// Il trattino basso iniziale (_) rende la classe privata al file (simile a "static" in C++).
class _SoundTestPageState extends State<SoundTestPage> {
  // Il lettore audio: creato una volta e riusato. "final" = il riferimento non cambia
  // (come un puntatore const, ma l'oggetto puntato può cambiare stato).
  final AudioPlayer _player = AudioPlayer();

  // Stato della schermata: contatore degli allarmi.
  int _alarmCount = 0;

  // "async" + "await": la funzione può aspettare un'operazione lenta (es. avvio audio)
  // SENZA bloccare l'interfaccia. Simile a una std::future, ma con sintassi lineare.
  Future<void> _playAlarm() async {
    await _player.stop(); // se sta già suonando, riparte da capo
    await _player.play(AssetSource('sounds/alarm.wav')); // percorso relativo a "assets/"

    // setState: modifica lo stato e chiede a Flutter di richiamare build().
    setState(() {
      _alarmCount++;
    });
  }

  // dispose(): chiamato quando la schermata viene distrutta, come un distruttore C++.
  // Qui liberiamo le risorse native del lettore audio.
  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Scaffold = struttura base di una pagina (barra in alto + corpo).
    return Scaffold(
      appBar: AppBar(title: const Text('Landing Detector – test suono')),
      body: Center(
        // Column dispone i figli in verticale.
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            FilledButton.icon(
              onPressed: _playAlarm, // funzione chiamata al tocco (una callback)
              icon: const Icon(Icons.campaign, size: 40),
              label: const Text('LANDING!', style: TextStyle(fontSize: 32)),
              style: FilledButton.styleFrom(
                backgroundColor: Colors.red,
                padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 24),
              ),
            ),
            const SizedBox(height: 24), // spazio vuoto verticale
            Text('Allarmi suonati: $_alarmCount', style: const TextStyle(fontSize: 20)),
          ],
        ),
      ),
    );
  }
}
