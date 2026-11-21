// Passo 2: anteprima della camera a schermo (+ il pulsante del passo 1 sovrapposto).
//
// Concetti nuovi rispetto al passo 1:
// - Pacchetto "camera": parla con la camera nativa del telefono (CameraX su Android,
//   AVFoundation su iOS). Noi usiamo solo l'interfaccia Dart, la parte nativa è nel plugin.
// - Permessi: il permesso CAMERA va dichiarato nel manifest Android (lo fa già il plugin,
//   e l'abbiamo scritto anche noi per chiarezza) e va CONCESSO dall'utente a runtime.
//   La finestra "Consenti a landing_detector di scattare foto?" compare da sola la prima
//   volta che chiamiamo controller.initialize().
// - Ciclo di vita dell'app: quando l'app va in background la camera va rilasciata
//   (altrimenti Android la "ruba" o la app crasha), e riaperta quando si torna.
// - Tipi "nullable": `CameraController?` può valere null (come un puntatore che può essere
//   nullptr). Dart ti obbliga a controllarlo prima di usarlo; `x!` significa
//   "so che non è null" (se sbagli, eccezione a runtime).

import 'package:audioplayers/audioplayers.dart';
import 'package:camera/camera.dart';
import 'package:material_ui/material_ui.dart';

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
      home: const CameraPage(),
    );
  }
}

/// Schermata principale: anteprima della camera con il pulsante di allarme sopra.
class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

// "with WidgetsBindingObserver" è un *mixin*: aggiunge alla classe i metodi di un'altra
// (qui le callback del ciclo di vita dell'app). In C++ somiglia a ereditare anche da una
// classe-interfaccia con metodi virtuali già implementati vuoti: sovrascriviamo solo quello
// che ci serve (didChangeAppLifecycleState). È il classico pattern "observer".
class _CameraPageState extends State<CameraPage> with WidgetsBindingObserver {
  // --- Camera ---
  CameraController? _controller; // null finché la camera non è pronta
  String? _errorMessage; // se non è null, mostriamo un errore al posto dell'anteprima

  // --- Audio (dal passo 1) ---
  final AudioPlayer _player = AudioPlayer();
  int _alarmCount = 0;

  // initState(): chiamato UNA volta quando lo State nasce, come un costruttore.
  // Non può essere async, quindi lanciamo _initCamera() senza aspettarla: andrà avanti
  // "in sottofondo" e quando finisce chiamerà setState() per ridisegnare.
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this); // iscriviti agli eventi del ciclo di vita
    _initCamera();
  }

  /// Cerca la camera posteriore e la apre.
  Future<void> _initCamera() async {
    try {
      // Elenco delle camere del telefono (posteriore, frontale, a volte più posteriori).
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        _showError('Nessuna camera trovata su questo dispositivo.');
        return;
      }

      // Prendi la prima camera posteriore; se non c'è, la prima disponibile.
      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        camera,
        // "high" = 720p: sufficiente per la rete (che lavora a 640 px) e leggero da gestire.
        ResolutionPreset.high,
        enableAudio: false, // non registriamo audio → non serve il permesso microfono
      );

      // Qui Android chiede il permesso all'utente (solo la prima volta).
      await controller.initialize();

      // Mentre aspettavamo, la schermata potrebbe essere stata chiusa: "mounted" dice se
      // questo State è ancora attivo. Se no, liberiamo la camera ed usciamo.
      if (!mounted) {
        await controller.dispose();
        return;
      }

      setState(() {
        _controller = controller;
        _errorMessage = null;
      });
    } on CameraException catch (e) {
      // Errori specifici della camera, ad es. permesso negato.
      if (e.code == 'CameraAccessDenied') {
        _showError('Permesso camera negato.\n'
            'Abilitalo da Impostazioni → App → landing_detector → Autorizzazioni.');
      } else {
        _showError('Errore camera: ${e.code}\n${e.description ?? ''}');
      }
    } catch (e) {
      // Qualunque altro errore (es. nei test automatici, dove la camera non esiste).
      _showError('Errore camera: $e');
    }
  }

  void _showError(String message) {
    if (!mounted) return;
    setState(() => _errorMessage = message);
  }

  // Chiamata dal sistema quando l'app cambia stato (in primo piano, in background, ...).
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // debugPrint ≈ std::cout per il debug: compare nel terminale come "I/flutter (...)".
    debugPrint('Ciclo di vita: $state');
    final controller = _controller;
    if (state == AppLifecycleState.inactive) {
      // L'app sta perdendo il primo piano: rilascia la camera.
      if (controller == null) return;
      setState(() => _controller = null); // prima togli l'anteprima dallo schermo...
      controller.dispose(); // ...poi chiudi la camera
    } else if (state == AppLifecycleState.resumed) {
      // L'app torna in primo piano: riapri la camera (se non è già aperta).
      if (controller == null) _initCamera();
    }
  }

  Future<void> _playAlarm() async {
    await _player.stop();
    await _player.play(AssetSource('sounds/alarm.wav'));
    setState(() => _alarmCount++);
  }

  // Distruttore: smetti di osservare il ciclo di vita e libera camera e audio.
  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose(); // "?." = chiama dispose() solo se _controller non è null
    _player.dispose();
    super.dispose();
  }

  /// Il contenuto centrale: errore, rotellina di attesa oppure anteprima.
  Widget _buildPreview() {
    if (_errorMessage != null) {
      return Padding(
        padding: const EdgeInsets.all(24),
        child: Text(
          _errorMessage!,
          textAlign: TextAlign.center,
          style: const TextStyle(color: Colors.white, fontSize: 18),
        ),
      );
    }
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return const CircularProgressIndicator(); // camera in apertura
    }
    // CameraPreview mostra il flusso video mantenendo le proporzioni corrette.
    return CameraPreview(controller);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(title: const Text('Landing Detector – camera')),
      // Stack sovrappone i figli uno sopra l'altro (il primo è quello più in basso),
      // come livelli in un programma di grafica.
      body: Stack(
        children: [
          Center(child: _buildPreview()),
          // Positioned fissa un figlio dello Stack rispetto ai bordi: qui in basso al centro.
          Positioned(
            left: 0,
            right: 0,
            bottom: 24,
            child: Column(
              mainAxisSize: MainAxisSize.min, // la colonna occupa solo lo spazio necessario
              children: [
                FilledButton.icon(
                  onPressed: _playAlarm,
                  icon: const Icon(Icons.campaign, size: 32),
                  label: const Text('LANDING!', style: TextStyle(fontSize: 24)),
                  style: FilledButton.styleFrom(backgroundColor: Colors.red),
                ),
                const SizedBox(height: 8),
                Text(
                  'Allarmi suonati: $_alarmCount',
                  style: const TextStyle(color: Colors.white, fontSize: 16),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
