// Passo 4: logica del fallo di landing, punteggio mobile e allarme automatico.
//
// Pipeline per ogni frame analizzato:
//   1. il plugin "ultralytics_yolo" (YOLOView, codice nativo) esegue la rete sulla GPU e ci
//      passa i rilevamenti con le maschere tramite la callback onResult;
//   2. LandingLogic.process() (logica pura, senza grafica, testabile) decide se il frame è un
//      landing e aggiorna il punteggio mobile;
//   3. se il punteggio raggiunge la soglia, suona l'allarme;
//   4. LogicOverlayPainter disegna sopra la camera cosa "pensa" la logica: zona in memoria,
//      corpo del portatore e piedi (rossi se nella zona).
//
// Formato delle maschere:
//   result.mask è una matrice (lista di righe) che copre TUTTO il frame a 1/4 della
//   risoluzione del modello (frame 4:3, imgsz 640 → 160x120 in orizzontale, 120x160 in
//   verticale). Ogni cella contiene un "logit": > 0 significa "questo pixel appartiene
//   all'oggetto", 0 = fuori. Tutte le maschere di un frame usano la stessa griglia.

import 'dart:math' as math; // min, max (come <algorithm> in C++)
import 'dart:typed_data'; // Float64List: array di double contigui, come std::vector<double>

import 'package:audioplayers/audioplayers.dart';
import 'package:material_ui/material_ui.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

// ============================================================================================
// Costanti: modello e camera
// ============================================================================================

/// Dimensione d'ingresso (imgsz) del modello da usare: deve esistere il file
/// `assets/models/landing_<kModelSize>.tflite`, esportato con lo stesso imgsz.
///
/// ATTENZIONE: il plugin copia il modello dagli asset nella memoria dell'app solo la PRIMA
/// volta e poi riusa la copia se il nome è lo stesso. Per questo il nome contiene la
/// dimensione: se riesporti un modello con lo stesso imgsz, cambia nome (es. landing_640_v2)
/// oppure cancella i dati dell'app dal telefono (Impostazioni → App → Spazio di archiviazione).
const int kModelSize = 640;

/// Percorso del modello dentro gli asset dell'app (cartella dichiarata in pubspec.yaml).
/// '$kModelSize' inserisce il numero nella stringa (come std::format in C++).
const String kModelPath = 'assets/models/landing_$kModelSize.tflite';

/// Risoluzione dei frame passati alla rete: 4:3 come l'anteprima, lato lungo = imgsz.
const Size kAnalysisResolution = Size(kModelSize * 1.0, kModelSize * 0.75);

/// Frequenza di analisi variabile (inferenze al secondo):
/// - "riposo": nessun portatore di palla in vista → si analizza poco, il telefono si scalda meno;
/// - "azione": c'è un portatore → si analizza il più spesso possibile.
///   kActiveHz è solo un TETTO: la rete non supera comunque ~11 analisi/s; mettendolo più
///   alto, il plugin analizza un nuovo frame appena la GPU si libera.
const int kIdleHz = 2;
const int kActiveHz = 30;

/// Quanto restare in modalità "azione" dopo l'ultimo portatore visto.
const Duration kActiveHold = Duration(seconds: 2);

// ============================================================================================
// Costanti: logica del fallo (da tarare durante i test)
// ============================================================================================

/// Confidenza minima per considerare un portatore (carrier_ground / carrier_jump).
/// Bassa perché il modello attuale vede il landing vero a ~0.3: da alzare dopo il riaddestramento.
const double kCarrierMinConf = 0.25;

/// Confidenza minima perché un rilevamento di forbidden_area aggiorni la memoria della zona.
const double kZoneMinConf = 0.25;

/// Filtro passa-basso della zona: memoria = α·memoria + (1−α)·misura.
/// Più vicino a 1 = più stabile ma più lento ad adattarsi.
const double kZoneAlpha = 0.8;

/// Una cella fa parte della zona se la sua memoria supera questa soglia ("più sì che no").
const double kZoneThreshold = 0.5;

/// Se la zona non viene vista per questo tempo, la memoria si azzera (telefono spostato?).
const Duration kZoneForget = Duration(seconds: 10);

/// Frazione più bassa della maschera del portatore considerata "piedi" (0.15 = ultimo 15%
/// dell'altezza del giocatore).
const double kFeetFraction = 0.15;

/// Perché il frame conti come landing, le celle dei piedi nella zona devono essere ALMENO
/// kFeetMinCells (contro il rumore delle maschere piccole) E almeno kFeetMinFraction del totale
/// dei piedi (per i giocatori grandi/vicini, dove 3 celle sono una frazione minima).
/// 0.3 lascia margine al caso "un piede dentro e uno fuori" (~50%).
const int kFeetMinCells = 3;
const double kFeetMinFraction = 0.3;

/// Punteggio mobile: +1 per frame con landing, −1 senza (minimo 0, massimo kScoreMax).
/// L'allarme suona quando il punteggio RAGGIUNGE kScoreThreshold; poi si "riarma" solo
/// quando il punteggio torna a 0 (così un landing lungo non fa suonare a ripetizione).
const int kScoreThreshold = 3;

/// Tetto del punteggio: senza, un landing lungo accumulerebbe tanti punti e ci vorrebbe
/// molto tempo per tornare a 0 e riarmare l'allarme.
const int kScoreMax = 5;

YOLOStreamingConfig streamingConfigFor(int hz) => YOLOStreamingConfig.custom(
      includeMasks: true, // servono le maschere in Dart per la logica
      inferenceFrequency: hz,
      analysisResolution: kAnalysisResolution,
    );

// ============================================================================================
// Logica del fallo (Dart puro: nessun widget, quindi si può testare senza telefono)
// ============================================================================================

/// Esito dell'analisi di un frame: serve per il log, per lo schermo e per i test.
class FrameReport {
  final YOLOResult? carrier; // portatore scelto (il più sicuro), null se nessuno
  final int feetCells; // celle dei piedi del portatore (0 se non è carrier_ground)
  final int feetInZone; // di queste, quante sono nella zona proibita
  final int feetTop; // prima riga della fascia dei piedi (-1 se non calcolata): serve all'overlay
  final int zoneCells; // celle della zona in memoria
  final bool zoneSeen; // la rete ha visto la zona in QUESTO frame
  final bool landing; // questo frame conta come landing
  final int score; // punteggio mobile dopo questo frame
  final bool alarm; // l'allarme deve suonare ADESSO

  const FrameReport({
    required this.carrier,
    required this.feetCells,
    required this.feetInZone,
    required this.feetTop,
    required this.zoneCells,
    required this.zoneSeen,
    required this.landing,
    required this.score,
    required this.alarm,
  });

  /// Percentuale dei piedi nella zona (0..100).
  int get feetPercent => feetCells == 0 ? 0 : (100 * feetInZone / feetCells).round();

  /// Riga di log, es.:
  /// "carrier_ground 0.31 | piedi 6/24 (25%) nella zona | zona 1350 celle (vista) | punteggio 2 (allarme a 3)"
  String describe() {
    final c = carrier;
    final who = c == null ? 'nessun portatore' : '${c.className} ${c.confidence.toStringAsFixed(2)}';
    final feet = c?.className == 'carrier_ground'
        ? 'piedi $feetInZone/$feetCells ($feetPercent%) nella zona${landing ? ' → LANDING' : ''}'
        : '-';
    final zone = 'zona $zoneCells celle${zoneSeen ? ' (vista)' : ' (memoria)'}';
    return '$who | $feet | $zone | punteggio $score (allarme a $kScoreThreshold)'
        '${alarm ? ' | *** ALLARME ***' : ''}';
  }
}

/// Contiene lo stato che sopravvive tra un frame e l'altro: memoria della zona e punteggio.
/// La decisione "landing sì/no" invece usa solo il frame corrente (analisi a frame singolo).
class LandingLogic {
  // Memoria della zona: un valore in [0, 1] per ogni cella della griglia, salvato riga per
  // riga in un array piatto (cella (x, y) → indice y * _gridW + x), come in C/C++.
  Float64List? _zoneMemory; // null = nessuna zona in memoria
  int _gridW = 0;
  int _gridH = 0;
  DateTime _lastZoneSeen = DateTime(2000);

  int _score = 0;
  bool _armed = true; // l'allarme può suonare (si riarma quando il punteggio torna a 0)

  int get score => _score;

  // Per l'overlay: dimensioni della griglia e appartenenza di una cella alla zona in memoria.
  int get gridW => _gridW;
  int get gridH => _gridH;
  bool isZoneCell(int x, int y) {
    final memory = _zoneMemory;
    if (memory == null || x < 0 || y < 0 || x >= _gridW || y >= _gridH) return false;
    return memory[y * _gridW + x] > kZoneThreshold;
  }

  /// Dimentica la zona (pulsante "Reset zona"): utile nei test con le foto, dove ogni foto
  /// nuova equivale a spostare il telefono su un altro campo. Il punteggio scende da solo.
  void resetZone() {
    _zoneMemory = null;
  }

  /// Analizza i rilevamenti di un frame. [now] si può passare nei test per simulare il tempo.
  FrameReport process(List<YOLOResult> results, {DateTime? now}) {
    now ??= DateTime.now(); // "??=" assegna solo se è null

    // 1. Un solo portatore per frame: quello con la confidenza più alta.
    YOLOResult? carrier;
    for (final r in results) {
      final isCarrier = r.className == 'carrier_ground' || r.className == 'carrier_jump';
      if (isCarrier && r.confidence >= kCarrierMinConf) {
        if (carrier == null || r.confidence > carrier.confidence) carrier = r;
      }
    }

    // 2. Aggiorna la memoria della zona con le zone viste in questo frame (anche più di una).
    final zoneMasks = [
      for (final r in results)
        if (r.className == 'forbidden_area' && r.confidence >= kZoneMinConf && _hasMask(r)) r.mask!,
    ];
    final zoneSeen = zoneMasks.isNotEmpty;
    if (zoneSeen) {
      _updateZone(zoneMasks);
      _lastZoneSeen = now;
    } else if (now.difference(_lastZoneSeen) > kZoneForget) {
      _zoneMemory = null; // zona non vista da troppo tempo: dimenticala
    }

    // 3. Piedi del portatore (solo se è a terra) dentro la zona in memoria.
    var feetCells = 0;
    var feetInZone = 0;
    var feetTop = -1;
    final carrierMask = carrier?.mask;
    if (carrier != null &&
        carrier.className == 'carrier_ground' &&
        carrierMask != null &&
        _zoneMemory != null &&
        _isSameGrid(carrierMask)) {
      // Un "record" (a, b, c): restituisce più valori insieme, come std::tuple + structured binding.
      (feetCells, feetInZone, feetTop) = _countFeetInZone(carrierMask);
    }
    final landing = feetInZone >= kFeetMinCells && feetInZone >= kFeetMinFraction * feetCells;

    // 4. Punteggio mobile e allarme.
    _score = landing ? math.min(_score + 1, kScoreMax) : math.max(_score - 1, 0);
    var alarm = false;
    if (_armed && _score >= kScoreThreshold) {
      alarm = true;
      _armed = false;
    }
    if (_score == 0) _armed = true;

    return FrameReport(
      carrier: carrier,
      feetCells: feetCells,
      feetInZone: feetInZone,
      feetTop: feetTop,
      zoneCells: _countZoneCells(),
      zoneSeen: zoneSeen,
      landing: landing,
      score: _score,
      alarm: alarm,
    );
  }

  bool _hasMask(YOLOResult r) => r.mask != null && r.mask!.isNotEmpty && r.mask![0].isNotEmpty;

  bool _isSameGrid(List<List<double>> mask) =>
      mask.length == _gridH && mask[0].length == _gridW;

  /// Filtro passa-basso cella per cella: memoria = α·memoria + (1−α)·misura (misura 0 o 1).
  void _updateZone(List<List<List<double>>> zoneMasks) {
    final h = zoneMasks[0].length;
    final w = zoneMasks[0][0].length;

    // Misura di questo frame: 1 se la cella è in almeno una delle zone viste.
    final measured = Float64List(w * h); // inizializzato a 0
    for (final mask in zoneMasks) {
      if (mask.length != h || mask[0].length != w) continue; // griglie diverse: ignora
      for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
          if (mask[y][x] > 0) measured[y * w + x] = 1;
        }
      }
    }

    final memory = _zoneMemory;
    if (memory == null || w != _gridW || h != _gridH) {
      // Prima zona vista (o griglia cambiata, es. telefono ruotato): parti dalla misura,
      // così la zona è utilizzabile subito invece che dopo 4 frame.
      _zoneMemory = measured;
      _gridW = w;
      _gridH = h;
      return;
    }
    for (var i = 0; i < memory.length; i++) {
      memory[i] = kZoneAlpha * memory[i] + (1 - kZoneAlpha) * measured[i];
    }
  }

  int _countZoneCells() {
    final memory = _zoneMemory;
    if (memory == null) return 0;
    return memory.where((v) => v > kZoneThreshold).length;
  }

  /// Conta le celle dei "piedi" (fascia più bassa della maschera del portatore) e quante
  /// di queste sono nella zona in memoria. Restituisce (celle piedi, celle nella zona, prima riga).
  (int, int, int) _countFeetInZone(List<List<double>> mask) {
    // Riga più alta e più bassa occupate dal giocatore.
    var top = -1;
    var bottom = -1;
    for (var y = 0; y < _gridH; y++) {
      if (mask[y].any((v) => v > 0)) {
        if (top < 0) top = y;
        bottom = y;
      }
    }
    if (top < 0) return (0, 0, -1); // maschera vuota

    final height = bottom - top + 1;
    final feetRows = math.max(1, (height * kFeetFraction).ceil());
    final feetTop = bottom - feetRows + 1;
    final memory = _zoneMemory!;

    var feet = 0;
    var inZone = 0;
    for (var y = feetTop; y <= bottom; y++) {
      for (var x = 0; x < _gridW; x++) {
        if (mask[y][x] > 0) {
          feet++;
          if (memory[y * _gridW + x] > kZoneThreshold) inZone++;
        }
      }
    }
    return (feet, inZone, feetTop);
  }
}

// ============================================================================================
// Overlay della logica (disegnato da Flutter sopra la camera)
// ============================================================================================

/// Disegna la griglia delle maschere sopra l'anteprima.
///
/// Un CustomPainter è come un onDraw/paintEvent in Qt: Flutter ci dà un Canvas e la
/// dimensione, noi disegniamo. Va rifatto a ogni frame (shouldRepaint → true).
///
/// Mappatura griglia → schermo: la camera è mostrata "a riempimento" (FILL_CENTER: l'immagine
/// è scalata finché copre tutto lo schermo e i bordi in eccesso sono tagliati). Usiamo la
/// stessa formula del plugin: scala = max(W/w, H/h), immagine centrata. Solo camera posteriore
/// (con quella frontale l'immagine sarebbe specchiata).
class LogicOverlayPainter extends CustomPainter {
  final LandingLogic logic;
  final FrameReport? report;

  LogicOverlayPainter(this.logic, this.report);

  static final Paint _zonePaint = Paint()..color = Colors.orange.withValues(alpha: 0.35);
  static final Paint _bodyPaint = Paint()..color = Colors.white.withValues(alpha: 0.30);
  static final Paint _feetOutPaint = Paint()..color = Colors.cyanAccent.withValues(alpha: 0.75);
  static final Paint _feetInPaint = Paint()..color = Colors.red.withValues(alpha: 0.90);

  @override
  void paint(Canvas canvas, Size size) {
    final mask = report?.carrier?.mask;
    final hasMask = mask != null && mask.isNotEmpty && mask[0].isNotEmpty;

    // Dimensioni della griglia: dalla maschera del portatore o, se manca, dalla memoria della zona.
    final gw = hasMask ? mask[0].length : logic.gridW;
    final gh = hasMask ? mask.length : logic.gridH;
    if (gw == 0 || gh == 0) return;

    final scale = math.max(size.width / gw, size.height / gh);
    final dx = (size.width - gw * scale) / 2; // negativo = bordo tagliato
    final dy = (size.height - gh * scale) / 2;

    // Disegna, riga per riga, i tratti orizzontali consecutivi di celle "dentro": un
    // rettangolo per tratto invece che uno per cella (migliaia di celle → poche decine di tratti).
    void drawRuns(int y, bool Function(int x) inside, Paint paint) {
      var x = 0;
      while (x < gw) {
        if (!inside(x)) {
          x++;
          continue;
        }
        final start = x;
        while (x < gw && inside(x)) {
          x++;
        }
        canvas.drawRect(
          Rect.fromLTWH(dx + start * scale, dy + y * scale, (x - start) * scale, scale),
          paint,
        );
      }
    }

    // 1. Zona in memoria (quella che la logica usa davvero, non quella grezza della rete).
    for (var y = 0; y < gh; y++) {
      drawRuns(y, (x) => logic.isZoneCell(x, y), _zonePaint);
    }

    // 2. Portatore: corpo bianco; se è a terra, fascia dei piedi rossa (nella zona) o azzurra.
    if (!hasMask) return;
    final feetTop = report!.feetTop;
    for (var y = 0; y < gh; y++) {
      final row = mask[y];
      if (feetTop < 0 || y < feetTop) {
        drawRuns(y, (x) => row[x] > 0, _bodyPaint);
      } else {
        drawRuns(y, (x) => row[x] > 0 && logic.isZoneCell(x, y), _feetInPaint);
        drawRuns(y, (x) => row[x] > 0 && !logic.isZoneCell(x, y), _feetOutPaint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant LogicOverlayPainter oldDelegate) => true;
}

// ============================================================================================
// Interfaccia
// ============================================================================================

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
      home: const DetectionPage(),
    );
  }
}

/// Schermata principale: camera + rete neurale + logica del fallo.
class DetectionPage extends StatefulWidget {
  const DetectionPage({super.key});

  @override
  State<DetectionPage> createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  final YOLOViewController _controller = YOLOViewController();
  final LandingLogic _logic = LandingLogic();

  // --- Stato mostrato a schermo ---
  FrameReport? _report; // esito dell'ultimo frame
  String _perfText = '';
  String? _modelError;

  // --- Misura del ritmo reale di analisi ---
  DateTime? _lastResultAt;
  double _measuredHz = 0; // media mobile esponenziale delle analisi al secondo

  // --- Frequenza variabile ---
  int _currentHz = kIdleHz;
  DateTime _lastCarrierSeen = DateTime(2000);

  // --- Audio ---
  final AudioPlayer _player = AudioPlayer();
  int _autoAlarms = 0; // allarmi decisi dalla logica
  int _manualAlarms = 0; // pressioni del pulsante (per segnare i landing nei test)

  // --- Overlay ---
  // false = si vede solo il nostro overlay (cosa usa la logica); true = anche box e maschere
  // grezzi disegnati dal plugin (cosa vede la rete). Si cambia con l'icona a forma di occhio.
  bool _showRaw = false;

  @override
  void initState() {
    super.initState();
    // Tiene lo schermo acceso: senza, Android lo spegne, l'app va in pausa e la camera si chiude.
    WakelockPlus.enable();
    // Il controller ricorda l'impostazione anche se la vista nativa non è ancora pronta.
    _controller.setShowOverlays(_showRaw);
  }

  void _toggleRaw() {
    setState(() => _showRaw = !_showRaw);
    _controller.setShowOverlays(_showRaw);
  }

  /// Pulsante "Reset zona": da premere nei test con le foto quando si cambia foto.
  void _onResetZone() {
    debugPrint('*** RESET ZONA ***');
    setState(() => _logic.resetZone());
  }

  /// Chiamata da YOLOView ogni volta che la rete ha analizzato un frame.
  void _onResult(List<YOLOResult> results) {
    _measureRate();

    final report = _logic.process(results);

    debugPrint('frame [${_measuredHz.toStringAsFixed(1)}/s]: '
        '${results.isEmpty ? '(niente)' : results.map(_describe).join(' | ')}');
    debugPrint('logica: ${report.describe()}');

    if (report.alarm) {
      _autoAlarms++;
      _playAlarm();
    }

    _updateFrequency(report.carrier != null);

    if (!mounted) return;
    setState(() => _report = report);
  }

  /// Ritmo reale di analisi: 1 / (tempo dall'ultimo risultato), filtrato con una media mobile.
  void _measureRate() {
    final now = DateTime.now();
    final last = _lastResultAt;
    _lastResultAt = now;
    if (last == null) return;
    final dtMs = now.difference(last).inMilliseconds;
    if (dtMs <= 0) return;
    _measuredHz = 0.8 * _measuredHz + 0.2 * (1000 / dtMs);
  }

  /// Frequenza alta se c'è un portatore (o c'era fino a kActiveHold fa), bassa altrimenti.
  void _updateFrequency(bool carrierVisible) {
    final now = DateTime.now();
    if (carrierVisible) _lastCarrierSeen = now;
    final wantedHz = now.difference(_lastCarrierSeen) < kActiveHold ? kActiveHz : kIdleHz;
    if (wantedHz != _currentHz) {
      _currentHz = wantedHz;
      debugPrint('Frequenza di analisi → $wantedHz Hz');
      _controller.setStreamingConfig(streamingConfigFor(wantedHz));
    }
  }

  /// Descrizione testuale di un singolo rilevamento.
  String _describe(YOLOResult r) {
    final conf = r.confidence.toStringAsFixed(2);
    final mask = r.mask;
    if (mask == null || mask.isEmpty) return '${r.className} $conf (senza maschera)';
    // fold è come std::accumulate: somma, riga per riga, le celle > 0.
    final pixels = mask.fold<int>(0, (sum, row) => sum + row.where((v) => v > 0).length);
    return '${r.className} $conf ($pixels px)';
  }

  void _onPerformance(YOLOPerformanceMetrics m) {
    debugPrint('tempi: rete ${m.inferenceMs.toStringAsFixed(0)} ms, '
        'totale ${m.processingTimeMs.toStringAsFixed(0)} ms');
    if (!mounted) return;
    setState(() {
      _perfText = 'rete ${m.inferenceMs.toStringAsFixed(0)} ms, '
          'totale ${m.processingTimeMs.toStringAsFixed(0)} ms';
    });
  }

  void _onModelError(Object error, String modelPath, YOLOTask? task) {
    debugPrint('Errore modello: $error');
    if (!mounted) return;
    setState(() => _modelError = 'Modello non caricato ($modelPath):\n$error');
  }

  /// Pulsante: segna nel log che QUI c'è un landing (per confrontarlo con l'allarme automatico).
  void _onManualPressed() {
    debugPrint('*** ALLARME MANUALE (landing segnato dall\'utente) ***');
    setState(() => _manualAlarms++);
    _playAlarm();
  }

  Future<void> _playAlarm() async {
    await _player.stop();
    await _player.play(AssetSource('sounds/alarm.wav'));
    if (mounted) setState(() {}); // aggiorna i contatori a schermo
  }

  @override
  void dispose() {
    WakelockPlus.disable();
    _player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    const whiteText = TextStyle(color: Colors.white, fontSize: 14);
    final report = _report;

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Landing Detector'),
        // "actions": pulsanti a destra nella barra in alto.
        actions: [
          IconButton(
            tooltip: _showRaw ? 'Nascondi rilevamenti grezzi' : 'Mostra rilevamenti grezzi',
            icon: Icon(_showRaw ? Icons.visibility : Icons.visibility_off),
            onPressed: _toggleRaw,
          ),
          IconButton(
            tooltip: 'Reset zona',
            icon: const Icon(Icons.layers_clear),
            onPressed: _onResetZone,
          ),
        ],
      ),
      body: Stack(
        children: [
          // Livello 1: camera + rete (se _showRaw, il plugin disegna anche box e maschere grezzi).
          YOLOView(
            modelPath: kModelPath,
            task: YOLOTask.segment,
            controller: _controller,
            confidenceThreshold: 0.25,
            streamingConfig: streamingConfigFor(kIdleHz),
            onResult: _onResult,
            onPerformanceMetrics: _onPerformance,
            onModelError: _onModelError,
          ),

          // Livello 2: il nostro overlay, grande quanto la camera (Positioned.fill).
          // IgnorePointer: i tocchi passano attraverso e arrivano alla camera (tap per mettere a fuoco).
          Positioned.fill(
            child: IgnorePointer(
              child: CustomPaint(painter: LogicOverlayPainter(_logic, report)),
            ),
          ),

          // Livello 3: pannello con lo stato della logica.
          Positioned(
            left: 8,
            right: 8,
            top: 8,
            child: Container(
              padding: const EdgeInsets.all(8),
              // Sfondo rosso se il frame corrente è un landing, altrimenti nero semitrasparente.
              color: (report?.landing ?? false) ? Colors.red.withValues(alpha: 0.7) : Colors.black54,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (_modelError != null)
                    Text(_modelError!, style: const TextStyle(color: Colors.redAccent))
                  else ...[
                    Text('$_perfText | analisi reali ${_measuredHz.toStringAsFixed(1)}/s', style: whiteText),
                    if (report != null) ...[
                      Text(
                        report.carrier == null
                            ? 'portatore: nessuno'
                            : 'portatore: ${report.carrier!.className} '
                                '${report.carrier!.confidence.toStringAsFixed(2)}',
                        style: whiteText,
                      ),
                      Text(
                          'piedi nella zona: ${report.feetInZone}/${report.feetCells} celle '
                          '(${report.feetPercent}%)',
                          style: whiteText),
                      Text('zona: ${report.zoneCells} celle ${report.zoneSeen ? '(vista)' : '(memoria)'}',
                          style: whiteText),
                      Text('punteggio: ${report.score}   (allarme a $kScoreThreshold)',
                          style: whiteText.copyWith(fontSize: 20, fontWeight: FontWeight.bold)),
                    ],
                    const Text('arancio = zona in memoria · bianco = portatore · '
                        'piedi: rosso = nella zona, azzurro = fuori',
                        style: TextStyle(color: Colors.white70, fontSize: 11)),
                  ],
                ],
              ),
            ),
          ),

          // Livello 4: pulsante per segnare manualmente un landing e contatori.
          Positioned(
            left: 0,
            right: 0,
            bottom: 24,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                FilledButton.icon(
                  onPressed: _onManualPressed,
                  icon: const Icon(Icons.campaign, size: 32),
                  label: const Text('LANDING!', style: TextStyle(fontSize: 24)),
                  style: FilledButton.styleFrom(backgroundColor: Colors.red),
                ),
                const SizedBox(height: 8),
                Text('Allarmi automatici: $_autoAlarms   manuali: $_manualAlarms', style: whiteText),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
