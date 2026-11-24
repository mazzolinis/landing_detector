// Passo 4b.1: zona proibita fissa, calibrata con la rete e confermata dall'utente.
//
// Zona proibita (macchina a stati, vedi ZoneState):
//   calibrazione (3 s) → proposta → [utente: Conferma] → attiva
//        ↺ zona non trovata: nuovo tentativo automatico
//        ↘ dopo kMaxCalibAttempts tentativi (o kMaxCalibTime): ANALISI FERMA (camera spenta,
//          per non scaldare il telefono se l'app è partita per errore, es. in tasca) → [Riprova]
//   Durante la calibrazione si raccolgono le maschere di forbidden_area e si prende la
//   mediana cella per cella. Una volta attiva, la zona è un riferimento FISSO: la rete non
//   la modifica più (così un errore della rete non si accumula sul riferimento).
//   Il giroscopio (MotionMonitor) dice se il telefono è fermo: la calibrazione raccoglie
//   frame solo a telefono fermo; a zona attiva il rilevamento va in pausa mentre si muove e,
//   se a fine movimento la camera risulta ruotata o il movimento è stato forte (velocità
//   angolare o accelerazione alte), si ricalibra. Una vibrazione lieve lascia la zona com'è.
//
// Pipeline per ogni frame analizzato:
//   1. il plugin "ultralytics_yolo" (YOLOView, codice nativo) esegue la rete sulla GPU e ci
//      passa i rilevamenti con le maschere tramite la callback onResult;
//   2. LandingLogic.process() (logica pura, senza grafica, testabile) aggiorna la calibrazione
//      oppure, a zona attiva, decide se il frame è un landing e aggiorna il punteggio mobile;
//   3. se il punteggio raggiunge la soglia, suona l'allarme;
//   4. LogicOverlayPainter disegna sopra la camera cosa "pensa" la logica: zona (gialla in
//      calibrazione/proposta, arancio se attiva), corpo del portatore e piedi (rossi se nella zona).
//
// Formato delle maschere:
//   result.mask è una matrice (lista di righe) che copre TUTTO il frame a 1/4 della
//   risoluzione del modello (frame 4:3, imgsz 640 → 160x120 in orizzontale, 120x160 in
//   verticale). Ogni cella contiene un "logit": > 0 significa "questo pixel appartiene
//   all'oggetto", 0 = fuori. Tutte le maschere di un frame usano la stessa griglia.

import 'dart:async'; // StreamSubscription (l'iscrizione a un flusso di eventi, come un subscriber ROS)
import 'dart:collection'; // ListQueue: coda a doppia estremità, come std::deque
import 'dart:math' as math; // min, max (come <algorithm> in C++)
import 'dart:typed_data'; // Uint8List/Uint16List: array contigui, come std::vector<uint8_t>

import 'package:audioplayers/audioplayers.dart';
import 'package:material_ui/material_ui.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

// ============================================================================================
// Costanti: modello e camera
// ============================================================================================

/// Modello in uso: nome del file (senza .tflite) dentro `assets/models/`.
/// Per passare a un'altra versione (o tornare al backup) basta cambiare questa riga.
///
/// Convenzione per i nomi: `landing_<architettura><imgsz>_v<versione>`,
/// es. `landing_n640_v2` = YOLO26n-seg, imgsz 640, seconda versione del dataset/training.
/// (`landing_640` è il modello v1, YOLO26s-seg addestrato a 1280: nome vecchio, va bene così.)
///
/// ATTENZIONE: il plugin copia il modello dagli asset nella memoria dell'app solo la PRIMA
/// volta e poi riusa la copia finché il nome non cambia. Quindi un modello nuovo deve avere
/// SEMPRE un nome nuovo: non sovrascrivere mai un file esistente con un altro modello.
const String kModelName = 'landing_n640_v2';

/// imgsz con cui il modello in uso è stato esportato (il numero nel nome del file).
const int kModelSize = 640;

/// Percorso del modello dentro gli asset dell'app (cartella dichiarata in pubspec.yaml).
/// '$kModelName' inserisce il valore nella stringa (come std::format in C++).
const String kModelPath = 'assets/models/$kModelName.tflite';

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

/// Confidenza minima perché un rilevamento di forbidden_area entri nella calibrazione.
const double kZoneMinConf = 0.25;

/// Durata della calibrazione della zona (dal primo frame analizzato dopo l'avvio/ricalibrazione).
const Duration kCalibDuration = Duration(seconds: 3);

/// La calibrazione riesce solo se la rete vede la zona in almeno kCalibMinFrames frame E in
/// almeno kCalibMinSeenFraction dei frame analizzati. La frazione evita di proporre una zona
/// vista solo nei primi frame (es. camera coperta subito dopo). Altrimenti: nuovo tentativo.
const int kCalibMinFrames = 5;
const double kCalibMinSeenFraction = 0.5;

/// Dopo kMaxCalibAttempts calibrazioni di fila senza zona l'app smette di analizzare (camera
/// spenta) finché l'utente non preme "Riprova". kMaxCalibTime copre il caso in cui il telefono
/// non è mai fermo (es. in tasca mentre si cammina): lì la calibrazione resta "in attesa" e i
/// tentativi non verrebbero mai contati.
const int kMaxCalibAttempts = 5;
const Duration kMaxCalibTime = Duration(seconds: 30);

/// Telefono "fermo" = per almeno kStillTime nessun campione "in movimento". Un campione è in
/// movimento se:
///  - |ω| ≥ kStillGyroRate (movimenti rapidi, urti). 0.08 rad/s ≈ 4.6°/s: in zone_calib_3 il
///    telefono fermo in mano sta a 0.01–0.05 rad/s;
///  - oppure l'angolo di cui il telefono è ruotato nell'ultimo kStillTime supera kStillAngleDeg
///    (movimenti lenti: 0.03 rad/s non supera nessuna soglia di velocità ragionevole, ma in 1 s
///    fa 1.7°). Il tremolio della mano oscilla avanti e indietro e quasi si annulla.
const double kStillGyroRate = 0.08;
const double kStillAngleDeg = 1.0;
const Duration kStillTime = Duration(seconds: 1);

/// A fine movimento, la camera è considerata SPOSTATA (→ ricalibrazione) se:
///  - è ruotata di almeno kMovedAngleDeg rispetto all'inizio (1° ≈ 2–3 celle della griglia);
///  - oppure il movimento è stato FORTE: picco |ω| ≥ kStrongGyroRate o accelerazione lineare
///    ≥ kStrongAccel. Dopo uno scossone l'angolo integrato è poco affidabile, e il giroscopio
///    non vede le traslazioni (telefono spostato di lato senza ruotarlo): meglio ricalibrare.
///    In zone_calib_4: movimenti lenti in mano 0.1–0.6 rad/s, rotazioni 1.5–3, scossoni 4–8.
///    L'accelerometro è nuovo: 3 m/s² (~0.3 g) è una prima stima, da tarare col log ("acc").
/// Altrimenti è una vibrazione: il rilevamento riprende con la zona di prima.
const double kMovedAngleDeg = 1.0;
const double kStrongGyroRate = 1.0;
const double kStrongAccel = 3.0;

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

/// Dice se il telefono è fermo e se, dopo un movimento, la camera è rimasta ruotata.
///
/// Il giroscopio misura la velocità angolare ω (rad/s attorno agli assi x, y, z del telefono).
///  - Velocità: |ω| di ogni campione, senza integrare.
///  - Angolo su una finestra breve: somma di ω·dt sugli ultimi kStillTime (per angoli piccoli
///    la rotazione è circa il vettore ∫ω dt). Si integra solo per 1 s, o per la durata di un
///    movimento: con il bias del giroscopio calibrato di Android (< 0.005 rad/s) l'errore resta
///    sotto ~0.3°, quindi niente deriva. A differenza della gravità (accelerometro), così si
///    vede anche la rotazione attorno all'asse verticale (camera che "panoramica").
class MotionMonitor {
  DateTime? _firstSampleAt; // null = nessun campione ricevuto (niente giroscopio, o test)
  DateTime? _lastMovingAt; // ultimo campione "in movimento"
  DateTime? _prevT; // istante del campione precedente (per dt)
  double _peakGyro = 0; // |ω| massimo dall'ultima lettura (per il log)
  double _peakAccel = 0; // |a| massima dall'ultima lettura (per il log)

  // Finestra degli ultimi kStillTime: contributi ω·dt di ogni campione e loro somma.
  final ListQueue<(DateTime, double, double, double)> _window = ListQueue();
  double _wx = 0, _wy = 0, _wz = 0;

  // Episodio di movimento in corso: rotazione accumulata dall'inizio del movimento e picchi.
  bool _inEpisode = false;
  double _ex = 0, _ey = 0, _ez = 0;
  double _epPeakGyro = 0, _epPeakAccel = 0;

  static const double _degToRad = math.pi / 180;

  /// Angolo ruotato nell'ultimo kStillTime, in gradi (per il log).
  double get windowAngleDeg => _norm(_wx, _wy, _wz) / _degToRad;

  /// Nuovo campione del giroscopio, con il suo istante [t].
  void addGyro(double x, double y, double z, DateTime t) {
    _firstSampleAt ??= t;
    final rate = _norm(x, y, z);
    if (rate > _peakGyro) _peakGyro = rate;

    // Integrazione (rettangoli): dt dal campione precedente, limitato se c'è stato un buco.
    final prev = _prevT;
    _prevT = t;
    var dt = prev == null ? 0.0 : t.difference(prev).inMicroseconds / 1e6;
    dt = dt.clamp(0.0, 0.05);
    final dx = x * dt, dy = y * dt, dz = z * dt;
    _window.addLast((t, dx, dy, dz));
    _wx += dx;
    _wy += dy;
    _wz += dz;
    // Toglie dalla finestra i campioni più vecchi di kStillTime.
    while (_window.isNotEmpty && t.difference(_window.first.$1) > kStillTime) {
      final (_, ox, oy, oz) = _window.removeFirst();
      _wx -= ox;
      _wy -= oy;
      _wz -= oz;
    }

    final moving = rate >= kStillGyroRate || windowAngleDeg >= kStillAngleDeg;
    if (moving) _markMoving(t, alreadyCounted: (dx, dy, dz));
    if (_inEpisode) {
      _ex += dx;
      _ey += dy;
      _ez += dz;
      if (rate > _epPeakGyro) _epPeakGyro = rate;
    }
  }

  /// Nuovo campione dell'accelerometro "lineare" (m/s², gravità già tolta da Android).
  /// Serve solo a vedere i movimenti forti, comprese le traslazioni che il giroscopio non vede.
  void addAccel(double x, double y, double z, DateTime t) {
    _firstSampleAt ??= t;
    final a = _norm(x, y, z);
    if (a > _peakAccel) _peakAccel = a;
    if (a >= kStrongAccel) _markMoving(t);
    if (_inEpisode && a > _epPeakAccel) _epPeakAccel = a;
  }

  /// Segna un campione "in movimento"; se non c'era un episodio in corso, lo inizia partendo
  /// dall'angolo già nella finestra (l'inizio di un movimento lento, prima che superasse la
  /// soglia). [alreadyCounted] = contributo del campione corrente, sommato subito dopo.
  void _markMoving(DateTime t, {(double, double, double) alreadyCounted = (0, 0, 0)}) {
    _lastMovingAt = t;
    if (_inEpisode) return;
    _inEpisode = true;
    final (dx, dy, dz) = alreadyCounted;
    _ex = _wx - dx;
    _ey = _wy - dy;
    _ez = _wz - dz;
    _epPeakGyro = 0;
    _epPeakAccel = 0;
  }

  /// Telefono fermo da almeno kStillTime. Senza giroscopio si considera sempre fermo
  /// (meglio non bloccare l'app che bloccarla per sempre).
  bool isStill(DateTime now) {
    final first = _firstSampleAt;
    if (first == null) return true;
    return now.difference(_lastMovingAt ?? first) >= kStillTime;
  }

  /// Da chiamare a ogni frame. Se un movimento è appena finito (telefono di nuovo fermo),
  /// ne restituisce il riassunto; altrimenti null.
  /// Il risultato è un "record con nomi": come una struct anonima, si legge con r.angleDeg ecc.
  ({double angleDeg, double peakGyro, double peakAccel, bool cameraMoved})? takeFinishedMovement(
      DateTime now) {
    if (!_inEpisode || !isStill(now)) return null;
    _inEpisode = false;
    final angle = _norm(_ex, _ey, _ez) / _degToRad;
    return (
      angleDeg: angle,
      peakGyro: _epPeakGyro,
      peakAccel: _epPeakAccel,
      cameraMoved: angle >= kMovedAngleDeg || _epPeakGyro >= kStrongGyroRate || _epPeakAccel >= kStrongAccel,
    );
  }

  /// Picchi di |ω| (rad/s) e |a| (m/s²) dall'ultima chiamata (poi si azzerano): per il log.
  ({double gyro, double accel}) takePeaks() {
    final p = (gyro: _peakGyro, accel: _peakAccel);
    _peakGyro = 0;
    _peakAccel = 0;
    return p;
  }

  static double _norm(double x, double y, double z) => math.sqrt(x * x + y * y + z * z);
}

/// Stato della zona proibita usata dalla logica.
/// Un "enum" Dart è come una enum class in C++.
enum ZoneState {
  calibrating, // si raccolgono le maschere della rete
  proposal, // calibrazione finita: zona proposta, aspetta la conferma dell'utente
  active, // zona di riferimento fissa: si rilevano i landing
  gaveUp, // zona mai trovata: analisi ferma (camera spenta) finché l'utente non riprova
}

/// Esito dell'analisi di un frame: serve per il log, per lo schermo e per i test.
class FrameReport {
  final YOLOResult? carrier; // portatore scelto (il più sicuro), null se nessuno
  final int feetCells; // celle dei piedi del portatore (0 se non è carrier_ground)
  final int feetInZone; // di queste, quante sono nella zona proibita
  final int feetTop; // prima riga della fascia dei piedi (-1 se non calcolata): serve all'overlay
  final ZoneState zoneState; // stato della zona dopo questo frame
  final String zoneText; // descrizione della zona per log e schermo
  final bool zoneSeen; // la rete ha visto forbidden_area in QUESTO frame (diagnostica)
  final bool still; // telefono fermo (dal giroscopio)
  final bool landing; // questo frame conta come landing
  final int score; // punteggio mobile dopo questo frame
  final bool alarm; // l'allarme deve suonare ADESSO

  const FrameReport({
    required this.carrier,
    required this.feetCells,
    required this.feetInZone,
    required this.feetTop,
    required this.zoneState,
    required this.zoneText,
    required this.zoneSeen,
    required this.still,
    required this.landing,
    required this.score,
    required this.alarm,
  });

  /// Percentuale dei piedi nella zona (0..100).
  int get feetPercent => feetCells == 0 ? 0 : (100 * feetInZone / feetCells).round();

  /// Riga di log, es.:
  /// "carrier_ground 0.31 | piedi 6/24 (25%) nella zona | zona attiva 700 celle (rete: vista) | punteggio 2 (allarme a 3)"
  String describe() {
    final c = carrier;
    final who = c == null ? 'nessun portatore' : '${c.className} ${c.confidence.toStringAsFixed(2)}';
    final feet = c?.className == 'carrier_ground' && zoneState == ZoneState.active
        ? 'piedi $feetInZone/$feetCells ($feetPercent%) nella zona${landing ? ' → LANDING' : ''}'
        : '-';
    return '$who | $feet | $zoneText${still ? '' : ' | IN MOVIMENTO'}'
        ' | punteggio $score (allarme a $kScoreThreshold)${alarm ? ' | *** ALLARME ***' : ''}';
  }
}

/// Raccoglie le maschere di forbidden_area durante la calibrazione e ne calcola la mediana
/// cella per cella.
///
/// Per valori 0/1 la mediana è un voto a maggioranza: una cella è zona se era dentro la
/// maschera in PIÙ DELLA METÀ dei frame in cui la rete ha visto la zona. Così un giocatore che
/// passa sopra la zona per pochi frame (e ci fa un "buco") non cambia il risultato.
/// I frame in cui la rete NON vede la zona non votano: contano solo per la statistica.
/// Basta un contatore per cella (non serve salvare i frame), come un istogramma.
class ZoneCalibrator {
  int _w = 0;
  int _h = 0;
  Uint16List _votes = Uint16List(0); // per cella: in quanti frame era dentro la zona
  int framesSeen = 0; // frame analizzati durante la calibrazione
  int framesWithZone = 0; // di questi, quanti con forbidden_area sopra kZoneMinConf

  int get width => _w;
  int get height => _h;

  /// Ricomincia da zero su una griglia w x h (array piatto: cella (x, y) → indice y * w + x).
  void reset(int w, int h) {
    _w = w;
    _h = h;
    _votes = Uint16List(w * h);
    framesSeen = 0;
    framesWithZone = 0;
  }

  /// Aggiunge un frame. [zoneMasks] = maschere di forbidden_area di questo frame (anche più di una).
  void addFrame(List<List<List<double>>> zoneMasks) {
    framesSeen++;
    if (zoneMasks.isEmpty || _w == 0) return;
    // Unione delle zone del frame: una cella vota una sola volta anche se è in due maschere.
    final inside = Uint8List(_w * _h);
    for (final mask in zoneMasks) {
      if (mask.length != _h || mask[0].length != _w) continue; // griglia diversa: ignora
      for (var y = 0; y < _h; y++) {
        for (var x = 0; x < _w; x++) {
          if (mask[y][x] > 0) inside[y * _w + x] = 1;
        }
      }
    }
    for (var i = 0; i < inside.length; i++) {
      _votes[i] += inside[i];
    }
    framesWithZone++;
  }

  /// La cella ha la maggioranza dei voti finora (per mostrare la calibrazione in corso).
  bool isMajority(int x, int y) {
    if (framesWithZone == 0 || x < 0 || y < 0 || x >= _w || y >= _h) return false;
    return _votes[y * _w + x] * 2 > framesWithZone;
  }

  /// Zona finale (1 = zona, 0 = fuori), oppure null se la rete non l'ha vista abbastanza.
  Uint8List? result() {
    if (framesWithZone < kCalibMinFrames) return null;
    if (framesWithZone < kCalibMinSeenFraction * framesSeen) return null;
    final zone = Uint8List(_w * _h);
    var cells = 0;
    for (var i = 0; i < zone.length; i++) {
      if (_votes[i] * 2 > framesWithZone) {
        zone[i] = 1;
        cells++;
      }
    }
    return cells == 0 ? null : zone;
  }
}

/// Contiene lo stato che sopravvive tra un frame e l'altro: zona proibita e punteggio.
/// La decisione "landing sì/no" invece usa solo il frame corrente (analisi a frame singolo).
class LandingLogic {
  ZoneState _state = ZoneState.calibrating; // all'avvio si calibra subito
  final ZoneCalibrator _calib = ZoneCalibrator();
  DateTime? _calibStart; // istante del primo frame della calibrazione in corso
  bool _waitingStill = false; // calibrazione ferma in attesa che il telefono sia fermo
  int _failedAttempts = 0; // tentativi di calibrazione falliti (zona non trovata) di fila
  String _lastFailure = ''; // statistica dell'ultimo tentativo fallito, per log e schermo
  DateTime? _calibSequenceStart; // primo frame dell'intera calibrazione (tutti i tentativi)

  // Zona di riferimento (attiva) o proposta: 1 = zona, salvata riga per riga in un array
  // piatto (cella (x, y) → indice y * _gridW + x), come in C/C++.
  Uint8List? _zone;
  Uint8List? _proposal;
  int _gridW = 0;
  int _gridH = 0;

  int _score = 0;
  bool _armed = true; // l'allarme può suonare (si riarma quando il punteggio torna a 0)

  int get score => _score;
  ZoneState get zoneState => _state;
  int get failedAttempts => _failedAttempts;

  // Per l'overlay: dimensioni della griglia corrente.
  int get gridW => _gridW;
  int get gridH => _gridH;

  /// La cella è nella zona ATTIVA (l'unica che conta per il landing).
  bool isZoneCell(int x, int y) {
    final zone = _zone;
    if (_state != ZoneState.active || zone == null) return false;
    if (x < 0 || y < 0 || x >= _gridW || y >= _gridH) return false;
    return zone[y * _gridW + x] == 1;
  }

  /// La cella va disegnata come zona: attiva, proposta o calibrazione in corso.
  bool isDisplayZoneCell(int x, int y) {
    if (x < 0 || y < 0 || x >= _gridW || y >= _gridH) return false;
    switch (_state) {
      case ZoneState.active:
        return _zone![y * _gridW + x] == 1;
      case ZoneState.proposal:
        return _proposal![y * _gridW + x] == 1;
      case ZoneState.calibrating:
        return _calib.isMajority(x, y);
      case ZoneState.gaveUp:
        return false;
    }
  }

  /// Celle della zona mostrata (attiva, proposta o maggioranza della calibrazione in corso).
  int get displayZoneCells {
    var n = 0;
    for (var y = 0; y < _gridH; y++) {
      for (var x = 0; x < _gridW; x++) {
        if (isDisplayZoneCell(x, y)) n++;
      }
    }
    return n;
  }

  /// Descrizione della zona per log e schermo.
  String get zoneText {
    switch (_state) {
      case ZoneState.calibrating:
        final attempt = _failedAttempts == 0
            ? ''
            : ' tentativo ${_failedAttempts + 1}/$kMaxCalibAttempts (precedente: zona non trovata, $_lastFailure)';
        if (_waitingStill) return 'zona: calibrazione$attempt in attesa che il telefono sia fermo';
        if (_calib.framesSeen == 0) return 'zona: calibrazione$attempt';
        return 'zona: calibrazione$attempt (vista in ${_calib.framesWithZone}/${_calib.framesSeen} frame)';
      case ZoneState.proposal:
        return 'zona: proposta $displayZoneCells celle, da confermare '
            '(vista in ${_calib.framesWithZone}/${_calib.framesSeen} frame)';
      case ZoneState.active:
        return 'zona: attiva $displayZoneCells celle';
      case ZoneState.gaveUp:
        final why = _failedAttempts >= kMaxCalibAttempts
            ? '$_failedAttempts tentativi'
            : '${kMaxCalibTime.inSeconds} s senza telefono fermo';
        return 'zona: non trovata ($why) — ANALISI FERMA';
    }
  }

  /// Avvia (o riavvia) la calibrazione: la zona precedente viene scartata.
  /// Il cronometro parte dal primo frame analizzato dopo questa chiamata.
  void startCalibration() {
    _state = ZoneState.calibrating;
    _calibStart = null;
    _zone = null;
    _proposal = null;
    _gridW = 0; // la griglia verrà presa dal primo frame con maschere
    _gridH = 0;
    _calib.reset(0, 0); // svuota i voti della calibrazione precedente
    _failedAttempts = 0;
    _lastFailure = '';
    _calibSequenceStart = null;
  }

  /// L'utente conferma la zona proposta: diventa il riferimento fisso.
  void confirmZone() {
    if (_state != ZoneState.proposal) return;
    _zone = _proposal;
    _proposal = null;
    _state = ZoneState.active;
  }

  /// Imposta direttamente una zona attiva (celle 0/1 su una griglia w x h).
  /// Servirà al disegno manuale (passo 4b.2); per ora la usano i test.
  void setZone(Uint8List cells, int w, int h) {
    assert(cells.length == w * h);
    _zone = cells;
    _proposal = null;
    _gridW = w;
    _gridH = h;
    _state = ZoneState.active;
  }

  /// Analizza i rilevamenti di un frame.
  /// [now] si può passare nei test per simulare il tempo; [still] = telefono fermo (giroscopio);
  /// [moved] = è appena finito un movimento che ha spostato la camera (vedi kMovedAngleDeg).
  FrameReport process(List<YOLOResult> results,
      {DateTime? now, bool still = true, bool moved = false}) {
    now ??= DateTime.now(); // "??=" assegna solo se è null

    // 1. Un solo portatore per frame: quello con la confidenza più alta.
    YOLOResult? carrier;
    for (final r in results) {
      final isCarrier = r.className == 'carrier_ground' || r.className == 'carrier_jump';
      if (isCarrier && r.confidence >= kCarrierMinConf) {
        if (carrier == null || r.confidence > carrier.confidence) carrier = r;
      }
    }

    // 2. Zona proibita.
    final zoneMasks = [
      for (final r in results)
        if (r.className == 'forbidden_area' && r.confidence >= kZoneMinConf && _hasMask(r)) r.mask!,
    ];
    final grid = _gridOf(results); // griglia delle maschere di questo frame (null se nessuna)

    // Camera spostata a zona proposta/attiva → la zona non vale più → ricalibra. Due segnali:
    //  - l'IMU: un movimento è finito con la camera ruotata o è stato forte (non una vibrazione);
    //  - la griglia delle maschere è cambiata (telefono girato tra verticale e orizzontale).
    final hasZone = _state == ZoneState.active || _state == ZoneState.proposal;
    final gridChanged = grid != null && (grid.$1 != _gridW || grid.$2 != _gridH);
    if (hasZone && (moved || gridChanged)) {
      startCalibration();
    }

    if (_state == ZoneState.calibrating) {
      if (grid != null && (grid.$1 != _gridW || grid.$2 != _gridH)) {
        // Primo frame con maschere, oppure rotazione durante la calibrazione (c'era già una
        // griglia): si riparte su questa griglia; con la rotazione riparte anche il cronometro.
        final rotated = _gridW != 0;
        _gridW = grid.$1;
        _gridH = grid.$2;
        _calib.reset(_gridW, _gridH);
        if (rotated) _calibStart = now;
      }
      _calibSequenceStart ??= now;
      if (!still) {
        // Il telefono si muove: i frame non descrivono la posizione finale della camera.
        // Si buttano i voti raccolti e si riparte da zero appena il telefono è fermo.
        _waitingStill = true;
        _calib.reset(_gridW, _gridH);
        _calibStart = null;
      } else {
        _waitingStill = false;
        _calibStart ??= now;
        _calib.addFrame(zoneMasks);
        if (now.difference(_calibStart!) >= kCalibDuration) {
          final zone = _calib.result();
          if (zone == null) {
            // Zona non trovata (camera coperta, area non inquadrata, rete incerta):
            // nuovo tentativo automatico, senza chiedere nulla all'utente.
            _failedAttempts++;
            _lastFailure = 'vista in ${_calib.framesWithZone}/${_calib.framesSeen} frame';
            _calib.reset(_gridW, _gridH);
            _calibStart = now;
          } else {
            _proposal = zone;
            _state = ZoneState.proposal;
          }
        }
      }
    }
    // Troppi tentativi falliti, o troppo tempo in calibrazione: ci si arrende (l'interfaccia
    // spegne la camera). Così l'app partita per errore (in tasca) non scalda il telefono.
    if (_state == ZoneState.calibrating &&
        (_failedAttempts >= kMaxCalibAttempts ||
            now.difference(_calibSequenceStart ?? now) >= kMaxCalibTime)) {
      _state = ZoneState.gaveUp;
    }
    // A zona attiva le rilevazioni di forbidden_area non servono: la rete non tocca il riferimento.

    // 3. Piedi del portatore (solo se è a terra) dentro la zona attiva.
    //    Se il telefono si muove, la zona fissa non corrisponde più all'immagine: pausa.
    var feetCells = 0;
    var feetInZone = 0;
    var feetTop = -1;
    final carrierMask = carrier?.mask;
    if (_state == ZoneState.active &&
        still &&
        carrier != null &&
        carrier.className == 'carrier_ground' &&
        carrierMask != null &&
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
      zoneState: _state,
      zoneText: zoneText,
      zoneSeen: zoneMasks.isNotEmpty,
      still: still,
      landing: landing,
      score: _score,
      alarm: alarm,
    );
  }

  bool _hasMask(YOLOResult r) => r.mask != null && r.mask!.isNotEmpty && r.mask![0].isNotEmpty;

  /// (larghezza, altezza) della griglia delle maschere del frame: tutte le maschere di un
  /// frame hanno la stessa griglia, basta la prima.
  (int, int)? _gridOf(List<YOLOResult> results) {
    for (final r in results) {
      if (_hasMask(r)) return (r.mask![0].length, r.mask!.length);
    }
    return null;
  }

  bool _isSameGrid(List<List<double>> mask) =>
      mask.length == _gridH && mask[0].length == _gridW;

  /// Conta le celle dei "piedi" (fascia più bassa della maschera del portatore) e quante
  /// di queste sono nella zona attiva. Restituisce (celle piedi, celle nella zona, prima riga).
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

    var feet = 0;
    var inZone = 0;
    for (var y = feetTop; y <= bottom; y++) {
      for (var x = 0; x < _gridW; x++) {
        if (mask[y][x] > 0) {
          feet++;
          if (isZoneCell(x, y)) inZone++;
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

  static final Paint _zoneActivePaint = Paint()..color = Colors.orange.withValues(alpha: 0.35);
  static final Paint _zoneCalibPaint = Paint()..color = Colors.yellow.withValues(alpha: 0.25);
  static final Paint _zoneProposalPaint = Paint()..color = Colors.yellow.withValues(alpha: 0.50);
  static final Paint _bodyPaint = Paint()..color = Colors.white.withValues(alpha: 0.30);
  static final Paint _feetOutPaint = Paint()..color = Colors.cyanAccent.withValues(alpha: 0.75);
  static final Paint _feetInPaint = Paint()..color = Colors.red.withValues(alpha: 0.90);

  @override
  void paint(Canvas canvas, Size size) {
    final mask = report?.carrier?.mask;
    final hasMask = mask != null && mask.isNotEmpty && mask[0].isNotEmpty;

    // Dimensioni della griglia: dalla maschera del portatore o, se manca, da quella della zona.
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

    // 1. Zona: arancio se attiva (quella che la logica usa davvero), gialla in calibrazione
    //    (maggioranza dei voti finora) e più intensa quando è proposta.
    final zonePaint = switch (logic.zoneState) {
      ZoneState.active => _zoneActivePaint,
      ZoneState.proposal => _zoneProposalPaint,
      _ => _zoneCalibPaint, // "_" = tutti gli altri casi, come default: in uno switch C++
    };
    for (var y = 0; y < gh; y++) {
      drawRuns(y, (x) => logic.isDisplayZoneCell(x, y), zonePaint);
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
  /// false nei test: lì non esiste il giroscopio nativo.
  final bool useSensors;

  const LandingApp({super.key, this.useSensors = true});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Landing Detector',
      theme: ThemeData(colorScheme: .fromSeed(seedColor: Colors.orange)),
      home: DetectionPage(useSensors: useSensors),
    );
  }
}

/// Schermata principale: camera + rete neurale + logica del fallo.
class DetectionPage extends StatefulWidget {
  final bool useSensors;

  const DetectionPage({super.key, this.useSensors = true});

  @override
  State<DetectionPage> createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  final YOLOViewController _controller = YOLOViewController();
  final LandingLogic _logic = LandingLogic();

  // --- IMU (giroscopio + accelerometro) ---
  final MotionMonitor _motion = MotionMonitor();
  StreamSubscription<GyroscopeEvent>? _gyroSub;
  StreamSubscription<UserAccelerometerEvent>? _accelSub;

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

    _startSensors();
  }

  /// Giroscopio e accelerometro a ~50 campioni/s. listen() registra la callback chiamata a ogni
  /// campione (come un subscriber ROS); l'iscrizione va cancellata quando non serve più.
  /// "widget." dà accesso ai parametri del widget (qui useSensors) dall'oggetto State.
  void _startSensors() {
    if (widget.useSensors) {
      _gyroSub = gyroscopeEventStream(samplingPeriod: SensorInterval.gameInterval).listen(
        // e.timestamp = istante della misura (orologio di sistema), più preciso dell'arrivo
        // in Dart, dove i campioni possono arrivare a gruppi.
        (e) => _motion.addGyro(e.x, e.y, e.z, e.timestamp),
        onError: (Object e) => debugPrint('Giroscopio non disponibile: $e'),
      );
      // Accelerometro "lineare": accelerazione del telefono senza la gravità (m/s²).
      _accelSub = userAccelerometerEventStream(samplingPeriod: SensorInterval.gameInterval).listen(
        (e) => _motion.addAccel(e.x, e.y, e.z, e.timestamp),
        onError: (Object e) => debugPrint('Accelerometro non disponibile: $e'),
      );
    }
  }

  void _stopSensors() {
    _gyroSub?.cancel();
    _accelSub?.cancel();
    _gyroSub = null;
    _accelSub = null;
  }

  /// Zona mai trovata: si spegne tutto ciò che consuma (camera + rete, sensori, schermo acceso).
  /// pause() del plugin "stacca" la camera ma tiene il modello caricato: niente più frame.
  void _stopAnalysis() {
    debugPrint('*** ANALISI FERMATA: zona non trovata, camera spenta ***');
    _controller.pause();
    _stopSensors();
    WakelockPlus.disable(); // lo schermo può spegnersi da solo
  }

  /// Pulsante "Riprova": riaccende tutto e riparte con una calibrazione nuova.
  void _resumeAnalysis() {
    debugPrint('*** RIPRESA ANALISI (utente) ***');
    WakelockPlus.enable();
    _startSensors();
    _controller.resume();
    setState(() => _logic.startCalibration());
  }

  void _toggleRaw() {
    setState(() => _showRaw = !_showRaw);
    _controller.setShowOverlays(_showRaw);
  }

  /// Pulsante "Ricalibra zona": da premere quando il telefono è stato spostato
  /// (o, nei test con le foto, quando si cambia foto).
  void _onRecalibrate() {
    if (_logic.zoneState == ZoneState.gaveUp) return _resumeAnalysis(); // analisi ferma: riaccendi
    debugPrint('*** RICALIBRAZIONE ZONA (utente) ***');
    setState(() => _logic.startCalibration());
  }

  /// Pulsante "Conferma zona": la zona proposta diventa il riferimento fisso.
  void _onConfirmZone() {
    setState(() => _logic.confirmZone());
    debugPrint('*** ZONA CONFERMATA (utente): ${_logic.zoneText} ***');
  }

  /// Chiamata da YOLOView ogni volta che la rete ha analizzato un frame.
  void _onResult(List<YOLOResult> results) {
    _measureRate();

    final now = DateTime.now();
    final still = _motion.isStill(now);
    final peaks = _motion.takePeaks();
    final gyroAngle = _motion.windowAngleDeg;
    final movement = _motion.takeFinishedMovement(now);
    final moved = movement?.cameraMoved ?? false; // "?." e "??": se movement è null → false
    if (movement != null) {
      // Durante la calibrazione non c'è una zona da buttare: il movimento l'ha già fatta ripartire.
      final effect = _logic.zoneState == ZoneState.calibrating
          ? '(calibrazione in corso)'
          : moved
              ? '→ ricalibrazione'
              : '(vibrazione, zona invariata)';
      debugPrint('*** FINE MOVIMENTO: ruotata di ${movement.angleDeg.toStringAsFixed(1)}°, '
          'picco ${movement.peakGyro.toStringAsFixed(2)} rad/s, '
          '${movement.peakAccel.toStringAsFixed(1)} m/s² $effect ***');
    }
    final stateBefore = _logic.zoneState;
    final failedBefore = _logic.failedAttempts;
    final report = _logic.process(results, now: now, still: still, moved: moved);
    if (_logic.zoneState != stateBefore || _logic.failedAttempts > failedBefore) {
      debugPrint('*** ${_logic.zoneText} ***');
    }
    if (_logic.zoneState == ZoneState.gaveUp && stateBefore != ZoneState.gaveUp) _stopAnalysis();

    debugPrint('frame [${_measuredHz.toStringAsFixed(1)}/s, gyro ${peaks.gyro.toStringAsFixed(2)} rad/s '
        '${gyroAngle.toStringAsFixed(1)}°/1s, acc ${peaks.accel.toStringAsFixed(1)} m/s²]: '
        '${results.isEmpty ? '(niente)' : results.map(_describe).join(' | ')}');
    debugPrint('logica: ${report.describe()}');

    if (report.alarm) {
      _autoAlarms++;
      _playAlarm();
    }

    // In calibrazione si analizza al massimo ritmo: più frame = mediana più affidabile.
    _updateFrequency(report.carrier != null || _logic.zoneState == ZoneState.calibrating);

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
    _stopSensors();
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
            tooltip: 'Ricalibra zona',
            icon: const Icon(Icons.refresh),
            onPressed: _onRecalibrate,
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
                      if (!report.still)
                        Text('telefono in movimento: rilevamento in pausa',
                            style: whiteText.copyWith(color: Colors.yellow)),
                      Text(report.zoneText,
                          style: whiteText.copyWith(
                              color: report.zoneState == ZoneState.active ? Colors.white : Colors.yellow)),
                      Text('punteggio: ${report.score}   (allarme a $kScoreThreshold)',
                          style: whiteText.copyWith(fontSize: 20, fontWeight: FontWeight.bold)),
                    ],
                    const Text('zona: gialla = calibrazione/proposta, arancio = attiva · '
                        'bianco = portatore · piedi: rosso = nella zona, azzurro = fuori',
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
                // Analisi ferma: messaggio e pulsante per riaccendere.
                if (_logic.zoneState == ZoneState.gaveUp) ...[
                  const Text('Zona non trovata: analisi ferma',
                      style: TextStyle(color: Colors.yellow, fontSize: 18)),
                  const SizedBox(height: 4),
                  FilledButton.icon(
                    onPressed: _resumeAnalysis,
                    icon: const Icon(Icons.play_arrow),
                    label: const Text('Riprova'),
                  ),
                  const SizedBox(height: 8),
                ],
                // Pulsanti della zona: compaiono solo quando servono ("collection if").
                if (_logic.zoneState == ZoneState.proposal)
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      FilledButton.icon(
                        onPressed: _onConfirmZone,
                        icon: const Icon(Icons.check),
                        label: const Text('Conferma zona'),
                        style: FilledButton.styleFrom(backgroundColor: Colors.green),
                      ),
                      const SizedBox(width: 12),
                      FilledButton.icon(
                        onPressed: _onRecalibrate,
                        icon: const Icon(Icons.refresh),
                        label: const Text('Ricalibra'),
                      ),
                    ],
                  ),
                const SizedBox(height: 12),
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
