// Passo 4b.1: zona proibita fissa, calibrata con la rete e confermata dall'utente.
//
// Zona proibita (macchina a stati, vedi ZoneState):
//   calibrazione (3 s) → proposta → [utente: Conferma] → attiva
//        ↺ zona non trovata: nuovo tentativo automatico
//        ↘ dopo kMaxCalibAttempts tentativi (o kMaxCalibTime): ANALISI FERMA (camera spenta,
//          per non scaldare il telefono se l'app è partita per errore, es. in tasca) → [Riprova]
//   In qualsiasi momento: [Disegna] → disegno a mano (pennello) → [Conferma] → attiva.
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
import 'dart:io'; // File, Directory, IOSink: scrivere il log su file
import 'dart:math' as math; // min, max (come <algorithm> in C++)
import 'dart:typed_data'; // Uint8List/Uint16List: array contigui, come std::vector<uint8_t>

import 'package:audioplayers/audioplayers.dart';
import 'package:material_ui/material_ui.dart';
import 'package:flutter/services.dart'; // SystemChrome, DeviceOrientation: orientamento dello schermo
import 'package:path_provider/path_provider.dart'; // cartelle dell'app sul telefono
import 'package:shared_preferences/shared_preferences.dart'; // impostazioni salvate sul telefono
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
const String kModelName = 'landing_n640_v6';

/// imgsz con cui il modello in uso è stato esportato (il numero nel nome del file).
const int kModelSize = 640;

/// Percorso del modello dentro gli asset dell'app (cartella dichiarata in pubspec.yaml).
/// '$kModelName' inserisce il valore nella stringa (come std::format in C++).
const String kModelPath = 'assets/models/$kModelName.tflite';

/// Risoluzione dei frame passati alla rete: 4:3 come l'anteprima, lato lungo = imgsz.
const Size kAnalysisResolution = Size(kModelSize * 1.0, kModelSize * 0.75);

/// Frequenza di analisi variabile (inferenze al secondo):
/// - "riposo": nessun portatore di palla in vista → si analizza poco, il telefono si scalda meno;
/// - "azione": c'è un portatore → si analizza il più spesso possibile, cioè SENZA limite
///   (inferenceFrequency = null).
///   Attenzione: il plugin conta l'intervallo 1/frequenza dalla FINE dell'analisi precedente,
///   non dall'inizio. Con 30 Hz aspettava 33 ms dopo ogni analisi e poi un frame nuovo della
///   camera (che ne produce uno ogni 33 ms): ~50 ms persi a ogni frame, ~8 analisi/s invece di
///   ~14. Per questo in azione il limite si toglie del tutto.
const int kIdleHz = 2;

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

/// Raggi del pennello per disegnare la zona (piccolo, medio, grande), in pixel dello schermo.
/// Prima era uno solo, 22 px: troppo grande, si aggiungeva o cancellava sempre troppo.
const List<double> kBrushSizesPx = [6, 12, 20];
const List<String> kBrushSizeNames = ['piccolo', 'medio', 'grande'];

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
/// kFeetMinCells (contro il rumore delle maschere) E almeno kFeetMinFraction del totale dei piedi.
/// La frazione è disattivata (0): a 0.3 scartava landing veri (piedi al 27–37%) e non distingueva
/// i falsi (12–39%). Se un giocatore vicino, fermo appena DAVANTI alla zona, risultasse landing per
/// le caviglie, conviene abbassare kFeetFraction (fascia dei piedi più sottile).
const int kFeetMinCells = 3;
const double kFeetMinFraction = 0;

/// Punteggio mobile: +1 per frame con landing, −1 senza (minimo 0, massimo kScoreMax).
/// L'allarme suona quando il punteggio RAGGIUNGE kScoreThreshold; poi si "riarma" solo
/// quando il punteggio torna a 0 (così un landing lungo non fa suonare a ripetizione).
const int kScoreThreshold = 3;

/// Tetto del punteggio: senza, un landing lungo accumulerebbe tanti punti e ci vorrebbe
/// molto tempo per tornare a 0 e riarmare l'allarme.
const int kScoreMax = 5;

// ============================================================================================
// Log
// ============================================================================================

/// Livello del log in uso, scelto nella schermata iniziale e cambiabile a metà sessione (⚙):
///   ridotto  → 1 riga RIEPILOGO ogni kSummaryPeriod + gli eventi (righe con ***)
///   completo → in più 3 righe per ogni frame e i tempi del plugin ("LandingTiming")
/// È una variabile "static": una sola per tutta l'app, come una variabile globale in C++.
class LogConfig {
  static bool full = false;
}

/// Ogni quanto si scrive la riga RIEPILOGO (in entrambe le modalità).
const Duration kSummaryPeriod = Duration(seconds: 10);

/// Classi di cui il plugin costruisce e invia la maschera. Ogni maschera 160x120 costava ~20 ms
/// per frame; dopo le modifiche al plugin ~1–2 ms (speed_3), ma si chiedono comunque solo quelle
/// che servono: i portatori sempre, forbidden_area solo durante la calibrazione (a zona attiva la
/// zona è un riferimento fisso). ball non serve mai alla logica: arriva solo il box.
/// (maskClasses è un parametro aggiunto da noi al plugin, vedi packages/ultralytics_yolo.)
const List<String> kCarrierClasses = ['carrier_ground', 'carrier_jump'];
const List<String> kCalibMaskClasses = [...kCarrierClasses, 'forbidden_area']; // "..." = spread

/// [active] = modalità azione (nessun limite di frequenza), altrimenti riposo (kIdleHz);
/// [calibrating] = servono anche le maschere della zona.
YOLOStreamingConfig streamingConfigFor({required bool active, required bool calibrating}) =>
    YOLOStreamingConfig.custom(
      includeMasks: true, // servono le maschere in Dart per la logica
      inferenceFrequency: active ? null : kIdleHz, // null = nessun limite
      analysisResolution: kAnalysisResolution,
      maskClasses: calibrating ? kCalibMaskClasses : kCarrierClasses,
      timingLogs: LogConfig.full, // le righe "LandingTiming" del plugin solo nel log completo
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
    DateTime now,
  ) {
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
  editing, // l'utente disegna/corregge la zona a mano: rilevamento in pausa
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

// ============================================================================================
// Disegno della zona (Dart puro, testabile)
// ============================================================================================

/// Conversione tra griglia delle maschere e schermo. È la stessa formula del plugin: la camera è
/// mostrata "a riempimento" (FILL_CENTER), scala = max(W/w, H/h), immagine centrata; i bordi che
/// escono dallo schermo sono tagliati (dx, dy negativi).
class GridMapping {
  final double scale; // pixel dello schermo per cella
  final double dx, dy; // posizione sullo schermo dell'angolo (0, 0) della griglia

  const GridMapping(this.scale, this.dx, this.dy);

  /// "factory": un costruttore che calcola i campi prima di creare l'oggetto.
  factory GridMapping.fit(Size screen, int gw, int gh) {
    final s = math.max(screen.width / gw, screen.height / gh);
    return GridMapping(s, (screen.width - gw * s) / 2, (screen.height - gh * s) / 2);
  }

  /// Punto della griglia (in celle, anche frazionarie) → punto dello schermo.
  Offset toScreen(Offset g) => Offset(dx + g.dx * scale, dy + g.dy * scale);

  /// Punto dello schermo (es. il dito) → punto della griglia.
  Offset toGrid(Offset p) => Offset((p.dx - dx) / scale, (p.dy - dy) / scale);
}

/// Riempie il poligono chiuso [pts] (coordinate di griglia): 1 = cella con il centro dentro.
/// Metodo a "scansione di righe": per ogni riga si cercano i punti in cui i lati del poligono
/// la attraversano e si riempie tra un attraversamento e il successivo (regola pari/dispari).
Uint8List fillPolygon(List<Offset> pts, int w, int h) {
  final out = Uint8List(w * h);
  if (pts.length < 3) return out;
  for (var y = 0; y < h; y++) {
    final cy = y + 0.5; // centro della riga
    final xs = <double>[];
    for (var i = 0; i < pts.length; i++) {
      final a = pts[i];
      final b = pts[(i + 1) % pts.length]; // l'ultimo punto si collega al primo: bordo chiuso
      if ((a.dy <= cy && b.dy > cy) || (b.dy <= cy && a.dy > cy)) {
        xs.add(a.dx + (cy - a.dy) * (b.dx - a.dx) / (b.dy - a.dy));
      }
    }
    xs.sort();
    for (var k = 0; k + 1 < xs.length; k += 2) {
      final from = math.max(0, (xs[k] - 0.5).ceil());
      final to = math.min(w, (xs[k + 1] - 0.5).ceil());
      for (var x = from; x < to; x++) {
        out[y * w + x] = 1;
      }
    }
  }
  return out;
}

/// Pennello: mette a [value] (1 aggiungi, 0 cancella) le celle entro [radius] celle da [c].
void paintDisc(Uint8List cells, int w, int h, Offset c, double radius, int value) {
  final r = math.max(radius, 0.6); // almeno la cella sotto il dito
  final x0 = math.max(0, (c.dx - r).floor()), x1 = math.min(w - 1, (c.dx + r).ceil());
  final y0 = math.max(0, (c.dy - r).floor()), y1 = math.min(h - 1, (c.dy + r).ceil());
  for (var y = y0; y <= y1; y++) {
    for (var x = x0; x <= x1; x++) {
      final ddx = x + 0.5 - c.dx, ddy = y + 0.5 - c.dy;
      if (ddx * ddx + ddy * ddy <= r * r) cells[y * w + x] = value;
    }
  }
}

/// Pennello trascinato da [from] a [to]: dischi ravvicinati lungo il segmento, così anche un
/// movimento veloce del dito (punti lontani tra due eventi) lascia un tratto continuo.
void paintStroke(Uint8List cells, int w, int h, Offset from, Offset to, double radius, int value) {
  final steps = math.max(1, ((to - from).distance / math.max(radius * 0.5, 0.3)).ceil());
  for (var i = 0; i <= steps; i++) {
    paintDisc(cells, w, h, Offset.lerp(from, to, i / steps)!, radius, value);
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
  Uint8List? _edit; // zona in corso di disegno (stato editing)
  ZoneState _stateBeforeEdit = ZoneState.calibrating; // dove tornare se il disegno è annullato
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
      case ZoneState.editing:
        return _edit![y * _gridW + x] == 1;
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
      case ZoneState.editing:
        return 'zona: disegno a mano $displayZoneCells celle';
    }
  }

  /// Avvia (o riavvia) la calibrazione: la zona precedente viene scartata.
  /// Il cronometro parte dal primo frame analizzato dopo questa chiamata.
  void startCalibration() {
    _state = ZoneState.calibrating;
    _calibStart = null;
    _zone = null;
    _proposal = null;
    _edit = null;
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
  /// La usano la conferma del disegno a mano e i test.
  void setZone(Uint8List cells, int w, int h) {
    assert(cells.length == w * h);
    _zone = cells;
    _proposal = null;
    _edit = null;
    _gridW = w;
    _gridH = h;
    _state = ZoneState.active;
  }

  // --- Disegno a mano ---

  /// Inizia il disegno partendo dalla zona che si vede adesso (attiva, proposta o calibrazione in
  /// corso), oppure da una zona vuota. [fallbackW] x [fallbackH] = griglia da usare se non se ne
  /// conosce ancora una (nessuna maschera ricevuta, es. dopo "analisi ferma").
  void startEditing({required int fallbackW, required int fallbackH}) {
    if (_state == ZoneState.editing) return;
    if (_gridW == 0 || _gridH == 0) {
      _gridW = fallbackW;
      _gridH = fallbackH;
    }
    final start = Uint8List(_gridW * _gridH);
    for (var y = 0; y < _gridH; y++) {
      for (var x = 0; x < _gridW; x++) {
        if (isDisplayZoneCell(x, y)) start[y * _gridW + x] = 1;
      }
    }
    _edit = start;
    _stateBeforeEdit = _state;
    _state = ZoneState.editing;
  }

  /// "Traccia bordo": il bordo disegnato col dito ([outline], coordinate di griglia) viene chiuso e
  /// riempito, e sostituisce la zona in disegno. false se il bordo non racchiude nessuna cella.
  bool traceOutline(List<Offset> outline) {
    if (_state != ZoneState.editing) return false;
    final filled = fillPolygon(outline, _gridW, _gridH);
    if (!filled.contains(1)) return false;
    _edit = filled;
    return true;
  }

  /// Pennello "Aggiungi" ([add] = true) o "Cancella" da [from] a [to] (coordinate di griglia).
  void brush(Offset from, Offset to, double radiusCells, {required bool add}) {
    if (_state != ZoneState.editing) return;
    paintStroke(_edit!, _gridW, _gridH, from, to, radiusCells, add ? 1 : 0);
  }

  /// La zona disegnata diventa il riferimento fisso. false se è vuota (resta in disegno).
  bool confirmEditing() {
    final edit = _edit;
    if (_state != ZoneState.editing || edit == null || !edit.contains(1)) return false;
    setZone(edit, _gridW, _gridH);
    return true;
  }

  /// Annulla il disegno: si torna alla zona attiva o proposta di prima, altrimenti si calibra.
  void cancelEditing() {
    if (_state != ZoneState.editing) return;
    _edit = null;
    if (_stateBeforeEdit == ZoneState.active && _zone != null) {
      _state = ZoneState.active;
    } else if (_stateBeforeEdit == ZoneState.proposal && _proposal != null) {
      _state = ZoneState.proposal;
    } else {
      startCalibration();
    }
  }

  /// Analizza i rilevamenti di un frame.
  /// [now] si può passare nei test per simulare il tempo; [still] = telefono fermo (giroscopio);
  /// [moved] = è appena finito un movimento che ha spostato la camera (vedi kMovedAngleDeg).
  FrameReport process(List<YOLOResult> results, {DateTime? now, bool still = true, bool moved = false}) {
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
    //    A zona attiva arrivano solo le maschere dei portatori: senza portatori in vista la
    //    griglia non si vede, ma una rotazione di 90° è comunque un movimento forte per l'IMU.
    final hasZone = _state == ZoneState.active || _state == ZoneState.proposal;
    final gridChanged = grid != null && (grid.$1 != _gridW || grid.$2 != _gridH);
    if (hasZone && (moved || gridChanged)) {
      startCalibration();
    }
    // Durante il disegno i movimenti non contano (la zona si disegna su ciò che si vede ora),
    // ma un cambio di griglia sì: la zona disegnata non avrebbe più senso → si ricalibra.
    if (_state == ZoneState.editing && gridChanged) {
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

  bool _isSameGrid(List<List<double>> mask) => mask.length == _gridH && mask[0].length == _gridW;

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
// Riepilogo periodico (log raggruppato per i test lunghi)
// ============================================================================================

/// Accumula i dati dei frame e li riassume in una riga, poi si azzera.
class LogSummary {
  int _frames = 0, _ground = 0, _jump = 0, _landing = 0, _alarms = 0;
  int _timed = 0; // frame di cui si conoscono i tempi (arrivano in una callback separata)
  double _netSum = 0, _totalSum = 0, _totalMax = 0;
  double _gyroMax = 0, _accelMax = 0;

  void addFrame(FrameReport r, {required double gyro, required double accel}) {
    _frames++;
    if (r.carrier?.className == 'carrier_ground') _ground++;
    if (r.carrier?.className == 'carrier_jump') _jump++;
    if (r.landing) _landing++;
    if (r.alarm) _alarms++;
    _gyroMax = math.max(_gyroMax, gyro);
    _accelMax = math.max(_accelMax, accel);
  }

  void addTiming(double netMs, double totalMs) {
    _timed++;
    _netSum += netMs;
    _totalSum += totalMs;
    _totalMax = math.max(_totalMax, totalMs);
  }

  /// Riga di riepilogo dei frame raccolti in [window]; [uptime] = tempo dall'avvio dell'app.
  String flush({required Duration window, required Duration uptime}) {
    String f(double v, [int d = 0]) => v.toStringAsFixed(d);
    final minutes = uptime.inMinutes.toString().padLeft(2, '0');
    final seconds = (uptime.inSeconds % 60).toString().padLeft(2, '0');
    final sec = window.inMilliseconds / 1000;
    final times = _timed == 0
        ? 'tempi -'
        : 'rete ${f(_netSum / _timed)} ms, totale ${f(_totalSum / _timed)} ms (max ${f(_totalMax)})';
    final text =
        '$minutes:$seconds | $_frames frame (${f(sec > 0 ? _frames / sec : 0, 1)}/s) | $times | '
        'ground $_ground, jump $_jump, nessuno ${_frames - _ground - _jump} | '
        'landing $_landing, allarmi $_alarms | gyro max ${f(_gyroMax, 2)}, acc max ${f(_accelMax, 1)}';
    _frames = _ground = _jump = _landing = _alarms = _timed = 0;
    _netSum = _totalSum = _totalMax = _gyroMax = _accelMax = 0;
    return text;
  }
}

/// Testo dello stato termico restituito dal plugin (null = non disponibile).
/// Le app non possono leggere le temperature di CPU e GPU: si vedono quella della batteria, lo
/// stato termico di Android (da 1 in su il sistema può limitare la velocità) e il "margine"
/// (1.00 = soglia della limitazione pesante).
String describeThermal(Map<dynamic, dynamic>? info) {
  if (info == null || info.isEmpty) return 'temperatura n/d';
  const names = ['normale', 'leggero', 'moderato', 'severo', 'critico', 'emergenza', 'spegnimento'];
  final battery = info['batteryC'];
  final status = info['thermalStatus'];
  final headroom = info['thermalHeadroom'];
  return [
    if (battery is num) 'batteria ${battery.toStringAsFixed(1)} °C',
    if (status is int) 'termico $status${status >= 0 && status < names.length ? ' (${names[status]})' : ''}',
    if (headroom is num) 'margine ${headroom.toStringAsFixed(2)}',
  ].join(', ');
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
  final List<Offset> stroke; // bordo che si sta tracciando col dito (coordinate di griglia)
  final Offset? brushAt; // punto del pennello sotto il dito (coordinate di griglia), se si dipinge
  final double brushRadiusPx; // raggio del pennello sullo schermo

  LogicOverlayPainter(
    this.logic,
    this.report, {
    this.stroke = const [],
    this.brushAt,
    this.brushRadiusPx = 0,
  });

  static final Paint _zoneActivePaint = Paint()..color = Colors.orange.withValues(alpha: 0.35);
  static final Paint _zoneCalibPaint = Paint()..color = Colors.yellow.withValues(alpha: 0.25);
  static final Paint _zoneProposalPaint = Paint()..color = Colors.yellow.withValues(alpha: 0.50);
  static final Paint _zoneEditPaint = Paint()..color = Colors.lightGreenAccent.withValues(alpha: 0.45);
  static final Paint _strokePaint = Paint()
    ..color = Colors.white
    ..style = PaintingStyle.stroke
    ..strokeWidth = 3;
  static final Paint _bodyPaint = Paint()..color = Colors.white.withValues(alpha: 0.30);
  static final Paint _feetOutPaint = Paint()..color = Colors.cyanAccent.withValues(alpha: 0.75);
  static final Paint _feetInPaint = Paint()..color = Colors.red.withValues(alpha: 0.90);

  @override
  void paint(Canvas canvas, Size size) {
    final editing = logic.zoneState == ZoneState.editing;
    final mask = editing ? null : report?.carrier?.mask; // durante il disegno niente portatore
    final hasMask = mask != null && mask.isNotEmpty && mask[0].isNotEmpty;

    // Dimensioni della griglia: dalla maschera del portatore o, se manca, da quella della zona.
    final gw = hasMask ? mask[0].length : logic.gridW;
    final gh = hasMask ? mask.length : logic.gridH;
    if (gw == 0 || gh == 0) return;

    final map = GridMapping.fit(size, gw, gh);
    final scale = map.scale;
    final dx = map.dx; // negativo = bordo tagliato
    final dy = map.dy;

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
        canvas.drawRect(Rect.fromLTWH(dx + start * scale, dy + y * scale, (x - start) * scale, scale), paint);
      }
    }

    // 1. Zona: arancio se attiva (quella che la logica usa davvero), gialla in calibrazione
    //    (maggioranza dei voti finora) e più intensa quando è proposta.
    final zonePaint = switch (logic.zoneState) {
      ZoneState.active => _zoneActivePaint,
      ZoneState.proposal => _zoneProposalPaint,
      ZoneState.editing => _zoneEditPaint,
      _ => _zoneCalibPaint, // "_" = tutti gli altri casi, come default: in uno switch C++
    };
    for (var y = 0; y < gh; y++) {
      drawRuns(y, (x) => logic.isDisplayZoneCell(x, y), zonePaint);
    }

    // Bordo in corso di tracciamento: una linea bianca che segue il dito.
    if (stroke.length > 1) {
      final path = Path()..moveTo(map.toScreen(stroke.first).dx, map.toScreen(stroke.first).dy);
      for (final p in stroke.skip(1)) {
        final s = map.toScreen(p);
        path.lineTo(s.dx, s.dy);
      }
      canvas.drawPath(path, _strokePaint);
    }

    // Pennello: cerchio sotto il dito, per vedere quanto prende.
    if (brushAt case final b?) {
      canvas.drawCircle(map.toScreen(b), brushRadiusPx, _strokePaint);
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

// ============================================================================================
// Impostazioni (schermata iniziale) e log su file
// ============================================================================================

enum AppMode { match, test } // Partita / Test

enum ClipQuality { low, high } // qualità delle micro-clip (le clip arriveranno con un passo futuro)

enum OrientationChoice { landscape, portrait, auto }

/// Le scelte della schermata iniziale, salvate sul telefono (SharedPreferences = un piccolo
/// archivio chiave → valore dell'app, che resta anche chiudendo l'app).
class AppSettings {
  final AppMode mode;
  final ClipQuality clipQuality;
  final bool fullLog; // solo in Test
  final OrientationChoice orientation;

  const AppSettings({
    this.mode = AppMode.test,
    this.clipQuality = ClipQuality.low,
    this.fullLog = false,
    this.orientation = OrientationChoice.landscape,
  });

  bool get isTest => mode == AppMode.test;

  /// Copia con alcuni campi cambiati (l'oggetto è immutabile, come una struct const).
  AppSettings copyWith({
    AppMode? mode,
    ClipQuality? clipQuality,
    bool? fullLog,
    OrientationChoice? orientation,
  }) => AppSettings(
    mode: mode ?? this.mode,
    clipQuality: clipQuality ?? this.clipQuality,
    fullLog: fullLog ?? this.fullLog,
    orientation: orientation ?? this.orientation,
  );

  /// Orientamenti permessi durante la sessione (vuoto = tutti, cioè "automatico").
  List<DeviceOrientation> get deviceOrientations => switch (orientation) {
    OrientationChoice.landscape => [DeviceOrientation.landscapeLeft, DeviceOrientation.landscapeRight],
    OrientationChoice.portrait => [DeviceOrientation.portraitUp],
    OrientationChoice.auto => [],
  };

  String describe() =>
      'modalità ${isTest ? 'Test' : 'Partita'} | orientamento ${orientation.name} | '
      'clip ${clipQuality.name} | log ${!isTest
          ? 'nessuno'
          : fullLog
          ? 'completo'
          : 'ridotto'}';

  /// Gli enum si salvano con il loro nome ("test", "landscape"…); un valore sconosciuto o mancante
  /// (primo avvio) torna al predefinito.
  static Future<AppSettings> load() async {
    final p = await SharedPreferences.getInstance();
    T pick<T extends Enum>(List<T> values, String key, T fallback) =>
        values.asNameMap()[p.getString(key)] ?? fallback;
    const d = AppSettings();
    return AppSettings(
      mode: pick(AppMode.values, 'mode', d.mode),
      clipQuality: pick(ClipQuality.values, 'clipQuality', d.clipQuality),
      fullLog: p.getBool('fullLog') ?? d.fullLog,
      orientation: pick(OrientationChoice.values, 'orientation', d.orientation),
    );
  }

  Future<void> save() async {
    final p = await SharedPreferences.getInstance();
    await p.setString('mode', mode.name);
    await p.setString('clipQuality', clipQuality.name);
    await p.setBool('fullLog', fullLog);
    await p.setString('orientation', orientation.name);
  }
}

/// Log su file (modalità Test): una sessione = un file, con l'ora su ogni riga.
/// Cartella: /sdcard/Android/data/com.example.landing_detector/files/logs/ (dal PC:
/// `adb pull /sdcard/Android/data/com.example.landing_detector/files/logs data_logs/phone`).
/// Ci finisce tutto ciò che l'app scrive con debugPrint, senza i messaggi di Android.
class LogFile {
  static IOSink? _sink;
  static String? path;
  static bool _hooked = false;

  static String _two(int v) => v.toString().padLeft(2, '0');

  /// "Aggancia" debugPrint una volta sola: ogni riga va anche nel file, se ce n'è uno aperto.
  /// debugPrint è una variabile che contiene una funzione: si può sostituire, come un
  /// puntatore a funzione in C++.
  static void _hook() {
    if (_hooked) return;
    _hooked = true;
    final original = debugPrint;
    debugPrint = (String? message, {int? wrapWidth}) {
      original(message, wrapWidth: wrapWidth);
      final t = DateTime.now();
      _sink?.writeln('${_two(t.hour)}:${_two(t.minute)}:${_two(t.second)} ${message ?? ''}');
    };
  }

  /// Apre il file di una nuova sessione. Se non ci riesce (es. nei test, dove le cartelle del
  /// telefono non esistono) lo scrive nel log e prosegue senza file.
  static Future<void> open(AppSettings settings) async {
    try {
      final base = await getExternalStorageDirectory() ?? await getApplicationDocumentsDirectory();
      final dir = Directory('${base.path}/logs');
      await dir.create(recursive: true);
      final now = DateTime.now();
      final name =
          'session_${now.year}${_two(now.month)}${_two(now.day)}_'
          '${_two(now.hour)}${_two(now.minute)}${_two(now.second)}.txt';
      final file = File('${dir.path}/$name');
      _hook();
      _sink = file.openWrite();
      path = file.path;
      debugPrint('SESSIONE $name | modello $kModelName | ${settings.describe()}');
    } catch (e) {
      debugPrint('Log su file non disponibile: $e');
    }
  }

  /// Svuota il buffer sul disco (ogni riepilogo e quando l'app va in background).
  static Future<void> flush() async => _sink?.flush();

  /// Chiude il file della sessione (ritorno alla schermata iniziale).
  static Future<void> close() async {
    final sink = _sink;
    _sink = null;
    path = null;
    await sink?.flush();
    await sink?.close();
  }
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized(); // serve prima di usare i plugin
  runApp(const LandingApp());
}

/// Strumenti del disegno a mano della zona.
enum BrushTool { trace, add, erase }

/// Widget radice dell'app: imposta tema e schermata iniziale. Non ha stato.
class LandingApp extends StatelessWidget {
  /// false nei test: lì non esiste il giroscopio nativo.
  final bool useSensors;

  const LandingApp({super.key, this.useSensors = true});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Landing Detector',
      // Tema scuro: si legge meglio sopra la camera e in palestra.
      theme: ThemeData(
        colorScheme: .fromSeed(seedColor: Colors.orange, brightness: Brightness.dark),
      ),
      home: StartPage(useSensors: useSensors),
    );
  }
}

// ============================================================================================
// Schermata iniziale
// ============================================================================================

/// Si apre a ogni avvio con le ultime scelte già impostate: un tocco su "Avvia" e si parte.
class StartPage extends StatefulWidget {
  final bool useSensors;

  const StartPage({super.key, this.useSensors = true});

  @override
  State<StartPage> createState() => _StartPageState();
}

class _StartPageState extends State<StartPage> {
  AppSettings? _settings; // null finché non sono state lette dal telefono

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([]); // la schermata iniziale gira liberamente
    // "then": quando la lettura (asincrona) finisce, si aggiorna la schermata.
    AppSettings.load().then((s) {
      if (mounted) setState(() => _settings = s);
    });
  }

  /// Applica una modifica alle impostazioni ATTUALI (non a una copia vecchia: due tocchi
  /// ravvicinati, prima che lo schermo si ridisegni, non si annullano a vicenda).
  void _change(AppSettings Function(AppSettings) update) => setState(() => _settings = update(_settings!));

  /// "Avvia": salva le scelte, apre il log (in Test), blocca l'orientamento e apre la camera.
  /// Navigator è una pila di schermate: push ne mette una sopra, il "indietro" la toglie.
  /// "await" su push aspetta che la schermata della camera venga chiusa.
  Future<void> _start() async {
    final s = _settings!;
    await s.save();
    LogConfig.full = s.isTest && s.fullLog;
    if (s.isTest) await LogFile.open(s);
    // Senza "await": l'orientamento si applica da solo, non serve aspettare la risposta di Android.
    // "unawaited" dice esplicitamente che non aspettiamo questo Future (una promessa di risultato).
    unawaited(SystemChrome.setPreferredOrientations(s.deviceOrientations));
    if (!mounted) return;
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => DetectionPage(settings: s, useSensors: widget.useSensors),
      ),
    );
    // Ritorno alla schermata iniziale: fine della sessione.
    debugPrint('*** FINE SESSIONE ***');
    await LogFile.close();
    unawaited(SystemChrome.setPreferredOrientations([]));
  }

  @override
  Widget build(BuildContext context) {
    final s = _settings;
    if (s == null) return const Scaffold(body: Center(child: CircularProgressIndicator()));
    final theme = Theme.of(context);

    final title = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text('Tchoukball Landing', style: theme.textTheme.headlineMedium),
        Text('modello $kModelName', style: theme.textTheme.bodySmall),
      ],
    );
    final startButton = FilledButton.icon(
      onPressed: _start,
      icon: const Icon(Icons.play_arrow, size: 32),
      label: const Text('AVVIA', style: TextStyle(fontSize: 22)),
      style: FilledButton.styleFrom(minimumSize: const Size(220, 64)),
    );

    // Le opzioni: una riga per scelta. SegmentedButton = pulsanti affiancati, uno selezionato.
    Widget option<T>(
      String label,
      String? help,
      T value,
      List<(T, String)> choices,
      void Function(T) onChanged, {
      Set<T> disabled = const {},
    }) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label, style: theme.textTheme.titleMedium),
            const SizedBox(height: 4),
            SegmentedButton<T>(
              segments: [
                for (final (v, text) in choices)
                  ButtonSegment(value: v, label: Text(text), enabled: !disabled.contains(v)),
              ],
              selected: {value},
              onSelectionChanged: (sel) => onChanged(sel.first),
              showSelectedIcon: false,
            ),
            if (help != null) Text(help, style: theme.textTheme.bodySmall),
          ],
        ),
      );
    }

    final options = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        option(
          'Modalità',
          s.isTest
              ? 'Raccoglie dati: log su file, contatori, pulsante "segna landing"'
              : 'Schermo pulito, nessun log',
          s.mode,
          const [(AppMode.match, 'Partita'), (AppMode.test, 'Test')],
          (v) => _change((cur) => cur.copyWith(mode: v)),
        ),
        if (s.isTest)
          option(
            'Log',
            s.fullLog ? 'Una riga per ogni frame (file grandi)' : 'Un riepilogo ogni 10 s più gli eventi',
            s.fullLog,
            const [(false, 'Ridotto'), (true, 'Completo')],
            (v) => _change((cur) => cur.copyWith(fullLog: v)),
          ),
        option(
          'Qualità clip',
          'Le clip arriveranno con un prossimo passo',
          s.clipQuality,
          const [(ClipQuality.low, 'Bassa'), (ClipQuality.high, 'Alta')],
          (v) => _change((cur) => cur.copyWith(clipQuality: v)),
          disabled: const {ClipQuality.high},
        ),
        option('Orientamento', 'Bloccato per tutta la sessione, tranne "Automatico"', s.orientation, const [
          (OrientationChoice.landscape, 'Orizzontale'),
          (OrientationChoice.portrait, 'Verticale'),
          (OrientationChoice.auto, 'Automatico'),
        ], (v) => _change((cur) => cur.copyWith(orientation: v))),
      ],
    );

    // Stessi pezzi, disposti in base all'orientamento: in orizzontale opzioni a sinistra, titolo e
    // "Avvia" a destra; in verticale tutto in colonna. SingleChildScrollView: se lo schermo è
    // piccolo si scorre invece di tagliare il contenuto.
    return Scaffold(
      body: SafeArea(
        child: OrientationBuilder(
          builder: (context, orientation) => orientation == Orientation.landscape
              ? Row(
                  children: [
                    Expanded(
                      flex: 2,
                      child: SingleChildScrollView(padding: const EdgeInsets.all(16), child: options),
                    ),
                    Expanded(
                      child: Padding(
                        padding: const EdgeInsets.all(24),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [title, startButton],
                        ),
                      ),
                    ),
                  ],
                )
              : SingleChildScrollView(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      title,
                      const SizedBox(height: 24),
                      options,
                      const SizedBox(height: 32),
                      Center(child: startButton),
                    ],
                  ),
                ),
        ),
      ),
    );
  }
}

/// Schermata principale: camera + rete neurale + logica del fallo.
class DetectionPage extends StatefulWidget {
  final AppSettings settings;
  final bool useSensors;

  /// Solo per i test: una logica già preparata (es. con la zona proposta), per controllare la
  /// schermata in ogni stato senza camera.
  final LandingLogic? logic;

  const DetectionPage({super.key, this.settings = const AppSettings(), this.useSensors = true, this.logic});

  @override
  State<DetectionPage> createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  final YOLOViewController _controller = YOLOViewController();
  // "late": inizializzata al primo uso, quando "widget" è già disponibile.
  late final LandingLogic _logic = widget.logic ?? LandingLogic();

  // --- IMU (giroscopio + accelerometro) ---
  MotionMonitor _motion = MotionMonitor(); // sostituito con uno nuovo quando i sensori ripartono
  StreamSubscription<GyroscopeEvent>? _gyroSub;
  StreamSubscription<UserAccelerometerEvent>? _accelSub;
  AppLifecycleListener? _lifecycle; // app visibile / in background

  // --- Stato mostrato a schermo ---
  FrameReport? _report; // esito dell'ultimo frame
  String _perfText = '';
  String? _modelError;

  // --- Riepilogo periodico e temperatura ---
  LogSummary _summary = LogSummary();
  final Stopwatch _uptime = Stopwatch()..start(); // tempo dall'avvio
  final Stopwatch _summaryWindow = Stopwatch()..start();
  Timer? _summaryTimer; // callback periodica, come un timer ROS; va cancellata
  String _thermalText = '';
  Map<dynamic, dynamic>? _thermalInfo; // ultima lettura grezza (per decidere se il telefono scotta)

  // --- Schermata dal vivo ---
  bool _showDetails = false; // pannello dettagli aperto (toccando la pillola, solo in Test)
  bool _alarmFlash = false; // bordo rosso acceso (lampeggia per qualche istante dopo un allarme)
  Timer? _alarmFlashTimer;

  // --- Misura del ritmo reale di analisi ---
  DateTime? _lastResultAt;
  double _measuredHz = 0; // media mobile esponenziale delle analisi al secondo

  // --- Frequenza variabile ---
  bool _activeMode = false; // modalità azione (analisi senza limite di frequenza)
  bool _zoneMasksOn = true; // il plugin invia anche le maschere di forbidden_area (calibrazione)
  DateTime _lastCarrierSeen = DateTime(2000);

  // --- Audio ---
  final AudioPlayer _player = AudioPlayer();
  int _autoAlarms = 0; // allarmi decisi dalla logica
  int _manualAlarms = 0; // pressioni del pulsante (per segnare i landing nei test)

  // --- Overlay (icona a forma di occhio) ---
  // Il nostro overlay (zona, portatore, piedi: cosa usa la logica) si vede sempre.
  // true = in più box, confidenze e maschere grezzi disegnati dal plugin (cosa vede la rete).
  // La differenza di velocità è trascurabile: ~1-2 ms per frame (speed_3).
  bool _showRaw = false;

  // --- Disegno a mano della zona ---
  BrushTool _tool = BrushTool.trace;
  int _brushSize = 1; // indice in kBrushSizesPx (0 piccolo, 1 medio, 2 grande)
  List<Offset> _stroke = []; // bordo in corso (Traccia), coordinate di griglia
  Offset? _lastBrush; // ultimo punto del pennello (Aggiungi/Cancella), coordinate di griglia

  @override
  void initState() {
    super.initState();
    debugPrint('*** INIZIO SESSIONE: ${widget.settings.describe()} ***');
    // Tiene lo schermo acceso: senza, Android lo spegne, l'app va in pausa e la camera si chiude.
    WakelockPlus.enable();
    // Tutto schermo, come l'app fotocamera: le barre di Android si riaprono scorrendo dal bordo
    // e si richiudono da sole ("immersiveSticky").
    unawaited(SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky));
    // Il controller ricorda l'impostazione anche se la vista nativa non è ancora pronta.
    _controller.setShowOverlays(_showRaw);

    _startSensors();
    _startSummary();

    // Ci avvisa quando l'app esce dallo schermo e quando ci torna. onHide/onShow = app davvero non
    // visibile (Home, altra app, blocco schermo). La tendina delle notifiche o una finestra di
    // permesso la rendono solo "inattiva": non contano. Va rilasciato in dispose().
    _lifecycle = AppLifecycleListener(onHide: _onHide, onShow: _onShow);
  }

  /// App non più visibile: la camera la ferma già il plugin, qui si fermano sensori e riepilogo.
  void _onHide() {
    debugPrint('*** APP IN BACKGROUND: sensori fermi ***');
    LogFile.flush();
    _stopSensors();
    _stopSummary();
  }

  /// App di nuovo visibile. In background Android può non consegnare i dati dei sensori, quindi non
  /// si sa se il telefono è stato spostato: la zona si ricalibra SEMPRE (con conferma).
  void _onShow() {
    if (_logic.zoneState == ZoneState.gaveUp) {
      debugPrint('*** RITORNO DAL BACKGROUND: analisi ancora ferma ***');
      return; // resta ferma finché l'utente non preme "Riprova"
    }
    debugPrint('*** RITORNO DAL BACKGROUND: ricalibrazione ***');
    _startSensors();
    _startSummary();
    setState(() => _logic.startCalibration());
  }

  void _startSummary() {
    _summary = LogSummary(); // scarta i dati di prima della pausa
    _summaryWindow.reset();
    _summaryTimer ??= Timer.periodic(kSummaryPeriod, (_) => _printSummary());
  }

  void _stopSummary() {
    _summaryTimer?.cancel();
    _summaryTimer = null;
  }

  /// Una riga che riassume gli ultimi kSummaryPeriod. La temperatura si chiede al codice nativo:
  /// "await" aspetta la risposta senza bloccare l'app (nel frattempo gira il resto).
  Future<void> _printSummary() async {
    final text = _summary.flush(window: _summaryWindow.elapsed, uptime: _uptime.elapsed);
    _summaryWindow.reset();
    final info = await _controller.getThermalInfo();
    final thermal = describeThermal(info);
    debugPrint('RIEPILOGO $text | zona ${_logic.zoneState.name} | $thermal');
    LogFile.flush();
    if (mounted) {
      setState(() {
        _thermalText = thermal;
        _thermalInfo = info;
      });
    }
  }

  /// Giroscopio e accelerometro a ~50 campioni/s. listen() registra la callback chiamata a ogni
  /// campione (come un subscriber ROS); l'iscrizione va cancellata quando non serve più.
  /// "widget." dà accesso ai parametri del widget (qui useSensors) dall'oggetto State.
  void _startSensors() {
    _motion = MotionMonitor(); // riparte pulito: niente movimenti rimasti in sospeso da prima
    if (widget.useSensors && _gyroSub == null) {
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
    _stopSummary();
    WakelockPlus.disable(); // lo schermo può spegnersi da solo
  }

  /// Riaccende camera, sensori, riepilogo e schermo dopo "analisi ferma".
  void _wakeUp() {
    WakelockPlus.enable();
    _startSensors();
    _startSummary();
    _controller.resume();
  }

  /// Pulsante "Riprova": riaccende tutto e riparte con una calibrazione nuova.
  void _resumeAnalysis() {
    debugPrint('*** RIPRESA ANALISI (utente) ***');
    _wakeUp();
    setState(() => _logic.startCalibration());
  }

  /// Pulsanti "Disegna" / "Correggi" / "Disegna a mano": entra nel disegno a mano.
  void _onStartDrawing() {
    if (_logic.zoneState == ZoneState.gaveUp) _wakeUp(); // per disegnare serve vedere il campo
    // Griglia se non se ne conosce ancora una: frame 4:3, una cella ogni 4 pixel del frame
    // (160x120 in orizzontale, 120x160 in verticale con imgsz 640).
    final long = kAnalysisResolution.width ~/ 4, short = kAnalysisResolution.height ~/ 4;
    final portrait = MediaQuery.of(context).orientation == Orientation.portrait;
    setState(() {
      _logic.startEditing(fallbackW: portrait ? short : long, fallbackH: portrait ? long : short);
      _tool = _logic.displayZoneCells == 0 ? BrushTool.trace : BrushTool.add;
      _stroke = [];
    });
    debugPrint('*** DISEGNO ZONA: inizio (${_logic.displayZoneCells} celle di partenza) ***');
  }

  void _onConfirmDrawing() {
    if (!_logic.confirmEditing()) {
      _showMessage('La zona è vuota: traccia il bordo o usa Aggiungi');
      return;
    }
    debugPrint('*** ZONA DISEGNATA A MANO CONFERMATA: ${_logic.zoneText} ***');
    setState(() {});
  }

  void _onCancelDrawing() {
    setState(() => _logic.cancelEditing());
    debugPrint('*** DISEGNO ZONA ANNULLATO: ${_logic.zoneText} ***');
  }

  /// Messaggio breve in basso (SnackBar), che sparisce da solo.
  void _showMessage(String text) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));
  }

  // Gesti del dito sull'immagine durante il disegno. [map] converte lo schermo in griglia.
  void _onDrawStart(Offset screen, GridMapping map) {
    final g = map.toGrid(screen);
    setState(() {
      if (_tool == BrushTool.trace) {
        _stroke = [g];
      } else {
        _lastBrush = g;
        _logic.brush(g, g, kBrushSizesPx[_brushSize] / map.scale, add: _tool == BrushTool.add);
      }
    });
  }

  void _onDrawUpdate(Offset screen, GridMapping map) {
    final g = map.toGrid(screen);
    setState(() {
      if (_tool == BrushTool.trace) {
        // Un punto ogni ~mezza cella basta e tiene il bordo leggero.
        if (_stroke.isEmpty || (_stroke.last - g).distance >= 0.5) _stroke = [..._stroke, g];
      } else {
        _logic.brush(_lastBrush ?? g, g, kBrushSizesPx[_brushSize] / map.scale, add: _tool == BrushTool.add);
        _lastBrush = g;
      }
    });
  }

  void _onDrawEnd() {
    if (_tool == BrushTool.trace && _stroke.isNotEmpty) {
      if (!_logic.traceOutline(_stroke)) {
        _showMessage('Bordo troppo piccolo: disegna tutto il contorno della zona');
      }
    }
    setState(() {
      _stroke = [];
      _lastBrush = null;
    });
  }

  /// ⚙: impostazioni durante la sessione. In Test si può passare al log completo (es. si è visto
  /// un bug) e tornare al ridotto; nel file resta la riga del cambio.
  void _openSettings() {
    // showModalBottomSheet: un pannello che sale dal basso. StatefulBuilder gli dà un suo
    // setState, perché il pannello non fa parte di questa schermata.
    showModalBottomSheet<void>(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setSheetState) => SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(widget.settings.describe()),
                if (LogFile.path != null) Text('file: ${LogFile.path!.split('/').last}'),
                const SizedBox(height: 8),
                if (widget.settings.isTest)
                  SwitchListTile(
                    title: const Text('Log completo'),
                    subtitle: const Text('Una riga per ogni frame, da qui in avanti'),
                    value: LogConfig.full,
                    onChanged: (v) {
                      setSheetState(() => _setFullLog(v));
                    },
                  )
                else
                  const Text('In modalità Partita il log non viene salvato.'),
                const SizedBox(height: 12),
                // Senza barra in alto non c'è la freccia "indietro" (resta il gesto di Android).
                OutlinedButton.icon(
                  onPressed: () {
                    Navigator.of(context).pop(); // chiude il pannello
                    Navigator.of(this.context).pop(); // chiude la camera → schermata iniziale
                  },
                  icon: const Icon(Icons.stop),
                  label: const Text('Termina sessione'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  void _setFullLog(bool full) {
    if (LogConfig.full == full) return;
    LogConfig.full = full;
    debugPrint('*** LOG → ${full ? 'completo' : 'ridotto'} ***');
    // I tempi del plugin ("LandingTiming") seguono il livello del log.
    _controller.setStreamingConfig(streamingConfigFor(active: _activeMode, calibrating: _zoneMasksOn));
  }

  void _toggleRaw() {
    setState(() => _showRaw = !_showRaw);
    debugPrint('*** OCCHIO ${_showRaw ? 'ACCESO' : 'SPENTO'} ***'); // per confrontare i tempi nel log
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
          : _logic.zoneState == ZoneState.editing
          ? '(disegno in corso: si disegna su ciò che si vede ora)'
          : moved
          ? '→ ricalibrazione'
          : '(vibrazione, zona invariata)';
      debugPrint(
        '*** FINE MOVIMENTO: ruotata di ${movement.angleDeg.toStringAsFixed(1)}°, '
        'picco ${movement.peakGyro.toStringAsFixed(2)} rad/s, '
        '${movement.peakAccel.toStringAsFixed(1)} m/s² $effect ***',
      );
    }
    final stateBefore = _logic.zoneState;
    final failedBefore = _logic.failedAttempts;
    final report = _logic.process(results, now: now, still: still, moved: moved);
    if (_logic.zoneState != stateBefore || _logic.failedAttempts > failedBefore) {
      debugPrint('*** ${_logic.zoneText} ***');
    }
    if (_logic.zoneState == ZoneState.gaveUp && stateBefore != ZoneState.gaveUp) _stopAnalysis();

    _summary.addFrame(report, gyro: peaks.gyro, accel: peaks.accel);
    if (LogConfig.full) {
      _logFrame(results, report, peaks.gyro, gyroAngle, peaks.accel);
    } else if (report.alarm) {
      debugPrint('*** ALLARME AUTOMATICO: ${report.describe()}');
    }

    if (report.alarm) {
      _autoAlarms++;
      _playAlarm();
      _flashAlarm();
    }

    // In calibrazione si analizza al massimo ritmo: più frame = mediana più affidabile.
    _updateFrequency(report.carrier != null || _logic.zoneState == ZoneState.calibrating);

    if (!mounted) return;
    setState(() => _report = report);
  }

  /// Log completo di un frame (solo con LogConfig.full).
  void _logFrame(List<YOLOResult> results, FrameReport report, double gyro, double angle, double accel) {
    debugPrint(
      'frame [${_measuredHz.toStringAsFixed(1)}/s, gyro ${gyro.toStringAsFixed(2)} rad/s '
      '${angle.toStringAsFixed(1)}°/1s, acc ${accel.toStringAsFixed(1)} m/s²]: '
      '${results.isEmpty ? '(niente)' : results.map(_describe).join(' | ')}',
    );
    debugPrint('logica: ${report.describe()}');
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
    // Durante il disegno l'analisi non serve: ritmo di riposo (l'anteprima resta fluida).
    final wantActive =
        _logic.zoneState != ZoneState.editing && now.difference(_lastCarrierSeen) < kActiveHold;
    final wantZoneMasks = _logic.zoneState == ZoneState.calibrating;
    if (wantActive != _activeMode || wantZoneMasks != _zoneMasksOn) {
      if (LogConfig.full && wantActive != _activeMode) {
        debugPrint('Frequenza di analisi → ${wantActive ? 'massima (azione)' : '$kIdleHz Hz (riposo)'}');
      }
      if (LogConfig.full && wantZoneMasks != _zoneMasksOn) {
        debugPrint('Maschere di forbidden_area → ${wantZoneMasks ? 'sì (calibrazione)' : 'no'}');
      }
      _activeMode = wantActive;
      _zoneMasksOn = wantZoneMasks;
      _controller.setStreamingConfig(streamingConfigFor(active: wantActive, calibrating: wantZoneMasks));
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
    _summary.addTiming(m.inferenceMs, m.processingTimeMs);
    if (LogConfig.full) {
      debugPrint(
        'tempi: rete ${m.inferenceMs.toStringAsFixed(0)} ms, '
        'totale ${m.processingTimeMs.toStringAsFixed(0)} ms',
      );
    }
    if (!mounted) return;
    setState(() {
      _perfText =
          'rete ${m.inferenceMs.toStringAsFixed(0)} ms, '
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

  /// Bordo rosso lampeggiante per ~1.5 s dopo un allarme (6 cambi ogni 250 ms).
  void _flashAlarm() {
    _alarmFlashTimer?.cancel();
    var ticks = 0;
    setState(() => _alarmFlash = true);
    _alarmFlashTimer = Timer.periodic(const Duration(milliseconds: 250), (t) {
      ticks++;
      if (!mounted) return t.cancel();
      setState(() => _alarmFlash = ticks < 6 && !_alarmFlash);
      if (ticks >= 6) t.cancel();
    });
  }

  /// Il telefono si sta scaldando troppo: stato termico Android "moderato" o peggio, oppure
  /// margine vicino alla soglia della limitazione pesante.
  bool get _phoneHot {
    final status = _thermalInfo?['thermalStatus'];
    final headroom = _thermalInfo?['thermalHeadroom'];
    return (status is int && status >= 2) || (headroom is num && headroom >= 0.9);
  }

  Future<void> _playAlarm() async {
    await _player.stop();
    await _player.play(AssetSource('sounds/alarm.wav'));
    if (mounted) setState(() {}); // aggiorna i contatori a schermo
  }

  @override
  void dispose() {
    _alarmFlashTimer?.cancel();
    unawaited(SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge)); // barre di nuovo visibili
    _lifecycle?.dispose();
    _stopSummary();
    _stopSensors();
    WakelockPlus.disable();
    _player.dispose();
    super.dispose();
  }

  // ------------------------------------------------------------------------------------------
  // Schermata dal vivo: camera a tutto schermo, pillola di stato, comandi su un bordo.
  // Stessi pezzi in verticale e in orizzontale: cambia solo dove vengono messi.
  // ------------------------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    final report = _report;
    final editing = _logic.zoneState == ZoneState.editing;
    final landscape = MediaQuery.orientationOf(context) == Orientation.landscape;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Livello 1: camera + rete (se _showRaw, il plugin disegna anche box e maschere grezzi).
          YOLOView(
            modelPath: kModelPath,
            task: YOLOTask.segment,
            controller: _controller,
            confidenceThreshold: 0.25,
            streamingConfig: streamingConfigFor(active: false, calibrating: true),
            onResult: _onResult,
            onPerformanceMetrics: _onPerformance,
            onModelError: _onModelError,
          ),

          // Livello 2: il nostro overlay, grande quanto la camera (Positioned.fill).
          // Normalmente IgnorePointer: i tocchi passano alla camera (tap per mettere a fuoco).
          // Durante il disegno un GestureDetector riceve il dito; LayoutBuilder dà la dimensione
          // del riquadro, che serve a convertire i punti dello schermo in celle della griglia.
          Positioned.fill(
            child: LayoutBuilder(
              builder: (context, constraints) {
                final paint = CustomPaint(
                  size: constraints.biggest,
                  painter: LogicOverlayPainter(
                    _logic,
                    report,
                    stroke: _stroke,
                    brushAt: _tool == BrushTool.trace ? null : _lastBrush,
                    brushRadiusPx: kBrushSizesPx[_brushSize],
                  ),
                );
                if (!editing) return IgnorePointer(child: paint);
                final map = GridMapping.fit(constraints.biggest, _logic.gridW, _logic.gridH);
                return GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onPanStart: (d) => _onDrawStart(d.localPosition, map),
                  onPanUpdate: (d) => _onDrawUpdate(d.localPosition, map),
                  onPanEnd: (_) => _onDrawEnd(),
                  onTapDown: (d) => _onDrawStart(d.localPosition, map), // un tocco = una pennellata
                  onTapUp: (_) => _onDrawEnd(),
                  child: paint,
                );
              },
            ),
          ),

          // Livello 3: bordo rosso lampeggiante dopo un allarme.
          if (_alarmFlash)
            Positioned.fill(
              child: IgnorePointer(
                child: Container(
                  decoration: BoxDecoration(border: Border.all(color: Colors.red, width: 14)),
                ),
              ),
            ),

          // Livello 4: tutto ciò che sta sopra l'immagine, dentro l'area sicura (SafeArea: lontano
          // da notch e angoli arrotondati).
          SafeArea(
            child: Stack(
              children: [
                // Pillola di stato (e pannello dettagli) in alto a sinistra.
                Positioned(left: 12, top: 12, right: landscape ? 120 : 12, child: _statusArea(report)),

                // Messaggio breve al centro (calibrazione, analisi ferma, errore del modello).
                if (_centerMessage() case final message?)
                  Align(alignment: const Alignment(0, -0.35), child: _messageBox(message)),

                // Comandi: colonna a destra (orizzontale) o riga in basso (verticale). Durante il
                // disegno diventano gli strumenti del pennello.
                landscape
                    ? Positioned(right: 12, top: 0, bottom: 0, child: Center(child: _commands(Axis.vertical)))
                    : Positioned(
                        left: 0,
                        right: 0,
                        bottom: 16,
                        child: Center(child: _commands(Axis.horizontal)),
                      ),

                // Contatori in basso a sinistra (solo Test).
                if (widget.settings.isTest && !editing)
                  Positioned(
                    left: 12,
                    bottom: landscape ? 12 : 96,
                    child: _messageBox('allarmi $_autoAlarms · segnati $_manualAlarms', small: true),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Etichetta e colore dello stato della zona, per la pillola.
  (String, Color) _zoneLabel() => switch (_logic.zoneState) {
    ZoneState.active => ('ZONA ATTIVA', Colors.greenAccent),
    ZoneState.calibrating => ('CALIBRAZIONE', Colors.yellow),
    ZoneState.proposal => ('ZONA DA CONFERMARE', Colors.yellow),
    ZoneState.editing => ('DISEGNO ZONA', Colors.lightGreenAccent),
    ZoneState.gaveUp => ('ANALISI FERMA', Colors.grey),
  };

  /// Pillola: pallino colorato + stato della zona; in Test anche ritmo e temperatura.
  /// In Test toccandola si apre/chiude il pannello dei dettagli.
  Widget _statusArea(FrameReport? report) {
    final (label, color) = _zoneLabel(); // "record": due valori restituiti insieme
    final battery = _thermalInfo?['batteryC'];
    final parts = [
      label,
      if (widget.settings.isTest) '${_measuredHz.toStringAsFixed(0)}/s',
      if (battery is num && (widget.settings.isTest || _phoneHot)) '${battery.toStringAsFixed(0)}°C',
      if (_phoneHot) 'CALDO',
      if (report != null && !report.still && _logic.zoneState != ZoneState.editing) 'in movimento',
      if (_logic.zoneState == ZoneState.editing) ...[
        '${_logic.displayZoneCells} celle',
        switch (_tool) {
          BrushTool.trace => 'traccia il bordo',
          BrushTool.add => 'aggiungi',
          BrushTool.erase => 'cancella',
        },
      ],
    ];
    final landingNow = report?.landing ?? false;
    final pill = GestureDetector(
      onTap: widget.settings.isTest ? () => setState(() => _showDetails = !_showDetails) : null,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: landingNow ? Colors.red.withValues(alpha: 0.8) : Colors.black54,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.circle, size: 12, color: color),
            const SizedBox(width: 8),
            // Flexible: se il testo è più largo dello spazio va a capo invece di uscire dallo schermo.
            Flexible(
              child: Text(parts.join(' · '), style: const TextStyle(color: Colors.white, fontSize: 14)),
            ),
            if (widget.settings.isTest) ...[
              const SizedBox(width: 4),
              Icon(_showDetails ? Icons.expand_less : Icons.expand_more, size: 18, color: Colors.white70),
            ],
          ],
        ),
      ),
    );
    if (!(_showDetails && widget.settings.isTest)) return Align(alignment: Alignment.topLeft, child: pill);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [pill, const SizedBox(height: 6), _detailsPanel(report)],
    );
  }

  /// Pannello dei dettagli (quello che prima stava sempre in alto): solo in Test, a richiesta.
  Widget _detailsPanel(FrameReport? report) {
    const text = TextStyle(color: Colors.white, fontSize: 13);
    return Container(
      constraints: const BoxConstraints(maxWidth: 460),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(8)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text('$_perfText | analisi reali ${_measuredHz.toStringAsFixed(1)}/s', style: text),
          if (_thermalText.isNotEmpty) Text(_thermalText, style: text),
          if (report != null) ...[
            Text(
              report.carrier == null
                  ? 'portatore: nessuno'
                  : 'portatore: ${report.carrier!.className} ${report.carrier!.confidence.toStringAsFixed(2)}',
              style: text,
            ),
            Text(
              'piedi nella zona: ${report.feetInZone}/${report.feetCells} celle (${report.feetPercent}%)',
              style: text,
            ),
            Text(report.zoneText, style: text),
            Text(
              'punteggio: ${report.score}   (allarme a $kScoreThreshold)',
              style: text.copyWith(fontSize: 18, fontWeight: FontWeight.bold),
            ),
          ],
          const Text(
            'zona: gialla = calibrazione/proposta, arancio = attiva · bianco = portatore · '
            'piedi: rosso = nella zona, azzurro = fuori',
            style: TextStyle(color: Colors.white70, fontSize: 11),
          ),
        ],
      ),
    );
  }

  /// Messaggio al centro, se serve (null = nessun messaggio). "switch" con condizioni ("when").
  String? _centerMessage() {
    if (_modelError != null) return _modelError;
    final waiting = _logic.zoneText.contains('attesa');
    return switch (_logic.zoneState) {
      ZoneState.calibrating when waiting => 'Calibrazione: tieni fermo il telefono',
      ZoneState.calibrating => 'Calibrazione della zona…',
      ZoneState.gaveUp => 'Zona non trovata: ▶ riprova · ✏ disegna a mano',
      ZoneState.proposal => 'Zona proposta: ✓ conferma · ✕ ricalibra · ✏ correggi',
      _ => null,
    };
  }

  Widget _messageBox(String text, {bool small = false}) => Container(
    padding: EdgeInsets.symmetric(horizontal: small ? 10 : 16, vertical: small ? 4 : 10),
    decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(12)),
    child: Text(
      text,
      style: TextStyle(color: Colors.white, fontSize: small ? 13 : 17),
    ),
  );

  /// Pulsante rotondo semitrasparente con un'icona. [big] = scelta importante (conferma zona…).
  Widget _roundButton(
    IconData icon,
    String tooltip,
    VoidCallback onPressed, {
    Color? color,
    bool big = false,
  }) => Padding(
    padding: const EdgeInsets.all(6),
    child: IconButton.filled(
      tooltip: tooltip,
      onPressed: onPressed,
      iconSize: big ? 40 : 28,
      icon: Icon(icon),
      style: IconButton.styleFrom(backgroundColor: color ?? Colors.black54, foregroundColor: Colors.white),
    ),
  );

  /// Strumento del disegno: arancio se è quello scelto.
  Widget _toolButton(IconData icon, String tooltip, BrushTool tool) => _roundButton(
    icon,
    tooltip,
    () => setState(() => _tool = tool),
    color: _tool == tool ? Colors.orange : null,
  );

  /// Dimensione del pennello: a ogni tocco piccolo → medio → grande → piccolo. Il pallino
  /// nell'icona è grande quanto il pennello scelto.
  Widget _brushSizeButton() => Padding(
    padding: const EdgeInsets.all(6),
    child: IconButton.filled(
      tooltip: 'Pennello ${kBrushSizeNames[_brushSize]}',
      onPressed: () => setState(() => _brushSize = (_brushSize + 1) % kBrushSizesPx.length),
      icon: SizedBox(
        width: 28,
        height: 28,
        child: Center(child: Icon(Icons.circle, size: 8.0 + 9 * _brushSize)),
      ),
      style: IconButton.styleFrom(backgroundColor: Colors.black54, foregroundColor: Colors.white),
    ),
  );

  /// I comandi, in colonna o in riga ("Flex" = Row o Column a seconda di [axis]).
  /// Cambiano con lo stato della zona: proposta → Conferma/Correggi/Ricalibra; analisi ferma →
  /// Riprova/Disegna a mano; altrimenti i comandi abituali (più occhio e ⚑ in Test).
  Widget _commands(Axis axis) {
    final test = widget.settings.isTest;
    final children = switch (_logic.zoneState) {
      // Solo icone (le scritte non entravano in verticale); il significato è nel messaggio al
      // centro e nei tooltip (tenendo premuto il pulsante).
      ZoneState.proposal => [
        _roundButton(Icons.check, 'Conferma zona', _onConfirmZone, color: Colors.green, big: true),
        _roundButton(
          Icons.close,
          'Ricalibra',
          _onRecalibrate,
          color: Colors.red.withValues(alpha: 0.8),
          big: true,
        ),
        _roundButton(Icons.brush, 'Correggi', _onStartDrawing, big: true),
        _roundButton(Icons.settings, 'Impostazioni', _openSettings),
      ],
      ZoneState.editing => [
        _toolButton(Icons.gesture, 'Traccia bordo', BrushTool.trace),
        _toolButton(Icons.brush, 'Aggiungi', BrushTool.add),
        _toolButton(Icons.auto_fix_normal, 'Cancella', BrushTool.erase),
        _brushSizeButton(),
        _roundButton(Icons.check, 'Conferma disegno', _onConfirmDrawing, color: Colors.green, big: true),
        _roundButton(
          Icons.close,
          'Annulla disegno',
          _onCancelDrawing,
          color: Colors.red.withValues(alpha: 0.8),
          big: true,
        ),
      ],
      ZoneState.gaveUp => [
        _roundButton(Icons.play_arrow, 'Riprova', _resumeAnalysis, color: Colors.green, big: true),
        _roundButton(Icons.brush, 'Disegna a mano', _onStartDrawing, big: true),
        _roundButton(Icons.settings, 'Impostazioni', _openSettings),
      ],
      _ => [
        _roundButton(Icons.settings, 'Impostazioni', _openSettings),
        if (test)
          _roundButton(
            _showRaw ? Icons.visibility : Icons.visibility_off,
            _showRaw ? 'Nascondi rilevamenti grezzi' : 'Mostra rilevamenti grezzi',
            _toggleRaw,
          ),
        _roundButton(Icons.brush, 'Disegna zona', _onStartDrawing),
        _roundButton(Icons.refresh, 'Ricalibra zona', _onRecalibrate),
        if (test)
          _roundButton(
            Icons.flag,
            'Segna landing',
            _onManualPressed,
            color: Colors.red.withValues(alpha: 0.7),
          ),
      ],
    };
    return Flex(direction: axis, mainAxisSize: MainAxisSize.min, children: children);
  }
}
