// Rianalisi sul PC: legge le rilevazioni salvate da tool/replay_video.py e le fa passare nella
// STESSA LandingLogic dell'app (zona, piedi, punteggio, allarmi).
//
// Uso (dalla cartella landing_detector):
//   flutter test tool/replay_test.dart --dart-define=REPLAY=data_logs/replay            (tutti i video)
//   flutter test tool/replay_test.dart --dart-define=REPLAY=data_logs/replay/uno__landing_n640_v6.json
// Opzioni (altri --dart-define=...):
//   VERBOSE=true      una riga per ogni frame analizzato (come il log completo dell'app)
//   PRECALIB=false    calibra la zona all'inizio del video come dal vivo (i primi 3 s non contano)
//
// Annotazioni: per il video "uno.mp4" si legge, nella stessa cartella del JSON, "uno_landing.txt".
// Una riga per istante ("0:05.0" o "5.0"), "#" = commento:
//   00:05.0              landing vero: deve suonare
//   ? 00:03.7            punto difficile, NON è un landing: se suona è un falso allarme "giustificato"
//
// La zona: la telecamera del video è ferma, quindi la zona si calibra prima sui primi 3 s e poi
// il video si analizza dall'inizio con la zona già attiva (come nell'app, dove la zona si
// conferma prima dell'azione). Senza questo i landing nei primi 3 s non potrebbero suonare.
//
// È scritto come un "test" solo perché `flutter test` sa eseguire codice che usa Flutter senza
// telefono. Senza REPLAY viene saltato. Altre differenze dall'app: la zona proposta è confermata
// da sola; l'IMU non c'è (telefono fermo); il tempo è quello del video.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:landing_detector/main.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

const String kReplayPath = String.fromEnvironment('REPLAY');
const bool kVerbose = bool.fromEnvironment('VERBOSE');
const bool kPreCalibrate = bool.fromEnvironment('PRECALIB', defaultValue: true);

/// Un allarme "corrisponde" a un istante annotato se arriva tra 0.5 s prima e 1.5 s dopo
/// (l'annotazione a mano non è precisa e l'allarme arriva dopo qualche frame).
const double kHitBeforeS = 0.5;
const double kHitAfterS = 1.5;

/// "12.4" oppure "0:12.4" / "1:05" → secondi.
double parseTime(String s) => s.trim().split(':').fold(0.0, (acc, p) => acc * 60 + double.parse(p));

String fmt(double s) {
  final m = s ~/ 60;
  return '${m.toString().padLeft(2, '0')}:${(s - m * 60).toStringAsFixed(1).padLeft(4, '0')}';
}

/// Lunghezze alternate di 0 e 1 (vedi rle() nello script Python) → maschera gh righe x gw colonne.
List<List<double>> decodeRle(String rle, int gw, int gh) {
  final rows = List.generate(gh, (_) => List<double>.filled(gw, 0));
  var pos = 0;
  var value = 0.0;
  for (final run in rle.split(',')) {
    final n = int.parse(run);
    if (value == 1) {
      for (var k = pos; k < pos + n; k++) {
        rows[k ~/ gw][k % gw] = 1;
      }
    }
    pos += n;
    value = 1 - value;
  }
  return rows;
}

/// Un frame del JSON → rilevamenti nel formato del plugin.
List<YOLOResult> parseFrame(Map<String, dynamic> f, int gw, int gh) => [
      for (final d in f['d'] as List)
        YOLOResult(
          classIndex: 0,
          className: d[0] as String,
          confidence: (d[1] as num).toDouble(),
          boundingBox: Rect.fromLTRB((d[2] as num).toDouble(), (d[3] as num).toDouble(),
              (d[4] as num).toDouble(), (d[5] as num).toDouble()),
          normalizedBox: Rect.fromLTRB((d[2] as num).toDouble(), (d[3] as num).toDouble(),
              (d[4] as num).toDouble(), (d[5] as num).toDouble()),
          mask: d[6] == null ? null : decodeRle(d[6] as String, gw, gh),
        ),
    ];

/// Annotazioni di un video: landing veri e punti difficili (righe con "?").
({List<double> landings, List<double> difficult})? readAnnotations(File f) {
  if (!f.existsSync()) return null;
  final landings = <double>[], difficult = <double>[];
  for (var line in f.readAsLinesSync()) {
    line = line.split('#').first.trim();
    if (line.isEmpty) continue;
    if (line.startsWith('?')) {
      difficult.add(parseTime(line.substring(1)));
    } else {
      landings.add(parseTime(line));
    }
  }
  return (landings: landings, difficult: difficult);
}

/// Risultato di un video, per la tabella finale.
typedef VideoResult = ({String name, int landings, int hits, int falseAlarms, int difficultAlarms, String note});

VideoResult replayFile(File json, void Function(String) log) {
  final data = jsonDecode(json.readAsStringSync()) as Map<String, dynamic>;
  final grid = (data['grid'] as List).cast<int>();
  final gw = grid[0], gh = grid[1];
  final frames = (data['frames'] as List).cast<Map<String, dynamic>>();
  final videoName = File(data['video'] as String).uri.pathSegments.last;
  final stem = videoName.contains('.') ? videoName.substring(0, videoName.lastIndexOf('.')) : videoName;
  final t0 = DateTime(2026, 1, 1, 12);
  DateTime at(double t) => t0.add(Duration(microseconds: (t * 1e6).round()));

  log('');
  log('=== $videoName (${File(data['model'] as String).uri.pathSegments.last}) | griglia ${gw}x$gh | '
      'video ${data['frame_size']} a ${(data['fps'] as num).toStringAsFixed(0)} fps | '
      '${frames.length} frame, un\'analisi ogni ${(data['cycle_ms'] as num).toStringAsFixed(0)} ms');

  final logic = LandingLogic();
  var note = '';
  if (kPreCalibrate) {
    // 1° passaggio: solo per calibrare la zona (la camera del video è ferma).
    final calib = LandingLogic();
    for (final f in frames) {
      calib.process(parseFrame(f, gw, gh), now: at((f['t'] as num).toDouble()));
      if (calib.zoneState != ZoneState.calibrating) break;
    }
    if (calib.zoneState == ZoneState.proposal) {
      calib.confirmZone();
      final cells = Uint8List(calib.gridW * calib.gridH);
      for (var y = 0; y < calib.gridH; y++) {
        for (var x = 0; x < calib.gridW; x++) {
          if (calib.isZoneCell(x, y)) cells[y * calib.gridW + x] = 1;
        }
      }
      logic.setZone(cells, calib.gridW, calib.gridH);
      log('zona calibrata in anticipo sui primi ${kCalibDuration.inSeconds} s: ${logic.zoneText}; '
          'analisi dall\'inizio');
    } else {
      note = 'zona non trovata';
      log('*** zona NON trovata nei primi frame (${calib.zoneText}): si calibra come dal vivo ***');
    }
  }

  // 2° passaggio: il video dall'inizio, frame per frame, come nell'app.
  final summary = LogSummary();
  final alarms = <double>[];
  var windowStart = 0.0;
  var lastT = 0.0;
  for (final f in frames) {
    final t = (f['t'] as num).toDouble();
    lastT = t;
    final before = logic.zoneState;
    final report = logic.process(parseFrame(f, gw, gh), now: at(t));
    summary.addFrame(report, gyro: 0, accel: 0);
    if (kVerbose) log('${fmt(t)} ${report.describe()}');
    if (logic.zoneState != before) log('${fmt(t)} *** ${logic.zoneText} ***');
    if (logic.zoneState == ZoneState.proposal) {
      logic.confirmZone();
      log('${fmt(t)} *** zona confermata automaticamente: ${logic.zoneText} ***');
    } else if (logic.zoneState == ZoneState.gaveUp) {
      logic.startCalibration();
      log('${fmt(t)} *** analisi ferma: riparte da sola (nell\'app servirebbe "Riprova") ***');
    }
    if (report.alarm) {
      alarms.add(t);
      log('${fmt(t)} *** ALLARME: ${report.describe()}');
    }
    if (t - windowStart >= kSummaryPeriod.inMilliseconds / 1000) {
      final window = Duration(microseconds: ((t - windowStart) * 1e6).round());
      log('RIEPILOGO ${summary.flush(window: window, uptime: Duration(seconds: t.round()))}');
      windowStart = t;
    }
  }
  if (lastT > windowStart) {
    final window = Duration(microseconds: ((lastT - windowStart) * 1e6).round());
    log('RIEPILOGO ${summary.flush(window: window, uptime: Duration(seconds: lastT.round()))}');
  }
  log('FINE: ${fmt(lastT)} di video, ${alarms.length} allarmi ${alarms.map(fmt).join(', ')}');

  // Confronto con le annotazioni.
  final ann = readAnnotations(File('${json.parent.path}/${stem}_landing.txt'));
  if (ann == null) {
    log('(nessun file ${stem}_landing.txt: niente confronto)');
    return (name: stem, landings: -1, hits: 0, falseAlarms: alarms.length, difficultAlarms: 0, note: note);
  }
  bool near(double a, double t) => a >= t - kHitBeforeS && a <= t + kHitAfterS;
  final used = <double>{};
  var hits = 0;
  for (final l in ann.landings) {
    final match = alarms.where((a) => !used.contains(a) && near(a, l)).toList();
    if (match.isEmpty) {
      log('landing ${fmt(l)}: NON segnalato');
    } else {
      used.add(match.first);
      hits++;
      log('landing ${fmt(l)}: segnalato a ${fmt(match.first)} (ritardo ${(match.first - l).toStringAsFixed(2)} s)');
    }
  }
  var difficultAlarms = 0;
  for (final d in ann.difficult) {
    final match = alarms.where((a) => !used.contains(a) && near(a, d)).toList();
    if (match.isEmpty) {
      log('punto difficile ${fmt(d)}: nessun allarme (bene)');
    } else {
      used.addAll(match);
      difficultAlarms += match.length;
      log('punto difficile ${fmt(d)}: ALLARME a ${match.map(fmt).join(', ')} (falso, ma in un caso difficile)');
    }
  }
  final others = alarms.where((a) => !used.contains(a)).toList();
  for (final a in others) {
    log('allarme ${fmt(a)}: FALSO (lontano da landing e punti difficili)');
  }
  return (
    name: stem,
    landings: ann.landings.length,
    hits: hits,
    falseAlarms: others.length,
    difficultAlarms: difficultAlarms,
    note: note,
  );
}

void main() {
  test(
    'replay',
    () {
      final target = FileSystemEntity.typeSync(kReplayPath) == FileSystemEntityType.directory
          ? (Directory(kReplayPath).listSync().whereType<File>().where((f) => f.path.endsWith('.json')).toList()
            ..sort((a, b) => a.path.compareTo(b.path)))
          : [File(kReplayPath)];
      final all = StringBuffer();
      final results = <VideoResult>[];
      for (final json in target) {
        final out = StringBuffer();
        void log(String line) {
          out.writeln(line);
          // ignore: avoid_print
          print(line);
        }

        results.add(replayFile(json, log));
        File(json.path.replaceAll(RegExp(r'\.json$'), '.report.txt')).writeAsStringSync(out.toString());
        all.write(out);
      }

      // Tabella finale.
      final table = StringBuffer()
        ..writeln('')
        ..writeln('video          landing segnalati   falsi allarmi   allarmi su punti difficili')
        ..writeln('-----------------------------------------------------------------------------');
      var totL = 0, totH = 0, totF = 0, totD = 0;
      for (final r in results) {
        final landings = r.landings < 0 ? '(senza file)' : '${r.hits}/${r.landings}';
        table.writeln('${r.name.padRight(14)} ${landings.padRight(19)} ${'${r.falseAlarms}'.padRight(15)} '
            '${r.difficultAlarms}${r.note.isEmpty ? '' : '   (${r.note})'}');
        if (r.landings > 0) totL += r.landings;
        totH += r.hits;
        totF += r.falseAlarms;
        totD += r.difficultAlarms;
      }
      table.writeln('TOTALE         ${'$totH/$totL'.padRight(19)} ${'$totF'.padRight(15)} $totD');
      // ignore: avoid_print
      print(table);
      if (target.length > 1) {
        File('${Directory(kReplayPath).path}/RIEPILOGO.txt').writeAsStringSync('$all$table');
        // ignore: avoid_print
        print('Resoconto completo in ${Directory(kReplayPath).path}/RIEPILOGO.txt');
      }
    },
    skip: kReplayPath.isEmpty ? 'serve --dart-define=REPLAY=cartella o file.json (vedi tool/replay_video.py)' : false,
    timeout: const Timeout(Duration(minutes: 30)),
  );
}
