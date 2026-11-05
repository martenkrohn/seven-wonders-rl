# Fortlaufendes Training über VS Code

Stand: 29. Oktober 2026. Der Dauerlauf ist eingerichtet und technisch geprüft.
Der eigentliche Trainingslauf wurde noch nicht gestartet.

## Bedienung

VS Code unter Ubuntu beziehungsweise WSL mit dem vorhandenen
`python.code-workspace` oder dem Ordner `python/` öffnen.
Unter **Terminal → Aufgabe ausführen** stehen diese Tasks bereit:

| Task | Wirkung |
| --- | --- |
| **Training starten / fortsetzen (fortlaufend)** | Startet einmal von null; spätere Aufrufe laden denselben Trainingsstand. |
| **Training sauber stoppen (nach aktuellem Update)** | Fordert das Ende nach dem aktuellen PPO-Update beziehungsweise der laufenden Evaluation an. Speichert anschließend den Stand. |
| **Trainings-Dashboard öffnen (fortlaufender Lauf)** | Öffnet das Dashboard dieses Dauerlaufs. Nach einer neuen Evaluation die Seite neu laden. |

Der Start-Task läuft weiter, bis er gestoppt wird. Der Stop-Task ist eine
separate Aufgabe und kann parallel zum laufenden Training aufgerufen werden.
Warten, bis im Trainings-Terminal **„Training gespeichert und gestoppt“** steht.
Danach setzt derselbe Start-Task den Lauf fort. Für einen normalen Stopp den
Stop-Task verwenden; das Schließen oder gewaltsame Beenden des Terminals ist
ein harter Abbruch.

Die Taskdefinition steht in
[tasks.json](../../python/.vscode/tasks.json). Alle Aufgaben verwenden die
vorhandene virtuelle Umgebung unter `python/.venv/`.

## Welche Einstellungen gelten?

Laufverzeichnis: `python/runs/continuous-pair-pfsp-v1-seed331/`.
Es ist getrennt vom bisher vorbereiteten begrenzten Lauf.

- **Kein Endbudget:** Training läuft über beliebig viele Sitzungen weiter.
  Der bisherige Rezeptwert für `total_timesteps` begrenzt den Dauerlauf nicht.
- **Evaluation:** zu Beginn und dann alle **262.144 Übergänge** gegen
  denselben separaten Kontrollplan, jeweils 96 Partien.
- **Sicherung:** alle **512 Übergänge**, im Standardsetup nach jedem PPO-Update.
- **Liga-Snapshots:** alle **65.536 Übergänge**; höchstens 64 neueste Kopien
  werden aktiv als Gegner berücksichtigt. Ältere Dateien bleiben erhalten.
- **Netz und Lernen:** weiterhin Memory, vier Köpfe, feste Actor-Priors mit
  Stärke 32, Standardreward und die ergebnisabhängige Gegnerpaar-Auswahl.

Der Dauerbetrieb ändert die wissenschaftliche Einordnung der Köpfe und
Actor-Priors nicht: Sie bleiben eine eigene forschungsinspirierte Erweiterung,
deren Zusatznutzen noch zu prüfen ist. Die genaue
[Gegnerregel](PAIR_PFSP_TRAINING.md#auswahlregel-und-parameter) bleibt bestehen.

Es werden keine Prozentwerte wie 25% eines Gesamtbudgets mehr verwendet.
Das Dashboard zeigt absolute Schrittzahlen und den nächsten Kontrollpunkt.
Im Auswahlfeld bleiben der neue Start und die zwölf neuesten Berichte sowie
die eine feste historische Referenz verfügbar;
die vollständige Lernkurve und sämtliche früheren Berichte bleiben gespeichert.
Ein älteres Einzeldashboard lässt sich beispielsweise aus
`evaluations/step_0262144/dashboard.html` öffnen.

Alte Experimente werden nicht mehr automatisch geladen. Die
[feste Referenz und der direkte Vergleich](EVALUATION.md#bereinigte-dashboard-auswahl)
sind im Evaluationsleitfaden beschrieben. Beim ersten Evaluieren wird das
Referenzmodell zusätzlich einmal auf dem neuen Kontrollplan ausgewertet und
unter `reference_evaluation/` gespeichert. Weitere Kontrollpunkte und
Wiederaufnahmen verwenden diese Ergebnisse weiter. Pro neuem Bericht werden
das aktuelle Modell und die feste Referenz angezeigt.

## Was passiert beim Stoppen oder bei einem Absturz?

Beim **sauberen Stopp** wird das aktuelle Rollout mit seinem PPO-Update beendet,
beziehungsweise die laufende Evaluation fertiggestellt. Danach werden Modell,
Optimizer, Zufallszustände und Liga-Schätzungen gespeichert. Eine angeforderte,
noch nicht begonnene Evaluation kann bis zur Wiederaufnahme warten.

Beim **harten Prozessabbruch** wird beim nächsten Start der letzte vollständig
gespeicherte Stand geladen. Im Standardsetup kann höchstens ein noch nicht
gespeicherter Update mit 512 Übergängen verloren gehen. Unfertige Engine-Partien
werden neu begonnen; ihr Ablauf wird nicht mitten im Spiel wiederhergestellt.
Das gilt auch beim sauberen Stopp. Bereits gespeicherte Lernergebnisse bleiben
erhalten. Die Wiederaufnahme ist deshalb keine bitgenaue Fortsetzung derselben
Spiele.

Unvollständige Evaluationen werden bei der Wiederaufnahme separat aufbewahrt
und am selben Modellstand erneut durchgeführt. Eine durch den Abbruch
abgeschnittene letzte Protokollzeile wird archiviert. Andere beschädigte
Protokollzeilen werden als Fehler gemeldet.

Nach einer Wiederaufnahme laufen **Schrittzähler, Optimizer und Liga weiter**.
Die Anfangsgewichte werden nicht neu ausgewürfelt. Ein zweiter Trainer für
dasselbe Verzeichnis wird durch die bestehende Verzeichnissperre verhindert.
Geänderte Lern- oder Evaluationskonfigurationen werden nicht stillschweigend
als derselbe Versuch fortgesetzt.

## Bedienung in der Konsole

Aus `python/` starten beziehungsweise fortsetzen:

```bash
.venv/bin/python scripts/ppo_training/run_fresh_pair_pfsp.py --continuous --auto-resume
```

In einem zweiten Terminal sauber stoppen:

```bash
.venv/bin/python scripts/ppo_training/run_fresh_pair_pfsp.py --continuous --stop
```

Dashboard öffnen:

```bash
.venv/bin/python scripts/open_dashboard.py --run-dir runs/continuous-pair-pfsp-v1-seed331
```

Zum Prüfen der aufgelösten Argumente ohne Trainingsstart:

```bash
.venv/bin/python scripts/ppo_training/run_fresh_pair_pfsp.py --continuous --auto-resume --dry-run
```

`--eval-every`, `--save-every`, `--snapshot-every` und `--out-dir` können vor
einem neuen Lauf überschrieben werden. Bei späteren Aufrufen müssen dieselben
Werte angegeben werden. Die VS-Code-Tasks verwenden einheitlich die oben
genannten Standardwerte. Ein bestehendes Verzeichnis ohne wiederherstellbaren
Checkpoint wird erhalten und als Fehler gemeldet.

## Dateien und wissenschaftliche Auswertung

`latest.pt` ist der Wiederaufnahmepunkt; `config.json` enthält die tatsächlich
verwendeten Einstellungen. `training_status.json` zeigt den aktuellen Status,
Schrittstand und nächsten Evaluationszeitpunkt. `learning_curve.json` führt
die Kontrollpunkte über alle Sitzungen fort. Die übrigen
[Ligaprotokolle](PAIR_PFSP_TRAINING.md#evaluation-und-dashboard) bleiben bestehen.
Einzelberichte und Checkpointdateien sammeln sich über die Laufzeit weiter an.

Der Dauerlauf ist ein Entwicklungsmodus. Ein besseres Ergebnis nach mehr
Training ist zunächst ein Befund über diesen Lauf. Für einen Methodenvergleich
werden anschließend gleiche Trainingsbudgets und unabhängige Wiederholungen
benötigt. Das wiederholt betrachtete Kontrollpanel ist kein unberührter finaler
Test; eine abschließende Prüfung wird separat geplant.

Die technischen Tests stehen in
[test_continuous_training.py](../../python/tests/test_continuous_training.py).
Geprüft werden sauberes Stoppen, automatisches Fortsetzen über das ehemalige
Budget hinaus, harte Abbrüche bei Schritt 0 und einer späteren Evaluation,
Reparatur einer abgeschnittenen Protokollzeile und das begrenzte HTML-Auswahlfeld.
Zusätzlich wurde der tatsächliche Vier-Kopf-Starter in zwei temporären
Testsitzungen von 0 auf 192 und anschließend auf 384 Übergänge fortgesetzt;
die sieben Kontrollpunkte von 0 bis 384 blieben erhalten.
Dieser kurze Funktionstest ist kein Spielstärkenachweis.
Die abschließende Gesamtprüfung bestand mit **197 erfolgreichen Tests und
einem übersprungenen Test**. Die Dauerlauf-Tests wurden nach der letzten
Anpassung der Budgetdarstellung zusätzlich erneut erfolgreich ausgeführt.
