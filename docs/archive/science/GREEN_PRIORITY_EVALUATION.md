# Mad Scientist: dasselbe Modell mit und ohne Grün-Priorität

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Dieser Vergleich evaluiert ein eingefrorenes trainiertes Modell zweimal:
unveränderte freie Policy und Grün-Priorität. Es findet kein Training statt.
Die Grün-Priorität gilt ab der ersten Entscheidung, nicht erst nach dem ersten
gebauten grünen Gebäude.

Wenn ein grünes Gebäude legal baubar ist, werden ausschließlich diese
Bauaktionen zugelassen. Bezahltes und kostenloses Bauen zählen; grüne Karten
als Wunderbaumaterial oder Ablage nicht. Wissenschaftsgilden sind violett und
werden nicht erzwungen. Zwischen mehreren grünen Bauaktionen entscheidet
weiterhin die bedingte Wahrscheinlichkeitsverteilung desselben Modells.
Ohne legal baubares Grün bleibt seine Auswahl unverändert. Modellgewichte,
Zustandsmerkmale, Reward und Gegner-Policies werden nicht verändert.

## Vorab festgelegter Vergleich

- Ausgangsmodell: finales Entropie-.03/Lambda-.95-Modell von Seed 31. Es hatte
  den höchsten gemessenen Sieganteil im bisherigen gemeinsamen Holdout, war dem
  Standard aber nicht eindeutig überlegen. Das ist eine konkrete Auswahl und
  kein Beweis, dass es allgemein das beste Modell ist.
- Gegner: dieselben eingefrorenen Standard-Checkpoints 131.072 und 262.144.
- Neuer Test-Seed **69001**, um die Auswahl nicht erneut auf Test-Seed 59001
  zu beurteilen. Beide Modi verwenden exakt denselben neuen Testplan und dieselben
  rollenbezogenen Policy-Zufallsseeds.
- 40 Seed-Blöcke, je drei Sitze und vier geordnete Gegnerpaare:
  **480 Partien je Modus, insgesamt 960**. Alle Entscheidungen sind stochastisch.
- Hauptauswertung: gepaarter Sieganteils- und VP-Abstandsunterschied
  Grün-Priorität minus freie Policy. 95%-Intervalle verwenden 2.000
  Bootstrap-Stichproben ganzer Engine-Seed-Blöcke.
- Ergänzend: Wissenschaftsmittelwerte, hohe Erträge ab 10/20/30 Punkten,
  Zeitalteraufbau, Kartenentscheidungen und Regelprüfung.

## Starten

Aus dem Projektverzeichnis `python/`:

```bash
cd /home/marten/dev/7Wonders/python
.venv/bin/python scripts/eval_green_priority.py \
  --checkpoint runs/science-diagnostic-v1-entropy-seed31/latest.pt \
  --reference runs/science-diagnostic-v1-baseline-seed31/checkpoint_131072.pt \
              runs/science-diagnostic-v1-baseline-seed31/checkpoint_262144.pt
```

Der Befehl wertet ausschließlich vorhandene Gewichte aus und gibt den
Ergebnisordner aus. Standardmäßig entsteht ein neuer Ordner
`runs/green-priority-<Kennung>/`, damit wiederholte Starts nicht an einem bereits
gefüllten Ordner scheitern oder alte Ergebnisse überschreiben. Eine abgebrochene
Evaluation bleibt erhalten, wird aber nicht automatisch fortgesetzt. Ein optionales
`--out-dir PFAD` muss leer sein. `--blocks 1` ist ausschließlich ein kurzer
Funktionstest mit zwölf Partien je Modus.

Die vollständige Evaluation wurde bei der Einrichtung nicht ausgeführt. Nach
Abschluss mit dem ausgegebenen Ergebnisverzeichnis wieder melden.

## Konsole und Dashboard

Die Konsole nutzt die bestehenden formatierten Phasen-, Fortschritts-, Restzeit-
und Ergebnisanzeigen. Die Modi werden ausdrücklich **Freie Policy** und
**Grün-Priorität** genannt; keiner davon ist untrainiert. Das Dashboard zeigt
dieselben Identitäten und die Regelstatistik. Der Workspace-Task
**Analyse-Dashboard im Browser öffnen** findet die neue Auswertung nach Abschluss.

Pro Partie werden ursprüngliche legale Angebote und tatsächlich ausgeführte
Aktionen gespeichert. Die Regelstatistik zählt bestätigte Entscheidungen:

- `green_opportunities`: Entscheidungen mit mindestens einer legalen grünen Bauaktion.
- `green_builds`: tatsächlich gebaute grüne Gebäude.
- `restricted_decisions`: Die Grün-Priorität hat mindestens eine sonst legale
  Aktion ausgeschlossen. Das heißt nicht, dass die freie Policy bei derselben
  Beobachtung tatsächlich anders entschieden hätte.
- `violations`: Grün war legal, wurde trotz Grün-Priorität aber nicht gebaut.
  Dieser Wert muss null sein; die Evaluation schlägt andernfalls fehl.

Die Berichte enthalten JSONL-Einzelpartien, `summary.json`, `analysis.json`,
Text/Markdown, CSV-Exporte und `dashboard.html`. Für bereits vorhandene Daten
kann man Berichte ohne neue Spiele neu erzeugen:

```bash
.venv/bin/python scripts/eval_green_priority.py --report-only --out-dir runs/green-priority-DEINE_KENNUNG
```

## Interpretation

Ein besseres Ergebnis zeigt ungenutztes Potenzial der Grün-Priorität in dieser
Modell-/Gegnerkombination. Gegner verhalten sich abhängig vom veränderten
Spielzustand weiter gemäß ihren eingefrorenen Policies; ihre Reaktionen werden
nicht ausgeschaltet. Ein positives Ergebnis beweist keine fehlende gegnerische
Reaktion oder Deep Exploration als einzige Ursache.

Ein schlechteres Ergebnis widerlegt eine gute Wissenschaftsstrategie nicht:
Ressourcenplanung und die Auswahl zwischen grünen Karten wurden nicht für
diesen Zwang nachtrainiert. Mehr Wissenschaftspunkte allein sind ebenfalls kein
Nachweis besserer Spielstärke. Das Ergebnis gilt zunächst für einen trainierten
Checkpoint und Gegner derselben Trainingslinie.
