# Gezielte Versuche

Hier werden die kommenden wissenschaftlich ausgerichteten Vergleiche geführt.
Die bisherige explorative Phase ist im [Archiv](../archive/README.md)
dokumentiert. Der erste neue Versuch ist vorbereitet:
[001 · Frischer Lerner mit Gedächtnis und eigener PFSP-Liga](001-fresh-pair-pfsp/PROTOCOL.md).
Der lange Trainingslauf ist noch nicht gestartet.

## Ein Versuch, eine beantwortbare Frage

Ausgangspunkt ist die [Forschungsagenda](../research/AGENDA.md), nicht ein
beliebiges neues Modul. Eine Versuchsentscheidung sollte sagen können:
Was vermuten wir, welche Änderung prüft das, und welches Ergebnis würde dagegen
sprechen? Grundsätzlich führen wir die Versuche nacheinander durch und
entscheiden nach ihrer Auswertung weiter.

Neue Versuche erhalten einen Ordner wie `001-kurzer-fragenname/` mit:

| Datei | Zweck |
| --- | --- |
| `PROTOCOL.md` | Vor dem Lauf festgelegte Frage, Vergleich, Messung und Entscheidungsregel |
| `RESULTS.md` | Tatsächlich durchgeführter Ablauf, Ergebnisse, Abweichungen und Schlussfolgerung |
| `summary.json` | Optional: kompakte strukturierte Auswertung mit Herkunftsangaben |

Große Rohdaten und Modelle bleiben im Laufverzeichnis. Das Protokoll verweist
auf die genaue Konfiguration; es dupliziert nicht sämtliche Skriptoptionen.
Beim Anlegen wird der Versuch unten im Register verlinkt. Eine neue Methodenwahl
erfordert nicht für jede Kleinigkeit ein zusätzliches Dokument.

## Vorlage für `PROTOCOL.md`

```markdown
# [Prüffrage]

Status: geplant | läuft | abgeschlossen | abgebrochen
Datum / Kennung:

## Frage und Hypothese
Welche Unsicherheit soll der Versuch verringern?
Welcher Befund würde für oder gegen die Hypothese sprechen?
Welche Quelle oder bestehende Beobachtung motiviert den Vergleich?

## Vergleich
Baseline, Kandidat und die gezielt veränderte Komponente.
Gemeinsame Ausgangsgewichte, Informationen, Reward und Simulatorversion.
Abweichungen von einem isolierten Vergleich ausdrücklich nennen.

## Durchführung und Herkunft
Konfigurationsdatei, exakter Befehl, Code-/Engine-/Modellidentität.
Trainingsseeds, geplantes Budget und Abbruchregel.
Gegnerpool, gemeinsame Gegnerprofile und Auswahlregel.
Ausgabeordner und benötigte vorhandene Artefakte.

## Prüfung und Entscheidung
Primäre Metrik und praktisch relevante Verbesserung.
Diagnosen für die vermutete Fähigkeit sowie mögliche Nebenwirkungen.
Sitze, Wunder, Spielseeds und Einheiten der Unsicherheitsberechnung.
Entwicklungs-/Dashboardpanel von abschließender Prüfung trennen.
Checkpointauswahl und Umgang mit mehreren Vergleichen festlegen.
Welcher Ausgang rechtfertigt Behalten, Verwerfen oder weitere Klärung?
```

`RESULTS.md` beantwortet anschließend dieselbe Frage: Was lief tatsächlich,
welche Daten sind vollständig, wie unsicher ist der Unterschied, was folgt
daraus und was bleibt offen? Auch fehlgeschlagene oder negative Versuche werden
eingetragen. Eine nachträglich geänderte Hypothese wird als solche ausgewiesen.

Hinweise zur Methodik stehen in der
[Evaluationsrecherche](../research/CONTENT_BANK.md#evaluation); die
[Bedienungsanleitung](../guides/EVALUATION.md) beschreibt die vorhandenen Werkzeuge.

## Register

Noch keine neuen Versuche beschlossen. Die historischen Laufbefehle im Archiv
werden durch diese Vorlage nicht erneut beauftragt.
