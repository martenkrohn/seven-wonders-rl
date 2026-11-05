# 001 · Frischer Lerner mit Gedächtnis und ergebnisgesteuerter Gegnerliga

Beschlossen und vorbereitet am 29. Oktober 2026. Status: **technische Validierung;
wissenschaftlicher Langlauf noch nicht gestartet**.

Zusätzlich ist inzwischen ein separater
[Dauerbetrieb über VS-Code-Tasks](../../guides/CONTINUOUS_TRAINING.md)
eingerichtet. Er verwendet dieselbe Lern- und Gegnerregel, aber kein Endbudget,
Evaluationen alle 262.144 und Sicherungen alle 512 Übergänge. Sein eigenes
Laufverzeichnis lautet `runs/continuous-pair-pfsp-v1-seed331/`.
Die folgenden festen Budget- und Prozentangaben beschreiben weiterhin die
begrenzte Versuchsvariante; ein Dauerlauf ist zunächst eine Entwicklungsreihe.

## Frage und Aussagegrenze

Kann ein zufällig initialisierter Spieler mit der verbesserten
Informationsgrundlage und einer automatisch wachsenden eigenen Liga innerhalb
des festgelegten Budgets brauchbare, zunehmend starke Spielweisen entwickeln?

Dies ist ein **Ausgangslauf für das neue Gesamtsetup**. Er isoliert weder den
Nutzen von Memory noch den von PFSP oder Priors. Ein Vergleich mit alten
Endmodellen bewertet den erreichten Spielstand, nicht den kausalen Vorteil
eines einzelnen Bausteins. Ein unabhängiger zweiter Trainingsseed oder eine
Gegenprobe mit gleichverteilter Paar-Auswahl ist ein späterer Versuch.

## Vor dem Lauf festgelegter Aufbau

- Rezept: [fresh_pair_pfsp_v1.json](../../../python/configs/fresh_pair_pfsp_v1.json).
- Neuer Seed 331; keine Startgewichte und keine alten Trainingsgegner.
- PPO, vier Köpfe, pro Partie feste Kopfwahl, Actor-Logit-Priors Stärke 32.
- Gemeinsame vollständige eigene Hand und expliziter Beobachtungsverlauf;
  öffentliche Karten- und Wunderinformationen aktiviert.
- Standardreward: eigener Punkteabstand zum stärksten Gegner geteilt durch 10,
  nur am Spielende; keine Spezialistenboni.
- Acht parallele Umgebungen, Horizont 64, Minibatches 256, vier PPO-Epochen,
  Lernrate 0,0003; weitere PPO-Parameter stehen vollständig in `config.json`.
- 2.097.152 Übergänge, 4096 PPO-Updates; ungefähr 100.000 volle Partien
  bei 21 Engine-Übergängen je Partie. Unfertige Schluss-/Resume-Partien
  reduzieren die tatsächlich abgeschlossene Spielanzahl.
- Liga ausschließlich aus eigenen eingefrorenen Kopien ab Schritt 0,
  anschließend alle 65.536 Übergänge; maximal 64 aktive Modelle.
- Geordnete Paare, Priorisierung mit Exponent 2, Kontrollanteil 0,10,
  neutrale Schätzung 1/3 mit Pseudo-Spielzahl 3, Halbwertszeit 131.072.
- Ergebnisse werden über Kopf-Mischungen pro Checkpoint gemittelt.
- Keine im Voraus benannten Strategien, Verhaltensfilter oder Spezialisten.

Die [Bedienung](../../guides/PAIR_PFSP_TRAINING.md) definiert Formel,
Startphase, Wiederaufnahme und Protokolldateien. Die Größen sind dokumentierte
Ausgangseinstellungen, keine für 7 Wonders nachgewiesenen Optima. Der Wert 32
für Priors wird aus dem bisherigen Setup übernommen; bei frischer
Initialisierung beziehen sich die Priors auf die aktuellen öffentlichen
Eingaben und werden nicht erneut aus alten Gewichten übernommen.

## Messung

Fester separater Kontrollplan:
[Population](../../../python/configs/fresh_pair_pfsp_v1_evaluation.json),
[Paarplan](../../../python/configs/fresh_pair_pfsp_v1_eval_panel.json).
Zwei historische Endmodelle, vier geordnete Paarungen, drei Sitze,
acht gemeinsame Seed-Blöcke pro Paar und Sitz. Evaluation bei 0/25/50/75/100%.
Keine Rückmeldung aus diesen Partien an die Liga.

Primäre Entwicklungskennzahl: mittlerer Punkteabstand zum stärksten Gegner
auf diesem festen Panel. Zusätzlich: Sieganteil mit geteilten Siegen,
Ergebnisse je Paar, Punkteherkunft, Ressourcenbau und Abwürfe nach Zeitalter.
Gemeinsame Seeds ermöglichen gepaarte Vergleiche. Die neuen Berichte zeigen
den Vergleich zum festgehaltenen bisherigen Endmodell, das einmal auf demselben
Plan ausgewertet wird; die Startdaten bleiben ebenfalls gespeichert.
Die Berichte enthalten Unsicherheitsintervalle. Acht Seed-Blöcke bleiben eine kleine
Stichprobe und erlauben keine starke Aussage für seltene Untergruppen.

Für die Liga werden Zahl und Häufigkeit der Paare, effektive Stichproben,
Alter der Ergebnisse und Kontrollauswahlen geprüft. Stark unterschiedliche
Checkpoints oder viele Paare allein belegen keine strategische Vielfalt.

## Entscheidungsregel nach dem Lauf

1. Zuerst technische Gültigkeit prüfen: keine erfundenen Übergänge, keine
   Evaluationsgegner im Trainingspool, persistierte und nachvollziehbare Auswahl.
2. Lernkurve und Ergebnisse nach Paar betrachten. Ein besserer Gesamtschnitt
   bei deutlichem Einbruch einzelner Konstellationen erfordert Untersuchung.
3. Verbesserung zum zufälligen Start ist ein Lernnachweis auf diesem Panel,
   noch kein Beleg einer Verbesserung gegenüber dem bisherigen Verfahren.
4. Vor Aussagen über Methodenüberlegenheit einen kontrollierten Vergleich
   oder unabhängige Trainingswiederholung planen. Abschließende neue Seeds
   und zusätzliche Gegnerprofile werden gesondert festgelegt.

Die Kontrollpunkte entscheiden nicht automatisch über Abbruch oder
Hyperparameteränderungen. Jede bewusste Änderung während des Versuchs wird
als Abweichung dokumentiert; das bestehende Laufverzeichnis wird erhalten.

## Nachweise und Quellen

Die technische Prüfung umfasst die Auswahlrichtung, Neutralstart,
Unsicherheits-Schrumpfung, zeitliche Abwertung, getrennte linke/rechte Sitze,
Informationsgrenzen, Checkpoint-Wiederaufnahme und ein laufendes Dashboard.
Tests: [test_pair_pfsp.py](../../../python/tests/test_pair_pfsp.py).

Methodischer Bezug: [PFSP/AlphaStar](../../research/CONTENT_BANK.md#pfsp).
Die Erweiterung auf geordnete Dreierspiel-Paare und die konkrete Schätzregel
sind unsere Implementierungsentscheidungen. Sie implizieren weder
Gleichgewichtskonvergenz noch eine Garantie für Deep Exploration.

Technischer Nachweis vor dem Langlauf: Die markierte Regression bestand mit
189 erfolgreichen Tests und einem übersprungenen Test. Die ergänzende Prüfung
der neuen Rezeptvalidierung und der unmarkierten vorhandenen Evaluationstests
ergibt insgesamt **192 unterschiedliche bestandene Tests, einen übersprungenen**.
Der tatsächliche Starter lief mit 256 Übergängen, fünf eigenen eingefrorenen
Modellen, 25 Paaren und fünf Evaluationen zu je zwölf Spielen durch.
Artefakte dieses reinen Funktionstests liegen unter
`/tmp/7w-fresh-pair-pfsp-validation-20261029/`; sie sind kein wissenschaftliches
Lernergebnis. Der Test einer unterbrochenen und wiederaufgenommenen Phase prüft
zusätzlich, dass Liga-Feedback ausschließlich aus abgeschlossenen Trainingsspielen
stammt und nicht aus den Kontrollpanel-Partien.
