# Ergebnis des langen Laufs mit öffentlichen Informationen

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Der Lauf `long-mixed-wonders-v1-seed131` ist abgeschlossen: 2.097.152 zusätzliche
Übergänge, 4.096 PPO-Updates, 99.864 abgeschlossene Trainingspartien und fünf
Kontrollevaluationen mit je 288 vollständig abgeschlossenen Spielen. Keine
Evaluationsfehler oder Abbrüche. Dauer 4 h 28 min 17 s, einschließlich Evaluationen
130,3 Übergänge/s. Die vier Köpfe erhielten jeweils etwa 524.000 Übergänge.

## Spielstärke

| Fortschritt | Sieganteil | Punkte Ø | Abstand zum stärksten Gegner Ø | Abstand ohne Wissenschaftsspezialist Ø |
| --- | ---: | ---: | ---: | ---: |
| Start | 36,63% | 42,96 | −2,95 | −3,37 |
| 25% | 50,00% | 44,42 | +0,23 | +0,22 |
| 50% | 49,65% | 44,27 | −0,94 | −0,66 |
| 75% | 48,38% | 46,07 | −0,56 | −0,78 |
| Ende | 49,94% | 47,47 | +0,03 | +0,50 |

Sieganteil zählt geteilte Siege anteilig. Das Panel enthält zwölf gleich gewichtete
geordnete Paarungen, nicht die Gewichtung des Trainingspools.

Ende gegenüber Start: +2,98 Abstandspunkte, gespeichertes gepaartes 95%-Intervall
[+0,67; +5,69], und +13,31 Prozentpunkte Sieganteil [+2,31; +25,35]. Die Intervalle
verwenden Bootstrap ganzer Engine-Seed-Blöcke, acht Blöcke insgesamt. Das ist
Fortschritt auf diesem Kontrollplan, keine Bestätigung gegen unbekannte Strategien.

Der Fortschritt ist diesmal nicht auf Tische mit Wissenschaftlern begrenzt:

- Ohne Wissenschaftsspezialist: 240 Spiele, Sieganteil 35,62% → 53,68%,
  Abstand −3,37 → +0,50; Zuwachs +3,87 [ca. +1,64; +6,73].
- Nur v1/v2-Referenzen: 96 Spiele, Abstand −3,90 → +0,39;
  Zuwachs +4,28 [ca. +1,81; +7,57].
- Mit Militärspezialist: 96 Spiele, Abstand −3,05 → +0,96;
  Zuwachs +4,01 [ca. +1,74; +6,95].
- Mit Wissenschaftsspezialist: nur 48 Spiele, Sieganteil 41,67% → 31,25%,
  Abstand −0,85 → −2,31. Die Verschlechterung ist unsicher:
  Zuwachs −1,46 [ca. −7,04; +3,42]. Mögliche stärkere Wissenschaftskonkurrenz
  bleibt eine Hypothese und muss mit mehr Spielen geprüft werden.

Untergruppenintervalle wurden offline mit 10.000 Resamples gerechnet; die
gespeicherte Gesamtauswertung verwendet 2.000. Kleine Abweichungen der Grenzen
sind Monte-Carlo-Rundung, kein anderer Spielplan. Untergruppen sind explorativ.

Das Ende ist im Punkteabstand nicht nachweislich besser als der 25%-Stand:
−0,19 [−2,49; +2,70], Sieganteil praktisch gleich. Die höheren eigenen Punkte
später im Lauf übersetzen sich nicht automatisch in einen höheren Vorsprung.
Alle Zwischenstände behalten; keinen endgültigen Sieger aus diesem kleinen
mehrfach betrachteten Kontrollpanel auswählen.

## Ressourcenplanung und Abwerfen

| Verhalten | Start | Ende |
| --- | ---: | ---: |
| Graue Bauten je Partie, Zeitalter I | 0,108 | 1,500 |
| Graue Bauten je Partie, Zeitalter II | 0,087 | 0,292 |
| Anteil angewandter Entscheidungen mit legalem Kartenbau in III | 81,12% | 92,62% |
| Abwürfe je Partie in III, insgesamt | 0,951 | 0,417 |
| Abwürfe in III ohne legalen Kartenbau | 0,931 | 0,340 |
| Abwürfe in III trotz legalem Kartenbau | 0,021 | 0,076 |

Die frühe Ressourceninvestition verändert sich deutlich, besonders nach der
Hälfte des Laufs. Fehlende Kartenbaumöglichkeiten in III werden viel seltener.
Abwürfe ohne legalen Kartenbau gehen um rund 63% zurück. Diese Bezeichnung
bedeutet nicht, dass keine Wunderbauoption existierte.

Fast der gesamte Rückgang später Abwürfe stammt aus weniger Situationen ohne
legalen Kartenbau. Abwürfe trotz vorhandener Kartenbauoption steigen dagegen
leicht. Das zeigt auch: Nicht jeder freiwillige Abwurf ist zwangsläufig ein
Planungsfehler. In I steigen solche Abwürfe ebenfalls von 0,212 auf 0,361 je
Partie; Münzplanung und der Wert der verfügbaren Karten wären separat zu prüfen.

Paarweise mittlere Verbesserung der III-Kartenbauverfügbarkeit je Partie:
+11,49 Prozentpunkte [ca. +7,79; +15,38]. Dieser Nenner ist pro Partie gemittelt;
die Tabelle gewichtet alle angewandten Entscheidungen direkt.

Dies stützt die Vermutung, dass frühere Modelle zu wenig für spätere Bauten
investierten. Es beweist keinen alleinigen kausalen Effekt grauer Karten:
Handel, Ketten, Wunder und die geänderten Gegner beeinflussen den Verlauf mit.

## Strategieverhalten und Köpfe

Militärpunkte steigen von 2,93 auf 8,42, Wissenschaft von 8,52 auf 14,00,
Gilden von 3,28 auf 6,21. Blaue Punkte fallen von 17,22 auf 9,16.
In I werden am Ende fast keine blauen/gelben Karten gebaut: 6 blaue und
2 gelbe Bauten in 288 Partien, gegenüber 340 bzw. 148 am Start. Das könnte
Investitionsplanung zeigen, ist aber auch ein Hinweis auf eine starre Präferenz.

Alle vier Köpfe bauen früh mehr Grau: je nach Kopf 1,57–2,05 graue Karten
in I+II je Partie. Die Änderung ist nicht nur einem zufällig erfolgreichen Kopf
zuzuschreiben. Der gleiche Zustand führt dennoch meistens zu ähnlichen Aktionen:
die gemessene Greedy-Abweichung relativ zu Kopf 0 liegt in den letzten
100 Updates bei rund 7% (einschließlich Kopf 0 selbst). Vier eigenständige,
breit kontextabhängige Strategien sind damit noch nicht belegt.

Die vorhandenen Prior-Gewichte sind zwischen Start und Ende bytegleich.
Die tatsächlich gezogenen Gegnergruppen entsprechen ungefähr 50/15/20/5/5/5%.
Anfangs fehlende dynamische Mitglieder wurden wie vorgesehen durch v1/v2 ersetzt;
im letzten Viertel bestehen die dynamischen Gruppen regulär aus eigenen Ständen.

## Nächster sinnvoller Schritt

Keinen weiteren langen Lauf automatisch anhängen. Zunächst Start, 25%-Stand
und Endmodell gemeinsam auf neuen, nicht zur Trainingskontrolle verwendeten
Seed-Blöcken prüfen, beispielsweise 32 Blöcke × 12 Paarungen × 3 Sitze:
1.152 Spiele je Modell, 3.456 insgesamt. Das ist ein Vorschlag, noch nicht gestartet.
Zuerst dieselben Referenzgegner für einen sauberen Checkpoint-Vergleich; weitere
unbenutzte Gegner anschließend als ausdrücklich getrennte Generalisierungsprüfung.

Dabei insbesondere Fortschritt ohne Wissenschaftler, Wissenschaftskonkurrenz,
III-Kartenbauverfügbarkeit und Verhalten nach Wunderseite vergleichen.
Erst anhand dieses Ergebnisses Trainingsbudget oder Gegnergewichte ändern.
Informationsumfang, Gegnerpool und Laufdauer wurden gemeinsam geändert;
dieser einzelne Versuch trennt ihre individuellen Wirkungen nicht.

Vollständige maschinenlesbare Analyse:
`python/runs/long-mixed-wonders-v1-seed131/post_training_analysis.json`.
Das vorhandene `dashboard.html` enthält weiterhin alle fünf Kontrollpunkte.

## Vorbereiteter frischer Vergleich

Aus `python/` starten:

```bash
.venv/bin/python scripts/ppo_training/eval_long_training_fresh.py
```

Vergleicht Start, 25% und Ende auf 32 neuen Seed-Blöcken, zwölf Paarungen und
allen drei Sitzen: 1.152 Spiele je Modell, 3.456 insgesamt. Kein Training.
Prüft vorab Überschneidungen mit gespeicherten Trainings-/Evaluations-Seeds,
kopiert Modelle und Referenzen und speichert den gemeinsamen Plan vor Spielbeginn.
Ausgabe: `runs/long-fresh-seed130001-blocks32/`, einschließlich `dashboard.html`,
gepaarter Vergleiche und Verhaltensdiagnostik. Der technische Kurztest verwendet
einen anderen Seed und nur einen Block; der volle Vergleich startet manuell.

Optional beide Prüfungen in einem Aufruf:

```bash
.venv/bin/python scripts/ppo_training/eval_long_training_fresh.py --include-direct
```

Zusätzlich treten Start, 25%-Stand und Ende zusammen an einem Tisch an, mit
beiden Gegnerreihenfolgen und allen Sitzen. Je Modell 192 weitere Spiele,
zusammen 576, insgesamt mit dem Referenztest **4.032 Spiele**. Der direkte Test
verwendet einen eigenen Seed und erzeugt `direct/dashboard.html`. Es sind
ausgewogene Evaluationen jedes fokalen Spielers mit separaten Zufallsströmen,
keine dreifach gezählten Aufzeichnungen derselben Partie. Deshalb wird für den
Direktvergleich kein gepaarter Zugewinn gegen identische Gegner behauptet.
