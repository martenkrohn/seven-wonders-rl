# Auswertung des moderaten Self-Play-Laufs vom 23.10.2026

Lauf: `python/runs/poc-moderate-seed31-v2`, Seed 31, 262.144 Schritte,
512 Updates, 2 Umgebungen. Technisch vollständig abgeschlossen in 2.158 Sekunden
(rund 36 Minuten). Alle 12.482 protokollierten Trainingsspiele wurden abgeschlossen;
die Spielerpositionen waren nahezu gleich verteilt (4.161/4.161/4.160).
Alle neun regulären Evaluationen mit je 144 Spielen waren vollständig.

Die Referenzen während des Trainings waren untrainierte Anfangsmodelle. Der
Sieganteil stieg von 34,0 % auf 100 % bei Schritt 65.536 und lag am Ende bei
96,5 %. Das zeigt Lernen, beantwortet aber nicht, ob spätere Updates gegenüber
bereits trainierten Modellen hilfreich waren.

## Separater Test gegen trainierte Vorgänger

Quelle: `python/runs/poc-moderate-seed31-v2-holdout/summary.json` sowie
`candidate.jsonl` und `initial.jsonl`. Evaluation mit Seed 29001, 40 unabhängigen
Engine-Seed-Blöcken, allen drei Spielerpositionen und vier Gegnerkombinationen.
Das finale Modell und das untrainierte Ausgangsmodell spielten je 480 Spiele;
alle 960 Spiele wurden ohne Fehler oder Abbrüche abgeschlossen.

| Kennzahl | Untrainiert | Finales Modell |
|---|---:|---:|
| Alleinige Siege | 0 % | 55,21 % |
| Sieganteil einschließlich anteiliger Gleichstände | 0 % | 58,02 % |
| Eigene Punkte im Mittel | 17,59 | 41,50 |
| Punktevorsprung gegen den besten Gegner | −30,07 | +1,55 |

Der gepaarte Zuwachs des Punktevorsprungs gegenüber dem **untrainierten** Modell
beträgt +31,61 Punkte, mit einem 95%-Bootstrapintervall von [+30,20; +33,07].
Diese Zahl ist kein direkter Abstand zum 65.536er- oder 131.072er-Modell.

| Gegner links/rechts | Spiele | Finaler Sieganteil | Finaler Punktevorsprung |
|---|---:|---:|---:|
| 65.536 / 65.536 | 120 | 64,17 % | +2,71 |
| 131.072 / 131.072 | 120 | 50,83 % | −0,35 |
| 65.536 / 131.072 | 120 | 58,75 % | +1,22 |
| 131.072 / 65.536 | 120 | 58,33 % | +2,61 |

Nach Spielerposition beträgt der finale Sieganteil 51,88 %, 69,06 % und 53,13 %
(je 160 Spiele). Das Ergebnis beruht somit nicht ausschließlich auf einer Position.
Die Aufschlüsselungen sind deskriptiv; dafür wurden keine separaten Signifikanztests durchgeführt.

Zusätzliche absolute Unsicherheitsintervalle wurden aus den Einzelspielen berechnet:
pro Engine-Seed Mittelwerte über dessen 12 Spiele, anschließend 2.000 Bootstrap-
Resamples ganzer Blöcke mit lokalem NumPy-RNG-Seed 0 und 2,5/97,5-%-Quantilen.
Der finale Sieganteil hat so ein ungefähres 95%-Intervall von [52,40 %; 63,96 %],
der absolute Punktevorsprung von [+0,32; +2,76]. Diese sind zusätzliche
Auswertungen, keine Felder des bisherigen `summary.json`.

## Schlussfolgerung und nächster Schritt

Der Proof of Concept ist für diesen Lauf überzeugend: Lernen aus neuronalen
Self-Play-Spielen funktioniert, und der finale Stand schlägt trainierte Vorgänger
auf neuen Spiel-Seeds häufiger als den Ein-Drittel-Orientierungswert gleich
starker Dreispieler-Policies. Die letzten Updates waren somit nicht nur gegen
untrainierte Anfangsmodelle erfolgreich. Mehr Siege bei leicht negativem mittleren
Punktevorsprung gegen zwei 131.072er-Gegner sind vereinbar: wenige hohe Niederlagen
können viele knappe Siege im Punktemittel ausgleichen.

Ein einzelner Trainings-Seed und Gegner aus derselben Trainingslinie belegen
weder zuverlässige Verbesserung jedes Zwischenstands noch Stärke gegen fremde
Strategien oder Menschen. Die neuere Edition und Wunder-Ausbauschritte bleiben
unvollständig abgebildet. Vor einer größeren Investition empfiehlt sich ein
zweiter gleich konfigurierter Trainingslauf mit anderem Trainings-Seed, gefolgt
von einem vorher festgelegten Test gegen dessen eigene Vorgänger und möglichst
zusätzlich gegen eingefrorene Modelle dieses Laufs.
