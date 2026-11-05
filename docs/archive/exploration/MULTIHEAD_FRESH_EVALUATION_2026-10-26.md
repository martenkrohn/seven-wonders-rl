# Frische Evaluation von v1 und v2, 26. Oktober 2026

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

**Gesamturteil:** Der Vorteil der Priors bleibt auf neuen Seeds bestehen, fällt
aber kleiner aus. Ein breiter Fortschritt gegen reine Standardgegner bestätigt
sich nicht. Die Zugewinne konzentrieren sich auf Tische mit Wissenschaftsspezialisten.
Die Kartenbauverfügbarkeit im letzten Zeitalter verbessert sich nicht.

## Vorab festgelegter Test und Prüfung

- Nur die beiden fertigen Zufallskopf-Policies, keine Einzelkopf-Evaluation und
  kein weiteres Training. Je Spiel ein uniform gezogener Kopf, stochastische Aktionen.
- Test-Seed **99001**, **32 neue Engine-Seed-Blöcke**, alle 16 geordneten Paare
  derselben vier festen Gegner und alle drei Lernersitze: **1.536 Spiele je Modell**,
  insgesamt **3.072**. Beide Modelle erhalten identische Anfangszustände, Kopf-
  zuweisungen und Gegner. Kein Seed wurde nach Betrachtung der Ergebnisse ersetzt.
- Vor den Spielen wurden 34 vorhandene Trainings-, Validierungs-, Holdout- und
  Kalibrierungsdateien geprüft; keine Überschneidung der neuen Engine-Seeds mit
  deren aufgezeichneten Seeds. Die vollständige Dateiliste steht im Manifest.
- Modell- und Gegnerdateien für diesen Versuch kopiert und per SHA-256 geprüft.
  Nach Abschluss stimmen Modellkopien und Quelldateien weiterhin überein.
- Alle Spiele abgeschlossen; vollständige End-/Zeitalterzustände und Aktionsdaten,
  keine Punktsummenfehler, unbekannten gebauten Karten oder Engine-Ersatzabwürfe.
- Beide Modelle wurden auf allen **14 eigenen Wunderseiten** ausgewertet. Das
  schafft breitere Abdeckung als der alte Acht-Block-Test, ist aber weiterhin
  keine umfassende Prüfung aller Wunder-/Karten-/Gegnerkombinationen.
- Rund **3m45s** für die beiden Evaluationsserien und ihre primäre Auswertung;
  Berichtserstellung und nachgelagerte Offline-Diagnose kommen hinzu.

## Gesamtleistung

Sieganteil berücksichtigt geteilte Siege; der Abstand ist eigene VP minus VP des
stärksten Gegners. Jeder Wert verwendet den neuen gemeinsamen Testplan.

| Policy | Sieganteil | Strikte Siege | Eigene VP | Abstand | Wissenschaft | Militär |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1, ohne Priors | 41,23% | 39,91% | 45,34 | -2,07 | 3,74 | 4,03 |
| v2, mit Priors | **44,54%** | **42,84%** | **46,27** | **-0,69** | **8,43** | 1,68 |

Gepaarter Vorteil von v2:

- **+1,38 Abstandspunkte**, 95%-Intervall **[+0,58; +2,21]**.
- **+3,31 Prozentpunkte Sieganteil**, Intervall **[+0,36; +6,22]**.

Bootstrap mit 10.000 Stichproben über ganze Engine-Seed-Blöcke, Diagnose-Seed 0.
Die Intervalle erfassen Variation zwischen diesen Spielsituationen, keine Variation
über erneutes Training oder neue zufällige Prior-Netze.

Der alte, bereits untersuchte Acht-Block-Test ergab +2,49 Abstandspunkte und
+11,94 Prozentpunkte Sieganteil. Die neue Evaluation bestätigt die Richtung des
Gesamteffekts, nicht dessen damalige Größe. Alte und neue Ergebnisse werden nicht
zu einem vermeintlich unberührten Bestätigungstest zusammengelegt.

## Gegnerabhängigkeit: zentrale Einschränkung

| Neue Spiele | Spiele/Modell | Sieganteil v1 → v2 | Abstandszuwachs v2 |
| --- | ---: | ---: | ---: |
| Standard / Standard | 384 | **33,5% → 29,4%** | **-0,02** [-1,44; +1,50] |
| Mindestens ein Wissenschaftler | 672 | **44,6% → 55,3%** | **+3,17** [+2,15; +4,19] |
| Kein Wissenschaftler | 864 | **38,6% → 36,2%** | **-0,02** [-1,24; +1,21] |

Der zuvor auf acht Blöcken beobachtete deutliche Vorteil gegen reine Standardtische
bestätigt sich nicht. Ein Nachteil dort ist ebenfalls nicht klar belegt:
Sieganteiländerung -4,08 Prozentpunkte, Intervall [-10,46; +2,43].

Die größten gesicherten Abstandszuwächse in den geordneten Untergruppen liegen bei
Standard/Wissenschaft (+4,77), Militär/Wissenschaft (+3,64), Wissenschaft/Militär
(+3,40) und Wissenschaft/Standard (+2,21). Bei Militär/Militär ist der vormals große
Gewinn auf den neuen Seeds klein und unsicher: +0,46 [-1,63; +2,49].

Diese Untergruppen sind diagnostische, teilweise nachträglich gebildete Vergleiche;
es wurde keine Korrektur für mehrere Vergleiche vorgenommen. Sie begründen eine
gezielte Folgefrage, keine allgemeine Aussage über alle strategischen Populationen.

### Gewichtung des Gegnermixes

Die gleichmäßige Abdeckung der vier Referenzen in der Evaluation gibt dem
Wissenschaftler 25% Wahrscheinlichkeit je Gegnerplatz. Im Training sind es nur
16,7%; dort sind Tische ohne Wissenschaftler entsprechend häufiger.

Mit den Trainingswahrscheinlichkeiten umgewichtet beträgt der v2-Vorteil:
**+0,98 Abstandspunkte** [+0,04; +1,95], aber nur **+0,99 Prozentpunkte Sieganteil**
[-2,72; +4,65]. Für diesen Mix ist eine höhere Sieghäufigkeit somit nicht klar belegt.
Das ist eine Offline-Umgewichtung derselben Spiele, kein weiterer Simulationslauf.

## Wissenschaft und Ressourcenzugang

Wissenschaft steigt insgesamt von 3,74 auf 8,43 Punkte, Blau bleibt mit
22,67 → 22,31 ähnlich. Militär fällt von 4,03 auf 1,68, während Münz- und
Handelspunkte ebenfalls sinken. Die bessere Wissenschaft ersetzt also teilweise
andere Punktequellen.

Ohne Wissenschaftsgegner steigen Wissenschaftspunkte sogar von 5,62 auf 11,60,
aber der Punkteabstand bleibt unverändert. Mehr Wissenschaft ist in diesem Fall
kein Beleg für eine insgesamt bessere Strategie.

Frühe graue Bauten (I+II) steigen von **98 auf 188** in 1.536 Spielen:
0,064 → **0,122 pro Partie**. Das ist ungefähr eine graue Karte je acht Partien.
Eigene Manufakturproduktion bleibt selten.

Ein plausibler Zusammenhang ist die Versorgung durch Nachbarn: Mit mindestens
einem Wissenschaftler liegen am Ende von I im Mittel rund **2,7 graue Karten bei
den beiden Nachbarn**, ohne Wissenschaftler nur rund **0,2–0,25**. Die eigene graue
Bauhäufigkeit von v2 ist mit Wissenschaftler besonders niedrig (0,030 frühe graue
Bauten pro Partie), ohne Wissenschaftler höher (0,194).

Das passt zur Hypothese, dass manche Fortschritte von günstiger Handelsversorgung
abhängen. Es beweist sie nicht: Anzahl grauer Nachbarkarten ist kein vollständiges
Maß für Kosten, tatsächlich getätigte Käufe oder alle anderen Strategieeffekte.
Ein kausaler Handelstest wurde nicht durchgeführt.

## Abwürfe und echte Baubarkeit

| Pro Partie | v1 | v2 |
| --- | ---: | ---: |
| Abwürfe I | 0,234 | **0,158** |
| Abwürfe II | 0,533 | **0,314** |
| Abwürfe III | 1,025 | **0,694** |
| Alle Abwürfe | 1,792 | **1,165** |
| III: Abwurf ohne legalen Kartenbau | 0,681 | **0,654** |

Die Gesamtabwürfe sinken um rund 35%, die Abwürfe in III um rund 32%. Der Haupt-
mechanismus entspricht dem alten Test: **Abwürfe trotz legalem Kartenbau in III
sinken von 528 auf 61**. Vorhandene Bauoptionen werden wesentlich häufiger genutzt.
Ob jede einzelne Alternative besser war, ist damit nicht entschieden.

| Tatsächlich ausgeführte Entscheidungen in III | v1 | v2 |
| --- | ---: | ---: |
| Entscheidungen insgesamt | 9.291 | 9.275 |
| Ohne legalen Kartenbau | 1.204 | 1.227 |
| Davon Abwurf | 1.046 | 1.005 |
| Davon Wunderbau | 158 | 222 |
| Anteil mit legalem Kartenbau | **87,04%** | **86,77%** |

Die Aktionszahlen unterscheiden sich leicht durch unterschiedlich genutzte
Sonderfähigkeiten, etwa zusätzliche Kartenaktionen. Nenner sind echte ausgeführte
Aktionen, keine pauschal angenommenen 21 Protokollschritte.

Es gibt **keine Verbesserung der späten Kartenbauverfügbarkeit**. Die geringere
Zahl von Abwürfen ohne Kartenbauoption entsteht bei v2 durch mehr Wunderbau in
solchen Situationen. Ein fehlender Kartenbau bedeutet nicht, dass ein Abwurf
erzwungen ist; Wunderaktionen können weiterhin möglich sein.

## Einordnung und nächster Schritt

Die Priors zeigen auf neuen Seeds einen nützlichen, aber begrenzten Gesamteffekt.
Die stärkere Aussage einer allgemeinen Verbesserung gerade gegen Standardmodelle
muss nach diesem Test zurückgenommen werden. Die Ressourcenzugangsfrage bleibt
weitgehend ungelöst.

Eine Budgetskalierung kann weiter sinnvoll sein, sollte aber als eigener Versuch
mit festen Einstellungen und Zwischenmessungen beurteilt werden. Der wichtigste
Prüfbereich sind Tische ohne Wissenschaftler sowie echte Kartenbauverfügbarkeit,
nicht nur Gesamtsieganteil oder Wissenschaftspunkte. Zuvor ist sauberes Fortsetzen
mit korrekter Behandlung von Optimizer, Zufallszuständen und laufenden Partien
vorzubereiten; die bisherige Gewichtsinitialisierung allein ist kein exaktes Resume.
Für diese Aufgabe wurden weder Resume noch weiteres Training gestartet.

## Reproduktion und Artefakte

```bash
cd python
.venv/bin/python scripts/ppo_training/eval_multihead_fresh.py
```

Der Befehl verweigert das Überschreiben des abgeschlossenen Standard-Ausgabeordners.
Für einen anderen Testplan müssen Seed und Ausgabeordner explizit geändert werden.

- `python/runs/multihead-fresh-seed99001-blocks32/dashboard.html`
- Dort auch `summary.json`, `analysis.json`, `exploration.json`, `plan.json`,
  `manifest.json`, beide Rohdatensätze und unveränderliche Modell-/Referenzkopien.
- Offline-Prüfung: `audit.py`, Ergebnis `audit.json` im selben Ordner.
- Die ergänzende Science-Gruppenanalyse liegt auch in `science-opponent-analysis.json`.
- Zusammengefasste Kopie: `docs/archive/exploration/MULTIHEAD_FRESH_EVALUATION_2026-10-26.json`.
