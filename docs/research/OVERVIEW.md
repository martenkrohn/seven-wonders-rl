# Trainingsforschung: der schnelle Überblick

Stand: **29. Oktober 2026**. Kurzfassung der
[ausführlichen Content-Bank](CONTENT_BANK.md).
Sie folgt den drei Punkten unserer
[vereinbarten Forschungsrichtung](AGENDA.md).

## 1. Was verwenden wir eigentlich?

Unsere Grundlage ist **PPO mit Actor und Critic sowie GAE**. Der Actor wählt
Aktionen, der Critic schätzt den späteren Erfolg. Beide sind schon im frühesten
verfügbaren PPO-Code vorhanden. Das ist kein nachträglich eingebautes
Explorationsmodul.

Die **vier Köpfe, zufälligen Prior-Netze und der 60/20/20-Gegnerpool** sind
zusätzliche Entscheidungen. Insbesondere sind unsere Priors auf
Aktionspräferenzen eine eigene Übertragung: Die zugrunde liegende Forschung
arbeitet mit zufälligen Wertfunktionen. Deren theoretische Ergebnisse gelten
deshalb nicht automatisch für unsere Umsetzung.

Der Reward belohnt ausschließlich den **Punkteabstand zum stärksten Gegner am
Spielende**. Das ist sinnvoll definierbar, entspricht aber nicht genau dem
Maximieren der Siegchance. Diese Zielentscheidung sollten wir bewusst treffen.

**Der neue Gedächtnismodus adressiert zwei Informationslücken:** Er stellt die
vollständige eigene Hand und rechtmäßig zuvor gesehene Hände beiden Netzteilen
bereit. Im Dreierspiel kehrt dasselbe Kartenpaket in Runde 4 zurück; die
Erinnerung hilft außerdem, mögliche gegnerische Entscheidungen einzuschätzen.
Die Erweiterung ist im Trainer standardmäßig aktiv;
`--no-observation-memory` erlaubt Vergleichsläufe ohne Gedächtnis. Ihr Einfluss
auf die Spielstärke bleibt zu prüfen; die bisherigen Ergebnisse betreffen
Modelle ohne diese Erweiterung.
[Bedienung und Informationsgrenzen](../guides/OBSERVATION_MEMORY.md).

→ [Grundmethode und Herkunft](CONTENT_BANK.md#actor-critic)
· [Reward](CONTENT_BANK.md#reward)
· [Implementierungsprüfung](CONTENT_BANK.md#implementierungspruefung)
· [Prioritäten für die nächste Phase](CONTENT_BANK.md#prioritaeten)

## 2. Wie gehen wir Exploration und Gegnerauswahl an?

Zunächst müssen wir verschiedene mögliche Ursachen auseinanderhalten:

- **Nicht entdeckt:** Das Modell versucht eine vollständige Investitionsfolge kaum.
- **Schlecht gelernt:** Die Folge kommt vor, ihr späterer Erfolg wird aber frühen Entscheidungen schlecht zugerechnet.
- **Information fehlt:** Der Spieler könnte etwas wissen oder erinnern, das sein Netz nicht verarbeitet.
- **Gelegenheit fehlt:** Die Strategie lohnt sich gegen die gezogenen Gegner selten.

Mehr Zufälligkeit oder mehr Köpfe lösen diese Probleme nicht gleichermaßen.
Die Literatur bietet dafür unterschiedliche Richtungen: zusammenhängende
Variation über ganze Partien, Neuheitsanreize, unterscheidbare Skills,
Rückkehr zu interessanten Zuständen und bessere Zuordnung späterer Rewards.
**Wir sollten nach dem Engpass auswählen, nicht möglichst viele Module ergänzen.**

Bei den Gegnern ist weder „alt“ noch „stark“ allein das richtige Auswahlkriterium.
Unser Archiv streut nach Alter, nicht nach gemessenem strategischem Nutzen.
Dass die festen Anker bereits wertlos wären, ist durch die aktuellen Daten
ebenfalls nicht belegt.

Zwei Forschungsrichtungen verdienen einen genaueren Vergleich:

- **Leistungsabhängige Priorisierung, etwa PFSP:** Gegner nach gemessener
  Schwierigkeit und Vergessen gewichten, statt feste Gruppenanteile immer neu
  von Hand festzulegen.
- **Empirische Spiele und Populationen, insbesondere JPSRO:** Vorhandene
  Strategien gemeinsam auswerten und gezielt neue Antworten lernen. Dieser
  Mehrspieler-Bezug ist wichtig; er verlangt aber mehr als unseren heutigen
  Snapshot-Pool.

Bei uns muss stets das **Gegnerpaar** betrachtet werden. Ob eine Strategie
gegen B funktioniert, hängt auch von C ab. Eine Populationslösung ist außerdem
nicht automatisch die beste einzelne Policy für beliebige fremde Gegner.

**Inzwischen beschlossen und eingerichtet:** Ein frischer Lerner mit einer
Liga ausschließlich eigener eingefrorener Zwischenstände. Geordnete
Gegnerpaare werden anhand diskontierter Trainings-Sieganteile priorisiert;
gleichverteilte Kontrollauswahlen erhalten die Grundabdeckung. Historische
Endmodelle dienen nur der separaten Evaluation. Das ist eine PFSP-inspirierte
Ausgangsvariante, kein JPSRO und noch kein Nachweis einer Methodenverbesserung.
[Start und genaue Regel](../guides/PAIR_PFSP_TRAINING.md) ·
[Versuchsprotokoll](../experiments/001-fresh-pair-pfsp/PROTOCOL.md).

→ [Explorationsdiagnose](CONTENT_BANK.md#exploration)
· [Methodenvergleich](CONTENT_BANK.md#explorationsmethoden)
· [Gegnerpaare](CONTENT_BANK.md#gegnerpaare)
· [Populationsverfahren](CONTENT_BANK.md#populationen)

## 3. Was können wir anschließend belastbar behaupten?

Die bisherigen Ergebnisse zeigen, dass unser Modell Ressourcenplanung und
Wissenschaft lernen konnte. Der erste lange Lauf verbesserte sich auch in
einer frischen Prüfung. Der zusätzliche Gesamtfortschritt des zweiten Laufs
ist bisher statistisch unsicher. Offen ist vor allem, wie breit,
situationsabhängig und reproduzierbar die Fähigkeiten sind.

Ein überzeugender Nachweis braucht eine festgelegte Hauptmetrik, passende
Vergleichsmodelle, Ergebnisse nach Gegnerkonstellation und eine abschließende
Prüfung außerhalb des laufend betrachteten Dashboardpanels. Neue Spielseeds
ersetzen weder neue Gegnertypen noch unabhängige Trainingswiederholungen.

**Die nächste Entscheidung sollte deshalb aus einer klaren Prüffrage folgen:**
Welcher Engpass ist plausibel, welche einzelne Änderung adressiert ihn, und
welches Ergebnis würde dafür oder dagegen sprechen? Anschließend passt ein
Versuch nach dem anderen weiterhin gut zu unserem Vorgehen.

Eine wissenschaftlich begründete und sauber getestete Verbesserung können
wir verteidigen. „Bestmöglicher Spieler“ oder „alle wichtigen Strategien
garantiert entdeckt“ können wir daraus nicht ableiten.

→ [Bisherige Befunde](CONTENT_BANK.md#befunde)
· [Belastbare Evaluation](CONTENT_BANK.md#evaluation)
· [Nächste Klärungen](CONTENT_BANK.md#naechste-klaerungen)
· [Primärquellen und Lesetiefe](CONTENT_BANK.md#quellen)
