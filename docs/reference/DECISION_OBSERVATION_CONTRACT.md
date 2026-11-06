# Beobachtungsvertrag: aktive Second-Edition-Policy mit Gedächtnis

Stand: **6. November 2026**. Implementiert im aktiven `DecisionEnv`, Modell und
PPO als `decision-ppo-v2`. Geltungsbereich: Grundspiel zweite Edition, drei
Spieler, Protokoll 3 / `decision-v1`. Frühere `decision-ppo-v1`-Checkpoints sind
inkompatibel. Die [Entwicklungsanleitung](../guides/DEVELOPMENT.md) beschreibt
die passenden Prüfungen für Änderungen an diesem Vertrag.

Die logischen verschachtelten Felder dieses Dokuments werden als flaches
Gymnasium-Dictionary mit Punktpfaden umgesetzt, z. B. `public.built_counts` und
`decision_history.public.built_counts`. Auch Skalare sind NumPy-Arrays mit Form
`()`. `features/decision_memory.py::field_specs()` definiert den vollständigen
Tensorvertrag; `feature_contract()` exportiert ihn samt Semantik und Schemahash.

## 1. Informationsgrenze

Eine Beobachtung gehört genau einem Sitz in genau einer Partie. Ihre Eingaben
sind ausschließlich die eigene aktuelle Hand, das eigene angebotene
Entscheidungsmenü, der sichtbare Tisch, statische veröffentlichte Regeln/Daten
und zuvor rechtmäßig beobachtete Informationen. Actor und Critic unterliegen
derselben Grenze; es gibt keinen privilegierten Trainings-Critic.

Der Adapter liest ausdrücklich freigegebene Felder aus dem Steuerungszustand.
Er kopiert niemals `DecisionClient.state` oder die rohe Eventliste ins Modell.
Nicht erlaubt sind insbesondere gegnerische Hände/Optionen, verdeckte Ablagen,
vergrabene Kartenidentitäten, Deckreihenfolge, RNG-/Engine-Seed, fremde
unbestätigte Entscheidungen und interne Produktionsallokationen. Physische IDs,
Partie-/Entscheidungs-/Request-IDs, Revisionen, Uhrzeiten und Engine-SHA bleiben
außerhalb der Modellmerkmale. Der Adapter darf IDs zur Zuordnung/Deduplizierung
verwenden, aber keine Bedeutung aus ihrer numerischen oder textuellen Ordnung lernen.

Die Identität einer eigenen unter eine Wunderstufe gelegten Karte darf über
die eigene bestätigte Aktion erinnert werden. Das enthüllt keine gegnerischen
Wunderkarten. Frühere Hände werden ausdrücklich nicht als aktuelle Gegnerhände
oder als sichere Zugehörigkeit zu einem Gegner kodiert.

## 2. Ordnung, Datentypen und Vokabulare

- Relative Plätze in allen aktuellen und historischen Feldern: `[selbst, links,
  rechts]`, mit links `(seat+2)%3`, rechts `(seat+1)%3`.
- `card_keys`: die 80 Datensatzschlüssel `(type_id, age)`, lexikografisch sortiert.
  Eine Karten-ID allein reicht nicht. Physische Kopien werden gezählt.
- `wonder_keys`: die 14 nach Engine-ID sortierten Wunderseiten. Seite A/B ist
  damit explizit unterschieden; keine skalare ID als einzige Wunderinformation.
- Ressourcen: `wood, ore, clay, stone, loom, glass, papyrus, coins`.
- Fähigkeiten: `first_color_free, first_age_card_free, last_age_card_free,
  play_seventh_card, discard_build_for_free`.
- Zahlenfelder sind endliche `float32`; Zählwerte bleiben unnormierte,
  nichtnegative ganzzahlige Werte in diesen Tensoren. Masken sind boolesch.
  Bestehende `public_table`- und Aktionsmerkmale behalten ihre v1-Skalierung.
- Jeder gepaddete Eintrag ist null und hat eine explizite Gültigkeitsmaske.
  Null darf nicht zugleich unbekannt bedeuten. Unbekanntheit hat eine eigene Maske.

## 3. Aktuelle Beobachtung

`C=80`, `W=14`, `S=4` maximale Wunderstufen. Der öffentliche Snapshot `P`
ist ein benannter Satz folgender Tensoren; diese Struktur wird auch historisch
verwendet, mit identischer Bedeutung und relativer Sitzordnung.

| Feld in `P` | Form | Bedeutung / Quelle |
| --- | --- | --- |
| `public_table` | `[3,24]` | Bestehende öffentliche Aggregate und Skalierungen aus `decision_features.py` |
| `built_counts` | `[3,C]` | Identitäten physisch gebauter öffentlicher Karten; keine Wunderkarten |
| `copied_guild_counts` | `[3,C]` | Öffentlich kopierte Gilden, getrennt von physischen Bauten; Alter 3 aus Datensatz |
| `wonder_side` | `[3,W]` | One-hot der öffentlich sichtbaren Wunderseite |
| `stage_exists`, `stage_built` | je `[3,S]` | Existenz und bereits gebaute Stufen |
| `stage_costs` | `[3,S,8]` | Kosten jeder existierenden Stufe, auch zukünftiger, aus neutralem Export |
| `stage_effects` | `[3,S,17]` | Effekte jeder Stufe gemäß untenstehender Reihenfolge |
| `ability_unlocked` | `[3,5]` | Durch bereits gebaute Stufen erworbene Fähigkeiten |
| `ability_available`, `ability_available_known` | je `[3,5]` | Öffentlich belegbare aktuelle Verfügbarkeit plus Wissensmaske |
| `choice_sources` | `[3,C+S,9]` | Gruppierte Wahlproduktionen, jeweils sieben Ressourcenbits, Menge, Handelbarkeit |
| `choice_source_mask` | `[3,C+S]` | Existenz einer solchen Quelle |

`stage_effects` enthält in fester Reihenfolge: `vp, coins, shields,
science_joker`, sieben Wahlressourcenbits, `choice_tradable` und die fünf
Fähigkeitsbits. Fehlende Stufen sind vollständig null/maskiert. Wahlmengen einer
Stufe sind im aktuellen Export eins. Ein neuer Effekt oder eine andere Menge
verlangt eine Vertragsänderung, keine stille Vereinfachung.

Bei `choice_sources` gehören die ersten C Positionen zu gebauten Kartentypen,
die letzten S zu gebauten Wunderstufen. Identische Quellen werden mit ihrer
Menge zusammengefasst; verschiedene Wahlmöglichkeiten bleiben getrennte Quellen.
Die Maske verhindert, dass Alternativen einer Quelle als gleichzeitig verfügbare
Produktion gelesen werden. Feste Produktion bleibt in den Aggregaten erhalten.
Statische Kartenmerkmale, Effekte und Kettenbeziehungen werden aus dem gepinnten
Datensatz geladen; dessen Hash und Kartenreihenfolge sind Teil des Vertrags.

`ability_unlocked` folgt sichtbaren Stufen/Regeldaten. `ability_available`
folgt nur öffentlich nachweisbarer Semantik, nicht fremden legalen Optionen.
Die vorhandene Engine-Liste `available_abilities` bedeutet bei Olympia teilweise
nur „Angebote prüfen“, nicht „jetzt nutzbar“: Farbe, Runde und gebaute Karten
müssen nach den Regeln einbezogen werden. Nicht eindeutig belegbare Verfügbarkeit
ist null mit Wissensmaske null, niemals eine erfundene sichere Aussage.
Die ersten vier Verfügbarkeiten bezeichnen die öffentliche Regelberechtigung
in einer offenen primären Auswahlphase (nicht das Vorhandensein einer passenden
legalen Handkarte). Eine aktive öffentliche Ablageberechtigung wird separat
erkannt. Bei erworbener Ablagefähigkeit ist ihre Abwesenheit außerhalb eines
aktiven Rechts unbekannt, da keine internen Grants ausgewertet werden.

Zusätzlich erhält die aktuelle Beobachtung:

| Feld | Form | Bedeutung |
| --- | --- | --- |
| `public` | `P` | Der aktuelle öffentliche Snapshot |
| `hand_counts` | `[C]` | Vollständige eigene Hand direkt aus eigenen `hand_cards` |
| `hand_visible` | Skalar | Bei einer eigenen Entscheidung wahr; leere sichtbare Hand bleibt sichtbar |
| `discard_offer_counts` | `[C]` | Sichtbare physische Karten im eigenen vollständigen Ablageangebot |
| `discard_offer_visible` | Skalar | Nur bei eigener Ablagestapelwahl wahr |
| `time` | `[14]` | Zeit-/Entscheidungskontext gemäß Abschnitt 5 |
| `discard_count` | Skalar | Öffentliche Anzahl im Ablagestapel |
| `action_feats` | `[128,97]` | Bestehende Merkmale der tatsächlich angebotenen auswählbaren Optionen |
| `action_mask` | `[128]` | Genau diese gültigen Optionen; Padding bleibt gesperrt |

Handzahlen entstehen unabhängig von den Optionen: Zahlweisen, Wunderbau und
Freimodi vervielfachen die Hand nicht. Beim Ablageangebot wird zuerst nach
physischer Instanz dedupliziert, dann nach `card_keys` gezählt. Auch sichtbare,
wegen Duplikaten nicht auswählbare Karten gehören zum Ablageangebot, aber nicht
zur Aktionsmaske. Außerhalb dieser eigenen Sonderwahl sind dessen Werte null
und die Sichtbarkeitsmaske falsch. Es wird kein vollständiger öffentlicher
Ablagestapel aus versteckten Engine-Informationen rekonstruiert.

Die alten acht `context`-Werte werden nicht unverändert weitergereicht:
Zeitalter/Runde, Entscheidungstyp, Ablageanzahl und eigene Handgröße sind durch
die neuen Felder abgedeckt. `len(waiting_for)` entfällt, weil Annahme- und
Transportfortschritt keine strategische Beobachtung sein soll.

## 4. Zwei vollständige Verläufe statt 21 starrer Zeitpositionen

**Eigene Entscheidungshistorie**, Kapazität `H=32`, chronologisch aufsteigend,
ohne Abschneiden alter Hände:

- `decision_history.public`: `[H,...]` für jedes Feld von `P`.
- `decision_history.hand_counts [H,C]`, `hand_visible [H]`.
- `decision_history.discard_offer_counts [H,C]`, `discard_offer_visible [H]`.
- `decision_history.time [H,14]`, `discard_count [H]`.
- `decision_history.applied_action_feats [H,97]`,
  `applied_card [H,C]` (One-hot), `applied_mask [H]` und `valid [H]`.

Ein Eintrag enthält den Snapshot **vor** der eigenen Entscheidung und die
zugehörige **bestätigte** eigene Aktion. Die derzeit offene Entscheidung steht
nur in der aktuellen Beobachtung. Intern wird ihr Snapshot sofort unveränderlich
vorgemerkt; erst `receipt.status == applied` übernimmt ihn samt gewählter Option
in den Verlauf für spätere Beobachtungen. Nicht angewandte Anfragen erzeugen
keinen Aktionsverlauf und keinen gültigen PPO-Übergang.

**Öffentlicher Verlauf**, Kapazität `J=64`, chronologisch aufsteigend:

- `public_history.public`: `[J,...]` für jedes Feld von `P`.
- `public_history.source_turn [J]`, `followup_seq [J]`.
- `public_history.phase_flags [J,5]`: initial, primary, discard, age_end, terminal.
- `public_history.discard_count [J]`, `valid [J]`.

Er enthält den Anfangszustand und alle anschließend beobachteten strategischen
öffentlichen Änderungen, auch in `_advance()` und vor gegnerischen Folgewahlen.
Er enthält weder fremde Hände noch fremde ausgewählte Optionen. Bauänderungen
werden aus sichtbaren Tischdifferenzen gewonnen. Bei reiner Transportänderung
entsteht kein neuer Eintrag. Der aktuelle öffentliche Snapshot darf dem letzten
Eintrag entsprechen. Nicht eigens beobachtete Zwischenzustände werden nicht
erfunden; atomar zusammen veröffentlichte Änderungen bleiben eine Gruppe.

Die Kapazitäten decken den gepinnten Regelstand konservativ ab: höchstens 21
primäre Entscheidungen je Sitz plus höchstens vier eigene Wunder-Ablagewahlen;
öffentlich höchstens 21 primäre Auflösungen, zwölf Wunder-Ablagewahlen, drei
Altersabschlüsse sowie Anfang/Ende. Bestätigungen, Polls und Wiederholungen
verbrauchen keine Kapazität. Diese Schranken sind bei der Umsetzung zu testen.
Ein Überlauf bricht mit einem aussagekräftigen Fehler ab; keine stille Kürzung,
kein Ringpuffer. Andere Regeln erfordern einen neuen Vertrag.

## 5. Zeit, Gleichzeitigkeit und Sonderentscheidungen

`time` besitzt genau diese 14 Werte, in dieser Reihenfolge:

1. Zeitalter (1–3, terminal ebenfalls 3).
2. Runde im Zeitalter (1–7, terminal 0).
3. Globaler Enginezug (0–20, terminal 21).
4. Weitergabe nach links (1 in Zeitalter 1/3, 0 in 2 und terminal).
5.–7. Primärtyp als One-hot: regular, seventh, none.
8.–11. Eigener angebotener Typ als One-hot: regular, seventh, discard, none.
12. Index des zuletzt sichtbaren öffentlichen Verlaufseintrags.
13. Beobachtete Ablage-Folge innerhalb derselben Resolution (primär 0).
14. Eigener Entscheidungsindex seit Partiebeginn (0-basiert).

Der Adapter hält die ursprüngliche Resolution und deren globalen Zug intern
fest, bevor er Anfragen sendet. Ein Ergebnis kann bereits den nächsten Zug
melden; die darin sichtbar gewordenen Bauten gehören zur aufgelösten alten
Resolution. Nachgelagerte Ablagewahlen erhalten innerhalb dieser Resolution
die Folge 1, 2, … nach tatsächlich beobachteter Spielabfolge.

Alle regulären bzw. siebten Kartenbauten einer gemeinsamen Auflösung bilden
eine **gleichzeitige Gruppe**. Sitziteration, Eventnummern und Empfangsreihenfolge
geben ihnen keine strategische Reihenfolge. Militärwertung/Zeitalterwechsel
können im selben Status wie der letzte Bau erscheinen; dafür sind mehrere
`phase_flags` wahr. Kein künstlicher Vorher-Zustand wird daraus abgeleitet.

Für die reine Zuordnung sind vorher bekannte Resolutionen und eigene Receipts
zulässig. Rohe Ereignisse mit `card_instance_id` bleiben ausgeschlossen,
insbesondere `regular_applied` und `automatic_discard`. Falls das aktuelle
Protokoll eine genaue Unterteilung nicht öffentlich belegt, bleibt die Gruppe
zusammen; ein zukünftiges öffentliches Eventformat benötigt einen eigenen Vertrag.

## 6. Lebenszyklus und Modellanbindung

- Speicher gehören zu `(env_instance, game_id, seat)`, nicht zum Modellobjekt.
  Gecachte Gegnergewichte teilen niemals einen Verlauf. Alle drei Plätze erhalten
  ausschließlich ihre jeweilige Sicht und dieselbe relative Sitzkonvention.
- Jede eigene Beobachtung ist eine unabhängige unveränderliche Kopie. Spätere
  Bestätigungen, öffentliche Änderungen, Padding und Reset ändern weder bereits
  ausgegebene Beobachtungen noch gespeicherte PPO-Rollouts.
- Bei einer neuen Partie werden beide Verläufe und vorgemerkte Entscheidungen
  geleert, auch bei Resume. Resume stellt keine laufende Engine-Partie wieder her.
- Terminale Beobachtungen enthalten den letzten sichtbaren Zustand und die
  Historie, aber keine legalen Aktionen. Das Modell wird darauf nicht zur
  Aktionswahl aufgerufen; terminaler Bootstrap ist null.
- Ein gemeinsamer Encoder verarbeitet aktuelle Hand, öffentlichen Zustand und
  beide geordneten maskierten Verläufe. Sein Output geht an **Actor und Critic**.
  Der Actor bewertet zusätzlich einzelne Aktionskandidaten. Die Kandidatenmittelung
  des Critic ersetzt keinen Handencoder.
- Erste Umsetzung: expliziter Feedforward-Verlaufsencoder mit fester Zeitordnung
  und Masken. Reine ungewichtete Mittelung der Historie reicht nicht. Es ist
  kein rekurrenter Zustand nötig. Encoderbreiten werden als Architekturkonfiguration
  gespeichert; sie ändern nicht die fachliche Sichtbarkeit.

Die implementierte Architektur `ordered-visible-memory-v1` verwendet standardmäßig
64 öffentliche/gemeinsame und 32 Hand-/Verlaufsmerkmale. Öffentlicher Encoder
und Handencoder teilen ihre Gewichte über aktuelle und historische Zustände.
Maskierte Verlaufseinträge werden vor und nach den Encodern genullt. Die
geordneten Zeilen werden zusammengeführt und projiziert; ihre Reihenfolge bleibt
damit lernbar. Zeitwerte werden erst innerhalb des Modells mit fest gespeicherten
Skalen normalisiert. Die Beobachtung selbst enthält weiterhin Rohwerte.

## 7. Versionierung und Abnahme

Implementierte Kennungen: `decision-observation-memory-v1` für diese Beobachtung,
`visible-decision-history-v1` für das Gedächtnis und `decision-ppo-v2` für neue
Checkpoints. Vor dem ersten Training speichert der implementierte Vertrag
alle Feldnamen, Reihenfolgen, Formen, dtypes, Skalierungen, Masken, Kapazitäten,
Sichtbarkeits-/Zeitregeln, Karten-/Wunder-Vokabulare sowie Daten-/Regelhashes in
kanonischem JSON mit Schemahash. Unbekannte IDs/Effekte und inkompatible
Checkpoints führen zu Fehlern. Legacy- und aktuelle v1-Gewichte werden nicht
automatisch übernommen oder durch Nullhistorie kompatibel gemacht.

Codecommit, Quelltexthashes, tatsächlich gemeldete Engine-SHA, Binaryhash, Seeds, Konfiguration
und Bibliotheksstände gehören zusätzlich in das Laufmanifest. Herkunft und
Schema-Kompatibilität sind getrennte Prüfungen; gleiche Daten garantieren keine
gleiche Simulationssemantik.
`metrics.json` und `manifest.json` protokollieren die Laufherkunft. Bei Resume
liegt der neue Episodenseed mindestens eins über allen bisher protokollierten
Trainingsseeds; ein erneut übergebener alter Startseed erzeugt keine doppelte
Trainingspartie. Evaluationsseeds stehen separat in `metrics.json`.

Vor Freigabe der Implementierung müssen Tests mindestens zeigen:

1. Vollständige Hand einschließlich unbaubarer Karten; keine Mehrfachzählung
   durch Optionen; Invarianz unter Hand-/Optionsreihenfolge nach Rückzuordnung.
2. Gleicher aktueller Zustand mit unterschiedlichen früheren Händen kann
   Actor-Logits **und** Value ändern; Gradienten erreichen den gemeinsamen
   Hand-/Verlaufsencoder. Damit ist Zugriff belegt, noch kein Spielstärkegewinn.
3. Änderungen verdeckter Gegnerhände, fremder Optionen, vergrabener Karten-IDs,
   Seeds/Transport-IDs und roher Events verändern weder Eingabe noch Ausgabe.
4. Alle drei Sitze, Olympia-Freimodi, siebte Karten und Ablagefolgen funktionieren;
   öffentliche Änderungen aus `_advance()` bleiben erhalten; gleichzeitige Bauten
   bleiben invariant gegenüber ihrer internen Abarbeitungsreihenfolge.
5. Wissensmasken, Padding und Kapazitätsgrenzen stimmen. Nicht sichtbare
   Ablageangebote sind null, sichtbare leere Hände bleiben unterscheidbar.
6. Sitz-/Partietrennung, Modellcache, Reset und Resume sind korrekt; spätere
   Aktionen ändern keine alten Beobachtungs- oder Rollout-Tensoren.
7. Echte PPO-Updates sowie exaktes Save/Load gelingen; neue, alte und abweichende
   Schema-/Daten-/Regelverträge werden kontrolliert angenommen oder abgewiesen.

Maßgebliche vorhandene Quellen: [aktive Umgebung](../../python/src/sevenwonders_rl/envs/decision_env.py),
[Features](../../python/src/sevenwonders_rl/features/decision_features.py),
[Modell/PPO](../../python/src/sevenwonders_rl/utils/decision_ppo.py),
[Engine-Status und Auflösung](../../engine/src/decision.cpp),
[neutrale Wunderdefinitionen](../../python/src/sevenwonders_rl/data/bga_wonders-v1.json).
