> Historical literature review and implementation audit, 29 October 2026.
> Implementation claims describe the retired protocol-2 system. Use
> [the current architecture](../reference/ARCHITECTURE.md) for today's code.
> Old source and experiment links point to an immutable archived commit.

# Wissenschaftliche Grundlage unseres 7-Wonders-Trainings: Content-Bank

Stand und letzter Quellenabruf: **29. Oktober 2026**. Ausgangspunkt sind die drei
Fragen in der [vereinbarten Forschungsrichtung](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/research/AGENDA.md).
Die [Kurzfassung](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/research/OVERVIEW.md) bietet den Einstieg.

Dieses Dokument verbindet Primärliteratur, Prüfung unseres Codes und vorhandene
Ergebnisse. Es begründet noch keine einzelne neue Trainingskonfiguration.

**Implementierungsnachtrag:** Der Codeaudit beschreibt den Stand vor dem
Gedächtnisumbau. Inzwischen sind die vollständige eigene Hand und der
Beobachtungsverlauf als gemeinsamer Kontext für Actor und Critic
implementiert und im Trainer standardmäßig aktiv. [Aktuelle Bedienung und Grenzen](../reference/DECISION_OBSERVATION_CONTRACT.md).
Die historischen Lernbefunde bleiben unverändert; ein Leistungsgewinn durch
die Erweiterung wurde damit noch nicht nachgewiesen.

**Nachtrag zur Gegnerentscheidung:** Inzwischen ist eine PFSP-inspirierte
Auswahl geordneter Gegnerpaare implementiert. Der nächste Lauf beginnt mit
zufälligen Gewichten und ausschließlich eigenen eingefrorenen Zwischenständen
als Trainingsgegnern. Historische Modelle dienen nur als Evaluationsreferenzen.
[Konkrete Regel](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/guides/PAIR_PFSP_TRAINING.md) und
[Protokoll](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/experiments/001-fresh-pair-pfsp/PROTOCOL.md).
Die folgenden Beschreibungen des 60/20/20-Pools bleiben der historische Audit.

## Inhalt

1. [Rechercheauftrag, Vorgehen und Belegstärke](#recherche)
2. [Unser tatsächliches Verfahren und seine Herkunft](#verfahren)
3. [Was unsere bisherigen Ergebnisse aussagen](#befunde)
4. [Deep Exploration: Problem und heutiger Mechanismus](#exploration)
5. [Weitere Forschungsrichtungen für Exploration und Lernfeedback](#explorationsmethoden)
6. [Gegnerwahl: Stärke, Vielfalt und drei Spieler](#gegner)
7. [Populationen aus der Forschung: PFSP, PSRO und JPSRO](#populationen)
8. [Direkte Spielforschung und andere große Spiel-KIs](#spielforschung)
9. [Was eine belastbare Evaluation benötigt](#evaluation)
10. [Entscheidungsgrundlage und offene Fragen](#entscheidungen)
11. [Kommentiertes Quellenverzeichnis und Lesetiefe](#quellen)
12. [Codebelege, Datenbelege und Begriffserklärungen](#projektbelege)

<a id="recherche"></a>
## 1. Rechercheauftrag, Vorgehen und Belegstärke

### 1.1 Die drei Leitfragen

**Bestehendes Verfahren einordnen:** Was ist etablierte Grundlage, was wurde
später ergänzt, und entspricht unsere Umsetzung den Arbeiten, auf die wir uns
beziehen?

**Exploration und Gegnerauswahl verstehen:** Welche Mechanismen erschließen
langfristige Investitionen? Welche Gegnerpopulationen erhalten strategische
Gelegenheiten, ohne dauerhaft bedeutungslose Gegner zu bevorzugen? Welche
Ergebnisse gelten auch für drei Spieler mit verdeckten Informationen?

**Erreichbare Aussage festlegen:** Wie können wir stärkere und anpassungsfähigere
Spieler nachweisen? Wie weit tragen unsere Tests, und welche Aussagen über
Optimalität bleiben unzulässig?

Die Frage nach erwarteten Gewinnen durch mehr Rechenleistung ist entsprechend
deiner Überarbeitung ausgenommen. Methodische Voraussetzungen und Unterschiede
im benötigten Systemaufbau werden trotzdem genannt, damit sich die Ansätze
einordnen lassen.

### 1.2 Vorgehen und Grenzen der Recherche

Die Recherche ging vom vorhandenen Code zu Originalarbeiten und von deren
Literaturverweisen zu verwandten Ansätzen. Gesucht wurde unter anderem nach:

- `PPO GAE implementation details invalid action masking`;
- `bootstrapped DQN randomized prior functions deep exploration`;
- `deep exploration delayed reward credit assignment RUDDER`;
- `prioritized fictitious self play AlphaStar opponent population`;
- `PSRO JPSRO multiplayer general sum correlated equilibrium NeuPL`;
- `reinforcement learning evaluation statistical uncertainty random seeds`;
- `7 Wonders reinforcement learning Monte Carlo tree search`, einschließlich
  der gesonderten Variante **7 Wonders Duel**;
- jüngeren PSRO-Arbeiten und Veröffentlichungsnachweisen bis zum Abrufdatum.

Technische Behauptungen stützen sich auf Originalarbeiten beziehungsweise deren
offizielle Publikationsseiten. Ein PSRO-Survey diente als Wegweiser. Bei wichtigen
Methoden wurden die relevanten Algorithmusabschnitte und Voraussetzungen gelesen;
bei ergänzenden Arbeiten teilweise nur Abstract und Einleitung. Das
[Quellenverzeichnis](#quellen) macht diese Unterschiede sichtbar. Bei AlphaStar
war der Primärtext über einen PDF-Spiegel zugänglich; zitiert wird auch das
Original beim Verlag.

Das ist eine gezielte, nachvollziehbare Recherche, keine registrierte
systematische Übersicht mit Vollständigkeitsanspruch. Insbesondere ist eine
nicht gefundene Veröffentlichung kein Beweis, dass keine existiert. Ergebnisse
aus Abstracts werden nicht als unabhängige Prüfung aller Experimente ausgegeben.

### 1.3 Vier Arten von Aussagen

| Kennzeichnung | Bedeutung | Was daraus nicht folgt |
| --- | --- | --- |
| **Codebefund** | In den untersuchten Dateien tatsächlich vorhanden | Dass die betreffende Wahl die gemessene Stärke verursacht |
| **Projektbefund** | In vorhandenen Laufdaten beziehungsweise Auswertungen beobachtet | Allgemeine Reproduzierbarkeit über unabhängige Trainingsläufe |
| **Literaturbefund** | Ergebnis oder Mechanismus einer Originalarbeit | Unveränderte Gültigkeit in unserem Spiel |
| **Übertragung / Hypothese** | Eigene Folgerung für dieses Projekt | Bereits experimentell nachgewiesene Wirkung |

Theoretische Garantien werden zusätzlich mit ihren Voraussetzungen genannt.
Eine empirisch erfolgreiche Spiel-KI und ein Konvergenzsatz beantworten
unterschiedliche Fragen.

<a id="verfahren"></a>
## 2. Unser tatsächliches Verfahren und seine Herkunft

Dieses Kapitel beschreibt die bisher trainierten Modelle und den Code vor
Aktivierung des neuen, inzwischen standardmäßig aktiven Gedächtnismodus. Dessen technische Umsetzung
steht im [aktuellen Gedächtnisleitfaden](../reference/DECISION_OBSERVATION_CONTRACT.md).

<a id="actor-critic"></a>
### 2.1 Actor und Critic gehören zur vorhandenen Grundlage

Der **Actor** bewertet die verfügbaren Aktionen und erzeugt daraus
Auswahlwahrscheinlichkeiten. Der **Critic** schätzt den noch zu erwartenden
Trainingsreward. Beim Lernen wird unter anderem verglichen, ob eine Erfahrung
besser oder schlechter war als diese Erwartung.

Actor und Critic sind bereits im frühesten hier verfügbaren Repository-Stand
der PPO-Modelldatei vorhanden: Commit `cd2d863` vom 18. Oktober 2026 enthält
`ActorCritic`, einen Actor-Ausgang und `value_head`. Der Critic ist somit kein
später hinzugefügtes Explorationsmodul. Über eine Vorgeschichte außerhalb dieser
Git-Historie lässt sich daraus nichts sagen.

Unsere Grundmethode ist **PPO mit Actor-Critic und GAE**. PPO verwendet eine
begrenzte Surrogat-Zielfunktion für Policy-Updates. GAE berechnet ein geglättetes
Signal dafür, welche Aktionen besser als erwartet waren. Das sind etablierte
Bausteine; mehrere Köpfe, Prior-Netze und unser konkreter Gegnerpool sind
zusätzliche Entscheidungen. PPO-Clipping garantiert dabei weder eine feste
Obergrenze jeder tatsächlichen Policy-Änderung noch einen global optimalen
Spieler. [PPO, §§3–5](https://arxiv.org/abs/1707.06347),
[GAE](https://arxiv.org/abs/1506.02438).

### 2.2 Welche Aufgabe unser Modell tatsächlich löst

Es spielt im vorhandenen Simulator ein Dreierspiel mit zufälligen Austeilungen,
Wundern und verdeckten gegnerischen Handkarten. Kartenauswahl, Handel und
Wissenschaftskonkurrenz verbinden die Entscheidungen der drei Plätze.

Wichtige Grenzen des aktuellen Lerninterfaces:

- Das Modell ist ein **Feedforward-Netz**. Es erhält die aktuelle Beobachtung,
  aber keinen rekurrenten Gedächtniszustand über zuvor gesehene Hände.
- Öffentliche gebaute Karten, Wunderidentitäten und Stufeninformationen sind
  inzwischen enthalten. Das ersetzt nicht die Erinnerung an weitergereichte
  Karten oder eine vollständige Wahrscheinlichkeitsverteilung über verdeckte
  Hände.
- Die Aktionsschnittstelle hat 28 Positionen für vier Aktionstypen und bis zu
  sieben Kartenslots. Die Engine entscheidet Handelsdetails; eine freie Wahl
  jeder möglichen Ressourcen-Bezahlkombination ist keine eigene Policy-Aktion.
- Besondere zusätzliche Auswahlen, etwa eine freie Kartenauswahl aus dem
  Ablagestapel beim entsprechenden Wundereffekt, sind über `extra` derzeit
  nicht als vollständiger zusätzlicher Entscheidungsraum repräsentiert.

**Konsequenz:** Aussagen betreffen zunächst diese Simulator- und
Aktionsschnittstelle. Eine vollständige Regeltreue gegenüber jeder Edition und
jeder Sonderfähigkeit wurde mit dieser Recherche nicht neu zertifiziert.
Mehr Training oder ein anderer Gegnerpool können eine fehlende
Entscheidungsmöglichkeit nicht ersetzen. Codebelege stehen in [Kapitel 12](#projektbelege).

<a id="reward"></a>
### 2.3 Reward: Punkteabstand, keine ausdrückliche Strategiebelohnung

In den beiden langen Generalistenläufen gilt ausschließlich am regulären
Spielende:

```text
Reward = (eigene Siegpunkte − Punkte des stärksten Gegners) / 10
```

Frühe graue Karten, Wissenschaft oder Militär erhalten keinen eigenen Bonus.
Die Spezialisten hatten andere Zielprofile; diese sind nicht das aktive Ziel
der langen Generalistenläufe. Es gibt ebenfalls keinen zusätzlichen Neuheitsreward
durch die aktuellen Prior-Netze.

Dieses Ziel maximiert idealisiert den **erwarteten Punkteabstand**, nicht
unmittelbar die Siegchance. Ein Gedankenbeispiel verdeutlicht den Unterschied:

| Wahl | Mögliche Punkteabstände | Siegchance | Erwarteter Abstand |
| --- | --- | ---: | ---: |
| A | Immer +1 | 100% | +1 |
| B | Mit 60% +10, sonst −1 | 60% | +5,6 |

Der aktuelle Reward bevorzugt B. Das ist kein Nachweis eines Fehlers, sondern
eine Zielentscheidung, die bei der Evaluation offengelegt werden muss.
Sieganteil und Abstand sollten deshalb gemeinsam ausgewiesen werden.

Auch die Summe der drei Abstandswerte ist nicht konstant: Punkte `(60,50,40)`
ergeben Abstände `(10,−10,−20)`, Punkte `(60,50,50)` dagegen `(10,−10,−10)`.
Unser Reward definiert daher kein Zwei-Spieler-Nullsummenspiel. Selbst ein
Sieganteil-Reward mit konstanter Gesamtsumme würde aus drei Spielern nicht zwei
machen. Das ist für die Übertragung von Gleichgewichtsgarantien entscheidend.

### 2.4 Konfiguration des zuletzt abgeschlossenen Laufs

Code und Startskript ergeben folgenden Stand:

| Bestandteil | Umsetzung |
| --- | --- |
| Policy | Vier Actor-Köpfe und vier Value-Köpfe mit gemeinsamem Encoder |
| Kopfwahl | Gleichverteilt zu Partiebeginn, Kopf-ID bleibt innerhalb der Partie bestehen |
| Aktionswahl | Stochastisch aus der maskierten Policy |
| Zufällige Priors | Pro Kopf eigenes eingefrorenes Netz, Addition auf Actor-Logits, Stärke 32 |
| Verarbeitete Merkmale | 645 globale Merkmale, 121 Merkmale je Aktionsposition |
| Eingaben der erhaltenen Priors | Alter Merkmalspräfix: 102 global + 99 aktionsbezogen |
| Lernverfahren | PPO; Clipping 0,2; vier Minibatch-Epochen; Minibatchgröße 256 |
| Optimierung | Adam, Lernrate 0,0003; Value-Gewicht 0,5; Entropiegewicht 0,01 |
| Rückmeldung | GAE mit `gamma=0.999`, `lambda=0.95` |
| Datensammlung | Acht Umgebungen × Horizont 64 = 512 Übergänge je regulärem Update |
| Gegner pro Platz | 60% jüngere Snapshots, 20% zeitlich gestreutes Archiv, 20% feste Anker |
| Kontrollpanel | Unverändert je Lauf, Evaluation bei 0/25/50/75/100% |

Die Kopf-ID bleibt konstant, die Gewichte des Lerners können zwischen
Rolloutabschnitten einer laufenden Partie aktualisiert werden. Das ist keine
vollständig eingefrorene Policy pro Trainingspartie. Gegner-Checkpoints bleiben
dagegen eingefroren.

Die heute geteilten Encoder ermöglichen Transfer zwischen Köpfen. Sie koppeln
aber auch die Köpfe: Ein Update eines Kopfes kann durch gemeinsame Merkmale das
Verhalten anderer Köpfe verändern. Vier Köpfe sind somit weder vier unabhängige
Trainingsläufe noch vier unabhängig gelernte vollständige Netze.

<a id="implementierungspruefung"></a>
### 2.5 Abgleich mit den wissenschaftlichen Bausteinen

| Befund im Code | Einordnung und praktische Bedeutung |
| --- | --- |
| PPO-Verhältnis verwendet gespeicherte Log-Wahrscheinlichkeiten des ausgewählten Kopfes einschließlich Prior und Aktionsmaske | Konsistenz zwischen Verhalten bei der Datensammlung und verwendeter Policy-Wahrscheinlichkeit ist gegeben |
| GAE rechnet zunächst getrennt nach Zeit und Umgebung; Endzustände stoppen das Bootstrapping | Verhindert die offensichtliche Vermischung verschiedener Partien in der Rückrechnung |
| Critic erhält nur den globalen Vektor | Eigene aktuelle Hand, Aktionsmerkmale und Aktionsmaske gehen nicht direkt in seinen Value-Ausgang ein |
| Actor erhält globale und aktionsbezogene Merkmale | Actor kann Informationen verwenden, die dem Critic fehlen |
| Priors wirken nur auf Actor-Logits | Kein Ensemble zufällig perturbierter Q-Funktionen mit eigenen Bellman-Zielen |
| Kein explizites Replay-Bootstrap pro Kopf | Die unterschiedlichen Erfahrungen entstehen durch Kopfwahl und Spielverläufe |
| Kein endliches Gradientennorm-Limit | `clip_grad_norm_(..., inf)` prüft die Norm auf endliche Werte, begrenzt sie aber nicht auf einen endlichen Schwellenwert |
| Kein KL-Abbruch und keine geklippte Value-Loss | Zulässige Implementierungsentscheidungen, keine vollständige Übernahme jeder verbreiteten PPO-Rezeptur |

**Hypothese zur Critic-Eingabe:** Dieselbe öffentliche Auslage kann mit einer
guten oder schlechten eigenen Hand ganz unterschiedliche Aussichten haben.
Ein weniger informierter Critic muss diese Fälle mitteln. Das kann das
Lernsignal unpräziser machen. Daraus folgt weder, dass der Policy-Gradient
automatisch ungültig ist, noch dass diese Einschränkung den bisherigen
Strategieverlauf verursacht hat. Es ist eine separat prüfbare Informations- und
Value-Frage.

Für die Aktionsmaskierung gibt es eine konkrete theoretische Grundlage:
zustandsabhängiges Ausblenden ungültiger Aktionen ist unter den untersuchten
Voraussetzungen mit Policy-Gradient-Lernen vereinbar. Das hilft nur bei
korrekter und konsistenter Maske; eine Maske beweist nicht die Korrektheit der
darunterliegenden Spielregeln.
[Huang und Ontañón, Invalid Action Masking](https://arxiv.org/abs/2006.14171).

Eine große empirische Studie zu On-Policy-RL zeigt, dass Implementierungsdetails
erhebliche Auswirkungen haben können. Untersucht wurden jedoch fünf
Continuous-Control-Aufgaben. Ihre günstigsten Einstellungen sind deshalb keine
nachgewiesenen 7-Wonders-Standards. Sinnvoll ist der Abgleich mit einer sauber
dokumentierten Referenzimplementierung, nicht das wahllose Kopieren aller
Defaults. [Andrychowicz et al., What Matters in On-Policy RL](https://arxiv.org/abs/2006.05990).

<a id="befunde"></a>
## 3. Was unsere bisherigen Ergebnisse aussagen

Diese Zahlen stammen aus bestehenden Auswertungen. Für die Recherche wurden
keine neuen Trainingspartien oder Evaluationsläufe gestartet.

| Vorhandener Befund | Tragfähige Aussage | Nicht dadurch geklärt |
| --- | --- | --- |
| Erster langer Lauf, frische Referenzprüfung: Ende gegen Start +3,80 Abstandspunkte, 95%-Intervall [+2,74; +4,78] | Der erreichte Spieler war unter diesem Prüfplan besser als sein Ausgangsstand | Welcher einzelne Bestandteil die Verbesserung verursachte |
| Dabei Bauverfügbarkeit in III von 81,26% auf 93,15%; Abwürfe von 1,018 auf 0,378 je Partie | Die anfangs vermutete schlechte Ressourcenversorgung wurde deutlich verbessert | Dass jede Ressourceninvestition optimal oder die Strategiebreite ausreichend ist |
| Zweiter langer Lauf: +1,14 Abstandspunkte [−0,72; +2,98] gegen Start | Positive Punktschätzung, zusätzlicher Gesamtfortschritt noch unsicher | Gesicherte Verbesserung, gesichertes Plateau oder gesicherter Rückschritt |
| Zweiter Lauf: Wissenschaftspunkte 16,29 → 21,44; blaue Punkte 9,18 → 15,41 | Die gelernte Punkteverteilung hat sich verändert | Ob die Policy passend zwischen Strategien umschaltet |
| Geteilte Köpfe und anhaltende Prior-Funktionen | Mechanismen für Verhaltensvariation sind vorhanden | Unabhängige Strategien oder garantierte Entdeckung langfristiger Pläne |

Quellen und vollständige Prüfpläne:
[frische Prüfung des ersten langen Laufs](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/archive/long-training/LONG_FRESH_RESULTS_2026-10-26.md),
[zweiter langer Lauf](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/archive/long-training/LONG_TRAINING_V2_RESULTS_2026-10-29.md).

Die beiden langen Phasen sind eine **zusammenhängende Warmstart-Lerntrajektorie**.
Der zweite Lauf beginnt mit den Endgewichten des ersten. Unterschiedliche
Phasenseeds machen daraus keine unabhängigen Wiederholungen desselben
Experiments. Ebenso waren Informationsumfang, Laufdauer und Gegnerwahl in
früheren Schritten nicht immer einzeln variiert worden.

Insbesondere sollte unsere heutige Frage nicht mehr lauten: „Warum lernt das
Modell grundsätzlich keine Ressourcen oder Wissenschaft?“ Vorhandene Daten
zeigen, dass es beides lernen konnte. Offen sind **Reproduzierbarkeit,
Kontextabhängigkeit, unentdeckte Alternativen und Robustheit gegenüber anderen
Gegnern**.

<a id="exploration"></a>
## 4. Deep Exploration: Problem und heutiger Mechanismus

### 4.1 Vier verschiedene Engpässe

| Engpass | Beispiel im Spiel | Was allein nicht genügt |
| --- | --- | --- |
| Exploration | Mehrere vorbereitende Ressourcenentscheidungen werden nie gemeinsam versucht | Etwas höhere Zufälligkeit jeder Einzelaktion |
| Zeitliche Zuordnung des Erfolgs | Der erfolgreiche spätere Bau wird den frühen Investitionen kaum zugerechnet | Dieselbe erfolgreiche Sequenz bloß einmal zu besuchen |
| Information / Gedächtnis | Eine zuvor gesehene und weitergereichte Hand wird vergessen | Mehr Köpfe über derselben unvollständigen Eingabe |
| Strategische Gelegenheit | Zwei Gegner nehmen bereits die relevanten Wissenschaftskarten weg | Beliebig viele weitere Versuche derselben Strategie in dieser Konstellation |

Diese Trennung ist eine Diagnosehilfe für unser Projekt. Ein schlechter
Endreward sagt nicht, welcher Engpass vorliegt. Das ist besonders relevant,
weil Wissenschaft von Ressourcen, Kartenausteilung und Gegnerentscheidungen
gleichzeitig abhängt.

### 4.2 Warum zusammenhängendes Ausprobieren etwas anderes ist

In einem vereinfachten Beispiel braucht eine neue Strategie sechs passende
Vorentscheidungen. Würde jede nur mit Wahrscheinlichkeit 1/2 getroffen und
wären sie unabhängig, läge die Wahrscheinlichkeit der ganzen Folge bei 1/64.
Das ist eine Illustration, keine gemessene Erfolgswahrscheinlichkeit unseres
Spiels. Ein ganzer Plan kann außerdem auf mehreren Wegen funktionieren.

Deep Exploration meint hier: Unsicherheit so in Verhalten übersetzen, dass
über mehrere Entscheidungen hinweg zusammenhängend andere Möglichkeiten
erkundet werden. Eine Policy, die bei jeder Entscheidung ungerichtet würfelt,
kann denselben kurzfristig attraktiven Bereich sehr häufig besuchen und
trotzdem keine tragfähige Investitionsfolge ausprobieren.

Die bsuite-Arbeit bietet dafür kleine Diagnoseaufgaben, unter anderem
**Deep Sea**, und trennt Exploration von Gedächtnis und Credit Assignment.
Solche Aufgaben prüfen einen Mechanismus kontrollierter als ein großes Spiel,
ersetzen aber dessen eigentlichen Leistungsnachweis nicht.
[Osband et al., bsuite](https://arxiv.org/abs/1908.03568).

<a id="bootstrapped-dqn"></a>
### 4.3 Bootstrapped DQN: der ursprüngliche Bezugspunkt

Bootstrapped DQN lernt mehrere Q-Funktionen. Ein Kopf wird zu Episodenbeginn
gewählt und steuert das Verhalten über die Episode. Die Köpfe besitzen eigene
Lernziele; Bootstrap-Masken beziehungsweise unterschiedliche Datengewichte
sind Bestandteile des Ansatzes, wobei auch eine Variante mit vollständiger
Datenüberlappung untersucht wurde. Die Arbeit zeigt Vorteile bei zeitlich
ausgedehnter Exploration in ihren Testumgebungen.
[Osband et al., 2016](https://arxiv.org/abs/1602.04621).

**Übertragung auf uns:** Die Auswahl eines Kopfes pro Partie übernimmt das
Prinzip zeitlich beständiger Variation. PPO ist aber kein Q-Lernverfahren:
Unsere Köpfe lernen stochastische Policies aus eigenen On-Policy-Erfahrungen,
ohne ein solches Replay- und Q-Zielsystem. Wir haben damit nicht unverändert
Bootstrapped DQN implementiert und können dessen Befunde nicht als direkten
Nachweis unserer vier Actor-Köpfe verwenden.

<a id="randomized-priors"></a>
### 4.4 Randomized Prior Functions: was gleich ist und was abweicht

Die Originalarbeit addiert feste zufällige Funktionen zu lernbaren
Wertfunktionen. Ihr Deep-RL-Algorithmus verwendet diese Kombination auch bei
den TD-Zielen und perturbiert die Erfahrungen der Ensemblemitglieder. Die
Arbeit begründet das Prinzip unter anderem in vereinfachten statistischen
Modellen und untersucht es empirisch; eine beliebige neuronale Umsetzung
erbt dadurch keine allgemeine Garantie.
[Osband, Aslanides und Cassirer, 2018](https://arxiv.org/abs/1806.03335).

Unser Code berechnet dagegen je Kopf und Aktion:

```text
Logit_h(o,a) = lernbarer Actor_h(o,a) + 32 × fester Prior_h(o,a)
Policy_h      = Softmax über die legalen Aktionen
```

**Codebefund:** Die Priors sind eingefroren, verwenden rohe Merkmalspräfixe und
bleiben bei der späteren Erweiterung öffentlicher Informationen unverändert.
Neue öffentliche Merkmale beeinflussen den lernbaren Teil, nicht die erhaltene
Prior-Funktion. Es gibt keinen Prior im Value-Ausgang und kein Bootstrap-Replay.

**Einordnung:** Das ist eine plausible eigene Übertragung auf Policy-Logits.
Die Prior-Funktion kann einem Kopf andere Präferenzen geben. Ein festes
Zufallsnetz enthält aber keinen ausdrücklich formulierten Plan „Baue früh
Glas, danach Wissenschaft“. Seine Präferenzen müssen im relevanten
Zustandsverlauf erst zusammenpassen. Der lernbare Actor kann sie zudem
kompensieren. Die gewählte Stärke 32 ist eine Projekteinstellung, kein
allgemeiner Literaturstandard.

Eine gleichverteilte Kopfwahl ist bei uns auch kein nachgewiesenes Sampling
aus einer korrekt kalibrierten Bayesianischen Verteilung möglicher optimaler
Policies. Dafür fehlen sowohl ein entsprechender Nachweis als auch eine
Messung der Unsicherheitskalibrierung.

Die ausführliche Arbeit zu randomisierten Wertfunktionen entwickelt Garantien
unter expliziten Modellannahmen, darunter endliche tabellarische Probleme.
Das begründet eine Forschungsrichtung, keine Garantie für unser neuronales
Mehrspielersystem. Hier wurde der Abstract mit seinen beanspruchten Grenzen
geprüft, nicht der gesamte lange Beweisapparat.
[Osband et al., JMLR 2019](https://www.jmlr.org/papers/v20/18-339.html).

<a id="credit-assignment"></a>
### 4.5 Unser Feedback reicht zurück, aber nicht fehlerfrei

Im aktuellen Code gilt vereinfacht:

```text
delta_t = r_t + gamma × V(o_{t+1}) − V(o_t)
Advantage_t = delta_t + (gamma×lambda)delta_{t+1} + …
```

Mit unseren Werten ist `gamma^20 ≈ 0,980`, aber
`(gamma×lambda)^20 ≈ 0,352`. Die erste Zahl betrifft die zeitliche Diskontierung,
die zweite das Gewicht eines bestimmten weiter entfernten TD-Residuums in
dieser Rückrechnung. **Sie bedeutet nicht**, dass nur 35% eines Enderfolgs
gelernt werden oder dass die Strategie mit 35% Wahrscheinlichkeit entdeckt
wird. Der Critic transportiert ebenfalls Vorhersagen über spätere Erträge.

GAE regelt einen Bias-Varianz-Kompromiss. Schlechte Value-Vorhersagen können
bei Bootstrapping relevant werden. Daraus folgt kein pauschales „Lambda auf
eins setzen“: Auch stärker schwankende Signale können Lernen erschweren.
[GAE, insbesondere Herleitung des Schätzers](https://arxiv.org/abs/1506.02438).

**Projektbezogene Prüfidee:** Zuerst feststellen, ob erfolgreiche langfristige
Verläufe bereits vorkommen, und wie der Critic vergleichbare frühe Situationen
einschätzt. Kommen die Verläufe vor, werden aber nicht stabil gelernt, ist die
Zuordnung des Erfolgs mindestens ebenso plausibel wie reine Nichtentdeckung.
Beobachtungsdaten allein trennen die Ursachen allerdings nicht abschließend.

<a id="explorationsmethoden"></a>
## 5. Weitere Forschungsrichtungen für Exploration und Lernfeedback

Die folgenden Richtungen erfüllen verschiedene Aufgaben. Sie sind eine
Auswahl zur Beantwortung unserer Fragen, keine Liste gleichzeitig einzubauender
Module.

### 5.1 Neuheitsbonus: Random Network Distillation

Bei RND erzeugt ein eingefrorenes Zufallsnetz Zielausgaben; ein zweites Netz
lernt, diese vorherzusagen. Hoher Vorhersagefehler liefert einen intrinsischen
Neuheitsbonus. Das ist ein anderer Mechanismus als unsere zufälligen
Actor-Logit-Priors. Die Arbeit zeigt unter anderem Atari-Explorationsergebnisse.
[Burda et al., RND](https://arxiv.org/abs/1810.12894).

**Übertragung:** Es könnte belohnen, bislang selten besuchte Spielzustände zu
erreichen. Bei uns erzeugen schon Kartenausteilung, Wunder und zwei Gegner
viele seltene Beobachtungen. Ein Bonus könnte dadurch irrelevante Neuheit
anstatt nützlicher strategischer Entdeckungen bevorzugen. Zu prüfen wären
Merkmalswahl, Bonusnormalisierung, Bonusgewicht und Verhalten nach Abschalten
des Bonus. Ohne diese Kontrolle wäre „mehr verschiedene Zustände“ kein
hinreichender Erfolg.

### 5.2 Unterschiedliche Verhaltensweisen lernen: DIAYN

DIAYN lernt unterscheidbare Skills über eine latente Kennung, hohe
Verhaltensentropie und einen Diskriminator, der den Skill aus besuchten
Zuständen erkennen soll. Das ursprüngliche unüberwachte Verfahren benötigt
dafür keinen Aufgabenreward und wurde unter anderem in Bewegungsaufgaben
untersucht. [Eysenbach et al., DIAYN](https://arxiv.org/html/1802.06070v1).

**Übertragung:** Ein Kopf könnte als latente Kennung dienen. Ein
Unterscheidbarkeitsziel garantiert aber keine nützlichen 7-Wonders-Strategien:
Köpfe könnten sich durch nebensächliche Kartenentscheidungen unterscheiden.
Eine Kombination mit unserem Spielreward wäre eine eigene Methode mit
zusätzlichem Zielkonflikt. Sie wäre erst dann überzeugend, wenn verschiedene
Köpfe in verschiedenen Gegnersituationen tatsächlich Vorteile bringen.

### 5.3 Gezielt zurückkehren und von dort weiterforschen: Go-Explore

Go-Explore archiviert interessante Zustände beziehungsweise abstrahierte
Zellen, kehrt zu ausgewählten Ausgangspunkten zurück und erkundet von dort.
Der Ansatz trennt Entdeckung und robuste Ausführung; die Arbeit behandelt
unter anderem wiederherstellbare Simulatorzustände und eine policybasierte
Variante. [Ecoffet et al., Nature 2021](https://www.nature.com/articles/s41586-020-03157-9),
[Autoren-PDF](https://adrien.ecoffet.com/files/go-explore-nature.pdf).

**Übertragung:** Das adressiert ausdrücklich das Problem, dass ein früher
Fortschritt beim erneuten Versuch ständig verloren geht. Unser Modellcheckpoint
ist aber kein gespeicherter vollständiger Enginezustand. Verdeckte Karten,
Zufallszustand, öffentliche Informationen und zwei Gegner müssten bei
Wiederaufnahme konsistent behandelt werden. Ein Spielzustandsarchiv und sein
Informationsschutz wären ein größerer Systemumbau. Die Idee ist relevant,
derzeit aber keine kleine Ergänzung unserer Trainingsschleife.

### 5.4 Späten Erfolg frühen Schritten zurechnen: RUDDER

RUDDER lernt Folgen zu bewerten und verteilt Erträge auf wichtige frühere
Ereignisse um. Die Arbeit behandelt Return-Äquivalenz unter definierten
Voraussetzungen und zeigt Ergebnisse bei verzögerten Rewards. Eine gelernte
Umverteilung ist dabei keine fehlerfreie Erklärung jedes Spielverlaufs.
[Arjona-Medina et al., NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html).

**Übertragung:** Interessant, wenn gute Ressourcenfolgen bereits besucht,
ihre frühen Beiträge aber schlecht gelernt werden. Das Verfahren entdeckt
eine noch nie beobachtete erfolgreiche Folge nicht automatisch. Benötigt
würden ein Sequenzmodell, vollständige Episodentrajektorien und eine Prüfung,
ob die Umverteilung wirklich das ursprüngliche Ziel erhält. Das ist eine
Richtung für Credit Assignment, keine Garantie für Deep Exploration.

### 5.5 Zwischenbelohnungen: Potential-based Reward Shaping

Die klassische Arbeit von Ng, Harada und Russell untersucht eine Form von
Zusatzreward, die unter ihren MDP-Annahmen die optimalen Policies erhält:

```text
F(s,s') = gamma × Phi(s') − Phi(s)
```

Endzustände und Potentiale müssen passend behandelt werden. Ein beliebiger
Bonus pro grauer Karte erfüllt diese Bedingung nicht automatisch.
[Ng et al., ICML 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

**Übertragung:** Die Richtung ist nützlich, wenn wir dichteres Feedback mit
einem nachvollziehbaren Zusammenhang zum ursprünglichen Ziel möchten.
Ein gutes Potential ist aber selbst ein ungelöstes Bewertungsproblem.
Teilbeobachtbarkeit und veränderliche Gegner müssen in der Modellierung
berücksichtigt werden. „Belohne unsere vermutete richtige Strategie“ wäre
eine zusätzliche Zielvorgabe und kein voraussetzungsloser Optimalitätsschutz.

### 5.6 Vergleich der Richtungen für unser Problem

| Richtung | Primärer Angriffspunkt | Was wir vor einer Auswahl klären müssten |
| --- | --- | --- |
| Episodenweise Köpfe / zufällige Wertfunktionen | Zeitlich zusammenhängende Variation | Entstehen unterschiedliche vollständige Pläne, und passen Theorie und Umsetzung zusammen? |
| RND | Seltene Zustände aufsuchen | Misst die Neuheit strategischen Fortschritt oder vor allem zufällige Karten? |
| DIAYN / Skill-Ziele | Unterscheidbares Verhalten | Sind die Unterschiede nützlich und situationsabhängig? |
| Go-Explore | Frühe Erkundungserfolge wiederverwenden | Gibt es sichere Zustandswiederherstellung und eine sinnvolle Zustandsabstraktion? |
| RUDDER | Zeitliche Zuordnung beobachteter Erfolge | Ist verspätetes Feedback tatsächlich der Engpass? |
| Potential-Shaping | Dichteres, begründetes Lernfeedback | Können wir ein sinnvolles Potential und passende Randbedingungen angeben? |
| Gedächtnis / bessere Value-Eingaben | Fehlende Information und Vorhersagequalität | Was fehlt heute, obwohl es dem Spieler rechtmäßig bekannt ist? |

**Zwischenergebnis:** Kein einzelner dieser Mechanismen deckt Entdeckung,
Gedächtnis, Erfolgserklärung und passende Gegnergelegenheiten vollständig ab.
Die Methodenwahl sollte dem nachgewiesenen Engpass folgen.

<a id="gegner"></a>
## 6. Gegnerwahl: Stärke, Vielfalt und drei Spieler

### 6.1 Deine Grundsorge ist methodisch berechtigt

Der aktuelle Trainingsreward wird gegenüber einer **bestimmten Verteilung
zweier Gegner** optimiert. Wenn beide regelmäßig dieselben Karten benötigen,
verändert das den Nutzen eigener Wissenschaft, Produktion und Handel.
Das bedeutet jedoch nicht, dass Wissenschaft bei zwei wissenschaftsorientierten
Gegnern immer falsch ist oder dass bestimmte Strategien grundsätzlich
gegenseitig ausgeschlossen sind. Kartenausteilung und Wunder können die
Bewertung verändern.

Eine perfekte Antwort auf ähnliche eigene Vorgänger ist außerdem nicht
automatisch eine gute Antwort auf unbekannte Spielweisen. Umgekehrt ist ein
Gegner nicht deshalb nützlich, weil er ungewöhnlich spielt: Eine ungewöhnliche,
aber stets bedeutungslos schwache Policy kann wenig zusätzlichen Lernwert
bieten. Diese beiden Behauptungen sind Überlegungen zum Trainingsziel,
keine gemessene Erklärung unseres letzten Laufs.

### 6.2 Alter, Gesamtstärke und strategischer Wert sind verschieden

Ein älterer Gegner kann leicht besiegbar sein und trotzdem eine Gegenstrategie
abdecken, die bei neueren Modellen verschwunden ist. Er kann auch nur einen
leicht ausnutzbaren Fehler wiederholen. Das lässt sich aus seinem Alter allein
nicht ableiten.

Ein hypothetischer Zyklus macht das Problem verständlich:

```text
A schlägt B, B schlägt C, C schlägt A.
```

Dann gibt es keine widerspruchsfreie Gesamtreihenfolge, die alle Paarungen
erklärt. Dieser Zyklus ist hier ein Beispiel, kein Nachweis eines solchen
konkreten Zyklus zwischen unseren Checkpoints. Er motiviert jedoch, Ergebnisse
nach Konstellationen aufzubewahren und Modelle nicht allein nach einem
Gesamtscore zu löschen.

**Wichtig für unsere Anker:** Im letzten Kontrollpanel lagen die mittleren
Abstände gegen zwei Exemplare des vorherigen Endmodells bei −3,50 und gegen
zwei Exemplare des damaligen 25%-Stands bei −2,71. Es waren nur jeweils 24
Spiele. Das belegt weder große Stärkeunterschiede noch die Behauptung, diese
Anker seien offensichtlich wertlose leichte Gegner.
[Projektbericht](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/archive/long-training/LONG_TRAINING_V2_RESULTS_2026-10-29.md).

<a id="gegnerpaare"></a>
### 6.3 Drei Spieler benötigen Gegnerprofile

Bei zwei Spielern genügt für einen Vergleich häufig eine Tabelle
„Policy A gegen Policy B“. Bei uns ist zusätzlich entscheidend, **wer auf
dem dritten Platz sitzt**. Für den fokalen Spieler i ist das Objekt:

```text
u_i(Policy_i, Policy_j, Policy_k; Sitz, Wunder, Austeilung)
```

Der heutige Pool zieht die beiden Gegner unabhängig. Hat ein Gegner j
Wahrscheinlichkeit `p(j)`, dann entsteht das geordnete Gegnerpaar mit:

```text
q(j,k) = p(j) × p(k)
```

Ein Beispiel mit den nominalen Gruppengewichten 60/20/20 zeigt die Bedeutung:
20% Anker **je Gegnerplatz** heißen bei unabhängigen Ziehungen 36% Partien mit
mindestens einem Anker (`1−0,8²`) und 4% mit zwei Ankern. Das ist nicht dasselbe
wie 20% aller Partien. Frühe Fallbacks können die tatsächliche Verteilung
zusätzlich verändern.

Auch „Gewinnwahrscheinlichkeit gegen Modell B“ ist ohne Festlegung des zweiten
Gegners unvollständig. Eine leistungsabhängige Auswahl müsste deshalb
Gegnerpaare oder sauber definierte bedingte Durchschnittswerte verwenden.
Mehrere Einzelgegner-Gewichte allein beschreiben nicht jede mögliche gemeinsame
Verteilung: Gezielt nur gemischte Paare zu ziehen lässt sich beispielsweise
nicht durch zwei unabhängige Ziehungen beliebig nachbilden.

### 6.4 Was unser aktueller Pool tatsächlich sichert

Er bewahrt jüngere und ältere Entwicklungsstände und verhindert dadurch eine
ausschließliche Konzentration auf den unmittelbaren Vorgänger. Das Archiv wird
aber **nach zeitlicher Streuung**, nicht nach gemessenem Verhalten oder
zusätzlichem strategischem Nutzen gewählt. Sechs zeitlich verteilte Modelle
können ähnlich spielen.

Die Mischung 60/20/20 ist eine Projektheuristik. Weder ihre Gewichte noch die
Anzahl der jüngeren und archivierten Modelle sind aus einer hier geprüften
Arbeit als optimale 7-Wonders-Werte abgeleitet. Der Pool berechnet keine
dynamischen Gewichte aus laufenden Ergebnissen und löst kein empirisches Spiel.

Eine wissenschaftlich besser begründete Weiterentwicklung muss deshalb zuerst
festlegen, **welchen Wert sie steuern soll**: Lernschwierigkeit, Schutz vor
Vergessen, strategische Abdeckung oder Annäherung an ein Gleichgewicht. Diese
Ziele können unterschiedliche Auswahlregeln rechtfertigen.

<a id="populationen"></a>
## 7. Populationen aus der Forschung: PFSP, PSRO und JPSRO

<a id="pfsp"></a>
### 7.1 PFSP und AlphaStar: aus Ergebnissen Gegner priorisieren

AlphaStar verwendet eine Liga mit Hauptlernern, historischen Policies und
gezielt trainierten Exploitern. Bei Prioritized Fictitious Self-Play werden
Gegner abhängig von der geschätzten Gewinnwahrscheinlichkeit `w` gewichtet,
beispielsweise mit `f(w)=(1−w)^p` für schwierige Gegner oder `f(w)=w(1−w)` für
ein Curriculum näher am eigenen Niveau. Die Arbeit untersucht auch Vergessen
und Schwachstellen gegenüber Gegenstrategien. Das ist empirische Evidenz
aus StarCraft II, keine Garantie für unser Dreierspiel.
[Vinyals et al., Nature 2019, Methods](https://doi.org/10.1038/s41586-019-1724-z),
[Primärtext als PDF](https://gwern.net/doc/reinforcement-learning/model-free/alphastar/2019-vinyals.pdf).

**Übertragung:** Eine solche leistungsabhängige Auswahl adressiert deine Frage
nach leichten Altgegnern direkter als bloß neue feste Prozentsätze. Für uns
wären Gegnerprofile, Messunsicherheit, Vergessensschutz und Aktualität der
Ergebnisschätzungen festzulegen. Einen kaum geprüften Gegner nach zwei Siegen
auf Gewicht null zu setzen wäre keine belastbare Regel. Konkrete Mischanteile
aus AlphaStar sind ebenfalls kein allgemeiner Standard.

**Grenze:** Gute Priorisierung bekannter Gegner erzeugt nicht automatisch eine
noch fehlende neue Gegenstrategie. Dafür dienen in einer Liga unter anderem
zusätzliche Lerner beziehungsweise Exploiter. Ein einzelner Hauptlerner mit
historischen Snapshots übernimmt diese Funktion nicht vollständig.

### 7.2 NFSP: Antworten gegen ein historisches Durchschnittsverhalten

Neural Fictitious Self-Play kombiniert eine angenäherte Best Response mit einem
separat gelernten historischen Durchschnittsverhalten. Die Originalarbeit
verwendet dafür RL und überwachte Lernspeicher und zeigt Resultate in
Zwei-Spieler-Poker. [Heinrich und Silver, NFSP](https://arxiv.org/abs/1603.01121).

**Übertragung:** Das Verfahren beantwortet die Sorge vor bloßem Hinterherlaufen
hinter dem letzten Gegner. Unser Snapshot-Archiv ist aber kein NFSP-
Durchschnittsnetz mit entsprechendem Speicher. Auch NFSP würde in unserem
Dreierspiel seine Gleichgewichtseigenschaften nicht allein durch den Namen
behalten. Es ist ein wichtiger Vergleichspunkt für die Bedeutung historischer
Gegner, keine unmittelbare Ersatzkonfiguration.

<a id="psro"></a>
### 7.3 PSRO: Strategien anhand eines empirischen Spiels erweitern

Policy-Space Response Oracles organisiert eine Schleife: Ergebnisse zwischen
vorhandenen Policies messen, eine Meta-Verteilung berechnen, dagegen eine
neue angenäherte Best Response trainieren und die Population erweitern. Die
Originalarbeit behandelt auch Generalisierung zwischen unabhängig trainierten
Policies. [Lanctot et al., PSRO](https://papers.nips.cc/paper_files/paper/2017/hash/3323fe11e9595c09af38fe67567a9394-Abstract.html).

**Übertragung:** Der Wert einer alten Policy hängt dann von ihrer Rolle im
empirischen Spiel ab. Unser System besitzt bisher Snapshots und eine
Auswahlheuristik, aber weder einen vollständigen empirischen Auszahlungstensor
noch einen Meta-Solver oder getrennte Antwortlernphasen. „Wir verwenden PSRO“
wäre deshalb derzeit eine falsche Methodenbeschreibung.

**Grenze:** Eine schwache angenäherte Best Response kann wichtige Strategien
übersehen. Der Solver kann nur vorhandene, hinreichend genau bewertete Policies
gewichten. Die Population löst das Explorationsproblem des Antwortlerners nicht
automatisch.

<a id="jpsro"></a>
### 7.4 JPSRO: der besonders relevante Mehrspieler-Bezug

JPSRO erweitert das Konzept auf gemeinsame Policy-Verteilungen in
Mehrspieler-General-Sum-Spielen und verwendet Meta-Solver für korrelierte
beziehungsweise grob korrelierte Gleichgewichte, CE und CCE. Die bewiesene
Konvergenz setzt entsprechende Best-Response- und Meta-Solver-Schritte voraus;
die isolierenden Experimente der Arbeit verwenden exakte Antworten und exakte
Spielauswertung. Ein begrenztes PPO-Training mit verrauschten Schätzungen erfüllt
das nicht automatisch. [Marris et al., ICML 2021, §5 und Sätze 6–7](https://proceedings.mlr.press/v139/marris21a.html).

**Übertragung:** Für drei Spieler ist dies ein fachlich passenderer theoretischer
Bezug als die unbesehene Übernahme von Zwei-Spieler-Nullsummen-Nash-Resultaten.
Erforderlich wären gemeinsame Auszahlungen aller Spieler und eine explizite
gemeinsame Verteilung über Policy-Profile.

**Wichtige Begrenzung des Ziels:** Ein korreliertes Gleichgewicht beschreibt
eine gemeinsame Verteilung mit einem Korrelationsmechanismus. Seine
Randverteilungen unabhängig zu ziehen erhält die Eigenschaft im Allgemeinen
nicht. Die Grundlagen hierzu erläutern
[Hart und Mas-Colell, Einleitung und Definition](https://ranger.uta.edu/~weems/NOTES6319/PAPERSTWO/hartMasColell.corrEq.pdf).
Es ist außerdem kein Beweis, dass eine daraus herausgegriffene einzelne
Policy gegen beliebige unbekannte Gegner unschlagbar ist. Wir müssten zuerst
entscheiden, ob wir eine gemeinsame Population oder einen einzeln einsetzbaren
Spieler verbessern möchten.

### 7.5 NeuPL-JPSRO: gemeinsame Netze mit Schutz bestehender Policies

NeuPL-JPSRO verbindet die Population mit einer konditionierten gemeinsamen
neuronalen Repräsentation. Die Arbeit verwendet unter anderem feste
Referenzpolicies sowie Distillation und Regularisierung. Ihre theoretische
Aussage nennt die exakte Ausführung dieser Schritte als Voraussetzung; bei
neuronaler Approximation ist das nicht automatisch erfüllt.
[Liu et al., AAMAS 2024, insbesondere §3.2](https://aamas.csc.liv.ac.uk/Proceedings/aamas2024/pdfs/p1247.pdf).

**Übertragung:** Besonders relevant ist die Frage, wie ein gemeinsames Netz neue
Antworten lernt, ohne alte Policies unkontrolliert zu verändern. Unsere vier
Köpfe teilen ebenfalls Merkmale, besitzen aber weder denselben konditionierten
Populationsaufbau noch diese Bestandssicherung. Die Arbeit ist daher eine
weiterführende Architektur-Richtung, keine Beschreibung unseres heutigen Netzes.

### 7.6 α-Rank: Populationen auswerten, ohne alles auf Elo zu reduzieren

α-Rank untersucht eine evolutionär motivierte Bewertung von Strategien und
Strategieprofilen, auch bei mehreren Spielern und nichttransitiven Beziehungen.
Es bietet einen anderen Blick auf die Population als eine einzige lineare
Stärkerangliste. [Omidshafiei et al., 2019](https://arxiv.org/abs/1903.01373).

**Übertragung:** Das kann die Analyse vorhandener Profile unterstützen.
Es ist kein eigenständiger Mechanismus zum Entdecken neuer Policies und
kein Zertifikat niedriger Ausnutzbarkeit gegenüber noch unbekannten Strategien.
Zunächst müssten genügend belastbare Profilergebnisse vorhanden sein.

### 7.7 Aktueller Literaturstand: auch neuere Arbeiten genau abgrenzen

Global PSRO ist als ICML-2026-Arbeit auf der offiziellen PMLR-Seite
verzeichnet. Sie richtet die Populationserweiterung auf Population Exploitability
aus, behandelt aber ausdrücklich **Zwei-Spieler-Nullsummenspiele**. Geprüft
wurden Publikationsnachweis und Abstract, nicht die vollständigen Experimente.
Sie zeigt eine aktuelle Forschungsrichtung, begründet hier jedoch keine
unmittelbare Empfehlung für unser Dreierspiel.
[Zhang et al., Global PSRO](https://proceedings.mlr.press/v306/zhang26ho.html).

Eine PSRO-Übersicht von 2024 half beim Auffinden von Methodenfamilien;
die tragenden Aussagen dieses Dokuments sind an den Originalarbeiten geprüft.
Anonyme oder nicht sicher einer begutachteten Endfassung zuzuordnende jüngere
Manuskripte werden nicht als etablierter Standard behandelt.
[Bighashdel et al., Übersicht](https://ijcai.org/proceedings/2024/880).

### 7.8 Welche Richtung beantwortet welche Frage?

| Richtung | Zentrale Auswahlgrundlage | Passt zu unserer Frage? | Zusätzliche Voraussetzung |
| --- | --- | --- | --- |
| Heutiger Pool | Alter und feste Gruppenanteile | Einfache historische Abdeckung | Strategischer Wert bleibt ungemessen |
| PFSP-orientierte Auswahl | Aktuelle Schwierigkeit / Vergessen | Direkte Richtung für den Nutzen alter Gegner | Stabile, profilbezogene Ergebnisschätzung |
| PSRO | Empirisches Spiel und Antwortlerner | Systematischer Aufbau einer Strategiemenge | Meta-Solver und wiederholte Antwortlernphasen |
| JPSRO | Gemeinsame Profilverteilung und (C)CE-Ziel | Theoretisch besonders relevant für mehrere Spieler | Klare Entscheidung über Populations- und Einsatz-Ziel |
| NeuPL-JPSRO | Konditionierte Population mit Bestandssicherung | Relevant bei gemeinsamem Netz | Zusätzliche Distillation / Regularisierung und deren Kontrolle |
| α-Rank | Evolutionäre Struktur gemessener Profile | Ergänzende Populationsdiagnose | Ausreichender Auszahlungstensor |

Ein begründetes nächstes Vorgehen ist, leistungsabhängige Auswahl und
mehrspielergerechte Populationsmodelle getrennt zu vergleichen. Das wählt noch
keinen universellen Sieger und setzt keinen vollständigen Liga-Umbau voraus.

<a id="spielforschung"></a>
## 8. Direkte Spielforschung und andere große Spiel-KIs

### 8.1 Forschung zum eigentlichen 7 Wonders

Robilliard, Fonlupt und Teytaud veröffentlichten 2014 eine Arbeit zu
Monte-Carlo Tree Search für 7 Wonders. Die Verlagsseite bestätigt den direkten
Spielbezug. Der vollständige Primärtext war hier nicht verlässlich zugänglich;
deshalb werden daraus keine präzisen Stärkezahlen oder Einzelheiten der besten
Variante übernommen.
[Originalpublikation](https://link.springer.com/chapter/10.1007/978-3-319-14923-3_5).

Ein SBGames-Beitrag von 2020 beschreibt eine 7-Wonders-Engine für KI-Forschung.
Das ist für Simulatorherkunft und Schnittstellen interessant, aber kein
Nachweis eines führenden Lernverfahrens.
[Jardim et al., Engine-Paper](https://www.sbgames.org/proceedings2020/ComputacaoShort/208553.pdf).

Ein institutionell nachgewiesener Bachelorabschluss von 2025 untersucht
7-Wonders-Agenten mit MCTS in Unity. Hier wurden Metadaten und Abstract
geprüft. Die Arbeit ist ein konkreter weiterer Recherchepfad, aber kein Ersatz
für einen belastbaren, identisch evaluierten Vergleich mit unserem Agenten.
[Minenko, Karls-Universität](https://dspace.cuni.cz/handle/20.500.11956/200807?locale-attribute=en).

**Folgerung:** Suchverfahren gehören zu den ernsthaften Vergleichsrichtungen.
Die gefundenen Quellen liefern jedoch keinen direkt übernehmbaren aktuellen
Champion und keinen fertigen Stärkemaßstab für unsere Simulatorversion.

### 8.2 ZeusAI für 7 Wonders Duel: naher Name, anderes Problem

ZeusAI kombiniert ein Transformer-Netz mit MCTS und Selbstspiel in **7 Wonders
Duel**. Der vollständige Bericht nennt 420.000 Selbstspielpartien und eine
kleine menschliche Prüfung mit 38 Partien. Die Initialisierung verwendet auch
einen regelbasierten Spieler; „ohne menschliche Supervision“ bedeutet hier
nicht, dass keinerlei vorgegebene Spielheuristik verwendet wurde.
[Paolini et al., CoG 2024 / Preprint](https://arxiv.org/html/2406.00741v1).

**Übertragung:** Duel hat zwei Spieler, einen anderen Kartenmechanismus und
unmittelbare militärische beziehungsweise wissenschaftliche Siegbedingungen.
Eine erfolgreiche Wissenschaftsstrategie dort löst unsere Konkurrenz- und
Investitionsfrage nicht direkt. Außerdem enthält die Entscheidung MCTS-Suche:
Ein Vergleich mit unserer reinen Policy müsste die Rechenzeit bei der
Aktionswahl berücksichtigen. Die Arbeit belegt eine interessante Kombination
aus Lernen und Suche in einem verwandten Spiel, keine geeignete Zahl von
Trainingspartien für unser Modell.

### 8.3 Warum nicht einfach die Methode einer berühmten Spiel-KI übernehmen?

| Originalarbeit | Was sie belegt | Übertragungsgrenze |
| --- | --- | --- |
| [Deep CFR, ICML 2019](https://proceedings.mlr.press/v97/brown19b.html) | Neuronale Approximation von Regret-Minimierung in großen unvollständig beobachtbaren Spielen | Die behandelten Gleichgewichtsgarantien beziehen sich auf zwei Spieler und Nullsumme; benötigt passenden Spielbaum-/Informationszustandszugriff |
| [ReBeL, NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html) | Verbindung von Lernen und Suche mit Public-Belief-Zuständen | Kein unmittelbarer Beweis für unser dreisitziges General-Sum-Problem; hier nur Abstract geprüft |
| [DeepNash, Science 2022](https://arxiv.org/html/2206.15378v1) | Erfolgreiches gleichgewichtsorientiertes Lernen in Stratego ohne Spielbaumsuche | Zwei-Spieler-Nullsummenstruktur und eigener Lernalgorithmus |
| [Pluribus, Science 2019](https://noambrown.github.io/papers/19-Science-Superhuman.pdf) | Starke empirische Ergebnisse im Sechs-Spieler-Poker | Gerade dort ergibt sich die Stärke nicht aus einer allgemeinen Mehrspieler-Konvergenzgarantie |

Die Existenz starker Alternativen spricht für passende Baselines. Sie bedeutet
nicht, dass ein vollständiger Wechsel von PPO automatisch besser wäre.
Umgekehrt ist PPO auch nicht schon deshalb optimal, weil es verbreitet ist.
Entscheidend sind Problemstruktur, zugängliche Informationen, faire Messung
und die Güte der tatsächlichen Umsetzung.

<a id="evaluation"></a>
## 9. Was eine belastbare Evaluation benötigt

### 9.1 Ziel, Prüffrage und Vergleich festlegen

Vor einer Methodenentscheidung muss klar sein, ob wir vor allem den
erwarteten Punkteabstand, den Sieganteil oder die Robustheit gegenüber
verschiedenen Gegnertypen verbessern möchten. Eine Metrik ist als primäre
Entscheidungsgröße festzulegen; die anderen bleiben wichtige Ergänzungen.

Es sind drei verschiedene Fragen:

1. **Hat sich dieser konkrete Checkpoint verbessert?** Vergleich mit seinem
   Ausgangsmodell unter identischem Prüfplan.
2. **Ist die Methode verlässlich besser?** Unabhängige Wiederholungen des
   Trainingsverfahrens bei vergleichbaren Bedingungen.
3. **Wie allgemein ist die Stärke?** Gegnertypen und Konstellationen, die nicht
   fortlaufend für die Methodenentwicklung verwendet wurden.

Eine wissenschaftliche Evaluation muss Zufall, Auswahlentscheidungen und
Unsicherheit offenlegen. Darauf zielen sowohl die Arbeit von Henderson et al.
als auch die neuere Auswertung von Agarwal et al.; ihre konkreten Benchmark-
Aggregationen sind nicht automatisch die passende Gegnergewichtung für uns.
[Henderson et al., Deep RL That Matters](https://arxiv.org/abs/1709.06560),
[Agarwal et al., Statistical Precipice](https://arxiv.org/abs/2108.13264).

<a id="seed-unsicherheit"></a>
### 9.2 Spielzufall und Trainingszufall getrennt behandeln

Unser letztes Kontrollpanel hat 14 Paarungen × drei Sitze × acht Engine-
Seed-Blöcke = 336 Spiele pro Stand. Dieselben Blöcke werden über Paarungen
und Sitze wiederverwendet. 336 Spiele sind deshalb nicht 336 unabhängig
gezogene Ausgangssituationen für die gemeinsame Auswertung.

Für den Vergleich zweier Checkpoints sollte die Differenz innerhalb desselben
Prüfblocks gebildet und der ganze Block gemeinsam resampelt werden. So bleiben
die vorgesehenen Paarungen und Abhängigkeiten erhalten. Acht Blöcke liefern
trotzdem nur eine begrenzte Grundlage für die Verteilung dieser Differenzen;
ein Bootstrap erzeugt keine neuen unabhängigen Informationen.

Zusätzlich gilt:

- Neue Evaluationsseeds prüfen andere Austeilungen und Zufallsverläufe, nicht
  automatisch andere gegnerische Strategien.
- Vier Köpfe teilen Netz und Training. Sie ersetzen keine vier unabhängig
  trainierten Replikate.
- Zwei aufeinanderfolgende Warmstart-Phasen mit verschiedenen Seeds sind
  ebenfalls keine solchen Replikate.
- Ein Intervall über Spielseeds beschreibt zunächst die Unsicherheit für
  diese trainierten Modelle und diesen Gegnerplan. Unsicherheit der Methode
  über Trainingswiederholungen kommt gesondert hinzu.

Die Unterscheidung ist die konkrete Übertragung der statistischen Literatur
auf unsere Datenstruktur, keine Forderung nach einer universellen Mindestzahl
von Trainingsseeds. [Henderson et al.](https://arxiv.org/abs/1709.06560),
[Agarwal et al.](https://arxiv.org/abs/2108.13264).

### 9.3 Kontrollpanel und abschließender Test haben verschiedene Rollen

Das Dashboardpanel ist nützlich, um Lernverläufe zu sehen und Fehler zu
entdecken. Wenn wir anhand seiner Ergebnisse Köpfe, Gegner, Hyperparameter
oder den besten Zwischenstand auswählen, wird es Teil der Entwicklung.
Es ist dann kein unberührter abschließender Test mehr.

Eine nachvollziehbare Trennung wäre:

| Bereich | Verwendung |
| --- | --- |
| Training | Erzeugt Daten und steuert gegebenenfalls die Gegnerpriorisierung |
| Entwicklung / Kontrollpanel | Dashboard, Fehleranalyse, Methoden- und Checkpointwahl |
| Abschließende Prüfung | Vorher fixierter Vergleich auf zurückgehaltenen Seeds und möglichst zurückgehaltenen Gegnerprofilen |

Nach jeder Einsicht neue Entscheidungen zu treffen und denselben „Holdout“
erneut zu verwenden macht auch diesen schrittweise zum Entwicklungstest.
Mehrere Zwischenstände und viele Untergruppen erhöhen außerdem die Chance,
zufällig einen günstigen Befund auszuwählen. Explorative Ergebnisse dürfen
weiter gezeigt werden; sie sollten als solche benannt und auf neuen Daten
bestätigt werden.

### 9.4 Welche Ergebnisse wir tatsächlich brauchen

**Spielstärke:** Sieganteil mit offengelegter Unentschiedenregel, Punkteabstand,
eigene Punkte und Ergebnisse je Gegnerprofil. Ein Durchschnitt kann deutliche
Schwächen verdecken. Gerade bei unterschiedlichen Untergruppen muss klar sein,
ob jeder Gegnertyp gleich oder nach einer vorgesehenen Einsatzverteilung zählt.

**Ressourcenplanung:** Bauverfügbarkeit in III, Abwürfe mit beziehungsweise ohne
legalen Kartenbau, frühe Produktion, Handelskosten und Kettenbauten. Ein
verfügbarer Wunderbau ist separat zu berücksichtigen: „kein Kartenbau“ bedeutet
nicht „überhaupt keine Bauaktion“. Ein absichtlicher Abwurf kann zudem Münzen
bringen oder einem Gegner eine Karte entziehen.

**Anpassung:** Verhalten und Ergebnisse nach sichtbarer Wissenschaftskonkurrenz,
Militärlage, eigener Produktion, Wunder und Sitz. Wichtig ist die Reihenfolge:
Das Modell soll auf Informationen reagieren, die es vor der Entscheidung
tatsächlich besitzen durfte.

**Kopfvielfalt:** Köpfe auf derselben legalen Beobachtungsmenge vergleichen,
anschließend vollständige Partien und Ergebnisse prüfen. Unterschiede in
Aktionsverteilungen sind ein Diagnosewert, aber noch kein Beleg nützlicher
Strategien. Wenn jeder Kopf nur auf seinen eigenen besuchten Zuständen
verglichen wird, vermischen sich Kopf- und Zustandsunterschiede.

**Entdeckung:** Vorab definieren, was eine langfristige Strategie hier erkennbar
macht: etwa eine Folge früher Kosten mit später genutzten Bauoptionen, nicht
lediglich mehr grüne Karten. Nachträglich nur die schönsten Partien auszuwählen
würde die Häufigkeit und den Misserfolg solcher Versuche verdecken.

### 9.5 Korrelation, Gegenprobe und Ursache auseinanderhalten

Mehr Wissenschaftspunkte können durch bessere Vorbereitung entstehen, aber
auch dadurch, dass Gegner zufällig mehr passende Karten passieren lassen.
Eine Korrelation zwischen grauen Bauten und späteren Punkten beweist deshalb
keinen kausalen Nutzen jeder grauen Karte.

Für eine stärkere Gegenprobe wäre ein zukünftiger Zustandswiederholungs-
Mechanismus hilfreich: Von demselben rechtmäßig beobachteten Ausgangspunkt
verschiedene Entscheidungen mit kontrollierten weiteren Zufallsströmen
vergleichen. Der heutige Trainingscheckpoint bietet das noch nicht. Auch ein
solcher Eingriff müsste die verdeckte Information sauber schützen und mehrere
Fortsetzungen betrachten.

Im jetzigen System sind stratifizierte Beobachtungsanalysen ein sinnvoller
erster Hinweis, aber kein Ersatz für diese Gegenprobe. Bei kleinen Wunder-
oder Gegneruntergruppen sind breite Intervalle und zurückhaltende Aussagen
zu erwarten.

<a id="ausnutzbarkeit"></a>
### 9.6 Exploiter und Ausnutzbarkeit richtig interpretieren

Ein gezielt gegen unseren Spieler trainierter Gegenagent kann eine übersehene
Schwachstelle finden. Das wäre ein starker Befund gegen umfassende Robustheit.
Scheitert dieser Gegenagent, haben wir dagegen nicht bewiesen, dass kein
besserer Gegenagent existiert: Sein eigener Lernprozess könnte gescheitert sein.

Eine ungefähr gelernte Best Response liefert bei korrekter Messung einen
gefundenen Abweichungsgewinn, keine gesicherte obere Schranke aller möglichen
Abweichungsgewinne. Niedrige gemessene Werte mit einem schwachen Antwortlerner
zertifizieren deshalb keine geringe wahre Ausnutzbarkeit.

Bei drei Spielern ist zusätzlich festzulegen, ob ein einzelner Spieler gegen
ein fixes Gegnerpaar abweicht oder zwei Gegner gemeinsam verändern dürfen.
Eine koordinierte Zweierkoalition ist ein anderes Robustheitsziel als eine
einseitige Abweichung. Nash-/CE-/CCE-Begriffe beziehen sich nicht beliebig auf
denselben Test. Bei CE steht die einseitige Abweichung nach einer eigenen
Empfehlung im Zentrum, nicht die freie Änderung aller anderen Spieler.
[Hart und Mas-Colell, §2](https://ranger.uta.edu/~weems/NOTES6319/PAPERSTWO/hartMasColell.corrEq.pdf).

### 9.7 Was ein Methodenvergleich konstant halten muss

Um einen Unterschied dem untersuchten Mechanismus zuordnen zu können, müssen
Ausgangsgewichte, Informationen, Reward, Simulatorversion, Gegnerregel,
Trainingsumfang und Testplan dokumentiert werden. Soweit möglich wird pro
Vergleich eine wesentliche Ursache verändert.

Ein Warmstart-Vergleich beantwortet „Was hilft diesem bereits trainierten
Modell?“. Ein Lernen-von-Anfang-an-Vergleich beantwortet eine andere Frage.
Auch die Auswahl eines besonders günstigen Ausgangsmodells ist zu berichten.

Gesamtaufwand umfasst gegebenenfalls weitere Lerner, Gegnerauswertung,
Hyperparametersuche und Suche bei der Aktionswahl. Sonst würden etwa eine
Population mit mehreren Antwortlernern und eine einzelne Policy nur dem Namen
nach mit gleichem Budget verglichen. Das ist eine Anforderung an faire
Methodenvergleiche, keine Prognose für mehr Rechenleistung.

Zur Nachvollziehbarkeit gehören Konfiguration, Modell- und Quelldateihashes,
Engineversion, Featureversion, Seeds, Checkpointwahl und fehlgeschlagene
Versuche. Der vorhandene strenge Checkpoint-/Schemaabgleich und die
protokollierten tatsächlich angewendeten Aktionen sind dafür gute Grundlagen.

<a id="entscheidungen"></a>
## 10. Entscheidungsgrundlage und offene Fragen

### 10.1 Antworten auf die drei Leitfragen

**1. Bestehendes Verfahren:** Die Grundlage ist etabliertes PPO mit
Actor-Critic und GAE. Die episodische Kopfwahl ist ein nachvollziehbarer
Explorationsgedanke. Unsere Actor-Logit-Priors und der feste Gegnergruppenmix
sind eigene Anpassungen. Sie verdienen kontrollierte Prüfung; sie sind keine
Umsetzung, für die bereits eine passende allgemeine Garantie vorliegt.

**2. Exploration und Gegner:** Für langfristige Variation, Credit Assignment,
Neuheitsanreize und Populationen gibt es unterschiedliche seriöse
Forschungsrichtungen. Leistungsabhängige Gegnerpriorisierung adressiert den
Lernwert alter Gegner; JPSRO adressiert ausdrücklich das Mehrspielerproblem.
Beides löst nicht von selbst die Entdeckung neuer Strategien. Die Engpässe
sollten vor einer Architekturentscheidung getrennt untersucht werden.

**3. Erreichbare Aussage:** Wir können eine reproduzierbare, sauber abgegrenzte
Verbesserung gegenüber benannten Baselines und in definierten Gegnersituationen
anstreben. Ein literaturgestützter Ansatz allein rechtfertigt keine Aussage
„bestmöglicher Spieler“ oder „garantiert alle wichtigen Strategien entdeckt“.

<a id="naechste-klaerungen"></a>
### 10.2 Welche Klärungen eine nächste Entscheidung tragen würden

Dies ist eine Reihenfolge von Erkenntnisfragen, kein bereits beauftragter
Trainings- oder Implementierungsplan.

| Frage | Sinnvolle Evidenz | Welche Entscheidung sie ermöglicht |
| --- | --- | --- |
| Ist unser Lernziel tatsächlich Siegchance oder Punkteabstand? | Explizite Zielwahl; getrennte vorhandene Sieg-/Abstandsergebnisse | Reward und primäre Testmetrik passend festlegen |
| Fehlen dem Lerner relevante Informationen? | Audit rechtmäßig bekannter Hand-/Verlaufsinformationen, Critic-Vorhersagen | Informations- und Gedächtnisfrage von Exploration trennen |
| Erkunden Köpfe vollständige unterschiedliche Pläne? | Vergleich auf gemeinsamen Beobachtungen plus vollständige Partien und Kontextauswertung | Bestehenden Mechanismus behalten, justieren oder Alternativen begründet prüfen |
| Welche Gegnerprofile sind leicht, schwierig oder einzigartig? | Profilbezogene Ergebnisse und Unsicherheit, nicht nur Checkpointalter | Leistungsabhängige Priorisierung und Archivkriterien begründen |
| Wollen wir einen Einzelspieler oder eine gemeinsame Population optimieren? | Geplanter Einsatz und Robustheitsziel | PFSP-orientierte Liga versus empirisches Spiel / JPSRO sinnvoll abgrenzen |
| Verbessert eine Änderung die gewünschte Fähigkeit? | Vorab definierter Vergleich, passende Diagnose, frische Bestätigung | Einzelne Methodenentscheidung verteidigen |

Die bisherige Präferenz, **einen Versuch nach dem anderen** durchzuführen und
danach neu zu entscheiden, bleibt damit vereinbar. Dafür muss die Prüffrage
des jeweiligen Versuchs eng genug sein; ein Wechsel mehrerer Bausteine
gleichzeitig erlaubt hauptsächlich Aussagen über das Gesamtpaket.

<a id="prioritaeten"></a>
### 10.3 Konkretisierte Anforderungen nach Besprechung des Überblicks

Am 29. Oktober 2026 wurden drei Arbeitsprioritäten festgehalten. Dies ist eine
Projektentscheidung über zu adressierende Lücken, kein zusätzlicher
Literaturbefund. A und B sind inzwischen umgesetzt und im Trainer standardmäßig aktiv;
für C ist inzwischen die im Nachtrag verlinkte Ausgangsvariante eingerichtet.
Der folgende Text dokumentiert Begründung und ursprüngliche Abwägung.

**A. Die eigene Hand muss in der Zustandsbewertung verfügbar sein.**
`build_obs()` liefert bereits `hand_ids`; `flatten_global()` übernimmt sie
aber nicht. Der Actor verarbeitet Merkmale einzelner Kandidaten, der Critic
nur die globale Einbettung. Dieser Aufbau ist schon im frühesten verfügbaren
`ActorCritic` vorhanden. Ein fachlicher Grund für die Auslassung ist dort
nicht dokumentiert; sie ist keine Vorgabe von PPO.

Eine Handdarstellung soll auch momentan unbaubare Karten enthalten, mit
Padding-/Gültigkeitsmaske. Nur gerade legale Bauaktionen zusammenzufassen wäre
unvollständig: Eine heute unbaubare Karte kann nach einer Investition wichtig
werden. Reine numerische Karten-IDs sind keine sinnvoll geordneten Messwerte;
Identitäten brauchen eine kategoriale Darstellung oder passende Kartenmerkmale.
Diese Handinformation kann beiden Netzteilen zugänglich gemacht werden.

Der bisherige Actor verarbeitet die Kandidaten einzeln zusammen mit dem
globalen Kontext. Die gemeinsame Softmax-Auswahl ersetzt keine ausdrücklich
gelernte Zusammenfassung der ganzen Hand. Bei der Erweiterung sollte daher
auch die Verarbeitung von Beziehungen zwischen Handkarten geprüft werden.
Ein präziserer Critic garantiert nicht allein eine stärkere Policy; nötig sind
eine technische Informationsprüfung und anschließend ein kontrollierter Vergleich.

**B. Rechtmäßig gesehene Hände müssen als Verlauf nutzbar sein.**
In unserem Dreiersimulator hat das ursprüngliche Paket bei seiner Rückkehr in
Runde 4 vier Karten: Zuvor haben alle drei Spieler jeweils eine entnommen.
Im normalen Spiel wird beim sechsten Auswählen aus zwei Karten eine gespielt
und die letzte abgelegt; es folgt keine reguläre siebte Auswahl aus dem
ursprünglichen Paket. Die Engine besitzt für die Siebte-Karte-Fähigkeit einen
zusätzlichen technischen Status, dessen Aktion sie für andere Spieler
überspringt. Diese Implementierung darf nicht als allgemeine siebte Spielrunde
beschrieben werden. Die Weitergaberichtung hängt vom Zeitalter ab.

Wer ein Paket gesehen hat, kennt damit einen Kandidatenbestand für spätere
Runden. Er weiß noch nicht sicher, welche Karten die anderen entnehmen.
Öffentlich gebaute Karten liefern weitere Hinweise; verdeckte Entnahmen
dürfen nicht aus dem vollständigen Enginezustand ergänzt werden.
Die Erinnerung dient auch der Einschätzung der nächsten gegnerischen
Entscheidungen: Welche Karten wurden welchem Nachbarn gerade angeboten und
welche davon passen zu dessen sichtbarer Auslage? Sie ist daher nicht auf die
Rückkehr des Pakets beschränkt.

Zwei Darstellungsrichtungen sind zunächst zu vergleichen: eine explizite,
begrenzte Folge eigener beobachteter Hände samt Runden-/Zeitalterbezug und
Aktionen oder ein gelerntes rekurrentes Gedächtnis. Die erste liefert die
beobachteten Fakten direkt; die zweite muss lernen, was sie speichert.
Rekurrente Netze sind für teilbeobachtbare RL-Aufgaben untersucht, aber nicht
dadurch automatisch die bessere Wahl in unserem Spiel.
[Hausknecht und Stone, DRQN](https://arxiv.org/abs/1507.06527).

**Gewählter und inzwischen implementierter erster Ansatz:** Den kurzen
Beobachtungsverlauf ausdrücklich speichern und als begrenzte zusätzliche
Eingabe bereitstellen. Pro eigenem Entscheidungspunkt werden Handkarten mit
Gültigkeitsmaske, Zeitalter/Runde, Weitergaberichtung, eigene ausgeführte Aktion
und rechtmäßig öffentliche Veränderungen festgehalten. Eine mengenbasierte
Darstellung der Karten verhindert, dass eine zufällige Reihenfolge innerhalb
der Hand als Spielinformation behandelt wird. Die Schritte bleiben zeitlich
geordnet. Ein gemeinsamer gelernter Encoder verarbeitet diesen Verlauf für
Actor und Critic; für den ersten Vergleich genügt eine einfache Verarbeitung
der fest begrenzten Eingabe. Ein rekurrentes Gedächtnis oder Attention bleibt
eine anschließend vergleichbare Architekturentscheidung.

Der Vorteil für unsere Untersuchung ist, dass wir gespeicherte Information
direkt prüfen können und den bisherigen PPO-Ablauf ohne rekurrente
Sequenz-Minibatches erweitern können. Das beweist keine generelle Überlegenheit
dieser Darstellung. Die aktuelle vollständige Hand und der Verlauf sollten
in aufeinanderfolgenden Vergleichen ergänzt werden, damit ihr jeweiliger
Beitrag unterscheidbar bleibt. Nur eine Liste aller irgendwann gesehenen
Karten ohne Zeit- und Paketbezug würde die Anforderung nicht erfüllen.

Unabhängig von der Architektur muss der Verlauf pro Partie und Spielerplatz
isoliert sein, darf nur eigene bisherige Beobachtungen und öffentliche Ereignisse
enthalten und muss beim Spielwechsel zurückgesetzt werden. Bei geteilten
Gegner-Modellcaches darf kein Spieler den Gedächtniszustand eines anderen
übernehmen. Sammlung, PPO-Update, Value-Bootstrap und Evaluation müssen
dieselbe Informationsdefinition verwenden.

**C. Gegnerprofile sollen nach überprüfbaren Regeln ausgewählt werden.**
Die gewünschte Richtung ist eine ergebnisgestützte Auswahl anstelle immer neu
händisch festgelegter Gruppenanteile. Dafür sind Schwierigkeit, Unsicherheit
der Schätzung, Vergessen und zusätzliche strategische Abdeckung getrennt zu
erfassen. Ein altes Modell wird weder allein wegen seines Alters entfernt noch
allein als historischer Anker dauerhaft gleich gewichtet.

Zunächst benötigt werden profilbezogene Ergebnisse mit Stichprobengröße und
Messzeitpunkt. Erst darauf lässt sich eine Priorisierungsregel begründen.
Einzelgegnerwerte müssen den jeweils zweiten Gegner klar definieren. Die
Trainingsauswahl und das abschließende Bewertungsfeld bleiben getrennt.
PFSP-orientierte Regeln und empirische Spielverfahren bleiben Vergleichsrichtungen;
ein konkreter Solver oder eine neue Prozentmischung ist damit nicht beschlossen.

Die Informationsgrundlage hat Vorrang vor der Deutung weiterer
Explorationsversuche: Fehlende Erinnerung lässt sich nicht allein durch mehr
zufällige Köpfe beheben. Für die Auswertung sollten die Eingabeänderung, der
Verlauf und die Gegnerregel möglichst getrennt geprüft werden; ein gemeinsamer
Umbau würde zunächst nur eine Aussage über das Gesamtpaket erlauben.

### 10.4 Was uns die Forschung derzeit nicht abnimmt

- Keine hier geprüfte Quelle garantiert das Entdecken aller lohnenden
  langfristigen Strategien in unserem neuronalen Dreierspiel.
- Es gibt keine aus diesen Arbeiten ableitbare universelle optimale Anzahl von
  Köpfen, Prior-Stärke oder Gegnergruppen-Gewichtung.
- Die relevante Strategiemenge im vollständigen 7 Wonders ist durch unsere
  bisherige Population nicht nachweislich abgedeckt.
- Eine gute Wissenschaftsquote oder ein hoher Durchschnittsreward ist kein
  vollständiger Maßstab für Anpassungsfähigkeit.
- Ohne kontrollierten Vergleich ist der kausale Beitrag der bisherigen
  Zusatzmechanismen offen, auch wenn das Gesamtmodell besser geworden ist.

Eine gute wissenschaftliche Grundlage besteht hier aus **passender
Problemformulierung, überprüfter Implementierung, begründeten Alternativen
und belastbarer Evaluation**. Der Algorithmusname allein ersetzt keinen
dieser Schritte.

<a id="quellen"></a>
## 11. Kommentiertes Quellenverzeichnis und Lesetiefe

Jahresangaben unterscheiden, soweit relevant, Preprint und begutachtete
Publikation. „Methodenteil geprüft“ meint die für die Aussagen relevanten
Abschnitte, nicht eine unabhängige Reproduktion oder vollständige Verifikation
jedes Beweises. Alle Links wurden im Rahmen dieser Recherche aufgerufen;
Zugriffslücken sind vermerkt.

### Grundlagen und Durchführung

| ID | Quelle | Geprüft / Verwendung |
| --- | --- | --- |
| G1 | Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | Primär-Preprint und PDF; Clipping- und Actor-Critic-Zielfunktion, §§3–5 |
| G2 | Schulman et al. (2015/2016), [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) | Herleitung und Bias-/Varianz-Argumentation im Primär-PDF |
| G3 | Huang, Ontañón (2020/2022), [Invalid Action Masking](https://arxiv.org/abs/2006.14171) | Primär-PDF, Gültigkeit und Grenzen maskierter Policy-Gradienten |
| G4 | Andrychowicz et al. (2020/2021), [What Matters in On-Policy RL?](https://arxiv.org/abs/2006.05990) | Haupttext der empirischen Implementierungsstudie; Aufgabenbereich ausdrücklich begrenzt |
| G5 | Hausknecht, Stone (2015, revidiert 2017), [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527) | Ergänzung bei der Besprechung des Überblicks: Abstract geprüft; Gedächtnis als Darstellungsrichtung, kein Beleg für Überlegenheit in 7 Wonders |

### Exploration und verzögerte Rückmeldung

| ID | Quelle | Geprüft / Verwendung |
| --- | --- | --- |
| E1 | Osband et al. (2016), [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621) | Methoden, Ensemble, Episodenkopf und Algorithmus im Anhang |
| E2 | Osband et al. (2018), [Randomized Prior Functions for Deep RL](https://arxiv.org/abs/1806.03335) | §§3–4 und Algorithmen; Abgrenzung unserer Actor-Logit-Variante |
| E3 | Osband et al. (2019), [Deep Exploration via Randomized Value Functions](https://www.jmlr.org/papers/v20/18-339.html) | Publikationsseite und Abstract; keine vollständige Prüfung der 62-seitigen Beweisarbeit |
| E4 | Burda et al. (2018/2019), [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894) | Hauptmechanismus und Motivation im Primär-PDF |
| E5 | Eysenbach et al. (2018/2019), [DIAYN](https://arxiv.org/html/1802.06070v1) | Volltext, Skill-Ziel und Diskriminator; keine Gleichsetzung mit nützlichen Spielstrategien |
| E6 | Ecoffet et al. (2021), [First return, then explore](https://www.nature.com/articles/s41586-020-03157-9) | Verlagsseite und [Autoren-PDF](https://adrien.ecoffet.com/files/go-explore-nature.pdf), Hauptmechanismen und Robustifizierung |
| E7 | Arjona-Medina et al. (2019), [RUDDER](https://papers.nips.cc/paper_files/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html) | Primär-PDF, Return-Umverteilung und Voraussetzungen in §2 |
| E8 | Ng et al. (1999), [Policy Invariance under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) | Primär-PDF, Potential-Form und Invarianzbedingungen |
| E9 | Osband et al. (2019/2020), [Behaviour Suite for RL](https://arxiv.org/abs/1908.03568) | Primär-PDF, Diagnoseaufgaben einschließlich Deep Sea und Gedächtnis |

### Gegnerpopulationen

| ID | Quelle | Geprüft / Verwendung |
| --- | --- | --- |
| P1 | Heinrich, Silver (2016), [Deep RL from Self-Play in Imperfect-Information Games](https://arxiv.org/abs/1603.01121) | NFSP-Haupttext, Durchschnittspolicy und Antwortlernen |
| P2 | Lanctot et al. (2017), [A Unified Game-Theoretic Approach to Multiagent RL](https://arxiv.org/abs/1711.00832) | PSRO-Haupttext und Algorithmus; empirisches Spiel und Meta-Solver |
| P3 | Vinyals et al. (2019), [AlphaStar](https://doi.org/10.1038/s41586-019-1724-z) | Primärartikel einschließlich Methods/PFSP; gelesen über den [PDF-Spiegel](https://gwern.net/doc/reinforcement-learning/model-free/alphastar/2019-vinyals.pdf), keine Sekundärzusammenfassung als Methodenbeleg |
| P4 | Marris et al. (2021), [Multi-Agent Training beyond Zero-Sum](https://proceedings.mlr.press/v139/marris21a.html) | Haupttext und Algorithmus, Konvergenzsätze und Beweisskizze §5.4; exakte experimentelle Orakel berücksichtigt |
| P5 | Liu et al. (2024), [NeuPL-JPSRO](https://aamas.csc.liv.ac.uk/Proceedings/aamas2024/pdfs/p1247.pdf) | Primär-PDF; Distillation/Regularisierung und explizite Exaktheitsannahmen |
| P6 | Omidshafiei et al. (2019), [α-Rank](https://arxiv.org/abs/1903.01373) | Abstract und grundlegende Motivation; keine vollständige Prüfung des evolutionären Beweisapparats |
| P7 | Zhang et al. (2026), [Global PSRO](https://proceedings.mlr.press/v306/zhang26ho.html) | Offizielle ICML/PMLR-Metadaten und Abstract; PDF-Abruf nicht auswertbar, keine Detailbehauptung über Versuche |
| P8 | Bighashdel et al. (2024), [Policy Space Response Oracles: A Survey](https://ijcai.org/proceedings/2024/880) | Sekundärquelle zur Navigation; konkrete technische Aussagen an Originalquellen abgeglichen |
| P9 | Hart, Mas-Colell (2000), [A Simple Adaptive Procedure Leading to Correlated Equilibrium](https://math.huji.ac.il/hart/abs/adapt.html) | Autorenseite und Primär-PDF über Universitätsspiegel; Einleitung und §2 zur Bedeutung korrelierter Verteilungen und einseitiger Abweichungen |

### Spiel-KIs und direkter Domänenbezug

| ID | Quelle | Geprüft / Verwendung |
| --- | --- | --- |
| S1 | Robilliard et al. (2014), [7-Wonders-MCTS-Publikation](https://link.springer.com/chapter/10.1007/978-3-319-14923-3_5) | Verlagsabstract; Volltextzugriff nicht verlässlich, deshalb keine Detailzahlen |
| S2 | Jardim et al. (2020), [7-Wonders-Engine für KI-Forschung](https://www.sbgames.org/proceedings2020/ComputacaoShort/208553.pdf) | Abstract und Einleitung des Primär-PDF; Simulatorbezug |
| S3 | Minenko (2025), [Bachelorarbeit zu 7 Wonders und MCTS](https://dspace.cuni.cz/handle/20.500.11956/200807?locale-attribute=en) | Institutionelle Metadaten und Abstract; Volltext nicht geprüft, geringere Belegstärke für Methodenüberlegenheit |
| S4 | Paolini et al. (2024), [ZeusAI für 7 Wonders Duel](https://arxiv.org/html/2406.00741v1) | Vollständiger kurzer Bericht, Aufbau, Initialisierung und Evaluation; CoG-Publikation zusätzlich institutionell nachgewiesen |
| S5 | Brown et al. (2019), [Deep CFR](https://proceedings.mlr.press/v97/brown19b.html) | Abstract und Einleitung des Primär-PDF; Theorie-Geltungsbereich |
| S6 | Brown et al. (2020), [ReBeL](https://papers.neurips.cc/paper_files/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html) | Offizieller Publikationsabstract; nur Einordnung, keine Implementierungsempfehlung |
| S7 | Perolat et al. (2022), [DeepNash](https://arxiv.org/html/2206.15378v1) | Abstract und Methodeneinordnung im Primärtext; Stratego-Bezug |
| S8 | Brown, Sandholm (2019), [Pluribus](https://noambrown.github.io/papers/19-Science-Superhuman.pdf) | Primär-PDF, Einleitung und theoretische Mehrspieler-Einschränkungen |

### Evaluation

| ID | Quelle | Geprüft / Verwendung |
| --- | --- | --- |
| V1 | Henderson et al. (2018), [Deep Reinforcement Learning That Matters](https://arxiv.org/abs/1709.06560) | Primär-Haupttext, Variabilität, Reproduzierbarkeit und Vergleichsprobleme |
| V2 | Agarwal et al. (2021), [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264) | Haupttext einschließlich Unsicherheit und Aggregation; Übertragung auf unsere Clusterstruktur ausdrücklich getrennt |

Weitere Suchtreffer wurden nicht als tragende Evidenz verwendet, wenn nur ein
Titel, ein unklarer Begutachtungsstatus oder ein nicht prüfbarer Volltext
vorlag. Insbesondere ergibt ein neueres Datum allein keine höhere
Übertragbarkeit. Offene Zugänge bei S1, S3 und P7 sind gezielte Nachlesepunkte,
falls eine dieser Richtungen später ausgewählt wird.

<a id="projektbelege"></a>
## 12. Codebelege, Datenbelege und Begriffserklärungen

### 12.1 Untersuchte Implementierung

Untersucht wurde der Arbeitsbaum am 29. Oktober 2026, basierend auf Repository-
HEAD `1c9ea06a06513f1f7b1ff46385d49d8bc996d944`, **einschließlich bereits
vorhandener nicht eingecheckter Änderungen**. Der Commit allein reproduziert
diesen Arbeitsbaum folglich nicht. Für spätere Experimente sind die tatsächlichen
Quelldateien und Konfigurationen zu versionieren beziehungsweise zu hashen.

| Beleg | Datei / relevante Stellen |
| --- | --- |
| Actor, Critic, geteilte Encoder, Prior-Funktion | [ppo_model.py](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/src/sevenwonders_rl/utils/ppo_model.py), Klassen `ActorCritic`, `MultiHeadActorCritic`, `forward_batch` |
| Globaler Critic-Input und Aktionsmaskierung | [ppo_utils.py](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/src/sevenwonders_rl/utils/ppo_utils.py), `flatten_global`, `action_features`, `masked_logits` |
| Öffentliche Karten-/Wundermerkmale | [public_state.py](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/src/sevenwonders_rl/features/public_state.py) und [obs_builder.py](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/src/sevenwonders_rl/features/obs_builder.py) |
| Terminaler Reward | [rewards/__init__.py](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/src/sevenwonders_rl/rewards/__init__.py), `RewardSpec.terminal_components` |
| GAE und Endzustandsbehandlung | [gae.py](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/src/sevenwonders_rl/utils/gae.py), `compute_gae` |
| Rollout, Kopfwahl, PPO-Loss und Optimierung | [train_unified.py](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/scripts/ppo_training/train_unified.py), `collect_rollout`, `ppo_update` und Umgebung-Reset |
| Historische Auswahl und Fallbacks | [opponent_pool.py](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/src/sevenwonders_rl/selfplay/opponent_pool.py), `MixedOpponentPool` |
| Konkrete Einstellungen des letzten Laufs | [run_long_training_v2.py](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/scripts/ppo_training/run_long_training_v2.py), [Gegnerkonfiguration](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/configs/long_mixed_v2_opponents.json), [Evaluationspanel](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/configs/long_mixed_v2_eval_panel.json) |
| Aktionsschnittstelle und Anwendung | [sevenwonders_env.py](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/src/sevenwonders_rl/envs/sevenwonders_env.py), [C++-Spielerlogik](https://github.com/martenkrohn/seven-wonders-engine/blob/f39c1d7ead271df437269b7f555a2c7a0d4b9ad1/src/player.cpp) |
| Checkpointvertrag und Evaluation | [checkpoints.py](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/src/sevenwonders_rl/utils/checkpoints.py), [milestones.py](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/src/sevenwonders_rl/evaluation/milestones.py), [selfplay.py](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/python/src/sevenwonders_rl/evaluation/selfplay.py) |

Historischer Nachweis des bereits vorhandenen Critic:

```bash
git show cd2d863:python/src/sevenwonders_rl/utils/ppo_model.py
```

Ältere Statusdokumente können inzwischen überholte Einschränkungen enthalten.
Für Aussagen zum aktuellen Verhalten wurden deshalb Code, konkrete
Laufkonfigurationen und neuere Ergebnisberichte höher gewichtet.

### 12.2 Vorhandene Ergebnisquellen

- [Multi-Head-Ergebnisse](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/archive/exploration/MULTIHEAD_RESULTS_2026-10-26.md): frühere Erkundungsphase.
- [Prior-Ergebnisse](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/archive/exploration/MULTIHEAD_PRIORS_RESULTS_2026-10-26.md) und
  [frische Prior-Prüfung](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/archive/exploration/MULTIHEAD_FRESH_EVALUATION_2026-10-26.md): Fortschritt und
  Grenzen der frühen Vergleiche.
- [Erster langer Lauf](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/archive/long-training/LONG_TRAINING_RESULTS_2026-10-26.md) und
  [frische gemeinsame Prüfung](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/archive/long-training/LONG_FRESH_RESULTS_2026-10-26.md): belastbarerer
  Vergleich auf neuen Spielseeds.
- [Zweiter langer Lauf](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/archive/long-training/LONG_TRAINING_V2_RESULTS_2026-10-29.md) mit
  [strukturierter Auswertung](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/archive/long-training/LONG_TRAINING_V2_RESULTS_2026-10-29.json): aktueller
  Bezugspunkt und Unsicherheit des zusätzlichen Fortschritts.

### 12.3 Begriffe in einfacher Sprache

| Begriff | Bedeutung hier |
| --- | --- |
| Policy | Regel des Modells, mit der es aus seinen Informationen eine Aktion auswählt |
| Value / Critic | Vorhersage, wie viel zukünftigen Reward eine Situation erwarten lässt |
| Q-Funktion | Bewertung einer Situation zusammen mit einer bestimmten Aktion |
| On-Policy | Lernen aus Erfahrungen der für die Sammlung verwendeten Policy; PPO verwendet diese Daten für mehrere begrenzte Updates |
| Bootstrap bei GAE | Eine Vorhersage über spätere Erträge verwenden, statt immer bis zum vollständigen Ende zu warten |
| Bootstrap bei Ensembles | Daten resampeln oder unterschiedlich gewichten, um mehrere Schätzer zu erzeugen; anderer Begriffskontext als bei GAE |
| Epistemische Unsicherheit | Unwissen des Lerners, das passende zusätzliche Erfahrungen verringern könnten |
| Zufallsunsicherheit | Zum Beispiel unbekannte Austeilung; sie verschwindet nicht schon durch mehr Lernen |
| Best Response | Beste Antwort auf eine festgelegte Verteilung anderer Spieler; bei RL gewöhnlich nur angenähert |
| Meta-Spiel | Kleineres Spiel, in dem komplette vorhandene Policies die auswählbaren Strategien sind |
| CE / CCE | Gleichgewichtsbegriffe für gemeinsame Empfehlungen; CE prüft Abweichungen nach Kenntnis der eigenen Empfehlung, CCE vor dieser Kenntnis |
| Nichttransitivität | A kann B schlagen und B C, ohne dass A deshalb auch C schlagen muss |
| Holdout | Daten oder Gegner, die vor der abschließenden Prüfung nicht für Auswahlentscheidungen verwendet wurden |
| Ablation | Vergleich, bei dem ein Bestandteil gezielt entfernt oder verändert wird, um seinen Beitrag zu untersuchen |

Zurück zur [Kurzfassung mit den wesentlichen Entscheidungen](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/research/OVERVIEW.md).
