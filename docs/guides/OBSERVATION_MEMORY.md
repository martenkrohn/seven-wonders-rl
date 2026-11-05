# Beobachtungsverlauf und gemeinsame Informationen für Actor und Critic

Implementiert am 29. Oktober 2026. Die Erweiterung wird im vereinheitlichten
Trainer **standardmäßig aktiviert**. `--observation-memory` kann weiterhin
explizit angegeben werden; `--no-observation-memory` schaltet sie für
Vergleichsläufe aus. Sie schaltet die öffentlichen
Karten-/Wunderinformationen automatisch hinzu. Ein langer Trainingslauf oder
eine neue Gegnerauswahl wird dadurch nicht automatisch gestartet.

## Was das Modell erhält

Actor und Critic verwenden denselben erweiterten Kontext:

- die bisherigen öffentlichen Tisch-, Punkte-, Ressourcen- und Wundermerkmale;
- die vollständige aktuelle eigene Hand einschließlich unbaubarer Karten;
- die legalen Aktionsmöglichkeiten und ihre vorhandenen Aktionsmerkmale;
- die zeitlich geordneten bisherigen Beobachtungen und bestätigten eigenen Aktionen.

Ein historischer Schritt enthält die damalige Hand als Kartenanzahlen,
eine Kennzeichnung, ob sie beobachtbar war, die öffentlichen Basismerkmale
einschließlich Runde/Zeitalter/Weitergaberichtung, öffentlich gebaute
Karten aller drei relativen Plätze und die bestätigte eigene Aktion samt Karte.
Die festen Zeitpositionen bleiben erhalten; nicht beobachtete Positionen sind
maskiert. Die Grenze von 21 Statuspositionen entspricht dem heutigen Simulator,
nicht 21 regulären Spielentscheidungen.

Das ist ein Beobachtungsprotokoll. Ein alter Kartenbestand wird nicht als sichere
aktuelle Gegnerhand dargestellt. Verdeckte gegnerische Ablage- und Wunderkarten
werden nicht gespeichert. Die zusätzliche technische Anfrage für eine siebte
Karte enthüllt Spielern ohne diese Fähigkeit keine neue Hand. Sie erhalten dort
eine neutrale, von der Engine übersprungene Auswahl.

## Netzaufbau und Datenfluss

Ein kleiner Hand-Encoder und ein Encoder für die legalen Aktionsmöglichkeiten
arbeiten gemeinsam mit einem Verlaufs-Encoder. Letzterer verarbeitet jeden
Schritt mit denselben Gewichten und führt die Ergebnisse in zeitlicher
Reihenfolge zusammen. Kartenreihenfolge innerhalb einer Hand ist keine
Gedächtnisinformation. Eine Maske verhindert Beiträge leerer Verlaufsschritte.

Der zusätzliche Kontext wird zum bisherigen globalen Kontext addiert und von
Actor und Critic verwendet. Der Actor hat weiterhin seinen Kandidatenzweig für
die Bewertung einzelner Aktionen. Es handelt sich um eine begrenzte explizite
Historie mit Feedforward-Verarbeitung, nicht um ein rekurrentes Netz.

Die Umgebung verwaltet drei getrennte Speicher pro Partie. Gemeinsame
Gegner-Modellcaches enthalten keinen eigenen Partieverlauf. Nach bestätigten
Aktionen werden nur die jeweiligen eigenen Aktionsfelder ergänzt. Bereits
ausgegebene Beobachtungen bleiben unveränderte Momentaufnahmen.

Die zusätzliche Eingabe wird an den bestehenden globalen Rollout-Tensor
angehängt. Dadurch speichern PPO-Datensammlung, Minibatch-Updates und
Critic-Bootstrap dieselbe zum jeweiligen Zeitpunkt verfügbare Information.
Der Folgezustand wird vor einem eventuellen Reset bewertet. Unterstützte
Milestone-Wiederaufnahmen beginnen weiterhin neue Engine-Partien und damit
leere Verlaufsspeicher; sie behaupten keine exakte Fortsetzung dieser Partien.

## Checkpoints und Weitertraining

Die Voreinstellung gilt für neue Trainingsphasen. Zum Fortsetzen eines alten
Laufs ohne Gedächtnis mit `--resume` muss `--no-observation-memory` angegeben
werden, damit die gespeicherte Konfiguration übereinstimmt. Beim Auswerten
bestimmt weiterhin der Checkpoint die Architektur.

Die Architekturversionen sind **4** für einen Kopf und **5** für mehrere Köpfe;
die Speicher-Eingabe besitzt einen eigenen Schemahash. Bestehende Architektur-
Versionen bleiben für alte Modelle ladbar. Ein neuer Checkpoint verlangt eine
Umgebung mit vollständigem Verlauf; fehlende Erinnerung wird nicht stillschweigend
durch Nullen ersetzt.

Beim Initialisieren aus passenden bisherigen Gewichten startet die letzte
Projektion des Zusatzpfads bei null. Auf identischen Eingaben bleiben Actor-
Logits und Value-Vorhersagen zunächst erhalten. Die eingefrorenen Priors behalten
ihre ursprünglichen Eingaben und Gewichte. Erst das anschließende Training lernt,
die neue Information zu nutzen. Dies garantiert keinen Leistungsgewinn.

Beispiel für einen **kurzen technischen Check** aus `python/`, mit einem neuen
Ausgabeverzeichnis und einem vorhandenen Vier-Kopf-Checkpoint:

```bash
.venv/bin/python scripts/ppo_training/train_unified.py \
  --init-checkpoint runs/long-mixed-wonders-v2-seed231/latest.pt \
  --observation-memory --num-heads 4 --prior-strength 32 \
  --total-timesteps 128 --n-envs 2 --rollout-horizon 32 \
  --snapshot-every 64 --eval-every 128 --eval-games-per-matchup 1 \
  --out-dir runs/memory-technical-check
```

Diese kleine Prüfung ist kein wissenschaftlicher Spielstärkevergleich und legt
keinen neuen Gegnerpool fest. Ohne separate Poolkonfiguration verwendet sie
die vorhandene normale Snapshot-Auswahl. Ein langer Versuch benötigt das
[eigene Protokoll](../experiments/README.md).

Die Hauptauswertung über `scripts/eval_selfplay.py` und ihre Worker erkennen
die Erweiterung anhand des Checkpoints. Auch neue Checkpoints als eingefrorene
Gegner bekommen ihre eigene vollständige Beobachtungshistorie. Für reine
Berichts- und Dashboard-Neuberechnung ändern sich die Bedienbefehle nicht.

## Prüfung und Aussagegrenze

Die Regressionen stehen in
[test_observation_memory.py](../../python/tests/test_observation_memory.py).
Sie prüfen insbesondere private Informationsgrenzen, vergangene Momentaufnahmen,
Kartenreihenfolge, Neutralität der Migration, Schemafehler, beide Netzausgänge,
das Ausblenden unspielbarer siebter Hände und den vollständigen PPO-/Evaluationsweg.

Technischer Nachweis vom 29. Oktober 2026: **182 Tests bestanden, einer
übersprungen** (markierte Regression plus die separat ausgeführten unmarkierten
Tests). Serielle, gebündelte und parallele Evaluation liefern im geprüften
Speicher-Setup dieselben Aktionen und Ergebnisse. Zusätzlich wurde das tatsächliche
`long-mixed-wonders-v2-seed231/latest.pt` in einer temporären technischen Phase
mit 128 Übergängen weitertrainiert. Vor dem ersten Update stimmen Ausgaben
aller vier Köpfe exakt mit dem Ausgangsmodell überein. Ausgangsdatei und feste
Priors bleiben unverändert; deren Eingabepräfix bleibt 102/99. Auch die separate
detaillierte Checkpoint-Auswertung wurde mit dem erweiterten Modell ausgeführt.

Die Funktion ist implementiert und technisch prüfbar. Ob sie bessere Vorbereitung,
Gegnerantizipation oder höhere Spielstärke lernt, bleibt eine experimentelle Frage.
Forschungseinordnung: [Content-Bank](../research/CONTENT_BANK.md#prioritaeten).
