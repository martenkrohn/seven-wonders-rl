/* Display translations only: saved experiment data and identifiers stay unchanged. */
(() => {
  'use strict';
  const messages = {
    '7 Wonders · Self-Play-Auswertung': '7 Wonders · Self-play evaluation',
    'SELF-PLAY LABOR': 'SELF-PLAY LAB', 'AUSWERTUNG': 'EVALUATION',
    'Auswertungsbereiche': 'Evaluation sections', 'Lokale Auswertung': 'Local evaluation',
    'Alle Daten bleiben in deinem Browser.': 'All data stays in your browser.',
    'Öffentliche Beispieldaten': 'Public example data',
    'Öffentlicher Demokontext': 'Public demo context',
    'Zusätzliche Dateien bleiben im Browser.': 'Additional files stay in your browser.',
    'TRAINING VERSTEHEN': 'UNDERSTANDING TRAINING', '+ Auswertung hinzufügen': '+ Add evaluation',
    '7 WONDERS / ANALYSE': '7 WONDERS / ANALYSIS', 'Datenauswahl': 'Data selection',
    'Sprache': 'Language', 'Auswertung': 'Evaluation', 'Modellstand': 'Model checkpoint',
    '7 Wonders · Lokale Trainingsanalyse': '7 Wonders · Local training analysis',
    '7 Wonders · Öffentliche Demo · Trainingsanalyse': '7 Wonders · Public demo · Training analysis',
    'Sortieren · Filtern · Vergleichen': 'Sort · Filter · Compare',
    'Bitte JavaScript aktivieren, um Tabellen zu sortieren und die Auswertung anzuzeigen.': 'Enable JavaScript to sort tables and view the evaluation.',
    'Dark Mode: an': 'Dark mode: on', 'Dark Mode: aus': 'Dark mode: off',
    'Zu heller Darstellung wechseln': 'Switch to light mode', 'Zu dunkler Darstellung wechseln': 'Switch to dark mode',
    'Blau / Zivilbauten': 'Blue / Civilian', 'Rot / Militär': 'Red / Military',
    'Grün / Wissenschaft': 'Green / Science', 'Gelb / Handel': 'Yellow / Commercial',
    'Violett / Gilden': 'Purple / Guilds', 'Wunder': 'Wonders', 'Münzen': 'Treasury',
    'Braun': 'Brown', 'Grau': 'Grey', 'Blau': 'Blue', 'Rot': 'Red', 'Grün': 'Green',
    'Gelb': 'Yellow', 'Violett': 'Purple', 'Unbekannt': 'Unknown',
    'Überblick': 'Overview', 'Die wichtigsten Ergebnisse und die Spielweise deines Modells.': 'The key results and how your model plays.',
    'Punkteherkunft': 'Score breakdown', 'Durchschnitt, Streuung und Unterschiede zu den Gegnern.': 'Averages, distributions and differences from opponents.',
    'Zeitalter': 'Ages', 'Wie Wirtschaft, Militär und Punkte im Laufe einer Partie entstehen.': 'How the economy, military and scores develop during a game.',
    'Karten & Aktionen': 'Cards & actions', 'Welche Karten gewählt werden – und welche legalen Möglichkeiten ungenutzt bleiben.': 'Which cards are chosen—and which legal opportunities are passed over.',
    'Vergleiche': 'Comparisons', 'Modellstände, Spielsituationen und geladene Trainingsläufe vergleichen.': 'Compare checkpoints, game situations and loaded training runs.',
    'Datenqualität': 'Data quality', 'Vollständigkeit, Testbedingungen und Grenzen der Auswertung.': 'Coverage, test conditions and limits of the evaluation.',
    'nicht erfasst': 'not recorded', 'nicht verfügbar': 'not available',
    'Kein Intervall verfügbar': 'No interval available', ' bis ': ' to ', ' Punkte': ' points',
    'Untrainiertes Ausgangsmodell (Schritt 0)': 'Untrained initial model (step 0)',
    'Testmodell': 'Test model', 'Trainingsende': 'Final model', 'Zwischenstand': 'Intermediate checkpoint',
    'Zusätzlicher Zwischenstand {number}': 'Additional checkpoint {number}',
    ' – Schritt ': ' – step ', 'Zeilen': 'Rows', 'Alle': 'All',
    'Vorherige Seite': 'Previous page', 'Nächste Seite': 'Next page', 'CSV exportieren': 'Export CSV',
    '{count} Einträge · Seite {page} von {pages}': '{count} entries · Page {page} of {pages}',
    'Keine passenden Daten vorhanden.': 'No matching data.',
    'Handel + Münzen (nicht getrennt)': 'Commercial + treasury (combined)', 'Gesamt': 'Total',
    'Sieganteil': 'Win share', 'Eigene Punkte Ø': 'Own points, mean', 'Vorsprung Ø': 'Margin, mean',
    'Spiele': 'Games', 'Punkteprofile wurden nicht erfasst.': 'Score profiles were not recorded.',
    'Durchschnittliche Punkte je Spiel. Die Mittellinie markiert 0; negative Werte liegen links.': 'Average points per game. The centre line marks zero; negative values extend to the left.',
    'Eigene Punkte': 'Own points', 'Durchschnitt je abgeschlossenem Spiel': 'Average per completed game',
    'Punktevorsprung': 'Score margin', 'Gegen den stärksten Gegner je Partie': 'Against the strongest opponent in each game',
    'Abgeschlossene Spiele': 'Completed games', 'Von {attempts} Versuchen': 'Out of {attempts} attempts',
    'Diese Auswertung enthält unvollständige Spiele. Kennzahlen beziehen sich auf abgeschlossene Partien.': 'This evaluation contains incomplete games. Metrics describe completed games.',
    'Für diesen Modellstand fehlen teilweise Zeitalter- oder Aktionsdaten. Fehlende Werte werden als „nicht erfasst“ angezeigt.': 'Some age or action data is missing for this checkpoint. Missing values appear as “not recorded”.',
    'Woher kommen die Punkte?': 'Where do the points come from?',
    'Das Punkteprofil des ausgewählten Modellstands.': 'The score profile of the selected checkpoint.',
    'Wissenschaft: {fraction} der Spiele ohne Punkte': 'Science: {fraction} of games with zero points',
    'Bei vorhandenen Punkten: Ø {mean} · Maximum: {max}': 'When positive: mean {mean} · Maximum: {max}',
    'Keine Spiele mit Wissenschaftspunkten.': 'No games with science points.',
    'Der Test im Kontext': 'Test context', 'Alle Modellstände spielen gegen denselben festgelegten Gegnerplan.': 'All checkpoints play against the same fixed opponent plan.',
    'Spiel-Seeds': 'Game seed blocks', 'Test-Seed': 'Evaluation seed',
    'Vorsprung, 95%-Intervall': 'Margin, 95% interval', 'Ausgewähltes Modell': 'Selected model',
    'Merkmalsversion': 'Feature version',
    'Ein Drittel Sieganteil ist der Orientierungswert gleich starker Dreispieler-Policies. Geteilte Siege zählen anteilig.': 'A one-third win share is the reference for equally strong three-player policies. Tied wins receive fractional credit.',
    'Entwicklung der Modellstände': 'Checkpoint development', 'Spaltenüberschriften anklicken, um die Reihenfolge zu ändern.': 'Click a column heading to change the sort order.',
    'Punkteherkunft und Verteilung': 'Score breakdown and distribution',
    'P10–P90 beschreibt den mittleren Bereich der Einzelspiele, kein Unsicherheitsintervall des Durchschnitts.': 'P10–P90 describes the middle range of individual games, not an uncertainty interval for the mean.',
    'Kategorie': 'Category', 'Mittelwert': 'Mean', 'Median': 'Median', 'Maximum': 'Maximum',
    'Ohne Punkte': 'Zero points', 'Ø bei positiven Punkten': 'Mean when positive',
    'Punkteprofil': 'Score profile', 'Abstand zum besten Gegner': 'Margin against the best opponent',
    'Je Partie derselbe Gegner, ausgewählt anhand seiner Gesamtpunkte.': 'The same opponent for each category within a game, selected by total score.', 'Abstand Ø': 'Margin, mean',
    'Kartenfarbe und Punkteherkunft sind verschieden: Wunder und Wissenschaftsgilden können zu Wissenschaft beitragen; Wunderschilde zu Militär. Kategorien-Maxima stammen möglicherweise aus unterschiedlichen Spielen.': 'Card colour and scoring category differ: wonders and Scientists Guild can contribute to science; wonder shields can contribute to military. Category maxima may come from different games.',
    'Vollständige Zeitalterfolgen: {covered} von {completed} Spielen. Mittelwerte können unterschiedliche verfügbare Teilmengen beschreiben.': 'Complete age sequences: {covered} of {completed} games. Means may describe different available subsets.',
    'Aufbau über drei Zeitalter': 'Development across three ages',
    'Zustände nach der Militärwertung; Zeitalter III zusätzlich nach finalen Gildeneffekten.': 'States after military scoring; Age III also includes final guild effects.',
    'ENDE ZEITALTER {age}': 'END OF AGE {age}', 'Punkte im Mittel · {games} Spiele': 'Mean points · {games} games',
    'Punktestand nicht erfasst': 'Score not recorded', 'Wertungsbereich': 'Scoring category',
    'Ende I': 'End of I', 'Ende II': 'End of II', 'Ende III': 'End of III',
    'Wirtschaft und Spielaufbau': 'Economy and development', 'Münzbestand': 'Coin balance',
    'Militärstärke / Schilde': 'Military strength / shields', 'Vollendete Wunderabschnitte': 'Completed wonder stages',
    'Feste Produktion Holz': 'Fixed wood production', 'Feste Produktion Erz': 'Fixed ore production',
    'Feste Produktion Lehm': 'Fixed clay production', 'Feste Produktion Stein': 'Fixed stone production',
    'Feste Produktion Stoff': 'Fixed textile production', 'Feste Produktion Glas': 'Fixed glass production',
    'Feste Produktion Papyrus': 'Fixed papyrus production', 'Kennzahl': 'Metric',
    'Zwischenstände sind Wertungen des aktuellen Zustands. Münzen können später ausgegeben werden; Wissenschaft und Gilden wachsen mit weiteren Karten. Ein Wertungszuwachs ist keine kausale Bewertung der zuletzt gewählten Aktion. Wahlproduktion ist nicht als feste Produktion eingerechnet.': 'Intermediate scores value the current state. Coins may be spent later; science and guild scores grow with further cards. A score increase is not the causal value of the last action. Choice production is excluded from fixed production.',
    'Einzelne Karten und legale Alternativen': 'Individual cards and legal alternatives',
    'Suche, Farbe und Zeitalter sind kombinierbar. CSV exportiert alle aktuell gefilterten Zeilen, auch über mehrere Seiten.': 'Combine search, colour and age filters. CSV exports all filtered rows, including those on other pages.',
    'Kartenname': 'Card name', 'Zum Beispiel Workshop': 'For example, Workshop',
    'Farbe': 'Colour', 'Alle Farben': 'All colours', 'Alle Zeitalter': 'All ages', 'Karte': 'Card',
    'Bauaktionen': 'Build actions', 'Partien mit Bau (#)': 'Games with a build (#)', 'Partien mit Bau (%)': 'Games with a build (%)',
    'Legale Angebote': 'Legal opportunities', 'Auswahlquote': 'Selection rate',
    'Partien mit Bau = Anteil der Spiele, in denen die Karte als Gebäude gebaut wurde. Auswahlquote = Bauaktionen / Entscheidungen mit dieser legal baubaren Karte. Wiederholte Angebote zählen mehrfach. Verkäufe und Wunderbau zählen nicht als Gebäude.': 'Games with a build = games in which the card was built as a structure. Selection rate = build actions / decisions where that card was legal to build. Repeated opportunities count separately. Discards and wonder construction do not count as buildings.',
    'Gebaute Farben und Aktionen je Zeitalter': 'Built colours and actions by age',
    'Durchschnitt tatsächlich bestätigter Aktionen je Spiel.': 'Average confirmed actions per game.',
    'Wunderbau': 'Wonder construction', 'Kartenverkauf': 'Discard for coins', 'Karten / Aktionen': 'Cards / actions',
    'Zeitalter I': 'Age I', 'Zeitalter II': 'Age II', 'Zeitalter III': 'Age III',
    'Modellstände auf demselben Testplan': 'Checkpoints on the same test plan',
    'Vergleiche funktionieren innerhalb dieser Auswertung gegen unveränderte Gegner und Spiel-Seeds.': 'Comparisons within this evaluation use unchanged opponents and game seeds.',
    'Direkter Fortschrittsvergleich': 'Paired progress comparison',
    'Das geprüfte End-/Testmodell wird jeweils mit einem früheren Stand auf demselben Plan verglichen.': 'The final/test model is compared with each earlier checkpoint on the same plan.',
    ' Prozentpunkte': ' percentage points', 'Hinweis auf Verbesserung': 'Evidence of improvement',
    'Hinweis auf Verschlechterung': 'Evidence of deterioration', 'Noch nicht eindeutig': 'Not yet conclusive',
    'Zu wenige Daten': 'Insufficient data', 'Verglichen mit': 'Compared with',
    'Vorsprung-Zuwachs': 'Margin gain', '95%-Intervall / Punkte': '95% interval / points',
    'Sieganteil-Zuwachs': 'Win-share gain', '95%-Intervall / Prozentpunkte': '95% interval / percentage points',
    'Einordnung des Vorsprungs': 'Margin interpretation', 'Unterschiede zwischen Spielsituationen': 'Differences between game situations',
    'Gruppieren nach': 'Group by', 'Gegnerpaar': 'Opponent pair', 'Spielerposition': 'Player seat',
    'Spielausgang': 'Outcome', 'Gesamtpunkte': 'Total points', 'Gruppe': 'Group',
    'Ausgewählte Punkte Ø': 'Selected points, mean', 'Geladene Trainingsläufe vergleichen': 'Compare loaded training runs',
    'Weitere analysis.json-Dateien kannst du oben über „Auswertung hinzufügen“ öffnen.': 'Open more analysis.json files using “Add evaluation” above.',
    'Modell': 'Model', 'Punkte Ø': 'Points, mean', 'Gegnerreferenzen': 'Reference opponents',
    'Laufübergreifende Werte sind nur bei vergleichbaren Gegnern und Testbedingungen direkt interpretierbar. Gleich benannte Checkpoints aus verschiedenen Trainingsläufen sind verschiedene Modelle. Untergruppen sind beschreibend, keine separaten Signifikanztests.': 'Cross-run values are directly interpretable only with comparable opponents and test conditions. Identically named checkpoints from different runs are different models. Subgroups are descriptive, not separate significance tests.',
    'Vollständigkeit der Spieldaten': 'Game data coverage',
    'Erfasste Spiele im Verhältnis zu den abgeschlossenen Partien des ausgewählten Modells.': 'Recorded games relative to completed games for the selected model.',
    'Information': 'Information', 'Erfasst / Anzahl': 'Recorded / count',
    'Endzustände aller Spieler': 'Final states for all players', 'Vollständige Zeitalterfolgen': 'Complete age sequences',
    'Aktionsprotokolle': 'Action traces', 'Fehlgeschlagene Spiele': 'Failed games',
    'Abgebrochene Spiele': 'Truncated games', 'Fehlerhafte Punktesummen': 'Score-sum mismatches',
    'Fehlerhafte Zeitalterfolgen': 'Invalid age sequences', 'Unbekannte gebaute Karten': 'Unknown built cards',
    'Testbedingungen und Modellidentität': 'Test conditions and model identity',
    'Unabhängige Spiel-Seeds': 'Independent game seed blocks', 'Modellpfad': 'Model path',
    'Modell SHA-256': 'Model SHA-256', 'Simulator SHA-256': 'Simulator SHA-256',
    'Kartendaten SHA-256': 'Card database SHA-256', 'Auswertungsdaten herunterladen': 'Download evaluation data',
    'Die Interpretation gilt für die verwendete Simulatorvariante. Gespeicherte Berichte älterer Läufe können andere Merkmalsversionen verwenden. Wissenschaftsarmut ist ein Beobachtungsbefund, kein Beweis für die optimale Strategie.': 'Interpretation applies to the simulator variant used. Older reports may use different feature versions. Limited science use is an observation, not proof of an optimal strategy.',
    '{games} Spiele · {seeds} Spiel-Seeds': '{games} games · {seeds} seed blocks',
    'Datei zu groß. Bitte analysis.json statt eines Spielprotokolls auswählen.': 'File too large. Select analysis.json rather than a game log.',
    'Bitte eine analysis.json aus der detaillierten Auswertung auswählen.': 'Select an analysis.json file from a detailed evaluation.',
    'Ungültige JSON-Datei.': 'Invalid JSON file.', 'Import nicht möglich: ': 'Import failed: ',
    'Sieg': 'Win', 'Niederlage': 'Loss', 'Gleichstand': 'Tie',
    'Öffentlicher Experimentstand · explorative Ergebnisse, kein allgemeiner Spielstärke-Benchmark.': 'Public experiment snapshot · exploratory results, not a general playing-strength benchmark.',
    'Exportiert: ': 'Exported: ', 'Quellcode und Experimentdokumentation': 'Source code & experiment documentation',
    'Die Beispieldaten sind öffentlich. Zusätzlich geöffnete Dateien werden lokal im Browser verarbeitet.': 'The example data is public. Additional files you open are processed locally in your browser.',
    'tabelle.csv': 'table.csv', 'modellvergleich.csv': 'model-comparison.csv', 'punkteherkunft.csv': 'score-breakdown.csv',
    'zeitalter-punkte.csv': 'scores-by-age.csv', 'zeitalter-wirtschaft.csv': 'economy-by-age.csv',
    'karten-gefiltert.csv': 'filtered-cards.csv', 'aktionen-je-zeitalter.csv': 'actions-by-age.csv',
    'fortschritt.csv': 'progress.csv', 'untergruppen.csv': 'subgroups.csv',
    'laufvergleich.csv': 'run-comparison.csv', 'datenqualitaet.csv': 'data-quality.csv'
  };
  const storageKey = 'sevenwonders-dashboard-language';
  let language = 'en';
  try {const saved=localStorage.getItem(storageKey);if(saved==='en'||saved==='de')language=saved;} catch (_) {}
  const listeners = [];
  const t = (key, values={}) => String(language==='en' ? messages[key]??key : key)
    .replace(/\{(\w+)\}/g,(match,name)=>Object.hasOwn(values,name)?String(values[name]):match);
  function apply(root=document) {
    document.documentElement.lang=language;
    for(const node of root.querySelectorAll('[data-i18n]'))node.textContent=t(node.getAttribute('data-i18n'));
    for(const attr of ['title','placeholder','aria-label']){
      for(const node of root.querySelectorAll('[data-i18n-'+attr+']'))node.setAttribute(attr,t(node.getAttribute('data-i18n-'+attr)));
    }
    for(const node of root.querySelectorAll('[data-language-en]')){
      node.textContent=node.getAttribute('data-language-'+language)||node.getAttribute('data-language-en');
    }
    const select=root.getElementById?.('language-select');if(select)select.value=language;
  }
  function setLanguage(value) {
    if(value!=='en'&&value!=='de')return false;
    language=value;
    try {localStorage.setItem(storageKey,value);} catch (_) {}
    if(typeof document!=='undefined')apply();
    listeners.forEach(fn=>fn());return true;
  }
  globalThis.SevenWondersI18n={t,locale:()=>language==='de'?'de-DE':'en-US',language:()=>language,
    setLanguage,apply,subscribe:fn=>listeners.push(fn)};
  if(typeof document!=='undefined')document.documentElement.lang=language;
})();
