/* Browser-local analysis: no server, remote assets or network calls. */
(() => {
  'use strict';
  const i18n = globalThis.SevenWondersI18n;
  const t = (key, values) => i18n?.t(key, values)??key;
  const locale = () => i18n?.locale()||'en-US';
  const numeric = value => typeof value === 'number' && Number.isFinite(value);
  function compareRows(a, b, key, direction = 1) {
    const x = a[key], y = b[key];
    const missing = v => v === null || v === undefined || (typeof v === 'number' && !Number.isFinite(v));
    if (missing(x) || missing(y)) return missing(x) === missing(y) ? 0 : missing(x) ? 1 : -1;
    return direction * (numeric(x) && numeric(y) ? x - y :
      new Intl.Collator(locale(), {numeric: true, sensitivity: 'base'}).compare(t(String(x)), t(String(y))));
  }
  function filterCards(cards, {query = '', color = '', age = ''} = {}) {
    const q = query.trim().toLocaleLowerCase(locale());
    return cards.filter(c => (!q || String(c.card).toLocaleLowerCase(locale()).includes(q)) &&
      (!color || c.color === color) && (!age || c.era === Number(age)));
  }
  function csvText(columns, rows) {
    const escape = (value, fromNumber=false) => {
      let text = value == null ? '' : String(value);
      if (!fromNumber && !numeric(value) && /^[=+\-@]/.test(text)) text = "'" + text;
      return '"' + text.replaceAll('"', '""') + '"';
    };
    return '\ufeff' + [columns.map(c => escape(t(c.label))).join(';'),
      ...rows.map(row => columns.map(c => {
        const value=row[c.key];
        const formatted=value!=null && c.format ? c.format(value) : value;
        return escape(typeof formatted==='string'?t(formatted):formatted, numeric(value));
      }).join(';'))].join('\r\n');
  }
  globalThis.SevenWondersDashboard = {compareRows, filterCards, csvText};
  if (typeof document === 'undefined') return;

  const CATEGORY = {blue:'Blau / Zivilbauten',military:'Rot / Militär',science:'Grün / Wissenschaft',yellow:'Gelb / Handel',guild:'Violett / Gilden',wonder:'Wunder',treasury:'Münzen'};
  const COLORS = {brown:'Braun',grey:'Grau',blue:'Blau',red:'Rot',green:'Grün',yellow:'Gelb',purple:'Violett',unknown:'Unbekannt'};
  const PALETTE = {blue:'#638cbe',military:'#bf6159',science:'#639768',yellow:'#d1ab45',guild:'#9274af',wonder:'#aa8c67',treasury:'#acaa9d'};
  const VIEWS = {
    overview:['Überblick','01','Die wichtigsten Ergebnisse und die Spielweise deines Modells.'],
    points:['Punkteherkunft','02','Durchschnitt, Streuung und Unterschiede zu den Gegnern.'],
    ages:['Zeitalter','03','Wie Wirtschaft, Militär und Punkte im Laufe einer Partie entstehen.'],
    cards:['Karten & Aktionen','04','Welche Karten gewählt werden – und welche legalen Möglichkeiten ungenutzt bleiben.'],
    comparisons:['Vergleiche','05','Modellstände, Spielsituationen und geladene Trainingsläufe vergleichen.'],
    quality:['Datenqualität','06','Vollständigkeit, Testbedingungen und Grenzen der Auswertung.']
  };
  let entries = JSON.parse(document.getElementById('evaluation-data').textContent);
  const state = {entry:0, model:'candidate', view:'overview', tables:new Map(),
    filters:{query:'',color:'',age:''}, groups:{kind:'opponents',category:'total'}, error:null};
  const $ = id => document.getElementById(id);
  globalThis.SevenWondersTheme?.attach($('theme-toggle'));
  const num = (value, digits=2) => numeric(value) ? new Intl.NumberFormat(locale(),{minimumFractionDigits:digits,maximumFractionDigits:digits}).format(value) : t('nicht erfasst');
  const pct = value => numeric(value) ? num(value*100,1)+' %' : t('nicht erfasst');
  const signed = value => numeric(value) ? (value>0?'+':'')+num(value) : t('nicht erfasst');
  const interval = (value, percent=false) => Array.isArray(value) && value.length===2 ?
    (percent ? pct(value[0])+t(' bis ')+pct(value[1]) : signed(value[0])+t(' bis ')+signed(value[1])+t(' Punkte')) : t('Kein Intervall verfügbar');
  const current = () => entries[state.entry];
  const entryTitle = entry => i18n?.language()==='de'&&entry.title_de?entry.title_de:entry.title;
  const report = () => current().report;
  const model = () => report().models[state.model];
  function el(tag, text, className) {
    const element = document.createElement(tag);
    if (text !== null && text !== undefined) element.textContent = typeof text==='string'?t(text):text;
    if (className) element.className = className;
    return element;
  }
  function modelLabel(entry, key) {
    const desc = entry.report.model_descriptors?.[key] || {};
    const match = String(desc.checkpoint || '').match(/checkpoint_(\d+)\.pt$/);
    const labelStep=String(entry.labels?.[key]||'').match(/Schritt ([\d.]+)/);
    const step = desc.timestep ?? (match ? Number(match[1]) : labelStep?Number(labelStep[1].replaceAll('.','')):null);
    if (key === 'initial') return t('Untrainiertes Ausgangsmodell (Schritt 0)');
    if(key!=='candidate'&&!numeric(step))return t('Zusätzlicher Zwischenstand {number}',{number:key.replace('comparison_','')});
    const name = key === 'candidate' ? (match ? 'Testmodell' : 'Trainingsende') : 'Zwischenstand';
    return t(name)+(numeric(step)?t(' – Schritt ')+num(step,0):'');
  }
  function panel(parent, title, description) {
    const box = el('section',null,'panel');
    box.append(el('h2',title));
    if (description) box.append(el('p',description,'muted'));
    parent.append(box);
    return box;
  }
  function notice(parent, text, good=false) {parent.append(el('div',text,'notice'+(good?' good':'')));}
  function button(text, handler) {
    const b=el('button',text); b.type='button'; b.addEventListener('click',handler); return b;
  }
  function download(text,name,type='text/csv;charset=utf-8') {
    const url=URL.createObjectURL(new Blob([text],{type})), link=el('a');
    link.href=url; link.download=t(name); document.body.append(link); link.click(); link.remove();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function selectControl(parent,label,options,onchange) {
    const wrapper=el('label',label), select=el('select');
    for(const [value,text] of options) {const option=el('option',text);option.value=value;select.append(option);}
    select.addEventListener('change',onchange);wrapper.append(select);parent.append(wrapper);return select;
  }
  const col=(key,label,type='number',format=null)=>({key,label:t(label),type,format});
  function table(parent,columns,getRows,{sort=null,direction=-1,filename='tabelle.csv'}={}) {
    const tableKey=[state.entry,state.model,state.view,filename].join(':');
    let {sortKey=sort||columns[0].key,sortDirection=direction,page=0,pageSize=25}=state.tables.get(tableKey)||{};
    const toolbar=el('div',null,'table-toolbar'), count=el('span'), controls=el('div');
    const sizeLabel=el('label','Zeilen'), size=el('select');
    for(const [value,label] of [['25','25'],['50','50'],['0','Alle']]) {const opt=el('option',label);opt.value=value;size.append(opt);}
    sizeLabel.append(size);controls.append(sizeLabel);
    size.value=String(pageSize);
    const previous=button('←',()=>{page--;draw();}), next=button('→',()=>{page++;draw();});
    previous.setAttribute('aria-label',t('Vorherige Seite'));next.setAttribute('aria-label',t('Nächste Seite'));
    controls.append(previous,next,button('CSV exportieren',()=>download(csvText(columns,sortedRows()),filename)));
    toolbar.append(count,controls);
    const wrap=el('div',null,'table-wrap');parent.append(toolbar,wrap);
    size.addEventListener('change',()=>{pageSize=Number(size.value);page=0;draw();});
    function sortedRows(){return [...getRows()].sort((a,b)=>compareRows(a,b,sortKey,sortDirection));}
    function draw(reset=false) {
      if(reset) page=0;
      const rows=sortedRows(), pages=pageSize?Math.max(1,Math.ceil(rows.length/pageSize)):1;
      page=Math.max(0,Math.min(page,pages-1));
      count.textContent=t('{count} Einträge · Seite {page} von {pages}',{count:num(rows.length,0),page:num(page+1,0),pages:num(pages,0)});
      state.tables.set(tableKey,{sortKey,sortDirection,page,pageSize});
      previous.disabled=page===0;next.disabled=page+1>=pages;wrap.replaceChildren();
      const renderedTable=el('table'),thead=el('thead'),tr=el('tr');
      for(const column of columns){
        const th=el('th',null,column.type==='number'?'numeric':'');
        const active=sortKey===column.key;
        th.setAttribute('scope','col');th.setAttribute('aria-sort',active?(sortDirection===1?'ascending':'descending'):'none');
        th.append(button(column.label+(active?(sortDirection===1?' ↑':' ↓'):' ↕'),()=>{
          sortDirection=active?-sortDirection:(column.type==='number'?-1:1);sortKey=column.key;page=0;draw();
        }));tr.append(th);
      }
      thead.append(tr);renderedTable.append(thead);const body=el('tbody');
      const shown=pageSize?rows.slice(page*pageSize,(page+1)*pageSize):rows;
      for(const row of shown){const tr=el('tr');
        for(const column of columns){const value=row[column.key];
          const text=value==null?t('nicht erfasst'):column.format?column.format(value):column.type==='number'?num(value):t(String(value));
          tr.append(el('td',text,column.type==='number'?'numeric':''));
        }body.append(tr);
      }
      if(!shown.length){const tr=el('tr'),td=el('td','Keine passenden Daten vorhanden.','empty');td.colSpan=columns.length;tr.append(td);body.append(tr);}
      renderedTable.append(body);wrap.append(renderedTable);
    }
    draw();return draw;
  }
  function scoreRows(m) {
    return [...Object.entries(CATEGORY),['commercial_combined','Handel + Münzen (nicht getrennt)'],['total','Gesamt']]
      .filter(([key])=>key!=='commercial_combined'||m.categories?.[key])
      .map(([key,label])=>({category:label,key,...(m.categories?.[key]||{})}));
  }
  const strengthRows=entry=>Object.entries(entry.report.models).map(([key,m])=>({
    model:modelLabel(entry,key),win_share:m.win_share,points:m.agent_vp_mean,margin:m.vp_diff_mean,games:m.completed}));
  function strengthTable(parent,entry=current()) {
    table(parent,[col('model','Modellstand','string'),col('win_share','Sieganteil','number',pct),col('points','Eigene Punkte Ø'),
      col('margin','Vorsprung Ø','number',signed),col('games','Spiele','number',v=>num(v,0))],()=>strengthRows(entry),{sort:'win_share',filename:'modellvergleich.csv'});
  }
  function pointBars(parent,m) {
    const rows=scoreRows(m).filter(r=>r.key!=='total'&&numeric(r.mean));
    if(!rows.length){parent.append(el('p','Punkteprofile wurden nicht erfasst.','empty'));return;}
    const max=Math.max(1,...rows.map(r=>Math.abs(r.mean)));
    for(const row of rows){const line=el('div',null,'bar-row'),label=el('span',row.category),swatch=el('span',null,'swatch');
      swatch.style.background=PALETTE[row.key]||'#999';label.prepend(swatch);
      const track=el('div',null,'bar-track'),fill=el('div',null,'bar-fill'),width=Math.abs(row.mean)/max*48;
      fill.style.width=width+'%';fill.style.left=(row.mean>=0?50:50-width)+'%';fill.style.background=PALETTE[row.key]||'#999';
      track.append(fill);line.append(label,track,el('span',num(row.mean),'bar-value'));parent.append(line);
    }
    parent.append(el('p','Durchschnittliche Punkte je Spiel. Die Mittellinie markiert 0; negative Werte liegen links.','definition'));
  }
  function overview(parent) {
    const m=model(), stats=el('div',null,'stats');
    for(const [label,value,detail] of [
      ['Sieganteil',pct(m.win_share),'95 %: '+interval(m.win_share_ci95,true)],
      ['Eigene Punkte',num(m.agent_vp_mean),'Durchschnitt je abgeschlossenem Spiel'],
      ['Punktevorsprung',signed(m.vp_diff_mean),'Gegen den stärksten Gegner je Partie'],
      ['Abgeschlossene Spiele',num(m.completed,0),t('Von {attempts} Versuchen',{attempts:num(m.attempted,0)})]
    ]){const tile=el('section',null,'stat');tile.append(el('div',label,'stat-label'),el('div',value,'stat-value'),el('small',detail));stats.append(tile);}
    parent.append(stats);
    if(!m.valid_for_selection) notice(parent,'Diese Auswertung enthält unvollständige Spiele. Kennzahlen beziehen sich auf abgeschlossene Partien.');
    if((m.coverage?.complete_age_sequences||0)<m.completed) notice(parent,'Für diesen Modellstand fehlen teilweise Zeitalter- oder Aktionsdaten. Fehlende Werte werden als „nicht erfasst“ angezeigt.');
    const grid=el('div',null,'grid-two');parent.append(grid);
    const points=panel(grid,'Woher kommen die Punkte?','Das Punkteprofil des ausgewählten Modellstands.');pointBars(points,m);
    const science=m.categories?.science;
    if(science){const box=el('div',null,'science-box');box.append(el('strong',t('Wissenschaft: {fraction} der Spiele ohne Punkte',{fraction:pct(science.zero_fraction)})),
      el('span',numeric(science.positive_mean)?t('Bei vorhandenen Punkten: Ø {mean} · Maximum: {max}',{mean:num(science.positive_mean),max:num(science.max,0)}):'Keine Spiele mit Wissenschaftspunkten.'));points.append(box);}
    const context=panel(grid,'Der Test im Kontext','Alle Modellstände spielen gegen denselben festgelegten Gegnerplan.');
    const refs=el('ul',null,'reference-list');for(const ref of report().references||[]) refs.append(el('li',String(ref.checkpoint).split(/[\\/]/).pop()));context.append(refs);
    const detail=el('div',null,'detail-list');for(const [label,value] of [['Spiel-Seeds',num(m.seed_blocks,0)],['Test-Seed',num(report().seed,0)],
      ['Vorsprung, 95%-Intervall',interval(m.vp_diff_ci95)],['Ausgewähltes Modell',modelLabel(current(),state.model)],
      ['Merkmalsversion',num(current().provenance?.feature_semantics_version,0)]]){
      const cell=el('div');cell.append(el('small',label),el('span',value));detail.append(cell);
    }context.append(detail);context.append(el('p','Ein Drittel Sieganteil ist der Orientierungswert gleich starker Dreispieler-Policies. Geteilte Siege zählen anteilig.','definition'));
    strengthTable(panel(parent,'Entwicklung der Modellstände','Spaltenüberschriften anklicken, um die Reihenfolge zu ändern.'));
  }
  function points(parent) {
    const box=panel(parent,'Punkteherkunft und Verteilung','P10–P90 beschreibt den mittleren Bereich der Einzelspiele, kein Unsicherheitsintervall des Durchschnitts.');
    table(box,[col('category','Kategorie','string'),col('n','Spiele','number',v=>num(v,0)),col('mean','Mittelwert'),col('median','Median'),col('p10','P10'),col('p90','P90'),col('max','Maximum'),
      col('zero_fraction','Ohne Punkte','number',pct),col('positive_mean','Ø bei positiven Punkten')],()=>scoreRows(model()),{sort:'mean',filename:'punkteherkunft.csv'});
    const grid=el('div',null,'grid-two');parent.append(grid);pointBars(panel(grid,'Punkteprofil'),model());
    const delta=panel(grid,'Abstand zum besten Gegner','Je Partie derselbe Gegner, ausgewählt anhand seiner Gesamtpunkte.');
    const rows=Object.entries(CATEGORY).map(([key,category])=>({category,...(model().category_margin_to_best?.[key]||{})}));
    table(delta,[col('category','Kategorie','string'),col('mean','Abstand Ø','number',signed),col('p10','P10'),col('p90','P90')],()=>rows,{sort:'mean'});
    notice(parent,'Kartenfarbe und Punkteherkunft sind verschieden: Wunder und Wissenschaftsgilden können zu Wissenschaft beitragen; Wunderschilde zu Militär. Kategorien-Maxima stammen möglicherweise aus unterschiedlichen Spielen.');
  }
  function ages(parent) {
    const m=model(),coverage=m.coverage?.complete_age_sequences||0;
    if(coverage<m.completed) notice(parent,t('Vollständige Zeitalterfolgen: {covered} von {completed} Spielen. Mittelwerte können unterschiedliche verfügbare Teilmengen beschreiben.',{covered:num(coverage,0),completed:num(m.completed,0)}));
    const box=panel(parent,'Aufbau über drei Zeitalter','Zustände nach der Militärwertung; Zeitalter III zusätzlich nach finalen Gildeneffekten.');
    const tiles=el('div',null,'age-grid');for(const age of [1,2,3]){const card=el('div',null,'age-card'),stat=m.age_points?.[age]?.total;
      card.append(el('strong',t('ENDE ZEITALTER {age}',{age})),el('div',num(stat?.mean)),el('small',stat?t('Punkte im Mittel · {games} Spiele',{games:num(stat.n,0)}):'Punktestand nicht erfasst'));tiles.append(card);
    }box.append(tiles);
    const rows=[...Object.entries(CATEGORY),['total','Gesamt']].map(([key,category])=>({category,a1:m.age_points?.[1]?.[key]?.mean,a2:m.age_points?.[2]?.[key]?.mean,a3:m.age_points?.[3]?.[key]?.mean}));
    table(box,[col('category','Wertungsbereich','string'),col('a1','Ende I'),col('a2','Ende II'),col('a3','Ende III')],()=>rows,{sort:'a3',filename:'zeitalter-punkte.csv'});
    const economy=panel(parent,'Wirtschaft und Spielaufbau');
    const metrics={coins:'Münzbestand',shields:'Militärstärke / Schilde',wonder_stage:'Vollendete Wunderabschnitte',wood:'Feste Produktion Holz',ore:'Feste Produktion Erz',clay:'Feste Produktion Lehm',stone:'Feste Produktion Stein',loom:'Feste Produktion Stoff',glass:'Feste Produktion Glas',papyrus:'Feste Produktion Papyrus'};
    table(economy,[col('metric','Kennzahl','string'),col('a1','Ende I'),col('a2','Ende II'),col('a3','Ende III')],()=>Object.entries(metrics).map(([key,metric])=>({metric,a1:m.age_economy?.[1]?.[key]?.mean,a2:m.age_economy?.[2]?.[key]?.mean,a3:m.age_economy?.[3]?.[key]?.mean})),{sort:'metric',direction:1,filename:'zeitalter-wirtschaft.csv'});
    notice(parent,'Zwischenstände sind Wertungen des aktuellen Zustands. Münzen können später ausgegeben werden; Wissenschaft und Gilden wachsen mit weiteren Karten. Ein Wertungszuwachs ist keine kausale Bewertung der zuletzt gewählten Aktion. Wahlproduktion ist nicht als feste Produktion eingerechnet.');
  }
  function cards(parent) {
    const m=model(), box=panel(parent,'Einzelne Karten und legale Alternativen','Suche, Farbe und Zeitalter sind kombinierbar. CSV exportiert alle aktuell gefilterten Zeilen, auch über mehrere Seiten.');
    const filters=el('div',null,'filters');box.append(filters);
    const searchLabel=el('label','Kartenname'),search=el('input');search.type='search';search.placeholder=t('Zum Beispiel Workshop');search.value=state.filters.query;searchLabel.append(search);filters.append(searchLabel);
    let refresh=()=>{};
    const color=selectControl(filters,'Farbe',[['','Alle Farben'],...Object.entries(COLORS)],()=>{state.filters.color=color.value;refresh(true);});
    const age=selectControl(filters,'Zeitalter',[['','Alle Zeitalter'],['1','I'],['2','II'],['3','III']],()=>{state.filters.age=age.value;refresh(true);});
    color.value=state.filters.color;age.value=state.filters.age;
    refresh=table(box,[col('card','Karte','string'),col('color_label','Farbe','string'),col('era','Zeitalter','number',v=>num(v,0)),col('built','Bauaktionen','number',v=>num(v,0)),
      col('games_built','Partien mit Bau (#)','number',v=>num(v,0)),col('build_game_fraction','Partien mit Bau (%)','number',pct),col('legal_opportunities','Legale Angebote','number',v=>num(v,0)),
      col('selection_fraction','Auswahlquote','number',pct)],()=>filterCards(m.cards||[],{query:search.value,color:color.value,age:age.value}).map(c=>({...c,color_label:COLORS[c.color]||c.color})),{sort:'built',filename:'karten-gefiltert.csv'});
    search.addEventListener('input',()=>{state.filters.query=search.value;refresh(true);});
    box.append(el('p','Partien mit Bau = Anteil der Spiele, in denen die Karte als Gebäude gebaut wurde. Auswahlquote = Bauaktionen / Entscheidungen mit dieser legal baubaren Karte. Wiederholte Angebote zählen mehrfach. Verkäufe und Wunderbau zählen nicht als Gebäude.','definition'));
    const actions=panel(parent,'Gebaute Farben und Aktionen je Zeitalter','Durchschnitt tatsächlich bestätigter Aktionen je Spiel.');
    const buckets={...COLORS,wonder:'Wunderbau',discard:'Kartenverkauf'};
    table(actions,[col('category','Karten / Aktionen','string'),col('a1','Zeitalter I'),col('a2','Zeitalter II'),col('a3','Zeitalter III'),col('total','Gesamt')],()=>Object.entries(buckets).map(([key,category])=>{
      const known=(m.coverage?.action_traces||0)>0, values=[1,2,3].map(a=>known?(m.age_actions?.[a]?.[key]??0):null);
      return {category,a1:values[0],a2:values[1],a3:values[2],total:known?values.reduce((a,b)=>a+b,0):null};
    }),{sort:'total',filename:'aktionen-je-zeitalter.csv'});
  }
  function comparisons(parent) {
    strengthTable(panel(parent,'Modellstände auf demselben Testplan','Vergleiche funktionieren innerhalb dieser Auswertung gegen unveränderte Gegner und Spiel-Seeds.'));
    const paired=panel(parent,'Direkter Fortschrittsvergleich','Das geprüfte End-/Testmodell wird jeweils mit einem früheren Stand auf demselben Plan verglichen.');
    const rows=Object.entries(report().paired_progress||{}).filter(([,p])=>p.paired_valid).map(([key,p])=>({model:modelLabel(current(),key),margin:p.vp_diff_gain,share:p.win_share_gain,
      margin_ci:interval(p.vp_diff_gain_ci95),share_ci:Array.isArray(p.win_share_gain_ci95)?num(100*p.win_share_gain_ci95[0])+t(' bis ')+num(100*p.win_share_gain_ci95[1])+t(' Prozentpunkte'):t('nicht verfügbar'),
      interpretation:Array.isArray(p.vp_diff_gain_ci95)?p.vp_diff_gain_ci95[0]>0?'Hinweis auf Verbesserung':p.vp_diff_gain_ci95[1]<0?'Hinweis auf Verschlechterung':'Noch nicht eindeutig':'Zu wenige Daten'}));
    table(paired,[col('model','Verglichen mit','string'),col('margin','Vorsprung-Zuwachs','number',signed),col('margin_ci','95%-Intervall / Punkte','string'),col('share','Sieganteil-Zuwachs','number',v=>signed(v*100)+t(' Prozentpunkte')),col('share_ci','95%-Intervall / Prozentpunkte','string'),col('interpretation','Einordnung des Vorsprungs','string')],()=>rows,{sort:'margin',filename:'fortschritt.csv'});
    const groups=panel(parent,'Unterschiede zwischen Spielsituationen');const controls=el('div',null,'group-controls');groups.append(controls);let update=()=>{};
    const kind=selectControl(controls,'Gruppieren nach',[['opponents','Gegnerpaar'],['seat','Spielerposition'],['wonder','Wunder'],['outcome','Spielausgang']],()=>{state.groups.kind=kind.value;update(true);});
    const category=selectControl(controls,'Punkteprofil',[['total','Gesamtpunkte'],...Object.entries(CATEGORY)],()=>{state.groups.category=category.value;update(true);});
    kind.value=state.groups.kind;category.value=state.groups.category;
    const groupLabel=group=>kind.value==='wonder'&&i18n?.language()==='en'?
      group.replace(/^(Gizah|Halikarnassos|Ephesos|Rhodos)\b/,name=>({Gizah:'Giza',Halikarnassos:'Halicarnassus',Ephesos:'Ephesus',Rhodos:'Rhodes'})[name]):t(group);
    update=table(groups,[col('group','Gruppe','string'),col('games','Spiele','number',v=>num(v,0)),col('share','Sieganteil','number',pct),col('margin','Vorsprung Ø','number',signed),col('points','Ausgewählte Punkte Ø')],()=>Object.entries(model().groups?.[kind.value]||{}).map(([group,g])=>({group:groupLabel(group),games:g.games,share:g.win_share,margin:g.vp_diff_mean,points:g.categories?.[category.value]?.mean})),{sort:'share',filename:'untergruppen.csv'});
    const runs=panel(parent,'Geladene Trainingsläufe vergleichen','Weitere analysis.json-Dateien kannst du oben über „Auswertung hinzufügen“ öffnen.');
    table(runs,[col('run','Auswertung','string'),col('model','Modell','string'),col('version','Merkmalsversion','number',v=>num(v,0)),col('share','Sieganteil','number',pct),col('points','Punkte Ø'),col('margin','Vorsprung Ø','number',signed),col('opponents','Gegnerreferenzen','string')],()=>entries.map(e=>{
      const key=e.report.models[state.model]?state.model:'candidate',m=e.report.models[key];return {run:entryTitle(e),model:modelLabel(e,key),share:m.win_share,points:m.agent_vp_mean,margin:m.vp_diff_mean,
        version:e.provenance?.feature_semantics_version,opponents:(e.report.references||[]).map(r=>String(r.checkpoint).split(/[\\/]/).pop()).join(' / ')};
    }),{sort:'share',filename:'laufvergleich.csv'});
    notice(parent,'Laufübergreifende Werte sind nur bei vergleichbaren Gegnern und Testbedingungen direkt interpretierbar. Gleich benannte Checkpoints aus verschiedenen Trainingsläufen sind verschiedene Modelle. Untergruppen sind beschreibend, keine separaten Signifikanztests.');
  }
  function quality(parent) {
    const m=model(),box=panel(parent,'Vollständigkeit der Spieldaten','Erfasste Spiele im Verhältnis zu den abgeschlossenen Partien des ausgewählten Modells.');
    const coverage=m.coverage||{};
    table(box,[col('metric','Information','string'),col('n','Erfasst / Anzahl','number',v=>num(v,0)),col('expected','Abgeschlossene Spiele','number',v=>num(v,0))],()=>[
      {metric:'Endzustände aller Spieler',n:coverage.final_states,expected:m.completed},
      {metric:'Vollständige Zeitalterfolgen',n:coverage.complete_age_sequences,expected:m.completed},
      {metric:'Aktionsprotokolle',n:coverage.action_traces,expected:m.completed},
      {metric:'Fehlgeschlagene Spiele',n:m.failed}, {metric:'Abgebrochene Spiele',n:m.truncated},
      {metric:'Fehlerhafte Punktesummen',n:coverage.score_sum_mismatches},
      {metric:'Fehlerhafte Zeitalterfolgen',n:coverage.invalid_age_sequences}, {metric:'Unbekannte gebaute Karten',n:coverage.unknown_built_cards}
    ],{sort:'metric',direction:1,filename:'datenqualitaet.csv'});
    const settings=panel(parent,'Testbedingungen und Modellidentität');
    const info=el('div',null,'detail-list');for(const [key,value] of [['Auswertung',entryTitle(current())],['Modellstand',modelLabel(current(),state.model)],
      ['Test-Seed',num(report().seed,0)],['Unabhängige Spiel-Seeds',num(m.seed_blocks,0)],['Modellpfad',report().model_descriptors?.[state.model]?.checkpoint||t('nicht erfasst')],
      ['Modell SHA-256',report().model_descriptors?.[state.model]?.sha256||t('nicht erfasst')],
      ['Merkmalsversion',num(current().provenance?.feature_semantics_version,0)],
      ['Simulator SHA-256',current().provenance?.engine_sha256||t('nicht erfasst')],
      ['Kartendaten SHA-256',current().provenance?.card_database_sha256||t('nicht erfasst')]]){const row=el('div');row.append(el('small',key),el('span',value));info.append(row);}settings.append(info);
    settings.append(button('Auswertungsdaten herunterladen',()=>download(JSON.stringify(report(),null,2),'analysis.json','application/json')));
    notice(parent,'Die Interpretation gilt für die verwendete Simulatorvariante. Gespeicherte Berichte älterer Läufe können andere Merkmalsversionen verwenden. Wissenschaftsarmut ist ein Beobachtungsbefund, kein Beweis für die optimale Strategie.');
  }
  function populateModels(){
    const select=$('model-select');select.replaceChildren();
    if(!report().models[state.model])state.model='candidate';
    for(const key of Object.keys(report().models)){const option=el('option',modelLabel(current(),key));option.value=key;select.append(option);}select.value=state.model;
  }
  function populateReports(){
    $('report-select').replaceChildren();entries.forEach((entry,i)=>{const option=el('option',entryTitle(entry));option.value=i;$('report-select').append(option);});$('report-select').value=state.entry;
  }
  function render(){
    const requested=location.hash.slice(1);state.view=VIEWS[requested]?requested:'overview';
    for(const link of $('navigation').children){const active=link.hash==='#'+state.view;link.classList.toggle('active',active);if(active)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');}
    $('page-title').textContent=t(VIEWS[state.view][0]);$('page-description').textContent=t(VIEWS[state.view][2]);
    $('dataset-meta').textContent=t('{games} Spiele · {seeds} Spiel-Seeds',{games:num(model().completed,0),seeds:num(model().seed_blocks,0)});
    $('error').hidden=!state.error;if(state.error)$('error').textContent=t('Import nicht möglich: ')+t(state.error);
    const parent=$('content');parent.replaceChildren();({overview,points,ages,cards,comparisons,quality})[state.view](parent);
  }
  function navigation(){
    $('navigation').replaceChildren();
    for(const [key,[name,index]] of Object.entries(VIEWS)){const a=el('a');a.href='#'+key;a.append(el('span',index),el('strong',name));$('navigation').append(a);}
  }
  $('report-select').addEventListener('change',()=>{state.entry=Number($('report-select').value);populateModels();render();});
  $('model-select').addEventListener('change',()=>{state.model=$('model-select').value;render();});
  $('import-file').addEventListener('change',async event=>{
    state.error=null;$('error').hidden=true;
    try{
      const incoming=[];
      for(const file of event.target.files){
        if(file.size>20*1024*1024)throw new Error('Datei zu groß. Bitte analysis.json statt eines Spielprotokolls auswählen.');
        let data;try{data=JSON.parse(await file.text());}catch(_){throw new Error('Ungültige JSON-Datei.');}
        if(!data?.models?.candidate?.categories||!data.models.candidate.groups)throw new Error('Bitte eine analysis.json aus der detaillierten Auswertung auswählen.');
        incoming.push({title:file.name+' · Seed '+(data.seed??'?')+' · '+(entries.length+incoming.length+1),report:data});
      }
      entries.push(...incoming);if(incoming.length)state.entry=entries.length-incoming.length;populateReports();populateModels();render();
    }catch(error){state.error=error.message;render();}
    event.target.value='';
  });
  window.addEventListener('hashchange',render);
  function refreshLanguage(){i18n?.apply();navigation();populateReports();populateModels();render();}
  i18n?.subscribe(refreshLanguage);
  $('language-select').addEventListener('change',event=>i18n?.setLanguage(event.target.value));
  refreshLanguage();
})();
