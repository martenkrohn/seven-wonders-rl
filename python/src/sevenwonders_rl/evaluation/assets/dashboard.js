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
    points:['Punkteherkunft','01','Woher kommen die Punkte – und wo liegen Stärken und Schwächen?'],
    ages:['Zeitalter','02','Produktion, Wirtschaft und Punkte am Ende jedes Zeitalters.'],
    cards:['Karten & Aktionen','03','Welche Karten und Aktionen prägen die Spielweise?'],
    training:['Trainingsverlauf','04','Fortschritt über Checkpoints und gezielte Vergleiche.']
  };
  let entries = JSON.parse(document.getElementById('evaluation-data').textContent);
  const context = JSON.parse(document.getElementById('dashboard-context')?.textContent || '{}');
  const state = {entry:0, model:'candidate', view:'points', tables:new Map(),
    filters:{query:'',color:'',age:''}, groups:{kind:'opponents',category:'total'},
    compare:'best', progress:'previous', metric:'vp_diff_mean', from:'', to:'', error:null};
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
  const isScience = entry => entry?.kind==='science_diagnostics';
  function el(tag, text, className) {
    const element = document.createElement(tag);
    if (text !== null && text !== undefined) element.textContent = typeof text==='string'?t(text):text;
    if (className) element.className = className;
    return element;
  }
  function modelLabel(entry, key) {
    const desc = entry.report.model_descriptors?.[key] || {};
    if(desc.label){
      const steps=desc.label.match(/^(\d[\d,.]*) Schritte$/);
      if(steps)return t('{step} Schritte',{step:num(Number(steps[1].replaceAll(',','').replaceAll('.','')),0)});
      return t(desc.label);
    }
    if(numeric(desc.head_id))return 'Head '+desc.head_id;
    if(desc.action_constraint==='none')return t('Freie Policy');
    if(desc.action_constraint==='green_priority')return t('Grün-Priorität');
    if(key==='initial' && desc.prior_strength>0)return t('Startmodell mit Priors (Schritt 0)');
    const match = String(desc.checkpoint || '').match(/checkpoint_(\d+)\.pt$/);
    const labelStep=String(entry.labels?.[key]||'').match(/Schritt ([\d.]+)/);
    const step = desc.timestep ?? (match ? Number(match[1]) : labelStep?Number(labelStep[1].replaceAll('.','')):null);
    if (key === 'initial' && desc.num_heads>1 && desc.initialization?.kind === 'checkpoint') return t('Startmodell');
    if (key === 'initial') return t(desc.initialization?.kind === 'checkpoint'
      ? 'Vortrainiertes Ausgangsmodell (Spezialisierungsschritt 0)' : 'Untrainiertes Ausgangsmodell (Schritt 0)');
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
  const col=(key,label,type='number',format=null,render=null)=>({key,label:t(label),type,format,render});
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
    const wrap=el('div',null,'table-wrap'),surface=el('div',null,'table-surface');
    surface.append(toolbar,wrap);parent.append(surface);
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
          const cell=el('td',column.render?null:text,column.type==='number'?'numeric':'');
          if(column.render)cell.append(column.render(value,row));tr.append(cell);
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
  const COLOR_HEX={brown:'#a5855e',grey:'#9ea8a0',blue:PALETTE.blue,red:PALETTE.military,green:PALETTE.science,yellow:PALETTE.yellow,purple:PALETTE.guild};
  const SCORE_ICONS={blue:'vp_civilian',military:'vp_military',science:'vp_scientific',yellow:'vp_commercial',guild:'vp_guild',wonder:'vp_wonder',treasury:'coins'};
  const RESOURCE_LABELS={wood:'Holz',ore:'Erz',clay:'Lehm',stone:'Stein',loom:'Stoff',glass:'Glas',papyrus:'Papyrus',coins:'Münzbestand',shields:'Militärstärke / Schilde',wonder_stage:'Vollendete Wunderabschnitte'};
  function artwork(kind,key,label,className='resource-icon') {
    const src=context.artwork?.[kind]?.[key];
    if(!src)return el('span',label,'artwork-fallback');
    const img=el('img');img.src=src;img.alt=t(label);img.className=className;img.loading='lazy';return img;
  }
  function details(parent,title,description) {
    const box=el('details',null,'panel disclosure');box.append(el('summary',title));
    if(description)box.append(el('p',description,'definition'));parent.append(box);return box;
  }
  function stars(p) {return !numeric(p)?'':p<.001?'***':p<.01?'**':p<.05?'*':'';}
  globalThis.SevenWondersDashboard.stars=stars;
  function testLabel(p) {return numeric(p)?(stars(p)||t('Nicht signifikant')):t('Nicht getestet');}
  function pText(p){return numeric(p)?(p<.0001?'< '+num(.0001,4):num(p,4)):t('Nicht getestet');}
  function strengthStats(parent) {
    const m=model(),stats=el('div',null,'stats compact-stats');
    for(const [label,value,detail] of [
      ['Eigene Punkte',num(m.agent_vp_mean),'Durchschnitt je abgeschlossenem Spiel'],
      ['Sieganteil',pct(m.win_share),'Geteilte Siege zählen anteilig.'],
      ['Punktevorsprung',signed(m.vp_diff_mean),'Gegen den stärksten Gegner je Partie']
    ]){const tile=el('section',null,'stat');tile.append(el('div',label,'stat-label'),el('div',value,'stat-value'),el('small',detail));stats.append(tile);}
    parent.append(stats);
  }
  function strengthTable(parent,entry=current()) {
    table(parent,[col('model','Modellstand','string'),col('win_share','Sieganteil','number',pct),col('points','Eigene Punkte Ø'),col('margin','Vorsprung Ø','number',signed)],
      ()=>Object.entries(entry.report.models).map(([key,m])=>({model:modelLabel(entry,key),win_share:m.win_share,points:m.agent_vp_mean,margin:m.vp_diff_mean})),{sort:'win_share',filename:'modellvergleich.csv'});
  }
  function progressChoices() {
    if(state.model!=='candidate')return [];
    const choices=[];
    if(report().previous_progress?.paired_valid)choices.push({key:'previous',label:t('Vorheriger Checkpoint')+(numeric(report().previous_step)?' · '+num(report().previous_step,0):''),data:report().previous_progress});
    for(const [key,data] of Object.entries(report().paired_progress||{}))if(data.paired_valid)choices.push({key,label:modelLabel(current(),key),data});
    return choices;
  }
  function progressPanel(parent,compact=false) {
    const choices=progressChoices();
    const box=panel(parent,'Fortschritt','Gepaarter Vergleich auf demselben Prüfplan.');
    if(!choices.length){box.append(el('p','Noch kein passender Fortschrittsvergleich verfügbar.','muted'));if(compact){const a=el('a','Trainingsverlauf öffnen');a.href='#training';box.append(a);}return;}
    let selected=choices.find(c=>c.key===state.progress)||choices[0];
    const controls=el('div',null,'group-controls');box.append(controls);
    const select=selectControl(controls,'Verglichen mit',choices.map(c=>[c.key,c.label]),()=>{state.progress=select.value;render();});select.value=selected.key;
    const d=selected.data, value=el('div',null,'progress-value '+(d.vp_diff_gain>0?'positive':d.vp_diff_gain<0?'negative':''));
    value.append(el('strong',signed(d.vp_diff_gain)+t(' Punkte')),el('span',testLabel(d.vp_diff_gain_p_value),'significance'));box.append(value);
    const info=details(box,'Statistik zum Vergleich');
    const dl=el('div',null,'detail-list');info.append(dl);
    for(const [label,v] of [['95%-Intervall / Punkte',interval(d.vp_diff_gain_ci95)],['p-Wert',pText(d.vp_diff_gain_p_value)],['Unabhängige Spiel-Seeds',num(d.seed_blocks,0)],['Sieganteil-Zuwachs',signed(d.win_share_gain*100)+t(' Prozentpunkte')]]){
      const row=el('div');row.append(el('small',label),el('span',v));dl.append(row);
    }
    info.append(el('p','* p < 0,05 · ** p < 0,01 · *** p < 0,001. Zweiseitiger gepaarter Test über Spiel-Seed-Blöcke. Sterne zeigen die Absicherung, das Vorzeichen die Richtung. Wiederholte Vergleiche sind explorativ und nicht für Mehrfachtests korrigiert.','definition'));
    if(compact){const a=el('a','Trainingsverlauf öffnen');a.href='#training';a.className='text-link';box.append(a);}
  }
  function comparisonOptions() {
    const options=[{value:'best',label:t('Bester Gegner je Partie'),m:null}];
    for(const key of Object.keys(report().models))if(key!==state.model)options.push({value:'model:'+key,label:modelLabel(current(),key),m:report().models[key]});
    if(report().plan_id)entries.forEach((e,i)=>{
      if(i!==state.entry&&!e.historical_reference&&e.report?.plan_id===report().plan_id&&e.report.models?.candidate)
        options.push({value:'entry:'+i,label:entryTitle(e),m:e.report.models.candidate});
    });
    return options;
  }
  function points(parent) {
    strengthStats(parent);
    const hero=panel(parent,'Punkteprofil','Durchschnittliche Punkte je Spiel.');
    const options=comparisonOptions(), selected=options.find(o=>o.value===state.compare)||options[0];
    const controls=el('div',null,'group-controls');hero.append(controls);
    const compare=selectControl(controls,'Vergleichen mit',options.map(o=>[o.value,o.label]),()=>{state.compare=compare.value;render();});compare.value=selected.value;
    const head=el('div',null,'score-heading');head.append(el('span','Punktekategorie'),el('span','Punkte Ø'),el('span','Abstand Ø'));hero.append(head);
    const rows=scoreRows(model()).filter(r=>r.key!=='total'), max=Math.max(1,...rows.filter(r=>numeric(r.mean)).map(r=>Math.abs(r.mean)));
    for(const row of rows){
      const line=el('div',null,'score-row');const label=el('div',null,'score-label');
      label.append(artwork('resources',SCORE_ICONS[row.key],row.category),el('span',row.category));
      const track=el('div',null,'profile-track'),fill=el('div',null,'profile-fill');
      if(numeric(row.mean)){fill.style.width=(Math.abs(row.mean)/max*48)+'%';fill.style.left=(row.mean<0?50-Math.abs(row.mean)/max*48:50)+'%';fill.style.background=PALETTE[row.key]||'#9ca79e';track.append(fill);}
      const amount=el('div',null,'score-amount');amount.append(track,el('strong',num(row.mean)));
      const delta=selected.m?(numeric(row.mean)&&numeric(selected.m.categories?.[row.key]?.mean)?row.mean-selected.m.categories[row.key].mean:null):model().category_margin_to_best?.[row.key]?.mean;
      const difference=el('span',signed(delta),'score-delta '+(delta>0?'positive':delta<0?'negative':''));line.append(label,amount,difference);hero.append(line);
    }
    hero.append(el('p','Abstand = eigene Punkte minus Vergleich. Die Mittellinie der Balken markiert null.','definition'));
    const distributions=details(parent,'Verteilungen · P10–P90 und Maximum','P10–P90 umfasst ungefähr die mittleren 80 % der einzelnen Partien. Es beschreibt keine Unsicherheit des Mittelwerts.');
    table(distributions,[col('category','Kategorie','string'),col('mean','Mittelwert'),col('range','Typischer Bereich · P10–P90','string'),col('max','Maximum')],
      ()=>scoreRows(model()).map(r=>({...r,range:numeric(r.p10)&&numeric(r.p90)?num(r.p10)+' – '+num(r.p90):null})),{sort:'mean',filename:'punkteherkunft.csv'});
    progressPanel(parent,true);
    const info=details(parent,'Datengrundlage und Export');quality(info);
    info.append(el('p','Kartenfarbe und Punkteherkunft sind verschieden: Wunder und Wissenschaftsgilden können zu Wissenschaft beitragen; Wunderschilde zu Militär. Kategorien-Maxima stammen möglicherweise aus unterschiedlichen Spielen.','definition'));
  }
  function ages(parent) {
    const m=model();
    const production=panel(parent,'Produktion nach Zeitalter','Durchschnittliche feste Produktionskapazität am Ende jedes Zeitalters. Wahlproduktion ist nicht enthalten.');
    const matrix=el('div',null,'production-table');production.append(matrix);
    const amount=value=>{const cell=el('span',num(value),'production-amount');if(numeric(value))cell.style.background='color-mix(in srgb, var(--gold) '+Math.min(45,Math.max(0,value)*15)+'%, transparent)';return cell;};
    table(matrix,[col('resource','Produktionsgut','string',null,(value,row)=>{
      const label=el('div',null,'resource-label');label.append(artwork('resources',row.key,value),el('span',value));return label;
    }),...['I','II','III'].map((age,i)=>col('a'+(i+1),t('Ende {age}',{age}),'number',null,amount))],
      ()=>['wood','ore','clay','stone','loom','glass','papyrus'].map(key=>({key,resource:RESOURCE_LABELS[key],...Object.fromEntries([1,2,3].map(age=>['a'+age,m.age_economy?.[age]?.[key]?.mean]))})),
      {sort:'resource',direction:1,filename:'produktion-je-zeitalter.csv'});
    const economy=panel(parent,'Wirtschaft und Spielaufbau');
    table(economy,[col('metric','Kennzahl','string',null,(v,r)=>{const w=el('div',null,'resource-label');if(r.key!=='wonder_stage')w.append(artwork('resources',r.key,v));w.append(el('span',v));return w;}),col('a1','Ende I'),col('a2','Ende II'),col('a3','Ende III')],
      ()=>['coins','shields','wonder_stage'].map(key=>({key,metric:RESOURCE_LABELS[key],...Object.fromEntries([1,2,3].map(age=>['a'+age,m.age_economy?.[age]?.[key]?.mean]))})),{sort:'metric',direction:1,filename:'zeitalter-wirtschaft.csv'});
    const box=panel(parent,'Punkte über die Zeitalter','Zustände nach der Militärwertung; Zeitalter III zusätzlich nach finalen Gildeneffekten.');
    const tiles=el('div',null,'age-grid');for(const age of [1,2,3]){const tile=el('div',null,'age-card');tile.append(el('strong',t('ENDE ZEITALTER {age}',{age})),el('div',num(m.age_points?.[age]?.total?.mean)),el('small','Punkte Ø'));tiles.append(tile);}box.append(tiles);
    const breakdown=details(box,'Punkte nach Kategorie');
    table(breakdown,[col('category','Wertungsbereich','string',null,(v,r)=>{const label=el('span',v),swatch=el('span',null,'swatch');swatch.style.background=PALETTE[r.key]||'var(--muted)';label.prepend(swatch);return label;}),col('a1','Ende I'),col('a2','Ende II'),col('a3','Ende III')],
      ()=>scoreRows(m).map(r=>({key:r.key,category:r.category,...Object.fromEntries([1,2,3].map(age=>['a'+age,m.age_points?.[age]?.[r.key]?.mean]))})),{sort:'a3',filename:'zeitalter-punkte.csv'});
  }
  function cardCell(value,row) {
    const box=el('details',null,'card-detail'),summary=el('summary',null,'card-summary');
    const swatch=el('span',null,'card-swatch');swatch.style.background=COLOR_HEX[row.color]||'var(--muted)';swatch.setAttribute('aria-label',t(COLORS[row.color]||'Unbekannt'));
    summary.append(swatch,artwork('cards',value,value,'card-thumbnail'),el('span',value));box.append(summary);
    const content=el('div',null,'card-expanded');content.append(artwork('cards',value,value,'card-image'),el('p',t('Legale Angebote: {count}',{count:num(row.legal_opportunities,0)}),'definition'));box.append(content);return box;
  }
  function cards(parent) {
    const m=model(),box=panel(parent,'Kartenwahl','Kartenbild anklicken, um die Karte und ihre Datengrundlage zu öffnen.');
    const filters=el('div',null,'filters');box.append(filters);
    const searchLabel=el('label','Kartenname'),search=el('input');search.type='search';search.placeholder=t('Zum Beispiel Workshop');search.value=state.filters.query;searchLabel.append(search);filters.append(searchLabel);
    let refresh=()=>{};
    const color=selectControl(filters,'Farbe',[['','Alle Farben'],...Object.entries(COLORS)],()=>{state.filters.color=color.value;refresh(true);});
    color.value=state.filters.color;
    const ageFilter=el('fieldset',null,'age-filter');ageFilter.append(el('legend','Zeitalter'));filters.append(ageFilter);
    const ageButtons=[];
    for(const [value,label] of [['','Alle Zeitalter'],['1','Zeitalter I'],['2','Zeitalter II'],['3','Zeitalter III']]){
      const choice=button(label,()=>{state.filters.age=value;updateAges();refresh(true);});
      ageButtons.push([choice,value]);ageFilter.append(choice);
    }
    function updateAges(){for(const [choice,value] of ageButtons)choice.setAttribute('aria-pressed',String(state.filters.age===value));}
    updateAges();
    refresh=table(box,[col('card','Karte','string',null,cardCell),col('era','Zeitalter','number',v=>num(v,0)),col('build_game_fraction','Partien mit Bau (%)','number',pct),col('selection_fraction','Auswahlquote','number',pct)],
      ()=>filterCards(m.cards||[],{query:search.value,color:color.value,age:state.filters.age}),{sort:'build_game_fraction',filename:'karten-gefiltert.csv'});
    search.addEventListener('input',()=>{state.filters.query=search.value;refresh(true);});
    box.append(el('p','Partien mit Bau = Anteil der Spiele, in denen die Karte als Gebäude gebaut wurde. Auswahlquote = Bauaktionen / Entscheidungen mit dieser legal baubaren Karte. Wiederholte Angebote zählen mehrfach. Verkäufe und Wunderbau zählen nicht als Gebäude.','definition'));
    const actions=panel(parent,'Gebaute Farben und Aktionen je Zeitalter','Durchschnitt tatsächlich bestätigter Aktionen je Spiel.');
    table(actions,[col('category','Karten / Aktionen','string',null,(v,r)=>{const label=el('span',v),swatch=el('span',null,'swatch');swatch.style.background=COLOR_HEX[r.key]||PALETTE.wonder;label.prepend(swatch);return label;}),col('a1','Zeitalter I'),col('a2','Zeitalter II'),col('a3','Zeitalter III'),col('total','Gesamt')],
      ()=>Object.entries({...COLORS,wonder:'Wunderbau',discard:'Kartenverkauf'}).filter(([key])=>key!=='unknown'||[1,2,3].some(a=>m.age_actions?.[a]?.unknown)).map(([key,category])=>{
        const known=(m.coverage?.action_traces||0)>0,v=[1,2,3].map(a=>known?(m.age_actions?.[a]?.[key]??0):null);return {key,category,a1:v[0],a2:v[1],a3:v[2],total:known?v.reduce((a,b)=>a+b,0):null};
      }),{sort:'total',filename:'aktionen-je-zeitalter.csv'});
  }
  function curveRows() {
    if(context.training?.evaluations?.length&&context.training.plan_id===report()?.plan_id&&!current()?.historical_reference)return context.training.evaluations;
    if(!report()?.plan_id)return [];
    return entries.filter(e=>!e.historical_reference&&e.report?.plan_id===report().plan_id&&numeric(e.report.training_step)).map(e=>({step:e.report.training_step,...e.report.models.candidate})).sort((a,b)=>a.step-b.step);
  }
  function curveChart(parent,rows) {
    const key=state.metric,percent=key==='win_share',fmt=percent?pct:signed;
    const points=rows.filter(r=>numeric(r[key]));
    if(!points.length){parent.append(el('p','Noch keine Lernkurve vorhanden. Einzelne Auswertungen bleiben unten vergleichbar.','empty'));return;}
    const reference=context.training&&context.training.plan_id===report()?.plan_id&&!current().historical_reference?context.training.reference?.[key]:null;
    const values=points.map(p=>p[key]);if(numeric(reference))values.push(reference);
    const minX=Math.min(...points.map(p=>p.step)),maxX=Math.max(minX+1,...points.map(p=>p.step));
    const bottom=Math.min(0,...values),top=Math.max(bottom+.1,...values),pad=(top-bottom)*.1;
    const x=v=>80+(v-minX)/(maxX-minX)*790,y=v=>260-(v-bottom+pad)/(top-bottom+2*pad)*220;
    const svg=(tag,attrs={},text=null)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!==null)n.textContent=text;return n;};
    const chart=svg('svg',{viewBox:'0 0 920 330',class:'training-chart',role:'img','aria-label':t('Trainingsverlauf')});chart.append(svg('title',{},t('Fortschritt nach Trainingsschritt')));
    for(let i=0;i<5;i++){const v=bottom-pad+(top-bottom+2*pad)*i/4,py=y(v);chart.append(svg('line',{x1:80,x2:870,y1:py,y2:py,class:'chart-grid'}),svg('text',{x:68,y:py+4,'text-anchor':'end'},fmt(v)));}
    if(numeric(reference))chart.append(svg('line',{x1:80,x2:870,y1:y(reference),y2:y(reference),class:'reference-line'}));
    const path=points.map((p,i)=>(i?'L':'M')+x(p.step)+','+y(p[key])).join(' ');chart.append(svg('path',{d:path,fill:'none',stroke:PALETTE.science,'stroke-width':3}));
    points.forEach(p=>{const dot=svg('circle',{cx:x(p.step),cy:y(p[key]),r:5,fill:PALETTE.science,tabindex:0,role:'button','aria-label':t('Schritt {step}: {value}',{step:num(p.step,0),value:fmt(p[key])})});
      dot.append(svg('title',{},num(p.step,0)+' · '+fmt(p[key])));
      const choose=()=>{const i=entries.findIndex(e=>!e.historical_reference&&e.report?.plan_id===report().plan_id&&e.report.training_step===p.step);if(i>=0){state.entry=i;state.model='candidate';populateReports();populateModels();render();}else{state.focusStep=p.step;render();}};
      dot.addEventListener('click',choose);dot.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}});chart.append(dot);
    });
    for(const step of [minX,(minX+maxX)/2,maxX])chart.append(svg('text',{x:x(step),y:292,'text-anchor':'middle'},num(step,0)));
    chart.append(svg('text',{x:475,y:322,'text-anchor':'middle'},t('Trainingsschritte')));parent.append(chart);
    if(numeric(reference))parent.append(el('p','Gestrichelte Linie: festes Referenzmodell auf demselben Prüfplan.','definition'));
    const focused=rows.find(r=>r.step===state.focusStep)||rows.find(r=>r.step===report().training_step)||rows[rows.length-1];
    if(focused){const info=el('div',null,'checkpoint-info');info.append(el('strong',t('Schritt {step}: {value}',{step:num(focused.step,0),value:fmt(focused[key])})));
      if(focused.previous_progress?.paired_valid){const p=focused.previous_progress;info.append(el('span',t('Zum vorherigen Checkpoint: {value}',{value:signed(p.vp_diff_gain)+t(' Punkte')+' · '+testLabel(p.vp_diff_gain_p_value)})));}

      const ci=key==='vp_diff_mean'?focused.vp_diff_ci95:key==='win_share'?focused.win_share_ci95:null;
      if(ci)info.append(el('span',t('95%-Intervall: {value}',{value:interval(ci,percent)})));
      if(!entries.some(e=>e.report?.training_step===focused.step)){const link=el('a','Älterer Detailbericht');link.href='evaluations/step_'+String(focused.step).padStart(7,'0')+'/dashboard.html#points';info.append(link);}
      parent.append(info);
    }
  }
  function training(parent) {
    const box=panel(parent,'Lernkurve','Wähle eine Kennzahl und klicke auf einen Checkpoint für Details.');
    const controls=el('div',null,'filters');box.append(controls);
    const metric=selectControl(controls,'Kennzahl',[['vp_diff_mean','Punktevorsprung'],['win_share','Sieganteil'],['agent_vp_mean','Eigene Punkte']],()=>{state.metric=metric.value;render();});metric.value=state.metric;
    const rows=curveRows();
    for(const [key,label] of [['from','Ab Schritt'],['to','Bis Schritt']]){const wrap=el('label',label),input=el('input');input.type='number';input.min='0';input.placeholder=key==='from'?'0':'∞';input.value=state[key];input.addEventListener('change',()=>{state[key]=input.value;render();});wrap.append(input);controls.append(wrap);}
    curveChart(box,rows.filter(r=>(state.from===''||r.step>=Number(state.from))&&(state.to===''||r.step<=Number(state.to))));
    if(context.training&&context.training.plan_id===report().plan_id&&!current().historical_reference){const status=context.training.status||{};box.append(el('p',t('Stand: {step} Schritte · Nächste Evaluation: {next}',{step:num(status.timestep,0),next:num(status.next_evaluation,0)}),'definition'));
      if(context.training.continuous)box.append(el('p','Fortlaufendes Training','definition'));
    }
    progressPanel(parent);
    const models=details(parent,'Modellstände auf demselben Testplan');strengthTable(models);
    const groups=details(parent,'Unterschiede zwischen Spielsituationen');const groupControls=el('div',null,'group-controls');groups.append(groupControls);let update=()=>{};
    const kind=selectControl(groupControls,'Gruppieren nach',[['opponents','Gegnerpaar'],['seat','Spielerposition'],['wonder','Wunder'],['outcome','Spielausgang']],()=>{state.groups.kind=kind.value;update(true);});
    const category=selectControl(groupControls,'Punkteprofil',[['total','Gesamtpunkte'],...Object.entries(CATEGORY)],()=>{state.groups.category=category.value;update(true);});kind.value=state.groups.kind;category.value=state.groups.category;
    const groupLabel=group=>kind.value==='wonder'&&i18n?.language()==='en'?group.replace(/^(Gizah|Halikarnassos|Ephesos|Rhodos)\b/,name=>({Gizah:'Giza',Halikarnassos:'Halicarnassus',Ephesos:'Ephesus',Rhodos:'Rhodes'})[name]):t(group);
    update=table(groups,[col('group','Gruppe','string'),col('games','Spiele','number',v=>num(v,0)),col('share','Sieganteil','number',pct),col('margin','Vorsprung Ø','number',signed),col('points','Ausgewählte Punkte Ø')],()=>Object.entries(model().groups?.[kind.value]||{}).map(([group,g])=>({group:groupLabel(group),games:g.games,share:g.win_share,margin:g.vp_diff_mean,points:g.categories?.[category.value]?.mean})),{sort:'share',filename:'untergruppen.csv'});
    const runs=details(parent,'Geladene Trainingsläufe vergleichen','Laufübergreifende Werte sind nur bei vergleichbaren Gegnern und Testbedingungen direkt interpretierbar.');
    table(runs,[col('run','Auswertung','string'),col('share','Sieganteil','number',pct),col('margin','Vorsprung Ø','number',signed)],()=>entries.filter(e=>!isScience(e)&&!e.historical_reference).map(e=>({run:entryTitle(e),share:e.report.models.candidate?.win_share,margin:e.report.models.candidate?.vp_diff_mean})),{sort:'share',filename:'laufvergleich.csv'});
    if(context.training?.league&&context.training.plan_id===report().plan_id&&!current().historical_reference){const league=details(parent,'Trainingsliga · Gegnerauswahl'),data=context.training.league;league.append(el('p',t('{count} aktive Modelle',{count:num(data.active_checkpoints?.length,0)}),'definition'));
      table(league,[col('left','Linker Gegner · Schritt'),col('right','Rechter Gegner · Schritt'),col('probability','Auswahlchance','number',pct)],()=>data.pairs||[],{sort:'probability',filename:'liga.csv'});
    }
  }
  function quality(parent) {
    const m=model(),info=el('div',null,'detail-list');parent.append(info);
    for(const [label,value] of [['Modellstand',modelLabel(current(),state.model)],['Abgeschlossene Spiele',num(m.completed,0)],['Unabhängige Spiel-Seeds',num(m.seed_blocks,0)],['Test-Seed',num(report().seed,0)],['Modellpfad',report().model_descriptors?.[state.model]?.checkpoint||t('nicht erfasst')],['Modell SHA-256',report().model_descriptors?.[state.model]?.sha256||t('nicht erfasst')]]){
      const row=el('div');row.append(el('small',label),el('span',value));info.append(row);
    }
    table(parent,[col('label','Information','string'),col('value','Anzahl','number',v=>num(v,0))],()=>[
      {label:'Fehlgeschlagene Spiele',value:m.failed},{label:'Abgebrochene Spiele',value:m.truncated},
      ...Object.entries({final_states:'Endzustände aller Spieler',complete_age_sequences:'Vollständige Zeitalterfolgen',action_traces:'Aktionsprotokolle',score_sum_mismatches:'Fehlerhafte Punktesummen',invalid_age_sequences:'Fehlerhafte Zeitalterfolgen',unknown_built_cards:'Unbekannte gebaute Karten'}).map(([key,label])=>({label,value:m.coverage?.[key]}))
    ],{filename:'datenqualitaet.csv'});
    parent.append(button('Auswertungsdaten herunterladen',()=>download(JSON.stringify(report(),null,2),'analysis.json','application/json')));
  }
  function populateModels(){
    const select=$('model-select');select.replaceChildren();select.disabled=!current()?.report;
    if(!current()?.report)return;
    if(!report().models[state.model])state.model=report().models.candidate?'candidate':Object.keys(report().models)[0];
    for(const key of Object.keys(report().models)){const option=el('option',modelLabel(current(),key));option.value=key;select.append(option);}select.value=state.model;
  }
  function populateReports(){
    $('report-select').replaceChildren();entries.forEach((entry,i)=>{const option=el('option',entryTitle(entry));option.value=i;$('report-select').append(option);});$('report-select').value=state.entry;
  }
  function render(){
    const requested=location.hash.slice(1),aliases={overview:'points',comparisons:'training',quality:'points',science:'training'};
    state.view=VIEWS[requested]?requested:aliases[requested]||'points';
    for(const link of $('navigation').children){const active=link.hash==='#'+state.view;link.classList.toggle('active',active);if(active)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');}
    $('page-title').textContent=t(VIEWS[state.view][0]);$('page-description').textContent=t(VIEWS[state.view][2]);
    $('error').hidden=!state.error;if(state.error)$('error').textContent=t('Import nicht möglich: ')+t(state.error);
    const parent=$('content');parent.replaceChildren();
    if(!current()?.report){
      $('dataset-meta').textContent='';
      notice(parent,context.training&&!entries.length?'Noch keine abgeschlossene Evaluation. Das Dashboard erscheint nach dem ersten Kontrollpunkt.':'Für diese Ansicht fehlt eine detaillierte Auswertung. Lade eine analysis.json; die Wissenschaftsdiagnose enthält keine vollständigen Testspiel-Daten.');return;
    }
    $('dataset-meta').textContent=t('{games} Spiele · {seeds} Spiel-Seeds',{games:num(model()?.completed,0),seeds:num(model()?.seed_blocks,0)});
    if(current().historical_reference)notice(parent,'Historische Referenz · ursprünglicher Prüfplan. Neue Trainingsstände erscheinen nach der ersten Evaluation.');
    if(model().valid_for_selection===false)notice(parent,'Diese Auswertung enthält unvollständige Spiele. Kennzahlen beziehen sich auf abgeschlossene Partien.');
    if((model().coverage?.complete_age_sequences||0)<model().completed)notice(parent,'Für diesen Modellstand fehlen teilweise Zeitalter- oder Aktionsdaten. Fehlende Werte werden als „nicht erfasst“ angezeigt.');
    if(model().coverage?.score_sum_mismatches>0)notice(parent,'Die Punktesummen enthalten Unstimmigkeiten. Details stehen unter Datengrundlage und Export.');
    ({points,ages,cards,training})[state.view](parent);
  }
  function navigation(){
    $('navigation').replaceChildren();for(const [key,[name,index]] of Object.entries(VIEWS)){const a=el('a');a.href='#'+key;a.append(el('span',index),el('strong',name));$('navigation').append(a);}
  }
  $('report-select').addEventListener('change',()=>{state.entry=Number($('report-select').value);state.compare='best';state.focusStep=null;populateModels();render();});
  $('model-select').addEventListener('change',()=>{state.model=$('model-select').value;render();});
  $('import-file').addEventListener('change',async event=>{
    state.error=null;
    try{
      const incoming=[];
      for(const file of event.target.files){
        if(file.size>20*1024*1024)throw new Error('Datei zu groß. Bitte analysis.json statt eines Spielprotokolls auswählen.');
        let data;try{data=JSON.parse(await file.text());}catch(_){throw new Error('Ungültige JSON-Datei.');}
        if(!data?.models?.candidate?.categories||!data.models.candidate.groups)throw new Error('Bitte eine analysis.json aus der detaillierten Auswertung auswählen.');
        incoming.push({title:file.name+' · Seed '+(data.seed??'?')+' · '+(entries.length+incoming.length+1),report:data});
      }
      entries.push(...incoming);if(incoming.length){state.entry=entries.length-incoming.length;state.model='candidate';state.compare='best';}populateReports();populateModels();render();
    }catch(error){state.error=error.message;render();}event.target.value='';
  });
  window.addEventListener('hashchange',render);
  function refreshLanguage(){i18n?.apply();navigation();populateReports();populateModels();render();}
  i18n?.subscribe(refreshLanguage);
  $('language-select').addEventListener('change',event=>i18n?.setLanguage(event.target.value));
  // Prefer the latest evaluated checkpoint; keep historical reference as fallback.
  const candidates=entries.map((e,i)=>({e,i})).filter(({e})=>e.report&&!e.historical_reference);
  if(candidates.length)state.entry=candidates.reduce((a,b)=>(b.e.report.training_step??-1)>(a.e.report.training_step??-1)?b:a).i;
  refreshLanguage();
})();
