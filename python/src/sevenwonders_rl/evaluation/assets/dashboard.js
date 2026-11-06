/* All report content is inserted as text; loading JSON never executes HTML. */
'use strict';
const copy = {
 en:{load:'Load report',eyebrow:'SECOND EDITION / THREE-PLAYER SELF-PLAY',title:'What has the agent learned?',intro:'An evaluation of one frozen PPO policy against a fixed panel of its own predecessors. Explore scores, opponent pairs and the effect of table position.',scopeTitle:'HOW TO READ THIS RESULT',scope:'The opponent panel comes from the same training run. These results describe this development panel, not general playing strength. The displayed model is a saved snapshot, not a live training feed. Confidence intervals resample whole engine-seed blocks; a single block cannot provide one.',categories:'Where the points come from',average:'Average VP per game',seats:'Position at the table',margin:'VP margin vs strongest rival',seatNote:'All three seats are evaluated. Opponent identities and engine seeds stay fixed within each comparison.',pairs:'The opponent panel',model:'Model',pair:'Left / right opponent',completed:'Completed',win:'Win share',points:'Own VP',marginShort:'VP margin',provenance:'Method and provenance',method:'Win share splits ties equally. The reward is the final VP margin against the strongest rival. Failed games are retained and invalidate progress comparisons. The panel covers every ordered opponent pair and all three seats.',footer:'Built to understand decisions, not just report a win rate.',source:'Project, method & source ↗',current:'Current snapshot',previous:'Previous evaluation',complete:'COMPLETE',incomplete:'INCOMPLETE',step:'Step',games:'games',blocks:'seed blocks',ci:'95% interval',noCI:'Interval unavailable',noComparison:'No earlier evaluation on this panel. A single score does not establish learning progress.',change:'Paired change in VP margin',seat:'Seat',plan:'Panel',engine:'Engine',schema:'Schema',ruleset:'Ruleset',hash:'Summary SHA256',error:'Cannot load this file. Select a Decision summary.json or a dashboard snapshot.json.',scientific:'Science',military:'Military',civilian:'Civilian',commercial:'Commerce',guild:'Guilds',wonder:'Wonder',coins:'Coins'},
 de:{load:'Bericht laden',eyebrow:'ZWEITE EDITION / SELF-PLAY MIT DREI SPIELERN',title:'Was hat der Agent gelernt?',intro:'Eine eingefrorene PPO-Policy spielt gegen ein festes Panel eigener Vorgänger. Erkunde Punkte, Gegnerpaare und den Einfluss der Sitzposition.',scopeTitle:'EINORDNUNG DES ERGEBNISSES',scope:'Die Gegner stammen aus demselben Trainingslauf. Die Ergebnisse gelten für dieses Entwicklungspanel und belegen keine allgemeine Spielstärke. Gezeigt wird ein gespeicherter Modellstand, kein Live-Training. Konfidenzintervalle resampeln ganze Engine-Seed-Blöcke; ein einzelner Block reicht dafür nicht.',categories:'Woher die Punkte kommen',average:'Mittlere Punkte pro Partie',seats:'Position am Tisch',margin:'Punkteabstand zum stärksten Gegner',seatNote:'Alle drei Sitze werden geprüft. Gegneridentitäten und Engine-Seeds bleiben innerhalb eines Vergleichs gleich.',pairs:'Das Gegnerpanel',model:'Modell',pair:'Gegner links / rechts',completed:'Abgeschlossen',win:'Sieganteil',points:'Eigene Punkte',marginShort:'Punkteabstand',provenance:'Methode und Herkunft',method:'Gleichstände teilen den Sieganteil. Der Reward ist der abschließende Punkteabstand zum stärksten Gegner. Fehlgeschlagene Spiele bleiben sichtbar und verhindern einen gültigen Fortschrittsvergleich. Das Panel prüft jedes geordnete Gegnerpaar auf allen drei Sitzen.',footer:'Entscheidungen verstehen, statt nur eine Siegquote zu melden.',source:'Projekt, Methode & Quellcode ↗',current:'Aktueller Modellstand',previous:'Vorherige Evaluation',complete:'VOLLSTÄNDIG',incomplete:'UNVOLLSTÄNDIG',step:'Schritt',games:'Partien',blocks:'Seed-Blöcke',ci:'95%-Intervall',noCI:'Intervall nicht verfügbar',noComparison:'Keine frühere Evaluation auf diesem Panel. Eine einzelne Punktzahl belegt noch keinen Lernfortschritt.',change:'Gepaarte Änderung im Punkteabstand',seat:'Sitz',plan:'Panel',engine:'Engine',schema:'Schema',ruleset:'Regelstand',hash:'SHA256 des Berichts',error:'Datei kann nicht geladen werden. Bitte eine Decision summary.json oder Dashboard-snapshot.json wählen.',scientific:'Wissenschaft',military:'Militär',civilian:'Zivilbauten',commercial:'Handel',guild:'Gilden',wonder:'Wunder',coins:'Münzen'}
};
function stored(key,fallback){try{return localStorage.getItem(key)||fallback;}catch{return fallback;}}
function save(key,value){try{localStorage.setItem(key,value);}catch{}}
let lang=stored('7w-language','en');if(!copy[lang])lang='en';
let data=JSON.parse(document.getElementById('report-data').textContent);
let selected='current';
const el=id=>document.getElementById(id);
const t=key=>copy[lang][key]||key;
const fmt=(n,d=1)=>typeof n==='number'&&Number.isFinite(n)?n.toLocaleString(lang,{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
const pct=n=>typeof n==='number'?fmt(n*100)+'%':'—';
function node(tag,text,className){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n;}
function ci(bounds,percent=false){return Array.isArray(bounds)?`${t('ci')}: ${percent?pct(bounds[0]):fmt(bounds[0])} – ${percent?pct(bounds[1]):fmt(bounds[1])}`:t('noCI');}
function normalize(value){
 const report=value.format==='decision-dashboard-v1'?value:{format:'decision-dashboard-v1',evaluation:value,provenance:{}};
 const e=report.evaluation;
 if(!e||!e.models||!e.models.current||!e.candidate||typeof e.candidate.step!=='number'||typeof e.valid!=='boolean'||typeof e.plan_id!=='string')throw Error('Invalid report');
 for(const m of Object.values(e.models)){
  if(!m||typeof m.completed!=='number'||typeof m.attempted!=='number'||!m.by_pair||!m.by_seat||!m.categories)throw Error('Invalid model');
 }
 return report;
}
function render(){
 const e=data.evaluation;const m=e.models[selected]||e.models.current;
 document.documentElement.lang=lang;
 document.querySelectorAll('[data-i18n]').forEach(n=>n.textContent=t(n.dataset.i18n));
 el('language').textContent=lang==='en'?'DE':'EN';
 el('status').textContent=t(e.valid?'complete':'incomplete');
 el('step').textContent=`${t('step')} ${fmt((selected==='previous'?e.previous:e.candidate)?.step,0)}`;
 el('games').textContent=`${fmt(m.completed,0)} / ${fmt(m.attempted,0)} ${t('games')}`;
 el('blocks').textContent=`${fmt(e.blocks,0)} ${t('blocks')}`;
 el('metrics').replaceChildren();
 for(const [key,label,percent] of [['win_share','win',true],['own_points','points',false],['margin','marginShort',false]]){
  const card=node('article',undefined,'metric');card.append(node('div',t(label),'label'),node('strong',percent?pct(m[key]):fmt(m[key])),node('span',ci(m.ci95?.[key],percent)));el('metrics').append(card);
 }
 const change=e.paired_change?.margin;
 el('comparison').textContent=change?`${t('change')}: ${fmt(change.change)} (${ci(change.ci95)}).`:t('noComparison');
 const select=el('model');select.replaceChildren();
 for(const name of Object.keys(e.models)){const option=node('option',t(name));option.value=name;option.selected=name===selected;select.append(option);}
 el('categories').replaceChildren();
 const categories=Object.entries(m.categories).filter(([key])=>key!=='total');
 const max=Math.max(1,...categories.map(([,value])=>value));
 for(const [key,value] of categories){
  const row=node('div',undefined,'barrow'),track=node('div',undefined,'track'),fill=node('div',undefined,'fill');
  fill.style.width=`${Math.max(0,Math.min(100,value/max*100))}%`;track.append(fill);row.append(node('span',t(key)),track,node('b',fmt(value)));el('categories').append(row);
 }
 el('seats').replaceChildren();
 for(const [seat,row] of Object.entries(m.by_seat)){
  const card=node('article',undefined,'seat');card.append(node('span',`${t('seat')} ${Number(seat)+1}`),node('strong',fmt(row.margin)),node('span',`${pct(row.win_share)} · ${fmt(row.completed,0)} ${t('games')}`));el('seats').append(card);
 }
 el('pairs').replaceChildren();
 const ref=index=>{const r=e.references?.[index];return r?`${r.label||Number(index)+1} · ${fmt(r.step,0)}`:index;};
 for(const [pair,row] of Object.entries(m.by_pair)){
  const tr=node('tr');const [left,right]=pair.split(',');
  for(const value of [`${ref(left)} / ${ref(right)}`,fmt(row.completed,0),pct(row.win_share),fmt(row.own_points),fmt(row.margin)])tr.append(node('td',value));el('pairs').append(tr);
 }
 el('provenance').replaceChildren();
 for(const [label,value] of [['ruleset',data.provenance?.ruleset],['plan',e.plan_id],['engine',data.provenance?.engine_commit],['schema',data.provenance?.schema_hash],['hash',data.provenance?.source_summary_sha256]])el('provenance').append(node('dt',t(label)),node('dd',value||'—'));
}
el('model').addEventListener('change',event=>{selected=event.target.value;render();});
el('language').addEventListener('click',()=>{lang=lang==='en'?'de':'en';save('7w-language',lang);render();});
document.documentElement.dataset.theme=stored('7w-theme',matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
el('theme').addEventListener('click',()=>{const theme=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=theme;save('7w-theme',theme);});
el('file').addEventListener('change',async event=>{
 try{const file=event.target.files[0];if(!file)return;if(file.size>10000000)throw Error('File too large');const next=normalize(JSON.parse(await file.text()));data=next;selected='current';render();el('error').hidden=true;}
 catch{el('error').textContent=t('error');el('error').hidden=false;}
});
el('csv').addEventListener('click',()=>{
 const e=data.evaluation,m=e.models[selected];
 const rows=[['model','pair','completed','attempted','win_share','own_points','margin'],...Object.entries(m.by_pair).map(([pair,r])=>[selected,pair,r.completed,r.attempted,r.win_share,r.own_points,r.margin])];
 const csv=rows.map(row=>row.map(value=>'"'+String(value??'').replaceAll('"','""')+'"').join(',')).join('\r\n');
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=node('a');a.href=url;a.download='seven-wonders-evaluation.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
data=normalize(data);render();
