/* Dependency-free DOM contract checks, executable in Node or a JavaScript isolate. */
async function runDashboardLanguageTests(sources) {
  const assert=(condition,message)=>{if(!condition)throw new Error(message);};
  class Element {
    constructor(tag){this.tagName=tag;this.children=[];this.attributes={};this.listeners={};this.style={};this._text='';this._value=undefined;this.className='';}
    set textContent(value){this._text=String(value);this.children=[];}
    get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
    set value(value){this._value=String(value);}
    get value(){return this._value??(this.tagName==='select'?this.children[0]?.value??'':'');}
    get hash(){return this.href?.includes('#')?'#'+this.href.split('#')[1]:'';}
    append(...nodes){for(const node of nodes){node.parent=this;this.children.push(node);}}
    prepend(node){node.parent=this;this.children.unshift(node);}
    replaceChildren(...nodes){this._text='';this.children=[];if(this.tagName==='select')this._value=undefined;this.append(...nodes);}
    remove(){if(this.parent)this.parent.children=this.parent.children.filter(c=>c!==this);}
    setAttribute(key,value){this.attributes[key]=String(value);}
    getAttribute(key){return this.attributes[key]??null;}
    removeAttribute(key){delete this.attributes[key];}
    get classList(){return {toggle:(name,on)=>{const set=new Set(this.className.split(' ').filter(Boolean));on?set.add(name):set.delete(name);this.className=[...set].join(' ');}};}
    addEventListener(name,fn){(this.listeners[name]??=[]).push(fn);}
    async fire(name){for(const fn of this.listeners[name]||[])await fn({target:this});}
    click(){if(!this.disabled)return this.fire('click');}
    querySelectorAll(selector){const key=selector.slice(1,-1);return all(this).filter(n=>n!==this&&Object.hasOwn(n.attributes,key));}
  }
  function all(root){return [root,...root.children.flatMap(all)];}
  const byText=(root,text)=>all(root).find(n=>n.textContent===text&&n.children.length===0);
  const controls=root=>all(root).filter(n=>n.tagName==='select'||n.tagName==='input');
  function fixture(){
    const category={n:480,mean:1.03,median:0,p10:0,p90:4,max:16,zero_fraction:0.577,positive_mean:2.44};
    const m={completed:480,attempted:480,failed:0,truncated:0,seed_blocks:40,valid_for_selection:true,
      win_share:0.4354167,agent_vp_mean:39.0375,vp_diff_mean:-2.03,win_share_ci95:[0.39,0.48],vp_diff_ci95:[-3.09,-0.98],
      categories:{science:category,blue:{...category,mean:12.58},yellow:null,total:{...category,mean:39.0375}},
      coverage:{final_states:480,complete_age_sequences:480,action_traces:480,score_sum_mismatches:0,invalid_age_sequences:0,unknown_built_cards:0},
      age_points:{1:{science:category,total:category}},age_economy:{1:{coins:category}},age_actions:{1:{green:0.05}},
      cards:Array.from({length:61},(_,i)=>({card:'Workshop '+i,color:'green',era:1,built:i,games_built:i,build_game_fraction:i/480,legal_opportunities:100,selection_fraction:i/100})),
      groups:{outcome:{Sieg:{games:203,win_share:1,vp_diff_mean:8,categories:{total:category}}},wonder:{'Gizah A (ID 0)':{games:30,win_share:0.5,categories:{total:category}}}},
      category_margin_to_best:{science:category}};
    return [{title:'Example experiment',title_de:'Beispielexperiment',report:{seed:59001,references:[{checkpoint:'checkpoint_65536.pt'}],models:{initial:{...m,win_share:0},candidate:m},
      model_descriptors:{initial:{checkpoint:'checkpoint_0.pt'},candidate:{checkpoint:'latest.pt',timestep:262144}},
      paired_progress:{initial:{paired_valid:true,vp_diff_gain:30,win_share_gain:0.43,vp_diff_gain_ci95:[28,32],win_share_gain_ci95:[0.39,0.48]}}}}];
  }
  function runtime({storage=new Map(),blocked=false}={}){
    const document=new Element('document'),html=new Element('html'),body=new Element('body');document.documentElement=html;document.body=body;document.append(html);html.append(body);
    const nodes={};
    for(const id of ['theme-toggle','language-select','report-select','model-select','navigation','page-title','page-description','dataset-meta','content','error','import-file','evaluation-data']){
      const tag=id.endsWith('select')?'select':id==='theme-toggle'?'button':'div';
      const node=new Element(tag);node.id=id;nodes[id]=node;body.append(node);
    }
    for(const language of ['en','de']){const option=new Element('option');option.value=language;option.textContent=language;nodes['language-select'].append(option);}
    const staticTitle=new Element('title');staticTitle.setAttribute('data-i18n','7 Wonders · Self-Play-Auswertung');body.append(staticTitle);
    const demo=new Element('p');demo.setAttribute('data-language-en','English experiment context');demo.setAttribute('data-language-de','Deutscher Experimentkontext');body.append(demo);
    nodes['evaluation-data'].textContent=JSON.stringify(fixture());
    document.createElement=tag=>new Element(tag);document.getElementById=id=>nodes[id];
    const window=new Element('window'),location={hash:'#overview'},g={};
    const localStorage={getItem:key=>{if(blocked)throw new Error('blocked');return storage.get(key)??null;},setItem:(key,value)=>{if(blocked)throw new Error('blocked');storage.set(key,value);}};
    const blobs=[];
    class Blob {constructor(parts){this.parts=parts;blobs.push(this);}}
    const URL={createObjectURL:()=> 'blob:test',revokeObjectURL:()=>{}};
    const media={matches:false,addEventListener:()=>{}};
    for(const name of ['i18n','theme','dashboard']){
      new Function('globalThis','document','window','localStorage','matchMedia','location','URL','Blob','setTimeout',sources[name])(
        g,document,window,localStorage,()=>media,location,URL,Blob,()=>{});
    }
    return {g,document,nodes,storage,blobs,staticTitle,demo,async view(view){location.hash='#'+view;await window.fire('hashchange');}};
  }
  const app=runtime();
  assert(app.document.documentElement.lang==='en','English must be the first-open default');
  assert(app.nodes['page-title'].textContent==='Overview','English overview');
  assert(app.staticTitle.textContent==='7 Wonders · Self-play evaluation','English document title');
  assert(app.nodes['model-select'].textContent.includes('Final model – step 262,144'),'English model identity and grouping');
  assert(app.nodes.content.textContent.includes('39.04'),'English decimal formatting');
  assert(app.nodes['theme-toggle'].textContent==='Dark mode: off','English theme label');
  assert(app.demo.textContent==='English experiment context','English public context');
  const views={overview:['Overview','Überblick'],points:['Score breakdown','Punkteherkunft'],ages:['Ages','Zeitalter'],cards:['Cards & actions','Karten & Aktionen'],comparisons:['Comparisons','Vergleiche'],quality:['Data quality','Datenqualität']};
  for(const [view,names] of Object.entries(views)){
    await app.view(view);
    assert(app.nodes['page-title'].textContent===names[0],'English '+view);
    assert(!/Wissenschaft|Zeitalter|Sieganteil|nicht erfasst|Punktevorsprung|Modellstand|Einträge|Auswahlquote/.test(app.nodes.content.textContent),'No German UI fragments in English '+view);
    app.g.SevenWondersI18n.setLanguage('de');
    assert(app.nodes['page-title'].textContent===names[1],'German '+view);
    assert(app.nodes['model-select'].textContent.includes('Trainingsende – Schritt 262.144'),'German model identity');
    app.g.SevenWondersI18n.setLanguage('en');
  }
  await app.nodes['theme-toggle'].click();
  app.nodes['language-select'].value='de';await app.nodes['language-select'].fire('change');
  assert(app.nodes['theme-toggle'].textContent==='Dark Mode: an','German theme label');
  assert(app.document.documentElement.getAttribute('data-theme')==='dark','Language preserves theme');
  assert(app.staticTitle.textContent==='7 Wonders · Self-Play-Auswertung','German document title');
  assert(app.demo.textContent==='Deutscher Experimentkontext','German public context');
  assert(app.g.SevenWondersDashboard.csvText([{key:'category',label:'Kategorie'},{key:'share',label:'Sieganteil',format:v=>String(v).replace('.',',')}],[{category:'Grün',share:0.5}]).includes('"Kategorie";"Sieganteil"'),'German CSV headings');
  const reopened=runtime({storage:app.storage});
  assert(reopened.document.documentElement.lang==='de','Manual language survives reopening');
  assert(reopened.document.documentElement.getAttribute('data-theme')==='dark','Saved theme survives reopening');
  app.g.SevenWondersI18n.setLanguage('en');await app.view('cards');
  let [search,colour,age]=controls(app.nodes.content);
  search.value='Workshop';await search.fire('input');colour.value='green';await colour.fire('change');age.value='1';await age.fire('change');
  const firstToolbar=all(app.nodes.content).find(n=>n.className==='table-toolbar');
  const pageSize=all(firstToolbar).find(n=>n.tagName==='select');pageSize.value='0';await pageSize.fire('change');
  const sortButton=byText(app.nodes.content,'Build actions ↓');assert(sortButton,'Card build sort heading');await sortButton.click();
  app.g.SevenWondersI18n.setLanguage('de');
  [search,colour,age]=controls(app.nodes.content);
  assert(search.value==='Workshop'&&colour.value==='green'&&age.value==='1','Language preserves card filters');
  assert(byText(app.nodes.content,'Bauaktionen ↑'),'Language preserves sort order');
  assert(all(app.nodes.content).find(n=>n.className==='table-toolbar').textContent.includes('61 Einträge'),'Filtered row count');
  app.g.SevenWondersI18n.setLanguage('en');
  await byText(app.nodes.content,'Export CSV').click();
  const csv=app.blobs.at(-1).parts.join('');
  assert(csv.includes('"Games with a build (#)"')&&csv.includes('"Selection rate"'),'English CSV headings');
  assert(csv.includes('"Green"')&&!csv.includes('Grün'),'English CSV values');
  assert(csv.includes('0.0 %')&&csv.includes('12.0 %'),'English formatted CSV percentages');
  assert(csv.split('\r\n').length===62,'CSV includes every filtered row');
  await app.view('comparisons');let [kind]=controls(app.nodes.content); // tables have page-size selects first
  const groupControls=all(app.nodes.content).find(n=>n.className==='group-controls');
  kind=controls(groupControls)[0];kind.value='outcome';await kind.fire('change');
  assert(byText(app.nodes.content,'Win'),'Translate stored outcome labels');
  kind.value='wonder';await kind.fire('change');assert(byText(app.nodes.content,'Giza A (ID 0)'),'Translate wonder names');
  app.g.SevenWondersI18n.setLanguage('de');assert(byText(app.nodes.content,'Gizah A (ID 0)'),'Restore German wonder names');
  const oldReport=app.nodes['report-select'].value;app.nodes['model-select'].value='initial';await app.nodes['model-select'].fire('change');
  app.g.SevenWondersI18n.setLanguage('en');assert(app.nodes['model-select'].value==='initial'&&app.nodes['report-select'].value===oldReport,'Language preserves report and model');
  const imported=fixture()[0].report;
  imported.models={candidate:{...imported.models.candidate,coverage:{},valid_for_selection:false}};
  imported.models.candidate.categories.commercial_combined={mean:4};
  app.nodes['import-file'].files=[{name:'older-analysis.json',size:100,text:async()=>JSON.stringify(imported)}];
  await app.view('overview');await app.nodes['import-file'].fire('change');
  assert(app.nodes.content.textContent.includes('This evaluation contains incomplete games.'),'English incomplete-game warning');
  assert(app.nodes.content.textContent.includes('Some age or action data is missing'),'English missing-data warning');
  app.g.SevenWondersI18n.setLanguage('de');app.g.SevenWondersI18n.setLanguage('en');
  assert(app.nodes['report-select'].value==='1'&&app.nodes['model-select'].value==='candidate','Imported report remains selected');
  await app.view('points');assert(app.nodes.content.textContent.includes('Commercial + treasury (combined)'),'Translate older combined score category');
  assert(app.nodes.content.textContent.includes('not recorded'),'Translate missing values without inventing zeros');
  app.nodes['import-file'].files=[{size:2,text:async()=>'{bad JSON'}];await app.nodes['import-file'].fire('change');
  assert(app.nodes.error.textContent==='Import failed: Invalid JSON file.','English import error');
  app.g.SevenWondersI18n.setLanguage('de');assert(app.nodes.error.textContent==='Import nicht möglich: Ungültige JSON-Datei.','Translate visible error on switch');
  const blocked=runtime({blocked:true});assert(blocked.document.documentElement.lang==='en','Blocked storage uses English');
  blocked.g.SevenWondersI18n.setLanguage('de');assert(blocked.nodes['page-title'].textContent==='Überblick','Blocked storage still permits switching');
  assert(blocked.g.SevenWondersI18n.setLanguage('fr')===false,'Reject unsupported languages');
  assert(runtime({storage:new Map([['sevenwonders-dashboard-language','fr']])}).document.documentElement.lang==='en','Invalid saved language defaults to English');
  const unsafe=app.g.SevenWondersDashboard.csvText([{key:'x',label:'Karte'}],[{x:'=SUM(1,2)'}]);
  assert(unsafe.includes("'=SUM(1,2)"),'CSV formula escaping is preserved');
  return {views:6,languages:2,persistence:true,blocked_storage:true,localized_csv:true,preserved_filters_and_selection:true};
}
if(typeof module!=='undefined')module.exports=runDashboardLanguageTests;
