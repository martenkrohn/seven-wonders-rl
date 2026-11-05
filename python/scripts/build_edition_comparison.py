#!/usr/bin/env python3
"""Build a local, offline edition report without changing simulator data.

The report embeds the user's old images. New images are optional, with stable
paths recorded in images_manifest.json. All code uses the standard library.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import copy
import csv
import hashlib
import html
import json
import mimetypes
from pathlib import Path
import re

import extract_card_db_from_cpp as cpp

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/reference/edition-comparison'
RESOURCE_NAMES = dict(wood='Holz', ore='Erz', clay='Lehm', stone='Stein',
                      loom='Stoff', glass='Glas', papyrus='Papyrus', coins='Münzen')
TOKENS = {'wood':'wood', 'ore':'ore', 'clay':'clay', 'stone':'stone',
          'cloth':'loom', 'loom':'loom', 'glass':'glass', 'papyrus':'papyrus',
          'press':'papyrus', 'coin':'coins', 'coins':'coins'}
ALIASES = {'craftmens guild':'Craftsmens Guild', 'univeristy':'University',
           'bazaar':'Bazar'}
COLORS = {'Raw Material':'brown', 'Manufactured Good':'grey',
          'Civilian Structure':'blue', 'Commercial Structure':'yellow',
          'Military Structure':'red', 'Scientific Structure':'green', 'Guild':'purple'}
COLOR_NAMES = dict(brown='Braun', grey='Grau', blue='Blau', yellow='Gelb',
                   red='Rot', green='Grün', purple='Violett')
FIELDS = {'name':'Name', 'cost':'Kosten', 'effects':'Effekt', 'chain_from':'Gratis mit', 'chain_to':'Ermöglicht gratis',
          'copies':'Kopien ab Spielerzahl', 'starting_resource':'Startressource', 'stages':'Stufen'}
OFFICIAL = 'https://cdn.svc.asmodee.net/production-rprod/storage/downloads/games/7wonders/en/'
SOURCES = {
 'official_cards': OFFICIAL + 'sev-en02-chain-list-1598888877Na4yW.pdf',
 'official_effects': OFFICIAL + 'sev-en02-effects-description-1716388911mLqDj.pdf',
 'official_faq': 'https://www.rprod.com/en/7-wonders/faq',
 'board_photos': 'https://assortedmeeples.com/7-wonders-board-game-strategy',
 'bga_tutorial': 'https://en.boardgamearena.com/tutorial/tutorial/frame.html?game=sevenwonders&tutorial=570',
}

BGA_RESOURCE = {'W':'wood','S':'stone','C':'clay','O':'ore','G':'glass','P':'papyrus','L':'loom'}
BGA_NAMES = {'Pawnshop':'Well','Apothocary':'Apothecary','Craftsmen Guild':'Craftsmens Guild'}
BGA_COLORS = dict(raw='brown',man='grey',civ='blue',com='yellow',mil='red',sci='green',gui='purple')
VERIFIED_BGA_COST_CORRECTIONS = {
    'town_hall_age3': {'stone':2,'ore':1,'glass':1},
    'arena_age3': {'stone':2,'ore':1},
    'circus_age3': {'stone':3,'ore':1},
    'craftsmens_guild_age3': {'stone':2,'ore':2},
}


def bga_name(value):
    return canonical(BGA_NAMES.get(value,value))


def bga_cost(item):
    result=dict(Counter(BGA_RESOURCE[r] for r in item.get('cost',[])))
    if item.get('coincost'):result['coins']=int(item['coincost'])
    return result


def bga_effects(item):
    if item.get('ressmode'):
        if item['ressmode']=='or':
            return dict(choice=sorted({BGA_RESOURCE[r] for r in item['ress']}),tradable=item['category'] in {'raw','man'})
        return dict(production=dict(Counter(BGA_RESOURCE[r] for r in item['ress'])),tradable=item['category'] in {'raw','man'})
    if item.get('discount'):
        d=item['discount']
        return {'discount': ('raw_'+d['direction']) if d['type']=='raw' else 'manufactured_both'}
    if item.get('shield'):return {'shields':int(item['shield'])}
    if item.get('science'):return {'science':{'1':'tablet','2':'gear','3':'compass','?':'joker'}[str(item['science'])]}
    result={}
    counts=dict(raw='brown',man='grey',civ='blue',com='yellow',mil='red',sci='green',rawmangui='brown_grey_purple',wonder='wonder_stages',strategist='defeat_tokens')
    directions={'all':'self_and_neighbors','both':'neighbors','none':'self'}
    for field, label in [('points','vp'),('coins','coins')]:
        if field not in item:continue
        value=item[field]
        if value['type'] in {'simple','decorators'}:
            result[label]=int(value['qt'])
            if value['type']=='decorators':result['condition']='wonder_complete'
        else:
            result[label+'_per']=int(value['qt']);result['count']=counts[value['type']];result['target']=directions[value['direction']]
            if value['type'] in {'com','rawmangui'} and value['direction']=='none':result['includes_self']=True
    return result


def bga_stage(item):
    result=dict(cost=bga_cost(item),vp=int(item.get('points',{}).get('qt',0)),effects=[])
    effects=result['effects']
    if item.get('shield'):effects.append(dict(type='add_shields',amount=int(item['shield'])))
    if item.get('coins'):effects.append(dict(type='add_coins',amount=int(item['coins']['qt'])))
    if item.get('science'):effects.append(dict(type='add_science'))
    if item.get('ressmode'):
        resources=set(item['ress'])
        if resources==set('CWOS'):effects.append(dict(type='choose_raw_nontradable'))
        elif resources==set('GLP'):effects.append(dict(type='choose_manuf_nontradable'))
        else:raise ValueError(f'Unexpected wonder production: {resources}')
    for flag, effect in {'pickDiscarded':'discard_build_for_free','sevenCard':'play_seventh_card','buildFirstEachColor':'first_color_free','buildFirst':'first_age_card_free','buildLast':'last_age_card_free'}.items():
        if item.get(flag):effects.append(dict(type=effect))
    return result


def apply_bga_reference(cards,wonders,directory):
    """Join independent definitions by names/age, retain discrepancies and IDs."""
    raw_cards=json.loads((directory/'card_types.json').read_text())
    raw_wonders=json.loads((directory/'wonders.json').read_text())
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['data_edition']!='second edition (game_edition=1)':
        raise ValueError('BGA reference is not the second-edition snapshot')
    for file in manifest['files']:
        p=directory/file['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=file['sha256']:
            raise ValueError(f'BGA asset hash mismatch: {p}')
    index={(bga_name(c['name']),int(c['age'])):(k,c) for k,c in raw_cards.items()}
    if len(index)!=80:raise ValueError('Unexpected BGA card name/age coverage')
    conflicts=[]
    for row in cards:
        key,item=index[(row['name'],row['age'])]
        if BGA_COLORS[item['category']]!=row['color']:raise ValueError('BGA category mismatch')
        new=dict(name=row['name'],cost=bga_cost(item),effects=bga_effects(item),
                 chain_from=sorted(bga_name(raw_cards[str(k)]['name']) for k in item['freeif']),
                 copies=marks_from_counts([int(item['qt'][str(n)]) for n in range(3,8)]) if row['color']!='purple' else [3])
        for field in ('cost','effects','chain_from','copies'):
            if row['new'][field]!=new[field]:
                if field!='cost' or VERIFIED_BGA_COST_CORRECTIONS.get(row['key'])!=new[field]:
                    raise ValueError(f'Unreviewed BGA discrepancy: {row["key"]} {field}; check the source and artwork before accepting it')
                conflicts.append(dict(key=row['key'],name=row['name'],field=field,previous=row['new'][field],bga=new[field],resolution='BGA definitions; differing card costs also checked on corresponding v2 sprite artwork'))
        row['previous_comparison']=copy.deepcopy(row['new'])
        row['new']=new;row['bga_id']=int(key);row['bga_source_name']=item['name']
        row['new_sprite']=dict(file='img/cards_v2.jpg',x=item['img']%10*255,y=item['img']//10*393,width=255,height=393)
        row['verification']={f:'BGA-Definitionen der zweiten Edition; zugeordnetes v2-Kartenbild vorhanden' for f in ('cost','effects','copies','chain_from')}
        if item['name']!=row['name']:
            row['notes'].append(f'BGA-Quellname „{item["name"]}“ wird im Vergleich als „{row["name"]}“ geführt; Zuordnung über Zeitalter und Sprite-Index.')
    for row in cards:
        row['new']['chain_to']=sorted({r['new']['name'] for r in cards if row['name'] in r['new']['chain_from']})
        row['edition_changes']=list(row['new']) if row['old'] is None else [f for f in row['new'] if row['new'][f]!=row['old'][f]]
    for row in wonders:
        bga_id=row['engine_id']+1;item=raw_wonders[str(bga_id)]
        new=dict(name=row['name'],starting_resource=BGA_RESOURCE[item['ress'][0]],stages=[bga_stage(s) for _,s in sorted(item['stages'].items(),key=lambda v:int(v[0]))])
        for f in ('starting_resource','stages'):
            if new[f]!=row['new'][f]:raise ValueError(f'BGA/photo wonder discrepancy: {row["key"]} {f}')
        row['bga_id']=bga_id;row['new_sprite']=dict(file='img/boards_v2.jpg',x=0,y=(bga_id-1)*223,width=450,height=223)
        row['verification']='Tafelfoto und BGA-Definitionen der zweiten Edition stimmen für Startressource, Kosten und Effekte überein.'
    return conflicts,manifest


def canonical(name):
    name = name.strip()
    return ALIASES.get(name.lower(), name)


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def cost(text):
    result = Counter()
    for amount, token in re.findall(r'(?:(\d+)\s*(?:x\s*)?)?\b(Wood|Ore|Clay|Stone|Cloth|Loom|Glass|Papyrus|Press|Coins?)\b', text, re.I):
        result[TOKENS[token.lower()]] += int(amount or 1)
    return dict(result)


def label_cost(value):
    return ' + '.join(f'{value[r]} {"Münze" if r=="coins" and value[r]==1 else name}' for r, name in RESOURCE_NAMES.items() if value.get(r)) or 'kostenlos'


def copies_from_marks(text):
    return [int(x) for x in re.findall(r'\d+', text)]


def marks_from_counts(values):
    previous, marks = 0, []
    for players, amount in zip(range(3, 8), values):
        if amount < previous:
            raise ValueError('Deck counts must be cumulative')
        marks.extend([players] * (amount - previous))
        previous = amount
    return marks


def basic_effect(name, color, provides):
    """Normalize printed effects; all values retain their conditional target."""
    if color == 'brown':
        resources = re.findall(r'\b(Wood|Ore|Clay|Stone)\b', provides, re.I)
        if 'OR' in provides:
            return {'choice': sorted({TOKENS[r.lower()] for r in resources}), 'tradable': True}
        return {'production': cost(provides), 'tradable': True}
    if color == 'grey':
        return {'production': cost(provides), 'tradable': True}
    if color == 'blue':
        return {'vp': int(re.search(r'\d+', provides).group())}
    if color == 'red':
        return {'shields': int(re.search(r'\d+', provides).group())}
    if color == 'green':
        return {'science': next(s for s in ['compass','gear','tablet'] if s in provides.lower())}
    effects = {
      'Tavern': {'coins':5},
      'East Trading Post': {'discount':'raw_right'},
      'West Trading Post': {'discount':'raw_left'},
      'Marketplace': {'discount':'manufactured_both'},
      'Forum': {'choice':['glass','loom','papyrus'], 'tradable':False},
      'Caravansery': {'choice':['clay','ore','stone','wood'], 'tradable':False},
      'Vineyard': {'coins_per':1, 'count':'brown', 'target':'self_and_neighbors'},
      'Bazar': {'coins_per':2, 'count':'grey', 'target':'self_and_neighbors'},
      'Haven': {'coins_per':1,'vp_per':1,'count':'brown','target':'self'},
      'Lighthouse': {'coins_per':1,'vp_per':1,'count':'yellow','target':'self','includes_self':True},
      'Chamber of Commerce': {'coins_per':2,'vp_per':2,'count':'grey','target':'self'},
      'Arena': {'coins_per':3,'vp_per':1,'count':'wonder_stages','target':'self'},
      'Ludus': {'coins_per':3,'vp_per':1,'count':'red','target':'self'},
      'Workers Guild': {'vp_per':1,'count':'brown','target':'neighbors'},
      'Craftsmens Guild': {'vp_per':2,'count':'grey','target':'neighbors'},
      'Traders Guild': {'vp_per':1,'count':'yellow','target':'neighbors'},
      'Philosophers Guild': {'vp_per':1,'count':'green','target':'neighbors'},
      'Spies Guild': {'vp_per':1,'count':'red','target':'neighbors'},
      'Magistrates Guild': {'vp_per':1,'count':'blue','target':'neighbors'},
      'Shipowners Guild': {'vp_per':1,'count':'brown_grey_purple','target':'self','includes_self':True},
      'Strategists Guild': {'vp_per':1,'count':'defeat_tokens','target':'neighbors'},
      'Decorators Guild': {'vp':7,'condition':'wonder_complete'},
      'Scientists Guild': {'science':'joker'},
      'Builders Guild': {'vp_per':1,'count':'wonder_stages','target':'self_and_neighbors'},
    }
    return effects[name]


def effect_label(e):
    if 'production' in e:
        return 'Produktion: ' + label_cost(e['production']) + ' pro Zug (handelbar)'
    if 'choice' in e:
        return '1× ' + ' / '.join(RESOURCE_NAMES[r] for r in e['choice']) + ' pro Zug (' + ('handelbar' if e['tradable'] else 'nur selbst nutzbar') + ')'
    if 'discount' in e:
        return {'raw_right':'Rohstoffe rechts für je 1 Münze', 'raw_left':'Rohstoffe links für je 1 Münze', 'manufactured_both':'Glas/Stoff/Papyrus bei beiden Nachbarn für je 1 Münze'}[e['discount']]
    if 'shields' in e:
        return f"{e['shields']} Schilde"
    if 'science' in e:
        return {'compass':'1 Zirkel', 'gear':'1 Zahnrad', 'tablet':'1 Tafel', 'joker':'1 frei zugeteilter Wissenschaftsjoker'}[e['science']]
    if 'count' in e:
        count = {'brown':'brauner Karte', 'grey':'grauer Karte', 'yellow':'gelber Karte', 'green':'grüner Karte', 'red':'roter Karte', 'blue':'blauer Karte', 'brown_grey_purple':'brauner, grauer oder violetter Karte', 'wonder_stages':'gebauter Wunderstufe', 'defeat_tokens':'Niederlagemarker'}[e['count']]
        target = {'self':'bei dir', 'neighbors':'bei den Nachbarn', 'self_and_neighbors':'bei dir und beiden Nachbarn'}[e['target']]
        parts = []
        if e.get('coins_per'):
            parts.append(f"sofort {e['coins_per']} "+('Münze' if e['coins_per']==1 else 'Münzen'))
        if e.get('vp_per'):
            parts.append(f"am Ende {e['vp_per']} "+('Punkt' if e['vp_per']==1 else 'Punkte'))
        return ' und '.join(parts) + f' je {count} {target}' + (' (einschließlich dieser Karte)' if e.get('includes_self') else '')
    if 'coins' in e:
        return f"sofort {e['coins']} Münzen"
    return f"{e.get('vp',0)} Punkte" + (' bei vollständig gebautem eigenem Wunder' if e.get('condition') else '')


def normalize_stage(stage):
    s = copy.deepcopy(stage)
    types = {e['type'] for e in s['effects']}
    s['effects'] = [e for e in s['effects'] if e['type'] not in {'add_vp_coin','add_shield_vp_coin','add_vp_discard'}]
    if 'add_vp_discard' in types:
        s['effects'].append({'type':'discard_build_for_free'})
    return s


def stage_effect(s):
    parts = [f"{s['vp']} Punkte"] if s['vp'] else []
    labels = {'add_science':'1 Wissenschaftsjoker', 'hand_build_for_free':'1 beliebige Handkarte pro Zeitalter gratis',
              'discard_build_for_free':'1 Karte aus der Ablage gratis bauen', 'choose_raw_nontradable':'1 Rohstoff nach Wahl pro Zug, nicht handelbar',
              'choose_manuf_nontradable':'1 Manufakturware nach Wahl pro Zug, nicht handelbar',
              'play_seventh_card':'zusätzliche letzte Karte jedes Zeitalters spielen', 'buy_raw_cheap':'Rohstoffe bei beiden Nachbarn für 1 Münze',
              'copy_guild':'am Ende beste Nachbargilde kopieren', 'first_color_free':'erste Karte jeder noch nicht vorhandenen eigenen Farbe gratis',
              'first_age_card_free':'erste reguläre Karte jedes Zeitalters gratis', 'last_age_card_free':'letzte reguläre (6.) Karte jedes Zeitalters gratis'}
    for e in s['effects']:
        if e['type'] == 'add_coins': parts.append(f"sofort {e['amount']} Münzen")
        elif e['type'] == 'add_shields': parts.append(f"{e['amount']} Schilde")
        else: parts.append(labels[e['type']])
    return '; '.join(parts)


def read_cards(catalog):
    old = read_csv(ROOT/'references/cards.csv')
    new = [r for r in read_csv(ROOT/'00_marten/final_cards.csv') if int(r['id']) <= 77]
    db = json.loads((ROOT/'python/src/sevenwonders_rl/data/card_db.json').read_text())
    by_name = {canonical(c['name']):c for c in db.values()}
    ids, _ = cpp.parse_enum(ROOT/'include/card.h','CARD_ID')
    types, _ = cpp.parse_enum(ROOT/'include/card.h','CARD_TYPE')
    declarations = cpp.parse_cards_from_main(ROOT/'main.cpp',ids,types)
    by_id_age = {(d['id'],d['era']):d for d in declarations}
    old_by_name_age = {(canonical(r['name']),int(r['age'])):r for r in old}
    rows = []
    for raw in new:
        name = canonical(raw['name']); color = COLORS[raw['type']]
        eras = [1,2] if color == 'grey' else [int(raw['age'] or 3)]
        old_name = {'Well':'Pawnshop', 'Decorators Guild':'Strategists Guild'}.get(name,name)
        for age in eras:
            previous = old_by_name_age.get((old_name,age))
            current = by_name.get(old_name)
            key = re.sub(r'[^a-z0-9]+','_',name.lower()).strip('_') + f'_age{age}'
            marks = copies_from_marks(raw['amount / players']) if color!='purple' else [3]
            if color=='grey': marks = [3,6] if age==1 else [3,5]
            nxt = dict(name=name, cost=cost(raw['cost']), effects=basic_effect(name,color,raw['provides']),
                       chain_from=sorted(catalog['chains'].get(name,[])), copies=marks)
            printed, engine = None, None
            if previous:
                printed = dict(name=old_name,cost=cost(previous['cost']), effects=basic_effect(old_name,color,previous['provides']),
                               chain_from=sorted(canonical(s) for s in previous['free with'].split(' OR ') if s.strip() not in {'','-'}),
                               copies=copies_from_marks(previous['amount / players']) if color!='purple' else [3])
                # A typo in the old reference CSV; verified against the local card image.
                if old_name=='Chamber of Commerce': printed['cost']={'clay':2,'papyrus':1}
                if old_name=='Arsenal': printed['copies']=[3,4,7]
                engine = copy.deepcopy(printed)
                dec = by_id_age[(current['id'],age)]
                engine['cost'] = {r:n for r,n in zip(RESOURCE_NAMES,dec['cost_vec']) if n}
                engine['copies'] = marks_from_counts(dec['amount_vec']) if color!='purple' else [3]
                predecessor = current.get('chain_from')
                id_names = {c['id_name']:canonical(c['name']) for c in db.values()}
                engine['chain_from'] = sorted([canonical(db[str(predecessor)]['name'])] if predecessor else
                    [id_names[x] for x in current.get('chain_from_any') or []])
                # Engine effect definitions match these printed effect families.
                # Runtime ordering/joker bugs remain separately noted, not inferred away.
                if color=='blue': engine['effects']={'vp':current['vp']['fixed']}
            old_filename = old_name
            if current: old_filename = current['name']
            image = ROOT/'images/cards'/f'{old_filename}{"2" if color=="grey" and age==2 else ""}.png'
            # These two filenames are reversed relative to Loom/Press: inspect
            # the printed 3+/6+ (age I) and 3+/5+ (age II) marks, not the suffix.
            if old_name=='Glassworks': image=ROOT/'images/cards'/('Glassworks2.png' if age==1 else 'Glassworks.png')
            if not image.exists():
                image = ROOT/'images/cards'/f'{old_name}.png'
            old_rel = str(image.relative_to(ROOT)) if image.exists() and previous else None
            edition_changes = list(nxt) if printed is None else [f for f in nxt if nxt[f]!=printed[f]]
            engine_changes = [] if engine is None else [f for f in printed if printed[f]!=engine[f]]
            notes = []
            if name=='Well':notes.append('Pawnshop → Well: Vergleichspaar, keine Gleichsetzung der IDs.')
            if name=='Decorators Guild':notes.append('Strategists Guild entfällt; Decorators Guild ersetzt sie im Grundspiel-Gildenbestand.')
            if name=='Lighthouse':notes.append('Engine-Sonderfall: Der Münzertrag zählt vor dem Einfügen der eigenen Karte; die Karte zählt dadurch beim Sofortertrag nicht mit.')
            if name in {'Scientists Guild'}:notes.append('Bekannter Engine-Wertungsfehler bei der optimalen Jokerverteilung; siehe Editionsaudit.')
            if name in {'Vineyard','Bazar'}:notes.append('Engine-Auflösung erfolgt sequenziell; gleichzeitige Nachbarkarten können den Ertrag beeinflussen.')
            if name in {'Sawmill','Quarry','Brickyard','Foundry'}:notes.append('C++ produziert 2 Einheiten; die Python-Datenbank verliert die Mengenangabe.')
            rows.append(dict(key=key, kind='card', name=name, old_name=old_name if previous else None, color=color, age=age,
                             engine_id=current['id'] if current else None, legacy_csv_id=int(raw['id']), old=printed,engine=engine,new=nxt,
                             edition_changes=edition_changes,engine_changes=engine_changes,old_image=old_rel,new_image=f'cards/{key}.png',
                             notes=notes, verification={'cost':'Lokale Vorarbeit; direkte neue Kartenbildprüfung offen',
                             'effects':'Offizielle Karten-/Effektübersicht', 'copies':'Offizielle Kartenübersicht',
                             'chain_from':'Offizielle Kettenübersicht'}, source_local_row=raw))
    for row in rows:
        for side in ('old','engine','new'):
            state = row[side]
            if state is None: continue
            state['chain_to'] = sorted({other[side]['name'] for other in rows
                                       if other[side] is not None and state['name'] in other[side]['chain_from']})
        row['edition_changes'] = list(row['new']) if row['old'] is None else [f for f in row['new'] if row['new'][f]!=row['old'][f]]
        row['engine_changes'] = [] if row['engine'] is None else [f for f in row['old'] if row['old'][f]!=row['engine'][f]]
    return rows


def read_wonders(catalog):
    old = json.loads((ROOT/'python/src/sevenwonders_rl/data/wonder_db.json').read_text())
    result = []
    for k, v in old.items():
        previous = dict(name=v['name'], starting_resource=v['starting_resource'],stages=[normalize_stage(s) for s in v['stages']])
        n = catalog['wonders'][k]
        nxt = dict(name=n['name'],starting_resource=n['starting_resource'],stages=n['stages'])
        changes = [f for f in ('starting_resource','stages') if previous[f]!=nxt[f]]
        result.append(dict(key=v['id_name'],kind='wonder',name=n['name'],old_name=v['name'],engine_id=int(k),
                           old=previous,engine=copy.deepcopy(previous),new=nxt,edition_changes=changes,engine_changes=[],
                           old_image=f'images/wonders/board_{k}.png',new_image=f'wonders/{v["id_name"]}.png',
                           source_image=n['source_image'],verification=n['verification'],notes=[]))
    return result


def value_label(field, v, row):
    if v is None: return 'nicht vorhanden'
    x = v[field]
    if field=='cost':return label_cost(x)
    if field=='effects':return effect_label(x)
    if field=='chain_from':return ' ODER '.join(x) or 'keine Baukette'
    if field=='chain_to':return ', '.join(x) or 'keine Folgekarte'
    if field=='copies':return 'Gildenpool: 1 Exemplar, davon Spielerzahl + 2 zufällig' if row.get('color')=='purple' else ', '.join(f'{n}+' for n in x)
    if field=='starting_resource':return '1 '+RESOURCE_NAMES[x]
    return str(x)


def data_url(path):
    if not path.is_file():return None
    raw=path.read_bytes()
    mime = ('image/webp' if raw[:4]==b'RIFF' and raw[8:12]==b'WEBP' else
            'image/png' if raw[:8]==b'\x89PNG\r\n\x1a\n' else
            'image/jpeg' if raw[:2]==b'\xff\xd8' else mimetypes.guess_type(path.name)[0])
    if mime not in {'image/png','image/jpeg','image/webp'}:raise ValueError(f'Unsupported image type: {path}')
    return 'data:'+mime+';base64,'+base64.b64encode(raw).decode()


def image_pair(row, new_dir):
    left = data_url(ROOT/row['old_image']) if row['old_image'] else None
    right = data_url(new_dir/row['new_image']) if new_dir else None
    def panel(src, label, filename, new=False):
        attrs = f' data-image-key="{html.escape(filename)}"' if new else ''
        if src:
            inner=f'<img src="{src}" alt="{html.escape(label)}" loading="lazy">'
        elif new and row.get('new_sprite'):
            s=row['new_sprite'];sheet='bga-cards' if row['kind']=='card' else 'bga-boards'
            text=(f'<text x="{s["x"]+s["width"]/2}" y="{s["y"]+12}" text-anchor="middle" font-family="serif" font-size="10" font-weight="bold" fill="#413d29">{html.escape(row["name"])}</text>' if row['kind']=='card' else '')
            clip='clip-'+row['key']
            inner=f'<svg class="sprite" viewBox="{s["x"]} {s["y"]} {s["width"]} {s["height"]}" role="img" aria-label="{html.escape(row["name"])} · BGA zweite Edition"><defs><clipPath id="{clip}"><rect x="{s["x"]}" y="{s["y"]}" width="{s["width"]}" height="{s["height"]}"/></clipPath></defs><g clip-path="url(#{clip})"><use href="#{sheet}"/>{text}</g></svg>'
        else:
            inner='<div class="placeholder">'+('Bild der neuen Edition folgt' if new else 'Keine Karte in der alten Ausgabe')+'</div>'
        return f'<figure{attrs}><figcaption>{html.escape(label)}</figcaption>{inner}</figure>'
    return '<div class="images">'+panel(left,'Alte Ausgabe · lokales Bild','')+panel(right,'Neue Ausgabe · 2020',row['new_image'],True)+'</div>'


def html_entry(row, new_dir):
    esc = html.escape
    changed = bool(row['edition_changes']); issue=bool(row['engine_changes'])
    badges = ('<span class="badge delta">Edition geändert</span>' if changed else '<span class="badge">Regeldaten gleich</span>')
    if issue:badges+='<span class="badge issue">Engine weicht ab</span>'
    name = row['name'] if not row['old_name'] or row['old_name']==row['name'] else row['old_name']+' → '+row['name']
    meta = f"Zeitalter {row['age']} · {COLOR_NAMES[row['color']]}" if row['kind']=='card' else f"Wunder-ID {row['engine_id']}"
    table = '<div class="table-wrap"><table><thead><tr><th>Merkmal</th><th>Alte Ausgabe</th><th>Engine aktuell</th><th>Neue Ausgabe 2020</th></tr></thead><tbody>'
    if row['kind']=='card':
        for f in ('cost','effects','chain_from','chain_to','copies'):
            table += '<tr><th>'+FIELDS[f]+'</th>'
            for side in ('old','engine','new'):
                cls = ' changed' if side=='new' and f in row['edition_changes'] else ' issuecell' if side=='engine' and f in row['engine_changes'] else ''
                table += f'<td class="{cls}">{esc(value_label(f,row[side],row))}</td>'
            table += '</tr>'
    else:
        table += '<tr><th>Startressource</th>'+''.join(f'<td class="{"changed" if side=="new" and "starting_resource" in row["edition_changes"] else ""}">{esc(value_label("starting_resource",row[side],row))}</td>' for side in ('old','engine','new'))+'</tr>'
        for i in range(max(len(row['old']['stages']),len(row['new']['stages']))):
            table += f'<tr><th>Stufe {i+1}</th>'
            for side in ('old','engine','new'):
                stages=row[side]['stages'];stage=stages[i] if i<len(stages) else None
                cls = 'changed' if side=='new' and (i>=len(row['old']['stages']) or stage!=row['old']['stages'][i]) else ''
                text = esc(label_cost(stage['cost']))+'<br><span class="effect">'+esc(stage_effect(stage))+'</span>' if stage else 'entfällt'
                table+=f'<td class="{cls}">{text}</td>'
            table+='</tr>'
    table+='</tbody></table></div>'
    notes=''.join('<p class="note">'+esc(s)+'</p>' for s in row['notes'])
    verify = 'Neue Ausgabe: BGA-Definitionen und zugeordnetes v2-Kartenbild. Effekte, Ketten und Kopien zusätzlich gegen offizielle Übersicht abgeglichen.' if row['kind']=='card' else row['verification']
    source = SOURCES['bga_tutorial'] if row['kind']=='card' else row['source_image']
    search = ' '.join(str(row.get(k,'')) for k in ('name','old_name','key','color'))+' '+meta
    return f'<article id="{row["key"]}" data-kind="{row["kind"]}" data-color="{row.get("color","")}" data-age="{row.get("age","")}" data-changed="{int(changed)}" data-issue="{int(issue)}" data-search="{esc(search.lower(),quote=True)}"><header><div><p class="eyebrow">{esc(meta)}</p><h2>{esc(name)}</h2></div><div>{badges}</div></header>'+image_pair(row,new_dir)+table+notes+f'<footer>{esc(verify)} <a href="{esc(source,quote=True)}" target="_blank" rel="noreferrer">Prüfquelle ↗</a></footer></article>'


def write_markdown(rows, wonders, path):
    lines=['# Vollständige Gegenüberstellung: Karten und Wunder', '', 'Stand: 5. November 2026. Grundspiel 2010 / aktueller Simulator / Grundspiel 2020.', '',
           '**Alle Positionen sind erfasst. Die neuen Kartenkosten sind mit den BGA-Definitionen der zweiten Edition abgeglichen; alle zugehörigen v2-Kartenbilder sind im lokalen Bericht enthalten.** Die 14 neuen Wunderseiten stimmen mit Tafelfotos und BGA-Definitionen überein.', '',
           'Die lokale Datei `00_marten/final_cards.csv` enthält bereits neue Kartenwerte. Sie war nicht in die Engine übernommen. Ihre IDs sind kein Engine-ID-Schema. Die sechs zusätzlichen Gilden ab CSV-ID 78 sind Erweiterungen/Promos und gehören nicht in diesen Grundspielvergleich.', '',
           '## Karten', '', 'Grau wird je Zeitalter getrennt geführt. Kopien „3+, 4+“ bedeuten zwei physische Karten mit diesen Aufdrucken; die Zahl ist keine Mindestspielerzahl des Kartentyps. Gilden: Pool mit zehn Karten, Spielerzahl + 2 werden gezogen.', '']
    for age in (1,2,3):
        lines += [f'### Zeitalter {age}', '', '| Karte / Engine-ID | Merkmal | Alte Ausgabe | Engine aktuell | Neue Ausgabe 2020 |', '| --- | --- | --- | --- | --- |']
        for r in rows:
            if r['age']!=age:continue
            name=r['name'] if r['old_name'] in (None,r['name']) else r['old_name']+' → '+r['name']
            for i,f in enumerate(('cost','effects','chain_from','chain_to','copies')):
                vals=[value_label(f,r[s],r).replace('|','/') for s in ('old','engine','new')]
                if f in r['edition_changes']:vals[2]='**'+vals[2]+'**'
                if f in r['engine_changes']:vals[1]='**'+vals[1]+'**'
                lines.append('| '+(' / '.join((name,str(r['engine_id']))) if i==0 else '')+' | '+FIELDS[f]+' | '+' | '.join(vals)+' |')
        lines.append('')
    lines += ['## Wunder', '', 'A → Tag; B → Nacht. Die ID benennt die heutige Engine-Zuordnung. Stufen werden nach Bauposition gegenübergestellt; bei Babylon Nacht entfällt die alte erste Punktestufe, die Fähigkeiten rücken nach vorn.', '']
    for r in wonders:
        lines += [f'### {r["old_name"]} → {r["name"]} (ID {r["engine_id"]})', '', '| Merkmal | Alte Ausgabe / Engine aktuell | Neue Ausgabe 2020 |','| --- | --- | --- |',
                  '| Start | '+value_label('starting_resource',r['old'],r)+' | '+value_label('starting_resource',r['new'],r)+' |']
        for i in range(max(len(r['old']['stages']),len(r['new']['stages']))):
            values=[]
            for side in ('old','new'):
                s=r[side]['stages'];values.append(label_cost(s[i]['cost'])+' → '+stage_effect(s[i]) if i<len(s) else 'entfällt')
            lines.append(f'| Stufe {i+1} | '+ ' | '.join(values)+' |')
        lines += ['', f'[Geprüftes Tafelfoto]({r["source_image"]})', '']
    lines += ['## Deckgrößen als Gegenprüfung', '', 'Je Zeitalter sind 7 × Spielerzahl Karten erforderlich. Die alte gedruckte Ausgabe und die neue Datentabelle erfüllen diese Bedingung für 3–7 Spieler. Die aktuellen Engine-Deklarationen liefern in Zeitalter III bei vier Spielern nur 26 statt 28 Karten, bei 5–7 Spielern jeweils eine zu wenig: Arsenal fehlt bei 4+, University kommt erst bei 5+ statt 4+ hinzu. Der derzeitige Dreispielerbetrieb erreicht trotzdem 21 Karten. Dies ist ein geerbter Datenfehler, keine Editionsänderung.', '',
              '## Grenzen und Quellen', '', '- Die Engine-Spalte beschreibt Kosten, Ketten, Verteilungen und Effektdefinitionen. Sie bestätigt nicht deren fehlerfreie Ausführung: Halikarnassos-Auswahl, Jokerwertung und simultane Erträge sind im [Audit](../EDITION_AUDIT.md) dokumentiert. Lighthouse zählt sich beim Sofort-Münzertrag in der aktuellen Ausführung nicht selbst mit.',
              '- Die alte Referenz-CSV nennt bei Chamber of Commerce fälschlich zwei Stein; das vorhandene Bild zeigt zwei Lehm. Dieser Quellenfehler wurde nur im Vergleich korrigiert.',
              '- Bei Arsenal fehlt im alten Referenz-CSV der Aufdruck 4+; das lokale Bild zeigt 3+, 4+, 7+. Die Engine hat diese 4+-Kopie ebenfalls nicht. Glassworks/Glassworks2 sind anhand der Aufdrucke dem richtigen Zeitalter zugeordnet.',
              '- Die alte Referenz-CSV beschreibt Shipowners Guild fälschlich als Nachbarwertung; Kartenbild und Engine beziehen sich auf die eigene Stadt. Im Vergleich wurde die gedruckte Fähigkeit verwendet.',
              '- Graue Karten sind in der neuen lokalen CSV nach Zeitalter zusammengeführt; für diesen Bericht anhand der offiziellen Übersicht getrennt.',
              '- Die neue Kartenkosten-Spalte wurde mit der unabhängigen BGA-Referenz abgeglichen. Vier frühere CSV-Werte wurden nach Definitionen und v2-Kartenbild korrigiert: Town Hall, Arena, Circus, Craftsmens Guild. Einzelheiten und beide Ausgangswerte stehen in [source_discrepancies.json](source_discrepancies.json).',
              '- BGA behält teils alte Namen oder Schreibvarianten: Pawnshop wird als Well, Apothocary als Apothecary und Craftsmen Guild als Craftsmens Guild zugeordnet. Die ursprünglichen Quellnamen und BGA-IDs bleiben dokumentiert. BGA-Serververhalten ist durch die statischen Definitionen nicht vollständig geprüft.', '']
    for k,url in SOURCES.items():lines.append(f'- [{k}]({url})')
    path.write_text('\n'.join(lines)+'\n')


def validate_comparison(cards, wonders):
    """Check completeness and independent deck/chain invariants before export."""
    if len(cards)!=80 or len(wonders)!=14:
        raise ValueError('Incomplete base-game inventory')
    if len({r['key'] for r in cards+wonders})!=94:
        raise ValueError('Duplicate comparison identity')
    for side in ('old','new'):
        names={r[side]['name'] for r in cards if r[side] is not None}
        for r in cards:
            state=r[side]
            if state is not None and not set(state['chain_from']).issubset(names):
                raise ValueError(f'Missing chain prerequisite: {r["key"]}')
        for players in range(3,8):
            for age in (1,2,3):
                count=sum(sum(mark<=players for mark in r[side]['copies']) for r in cards
                          if r[side] is not None and r['age']==age and r['color']!='purple')
                count += players+2 if age==3 else 0
                if count != 7*players:
                    raise ValueError(f'Wrong {side} deck size, age {age}, players {players}: {count}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'python/runs/edition-comparison')
    parser.add_argument('--new-images',type=Path,help='Root containing cards/<key>.png and wonders/<id_name>.png')
    parser.add_argument('--bga-reference',type=Path,default=ROOT/'references/bga-sevenwonders',help='Local second-edition BGA snapshot (definitions and sprite sheets)')
    args=parser.parse_args()
    catalog=json.loads((DOC/'catalog_2020.json').read_text())
    cards=read_cards(catalog);wonders=read_wonders(catalog);rows=cards+wonders
    conflicts,bga_manifest=apply_bga_reference(cards,wonders,args.bga_reference)
    validate_comparison(cards,wonders)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    inputs=['references/cards.csv','00_marten/final_cards.csv','main.cpp','src/player.cpp','src/wonder.cpp','python/src/sevenwonders_rl/data/card_db.json','python/src/sevenwonders_rl/data/wonder_db.json','docs/reference/edition-comparison/catalog_2020.json','python/scripts/build_edition_comparison.py','python/scripts/edition_comparison/report.html']
    summary=dict(card_rows=len(cards),old_card_types=len({r['old_name'] for r in cards if r['old_name']}),
                 new_card_types=len({r['name'] for r in cards}),wonder_sides=len(wonders),
                 changed_card_rows=sum(bool(r['edition_changes']) for r in cards),
                 engine_deviation_rows=sum(bool(r['engine_changes']) for r in cards),
                 changed_wonder_sides=sum(bool(r['edition_changes']) for r in wonders))
    bga_hashes={name:hashlib.sha256((args.bga_reference/name).read_bytes()).hexdigest() for name in ['card_types.json','wonders.json','manifest.json']}
    data=dict(checked_on=catalog['checked_on'],scope=catalog['edition'],summary=summary,sources=SOURCES,bga_input_sha256=bga_hashes,bga_asset_manifest=bga_manifest,source_discrepancies=conflicts,
              input_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in inputs},cards=cards,wonders=wonders)
    (DOC/'comparison.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    (DOC/'source_discrepancies.json').write_text(json.dumps(conflicts,indent=2,ensure_ascii=False)+'\n')
    write_markdown(cards,wonders,DOC/'COMPARISON.md')
    manifest=[dict(key=r['key'],name=r['name'],kind=r['kind'],old_image=r['old_image'],new_image=r['new_image'],bga_id=r['bga_id'],bga_sprite=r['new_sprite']) for r in rows]
    (DOC/'images_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    (output/'comparison.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    entries='\n'.join(html_entry(r,args.new_images) for r in rows)
    template=(Path(__file__).parent/'edition_comparison/report.html').read_text()
    report=template.replace('<!-- ENTRIES -->',entries).replace('__SUMMARY__',json.dumps(summary,ensure_ascii=False))
    defs='<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>'
    for ident,fn,width,height in [('bga-cards','cards_v2.jpg',2550,3144),('bga-boards','boards_v2.jpg',450,3122)]:
        defs+=f'<image id="{ident}" width="{width}" height="{height}" href="{data_url(args.bga_reference/"img"/fn)}"/>'
    report=report.replace('<!-- BGA_SPRITES -->',defs+'</defs></svg>')
    report=report.replace('__COUNT_CARDS__',str(len(cards))).replace('__COUNT_WONDERS__',str(len(wonders)))
    (output/'index.html').write_text(report)
    write_csv(cards,output/'cards.csv')
    write_wonder_csv(wonders,output/'wonders.csv')
    print(json.dumps({**summary,'html':str(output/'index.html')},ensure_ascii=False,indent=2))


def write_csv(rows,path):
    fields=['key','name','age','color','engine_id','edition_changes','engine_changes','old_cost','engine_cost','new_cost','old_effects','engine_effects','new_effects','old_chain_from','engine_chain_from','new_chain_from','old_chain_to','engine_chain_to','new_chain_to','old_copies','engine_copies','new_copies','new_cost_verification']
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            line={k:r[k] for k in fields[:5]};line.update(edition_changes=', '.join(r['edition_changes']),engine_changes=', '.join(r['engine_changes']))
            for side in ('old','engine','new'):
                for key in ('cost','effects','chain_from','chain_to','copies'):line[side+'_'+key]=value_label(key,r[side],r)
            line['new_cost_verification']=r['verification']['cost'];writer.writerow(line)


def write_wonder_csv(rows,path):
    fields=['key','engine_id','side','stage','cost','effect','starting_resource','source']
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            for side in ('old','engine','new'):
                for i,s in enumerate(r[side]['stages']):writer.writerow(dict(key=r['key'],engine_id=r['engine_id'],side=side,stage=i+1,cost=label_cost(s['cost']),effect=stage_effect(s),starting_resource=r[side]['starting_resource'],source=r['source_image'] if side=='new' else r['old_image']))


if __name__=='__main__':main()
