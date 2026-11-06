"""Public, seat-relative protocol-3 features from one versioned BGA dataset.

Every candidate represents an actual offered option, including physical discard
copies and free modes. No reconstructed legality or opponent private hands.
"""
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import numpy as np

RESOURCES=('wood','ore','clay','stone','loom','glass','papyrus','coins')
COLORS=('brown','grey','blue','yellow','red','green','purple')
SCIENCE=('gear','compass','tablet')
COUNTS=(*COLORS,'brown_grey_purple','wonder_stages')
KINDS=('build_structure','build_wonder','discard','build_discard_free')
FREE=('chain','olympia_first_color','olympia_first_age_card','olympia_last_age_card','discard_grant')
N_OPTIONS=128
PUBLIC_DIM=24
CONTEXT_DIM=8
FEATURE_NAMES=(
    tuple('kind_'+s for s in KINDS)+tuple('card_color_'+s for s in COLORS)
    +tuple('card_cost_'+s for s in RESOURCES)+tuple('card_production_'+s for s in RESOURCES[:7])
    +tuple('card_choice_'+s for s in RESOURCES[:7])+tuple('card_science_'+s for s in SCIENCE)
    +('card_joker','card_vp','card_coins','card_shields','card_type_id','card_age')
    +('payment_bank','payment_left','payment_right')+tuple('free_'+s for s in FREE)
    +tuple('effect_production_'+s for s in RESOURCES[:7])+tuple('effect_choice_'+s for s in RESOURCES[:7])
    +tuple('effect_science_'+s for s in SCIENCE)+('effect_joker','effect_vp','effect_coins','effect_shields')
    +tuple('effect_count_'+s for s in COUNTS)+('effect_vp_per','effect_coins_per')
    +tuple('effect_target_'+s for s in ('self','neighbors','self_and_neighbors'))
    +('effect_wonder_complete','effect_tradable','discount_raw_left','discount_raw_right','discount_manufactured_both')
    +('science_gain','next_stage','ability_first_color_free','ability_first_age_card_free',
       'ability_last_age_card_free','ability_play_seventh_card','ability_discard_build_for_free'))
K=len(FEATURE_NAMES)
INDEX={s:i for i,s in enumerate(FEATURE_NAMES)}

@lru_cache(maxsize=1)
def dataset():
    raw=[files('sevenwonders_rl.data').joinpath(f'bga_{k}-v1.json').read_bytes() for k in ('cards','wonders')]
    cards,wonders=map(json.loads,raw)
    if cards['format_version']!=1 or wonders['format_version']!=1 or cards['source']!=wonders['source']:
        raise ValueError('Incompatible BGA datasets')
    return {'cards':{(c['engine_id'],c['age']):c for c in cards['cards']},
            'wonders':{w['engine_id']:w for w in wonders['wonders']},
            'source_hash':cards['source']['sha256'],
            'dataset_hash':hashlib.sha256(b'\0'.join(raw)).hexdigest()}


def optimal_science(symbols,jokers):
    g,c,t=map(int,symbols)
    return max((g+x)**2+(c+y)**2+(t+jokers-x-y)**2
               +7*min(g+x,c+y,t+jokers-x-y)
               for x in range(jokers+1) for y in range(jokers-x+1))


def science_inventory(player):
    symbols=np.zeros(3,dtype=np.int32);jokers=0;data=dataset()
    for card in player['built_cards']:
        symbol=data['cards'][(card['type_id'],card['age'])]['effects'].get('science')
        if symbol in SCIENCE:symbols[SCIENCE.index(symbol)]+=1
        elif symbol=='joker':jokers+=1
    for stage in data['wonders'][player['wonder_id']]['stages'][:player['wonder_stage']]:
        jokers+=sum(e['type']=='add_science' for e in stage['effects'])
    return symbols,jokers


def stage_effects(player):
    stage=dataset()['wonders'][player['wonder_id']]['stages'][player['wonder_stage']]
    effect={'vp':stage['vp']}
    for item in stage['effects']:
        kind=item['type']
        if kind=='add_coins':effect['coins']=item['amount']
        elif kind=='add_shields':effect['shields']=item['amount']
        elif kind=='add_science':effect['science']='joker'
        elif kind=='choose_raw_nontradable':effect.update(choice=list(RESOURCES[:4]),tradable=False)
        elif kind=='choose_manuf_nontradable':effect.update(choice=list(RESOURCES[4:7]),tradable=False)
        else:effect[kind]=True
    return effect


def observation(state,seat,options):
    data=dataset();own=state['players'][str(seat)]
    table=np.zeros((3,PUBLIC_DIM),dtype=np.float32)
    for role,s in enumerate((seat,(seat+2)%3,(seat+1)%3)):
        p=state['players'][str(s)];symbols,jokers=science_inventory(p)
        colors=[0]*7
        for card in p['built_cards']:colors[COLORS.index(data['cards'][(card['type_id'],card['age'])]['color'])]+=1
        table[role]=[*(p['resources'][r]/4 for r in RESOURCES[:7]),p['resources']['coins']/20,
            p['resources']['shields']/6,*(symbols/4),jokers/2,p['wonder_stage']/4,p['wonder_id']/13,
            *(n/6 for n in colors),p['points']['total']/100,float(p['wonder_complete'])]
    decision=next((d for d in state['decisions'] if d['actor']==seat),None)
    kind=decision['kind'] if decision else None
    context=np.array([min(state['game']['age'],3)/3,(state['game']['round_in_age'] or 0)/7,
        float(kind=='regular'),float(kind=='seventh'),float(kind=='discard'),state['discard_count']/63,
        len(own['hand_cards'])/7,len(state['waiting_for'])/3],dtype=np.float32)
    feats=np.zeros((N_OPTIONS,K),dtype=np.float32);mask=np.zeros(N_OPTIONS,dtype=np.int8)
    symbols,jokers=science_inventory(own);before=optimal_science(symbols,jokers)
    if len(options)>N_OPTIONS:raise ValueError('Offered candidate capacity exceeded')
    for index,option in enumerate(options):
        row=feats[index];mask[index]=1
        def put(name,value=1):row[INDEX[name]]=value
        c=option['card'];card=data['cards'][(c['type_id'],c['age'])];raw=card['effects']
        put('kind_'+option['kind']);put('card_color_'+card['color'])
        put('card_type_id',c['type_id']/78);put('card_age',c['age']/3)
        for r,n in card['cost'].items():put('card_cost_'+r,n/4)
        for r,n in raw.get('production',{}).items():put('card_production_'+r,n/2)
        for r in raw.get('choice',[]):put('card_choice_'+r)
        if raw.get('science') in SCIENCE:put('card_science_'+raw['science'])
        elif raw.get('science')=='joker':put('card_joker')
        for key,scale in (('vp',8),('coins',9),('shields',3)):put('card_'+key,raw.get(key,0)/scale)
        payment=option['payment'];put('payment_bank',payment['bank']/10)
        put('payment_left',payment['to_seats'][str((seat+2)%3)]/10)
        put('payment_right',payment['to_seats'][str((seat+1)%3)]/10)
        if option['free_source'] in FREE:put('free_'+option['free_source'])
        effects=stage_effects(own) if option['kind']=='build_wonder' else {'coins':3} if option['kind']=='discard' else raw
        for r,n in effects.get('production',{}).items():put('effect_production_'+r,n/2)
        for r in effects.get('choice',[]):put('effect_choice_'+r)
        symbol=effects.get('science');after=symbols.copy();extra=jokers
        if symbol in SCIENCE:put('effect_science_'+symbol);after[SCIENCE.index(symbol)]+=1
        elif symbol=='joker':put('effect_joker');extra+=1
        put('science_gain',(optimal_science(after,extra)-before)/20)
        for key,scale in (('vp',8),('coins',9),('shields',3),('vp_per',3),('coins_per',3)):
            put('effect_'+key,effects.get(key,0)/scale)
        if effects.get('count') in COUNTS:put('effect_count_'+effects['count'])
        if 'target' in effects:put('effect_target_'+effects['target'])
        put('effect_wonder_complete',effects.get('condition')=='wonder_complete')
        put('effect_tradable',effects.get('tradable',False))
        if 'discount' in effects:put('discount_'+effects['discount'])
        for ability in ('first_color_free','first_age_card_free','last_age_card_free','play_seventh_card','discard_build_for_free'):
            put('ability_'+ability,effects.get(ability,False))
        if option['kind']=='build_wonder':put('next_stage',(own['wonder_stage']+1)/4)
    return {'public_table':table,'context':context,'action_feats':feats,'action_mask':mask}


def feature_contract(ruleset):
    return {'format':'decision-ppo-v1','feature_semantics_version':1,'protocol_version':3,'ruleset':ruleset,
        'source_hash':dataset()['source_hash'],'dataset_hash':dataset()['dataset_hash'],
        'actions':'offered-physical-options-v1','n_options':N_OPTIONS,'feature_names':list(FEATURE_NAMES),
        'public_shape':[3,PUBLIC_DIM],'context_dim':CONTEXT_DIM,
        'visibility':'own-hand-and-public-table','time_unit':'own-offered-decision'}
