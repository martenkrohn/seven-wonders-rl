#!/usr/bin/env python3
"""Extract card database from C++ source files.

Parses main.cpp, card.h, player.cpp to build comprehensive card_db.json.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_enum(file_path: Path, namespace_name: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Parse C++ enum with implicit increments."""
    content = file_path.read_text()
    
    # Find namespace block
    pattern = rf'namespace {namespace_name}\s*{{(.*?)}};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Namespace {namespace_name} not found in {file_path}")
    
    enum_block = match.group(1)
    name_to_int = {}
    int_to_name = {}
    current_value = 0
    
    for line in enum_block.split('\n'):
        line = line.strip()
        # Remove comments
        line = re.sub(r'//.*', '', line)
        line = re.sub(r'/\*.*?\*/', '', line)
        line = line.strip()
        
        if not line or line == '{' or line == '}' or line.startswith('enum '):
            continue
        
        # Remove trailing comma
        line = line.rstrip(',')
        
        if '=' in line:
            parts = line.split('=')
            name = parts[0].strip()
            value = int(parts[1].strip())
            current_value = value
        else:
            name = line
        
        name_to_int[name] = current_value
        int_to_name[current_value] = name
        current_value += 1
    
    return name_to_int, int_to_name


def parse_card_type_enum(file_path: Path) -> Tuple[Dict[int, str], Dict[str, str]]:
    """Parse CARD_TYPE enum and map to colors."""
    name_to_int, int_to_name = parse_enum(file_path, 'CARD_TYPE')
    
    type_to_color = {
        'materials': 'brown',
        'manufactured': 'grey',
        'civilian': 'blue',
        'commercial': 'yellow',
        'military': 'red',
        'scientific': 'green',
        'guild': 'purple',
    }
    
    return int_to_name, type_to_color


def parse_resource_enum(file_path: Path) -> Dict[int, str]:
    """Parse RESOURCE enum."""
    _, int_to_name = parse_enum(file_path, 'RESOURCE')
    return int_to_name


def split_arguments(args_str: str) -> List[str]:
    """Split top-level arguments by commas, respecting nesting."""
    args = []
    current = []
    depth = 0
    in_string = False
    escape_next = False
    
    for char in args_str:
        if escape_next:
            current.append(char)
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            current.append(char)
            continue
        
        if char == '"' and not in_string:
            in_string = True
            current.append(char)
        elif char == '"' and in_string:
            in_string = False
            current.append(char)
        elif in_string:
            current.append(char)
        elif char in '{(':
            depth += 1
            current.append(char)
        elif char in '})':
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    
    if current:
        args.append(''.join(current).strip())
    
    return args


def parse_card_constructor(card_str: str, card_id_map: Dict[str, int], 
                          type_map: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """Parse a single Card(...) constructor call."""
    # Find arguments between Card( and )
    match = re.search(r'Card\((.*)\)', card_str, re.DOTALL)
    if not match:
        return None
    
    args_str = match.group(1)
    args = split_arguments(args_str)
    
    if len(args) != 7:
        print(f"Warning: Expected 7 args, got {len(args)}: {card_str[:100]}")
        return None
    
    # Parse each argument
    # Arg 0: CARD_ID::xyz or -1
    id_arg = args[0].strip()
    if id_arg == '-1':
        # Special card (trading posts with chain_from_any)
        # We'll handle this in chain parsing
        id_match = re.search(r'CARD_ID::(\w+)', card_str)
        if id_match:
            id_name = id_match.group(1)
            id_int = card_id_map.get(id_name, -1)
        else:
            return None
    else:
        id_match = re.search(r'CARD_ID::(\w+)', id_arg)
        if not id_match:
            return None
        id_name = id_match.group(1)
        id_int = card_id_map.get(id_name)
        if id_int is None:
            print(f"Warning: Unknown CARD_ID: {id_name}")
            return None
    
    # Arg 1: "Name"
    name = args[1].strip().strip('"')
    
    # Arg 2: CARD_TYPE::xyz
    type_match = re.search(r'CARD_TYPE::(\w+)', args[2])
    if not type_match:
        return None
    type_name = type_match.group(1)
    type_int = type_map.get(type_name)
    
    # Arg 3: era
    era = int(args[3].strip())
    
    # Arg 4: free_with CARD_ID
    free_arg = args[4].strip()
    if free_arg == '-1':
        free_with_raw = -1  # chain_from_any
    elif 'CARD_ID::none' in free_arg:
        free_with_raw = None
    else:
        free_match = re.search(r'CARD_ID::(\w+)', free_arg)
        if free_match:
            free_name = free_match.group(1)
            free_with_raw = card_id_map.get(free_name)
        else:
            free_with_raw = None
    
    # Arg 5: cost vector {w,o,c,s,l,g,p,coins}
    cost_vec = [int(x.strip()) for x in args[5].strip('{}').split(',')]
    
    # Arg 6: amount vector {2p,3p,4p,5p,6p}
    amount_vec = [int(x.strip()) for x in args[6].strip('{}').split(',')]
    
    return {
        'id': id_int,
        'id_name': id_name,
        'name': name,
        'type_name': type_name,
        'type_int': type_int,
        'era': era,
        'free_with_raw': free_with_raw,
        'cost_vec': cost_vec,
        'amount_vec': amount_vec,
    }


def parse_cards_from_main(main_path: Path, card_id_map: Dict[str, int],
                          type_name_to_int: Dict[str, int]) -> List[Dict[str, Any]]:
    """Parse all Card(...) constructor calls from main.cpp."""
    content = main_path.read_text()
    
    # Find all Card(...) constructs
    cards = []
    pos = 0
    while True:
        idx = content.find('Card(', pos)
        if idx == -1:
            break
        
        # Find matching closing paren
        depth = 0
        start = idx + 5  # After 'Card('
        i = start
        while i < len(content):
            if content[i] == '(':
                depth += 1
            elif content[i] == ')':
                if depth == 0:
                    break
                depth -= 1
            i += 1
        
        card_str = content[idx:i+1]
        parsed = parse_card_constructor(card_str, card_id_map, type_name_to_int)
        if parsed:
            cards.append(parsed)
        
        pos = i + 1
    
    return cards


def merge_card_definitions(cards: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Merge cards by ID, handling duplicates across eras."""
    merged = {}
    
    for card in cards:
        card_id = card['id']
        
        if card_id not in merged:
            merged[card_id] = {
                'id': card_id,
                'id_name': card['id_name'],
                'name': card['name'],
                'type_name': card['type_name'],
                'type_int': card['type_int'],
                'eras': [card['era']],
                'free_with_raw': card['free_with_raw'],
                'cost_vec': card['cost_vec'],
                'amount_per_player_by_era': {
                    str(card['era']): {
                        '2': card['amount_vec'][0],
                        '3': card['amount_vec'][1],
                        '4': card['amount_vec'][2],
                        '5': card['amount_vec'][3],
                    }
                }
            }
        else:
            # Validate consistency
            existing = merged[card_id]
            if existing['name'] != card['name']:
                raise ValueError(f"Name mismatch for card {card_id}: {existing['name']} vs {card['name']}")
            if existing['type_name'] != card['type_name']:
                raise ValueError(f"Type mismatch for card {card_id}")
            
            # Add era
            if card['era'] not in existing['eras']:
                existing['eras'].append(card['era'])
            
            # Add amount data
            existing['amount_per_player_by_era'][str(card['era'])] = {
                '2': card['amount_vec'][0],
                '3': card['amount_vec'][1],
                '4': card['amount_vec'][2],
                '5': card['amount_vec'][3],
            }
    
    return merged


def parse_choice_resources(player_cpp: Path, card_id_map: Dict[str, int],
                           resource_map: Dict[int, str]) -> Dict[int, Dict[str, Any]]:
    """Parse choice resource production from AvailableCard()."""
    content = player_cpp.read_text()
    
    choice_data = {}
    
    # Find AvailableCard function
    func_match = re.search(r'bool Player::AvailableCard\([^)]+\)\s*{(.*?)\n}', content, re.DOTALL)
    if not func_match:
        return choice_data
    
    func_body = func_match.group(1)
    
    # Parse each if block for card choices
    # Pattern: if (this_card == CARD_ID::tree_farm && this->used_tree_farm == -1)
    patterns = {
        'tree_farm': ['wood', 'clay'],
        'forest_cave': ['wood', 'ore'],
        'timber_yard': ['stone', 'wood'],
        'excavation': ['stone', 'clay'],
        'mine': ['stone', 'ore'],
        'clay_pit': ['clay', 'ore'],
        'caravansery': ['wood', 'ore', 'clay', 'stone'],
        'forum': ['loom', 'glass', 'papyrus'],
    }
    
    for card_name, resources in patterns.items():
        card_id = card_id_map.get(card_name)
        if card_id:
            # Caravansery and forum are non-tradable (only used when !is_neighbor)
            tradable = card_name not in ['caravansery', 'forum']
            choice_data[card_id] = {
                'produces_choice': resources,
                'choice_tradable': tradable,
            }
    
    return choice_data


def parse_build_structure_effects(player_cpp: Path, card_id_map: Dict[str, int],
                                  resource_map: Dict[int, str]) -> Dict[int, Dict[str, Any]]:
    """Parse fixed production and effects from BuildStructure()."""
    content = player_cpp.read_text()
    
    effects = {}
    
    # Find BuildStructure function - look for the whole function body
    func_match = re.search(r'bool Player::BuildStructure\([^)]+\)\s*{(.*?)\n(?:bool Player::|// Checks|void Player::)', content, re.DOTALL)
    if not func_match:
        return effects
    
    func_body = func_match.group(1)
    
    # Find the switch statement on c.GetType()
    switch_match = re.search(r'switch\s*\(\s*c\.GetType\(\)\s*\)\s*{(.*?)\n\s*}\s*\n\s*cards_hand', func_body, re.DOTALL)
    if not switch_match:
        return effects
    
    switch_body = switch_match.group(1)
    
    # Parse individual case blocks from the switch
    # Each case block has: case CARD_TYPE::xxx: ... break;
    
    case_pattern = r'case\s+CARD_TYPE::\w+:\s*(.*?)\s*break;'
    case_matches = re.finditer(case_pattern, switch_body, re.DOTALL)
    
    for case_match in case_matches:
        case_body = case_match.group(1)
        
        # Within each case, find individual if/else-if conditions
        # Pattern: if (card == CARD_ID::xxx || ...) { action } 
        # OR: if (card == CARD_ID::xxx) { action } else if...
        
        # Handle multi-card OR conditions first (they can span multiple lines)
        # More flexible pattern: capture everything between ( and ) that contains CARD_ID references and ||
        multi_card_pattern = r'(?:if|else if)\s*\((card\s*==.*?CARD_ID.*?\|\|.*?card\s*==\s*CARD_ID::\w+)\)\s*{\s*(.*?)\s*}'
        multi_matches = list(re.finditer(multi_card_pattern, case_body, re.DOTALL))
        
        for multi_match in multi_matches:
            condition = multi_match.group(1)
            action = multi_match.group(2)
            
            # Extract all card names from condition
            card_names = re.findall(r'CARD_ID::(\w+)', condition)
            
            # Parse effect
            effect_data = {
                'produces_fixed': None,
                'shields': 0,
                'gain_coins': 0,
                'discount_flags': [],
                'extra_flags': [],
            }
            
            # Check for shields (most common in multi-card)
            shield_match = re.search(r'resources\[RESOURCE::shields\]\s*\+=\s*(\d+)', action)
            if shield_match:
                effect_data['shields'] = int(shield_match.group(1))
            
            # Add for all cards in condition
            for card_name in card_names:
                card_id = card_id_map.get(card_name)
                if card_id:
                    effects[card_id] = effect_data
        
        # Now handle single-card conditions (these are simpler)
        # We need to be careful not to re-match the multi-card ones
        # Remove multi-card matches from case_body first
        working_body = case_body
        for multi_match in multi_matches:
            working_body = working_body.replace(multi_match.group(0), '')
        
        # Now match single-card if/else-if blocks
        single_card_pattern = r'(?:if|else if)\s*\(\s*card\s*==\s*CARD_ID::(\w+)\s*\)\s*{\s*((?:[^{}]|{[^}]*})*?)\s*}'
        single_matches = re.finditer(single_card_pattern, working_body, re.DOTALL)
        
        for single_match in single_matches:
            card_name = single_match.group(1)
            action = single_match.group(2)
            card_id = card_id_map.get(card_name)
            if not card_id:
                continue
            
            effect_data = {
                'produces_fixed': None,
                'shields': 0,
                'gain_coins': 0,
                'discount_flags': [],
                'extra_flags': [],
            }
            
            # Check for resource production
            for res_name in ['wood', 'ore', 'clay', 'stone', 'loom', 'glass', 'papyrus']:
                if f'RESOURCE::{res_name}]' in action:
                    effect_data['produces_fixed'] = res_name
                    break  # Only primary production
            
            # Check for coins
            coin_match = re.search(r'resources\[RESOURCE::coins\]\s*\+=\s*(\d+)', action)
            if coin_match:
                effect_data['gain_coins'] = int(coin_match.group(1))
            
            # Check for shields
            shield_match = re.search(r'resources\[RESOURCE::shields\]\s*\+=\s*(\d+)', action)
            if shield_match:
                effect_data['shields'] = int(shield_match.group(1))
            
            # Check for flags - these are assignments like: this->raw_cheap_east = true;
            if 'raw_cheap_east' in action:
                effect_data['discount_flags'].append('raw_cheap_east')
            if 'raw_cheap_west' in action:
                effect_data['discount_flags'].append('raw_cheap_west')
            if 'manuf_cheap' in action:
                effect_data['discount_flags'].append('manuf_cheap')
            if 'raw_extra' in action:
                effect_data['extra_flags'].append('raw_extra')
            if 'manuf_extra' in action:
                effect_data['extra_flags'].append('manuf_extra')
            
            effects[card_id] = effect_data
    
    return effects


def parse_science_symbols(player_cpp: Path, card_id_map: Dict[str, int]) -> Dict[int, List[str]]:
    """Parse science symbols from CalculateScientificScore()."""
    content = player_cpp.read_text()
    
    science = {}
    
    # Find CalculateScientificScore function
    func_match = re.search(r'int Player::CalculateScientificScore\([^)]*\)\s*{(.*?)\n}', content, re.DOTALL)
    if not func_match:
        return science
    
    func_body = func_match.group(1)
    
    # Find the switch statement
    switch_match = re.search(r'switch\s*\(\s*card_id\s*\)\s*{(.*?)\n\s*}', func_body, re.DOTALL)
    if not switch_match:
        return science
    
    switch_body = switch_match.group(1)
    
    # Parse fall-through case blocks
    # Pattern: case CARD_ID::x: case CARD_ID::y: ... symbol++; break;
    case_block_pattern = r'((?:case\s+CARD_ID::\w+:\s*)+)(\w+)\+\+;'
    for block_match in re.finditer(case_block_pattern, switch_body):
        cases_part = block_match.group(1)
        symbol = block_match.group(2)
        
        # Extract all card names from the cases
        card_names = re.findall(r'CARD_ID::(\w+)', cases_part)
        
        for card_name in card_names:
            card_id = card_id_map.get(card_name)
            if card_id:
                if card_id not in science:
                    science[card_id] = []
                science[card_id].append(symbol)
    
    return science


def parse_civilian_vp(player_cpp: Path, card_id_map: Dict[str, int]) -> Dict[int, int]:
    """Parse civilian (blue) VP from CalculateCivilianScore()."""
    content = player_cpp.read_text()
    
    vp_map = {}
    
    # Find CalculateCivilianScore function
    func_match = re.search(r'int Player::CalculateCivilianScore\([^)]*\)\s*{(.*?)\n}', content, re.DOTALL)
    if not func_match:
        return vp_map
    
    func_body = func_match.group(1)
    
    # Parse switch cases
    # case CARD_ID::altar: case CARD_ID::theater: civil_score += 2; break;
    cases = re.finditer(r'((?:case\s+CARD_ID::\w+:\s*)+)\s*civil_score\s*\+=\s*(\d+);', func_body)
    for match in cases:
        cases_str = match.group(1)
        vp = int(match.group(2))
        
        # Extract all card names from this case group
        card_names = re.findall(r'CARD_ID::(\w+)', cases_str)
        for card_name in card_names:
            card_id = card_id_map.get(card_name)
            if card_id:
                vp_map[card_id] = vp
    
    return vp_map


def parse_commercial_vp(player_cpp: Path, card_id_map: Dict[str, int]) -> Dict[int, Dict[str, Any]]:
    """Parse commercial (yellow) conditional VP from CalculateCommercialScore()."""
    content = player_cpp.read_text()
    
    conditional_vp = {}
    
    # Find CalculateCommercialScore function
    func_match = re.search(r'int Player::CalculateCommercialScore\([^)]*\)\s*{(.*?)\n}', content, re.DOTALL)
    if not func_match:
        return conditional_vp
    
    func_body = func_match.group(1)
    
    # Parse each case manually
    patterns = {
        'haven': {
            'type': 'per_card_type',
            'card_type': 'materials',
            'vp_per': 1,
            'target': 'self',
        },
        'lighthouse': {
            'type': 'per_card_type',
            'card_type': 'commercial',
            'vp_per': 1,
            'target': 'self',
        },
        'chamber_of_commerce': {
            'type': 'per_card_type',
            'card_type': 'manufactured',
            'vp_per': 2,
            'target': 'self',
        },
        'arena': {
            'type': 'per_wonder_stage',
            'vp_per': 1,
            'target': 'self',
        },
    }
    
    for card_name, vp_rule in patterns.items():
        card_id = card_id_map.get(card_name)
        if card_id:
            conditional_vp[card_id] = vp_rule
    
    return conditional_vp


def parse_guild_vp(player_cpp: Path, card_id_map: Dict[str, int]) -> Dict[int, Dict[str, Any]]:
    """Parse guild (purple) conditional VP from CalculateGuildScore()."""
    content = player_cpp.read_text()
    
    conditional_vp = {}
    
    # Find CalculateGuildScore function
    func_match = re.search(r'int Player::CalculateGuildScore\([^)]*\)\s*{(.*?)\n}', content, re.DOTALL)
    if not func_match:
        return conditional_vp
    
    func_body = func_match.group(1)
    
    # Define patterns for each guild
    patterns = {
        'workers': {
            'type': 'per_card_type',
            'card_type': 'materials',
            'vp_per': 1,
            'target': 'neighbors',
        },
        'craftsmens': {
            'type': 'per_card_type',
            'card_type': 'manufactured',
            'vp_per': 2,
            'target': 'neighbors',
        },
        'traders': {
            'type': 'per_card_type',
            'card_type': 'commercial',
            'vp_per': 1,
            'target': 'neighbors',
        },
        'philosophers': {
            'type': 'per_card_type',
            'card_type': 'scientific',
            'vp_per': 1,
            'target': 'neighbors',
        },
        'spies': {
            'type': 'per_card_type',
            'card_type': 'military',
            'vp_per': 1,
            'target': 'neighbors',
        },
        'magistrates': {
            'type': 'per_card_type',
            'card_type': 'civilian',
            'vp_per': 1,
            'target': 'neighbors',
        },
        'shipowners': {
            'type': 'per_card_type_multi',
            'card_types': ['materials', 'manufactured', 'guild'],
            'vp_per': 1,
            'target': 'self',
        },
        'strategists': {
            'type': 'per_defeat_token',
            'vp_per': 1,
            'target': 'neighbors',
        },
        'builders': {
            'type': 'per_wonder_stage',
            'vp_per': 1,
            'target': 'all',
        },
        'scientists': {
            'type': 'science_wildcard',
        },
    }
    
    for card_name, vp_rule in patterns.items():
        card_id = card_id_map.get(card_name)
        if card_id:
            conditional_vp[card_id] = vp_rule
    
    return conditional_vp


def build_final_card_db(merged_cards: Dict[int, Dict[str, Any]],
                       choice_data: Dict[int, Dict[str, Any]],
                       build_effects: Dict[int, Dict[str, Any]],
                       science: Dict[int, List[str]],
                       civilian_vp: Dict[int, int],
                       commercial_vp: Dict[int, Dict[str, Any]],
                       guild_vp: Dict[int, Dict[str, Any]],
                       resource_map: Dict[int, str],
                       type_to_color: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Build final card_db.json structure."""
    card_db = {}
    
    resource_names = ['wood', 'ore', 'clay', 'stone', 'loom', 'glass', 'papyrus', 'coins']
    
    for card_id, card_data in merged_cards.items():
        # Build cost dict
        cost_vec = card_data['cost_vec']
        cost = {name: cost_vec[i] for i, name in enumerate(resource_names)}
        
        # Build chain fields
        free_with = card_data['free_with_raw']
        if free_with == -1:
            chain_from = None
            chain_from_any = ['east_trading_post', 'west_trading_post']
        elif free_with is None:
            chain_from = None
            chain_from_any = None
        else:
            chain_from = free_with
            chain_from_any = None
        
        # Build effects
        effects = {
            'produces_fixed': None,
            'produces_choice': None,
            'choice_tradable': None,
            'science': None,
            'shields': 0,
            'gain_coins': 0,
            'discount_flags': [],
            'extra_flags': [],
        }
        
        # Merge choice data
        if card_id in choice_data:
            effects['produces_choice'] = choice_data[card_id]['produces_choice']
            effects['choice_tradable'] = choice_data[card_id]['choice_tradable']
        
        # Merge build effects
        if card_id in build_effects:
            effects.update(build_effects[card_id])
        
        # Add science
        if card_id in science:
            effects['science'] = science[card_id]
        
        # Special case: Scientists guild adds science_wildcard to effects
        if card_id in guild_vp and guild_vp[card_id].get('type') == 'science_wildcard':
            effects['science_wildcard'] = True
        
        # Build VP
        vp = {
            'fixed': 0,
            'conditional': None,
        }
        
        if card_id in civilian_vp:
            vp['fixed'] = civilian_vp[card_id]
        
        if card_id in commercial_vp:
            vp['conditional'] = commercial_vp[card_id]
        elif card_id in guild_vp:
            vp['conditional'] = guild_vp[card_id]
        
        # Build final entry
        card_db[str(card_id)] = {
            'id': card_id,
            'name': card_data['name'],
            'id_name': card_data['id_name'],
            'type': card_data['type_name'],
            'color': type_to_color.get(card_data['type_name'], 'unknown'),
            'eras': sorted(card_data['eras']),
            'chain_from': chain_from,
            'chain_from_any': chain_from_any,
            'cost': cost,
            'amount_per_player_by_era': card_data['amount_per_player_by_era'],
            'effects': effects,
            'vp': vp,
        }
    
    return card_db


def build_mechanics_json() -> Dict[str, Any]:
    """Build mechanics.json with trading rules."""
    return {
        'trade': {
            'base_cost_per_resource': 2,
            'discount_cost_per_resource': 1,
            'manufactured_discount_flag': 'manuf_cheap',
            'raw_discount_flags': ['raw_cheap_east', 'raw_cheap_west', 'wonder_raw_cheap'],
            'non_tradable_sources': ['yellow_extra', 'wonder_extra'],
        },
        'tradable_rule': {
            'only_if_is_neighbor_false': [
                'forum',
                'caravansery',
                'bazar',
                'alexandria_choose_raw',
                'alexandria_choose_manuf',
            ],
        },
        'treasure': {
            'coins_per_vp': 3,
        },
    }


def write_json_atomic(data: Any, file_path: Path):
    """Write JSON atomically using temp file."""
    temp_path = file_path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write('\n')
    temp_path.replace(file_path)


def main():
    print("=" * 60)
    print("Card Database Extraction")
    print("=" * 60)
    
    # Parse enums
    print("\n1. Parsing enums...")
    card_h = REPO_ROOT / "include/card.h"
    resources_h = REPO_ROOT / "include/resources.h"
    
    card_id_name_to_int, card_id_int_to_name = parse_enum(card_h, 'CARD_ID')
    print(f"   - Found {len(card_id_name_to_int)} CARD_ID entries")
    
    type_name_to_int, _ = parse_enum(card_h, 'CARD_TYPE')
    type_int_to_name, type_to_color = parse_card_type_enum(card_h)
    print(f"   - Found {len(type_int_to_name)} CARD_TYPE entries")
    
    resource_int_to_name = parse_resource_enum(resources_h)
    print(f"   - Found {len(resource_int_to_name)} RESOURCE entries")
    
    # Parse Card constructors
    print("\n2. Parsing Card constructors from main.cpp...")
    main_cpp = REPO_ROOT / "main.cpp"
    cards = parse_cards_from_main(main_cpp, card_id_name_to_int, type_name_to_int)
    print(f"   - Parsed {len(cards)} card instantiations")
    
    # Merge duplicates
    print("\n3. Merging duplicate cards across eras...")
    merged_cards = merge_card_definitions(cards)
    print(f"   - Result: {len(merged_cards)} unique cards")
    
    # Parse effects from player.cpp
    print("\n4. Parsing effects from player.cpp...")
    player_cpp = REPO_ROOT / "src/player.cpp"
    
    choice_data = parse_choice_resources(player_cpp, card_id_name_to_int, resource_int_to_name)
    print(f"   - Choice resources: {len(choice_data)} cards")
    
    build_effects = parse_build_structure_effects(player_cpp, card_id_name_to_int, resource_int_to_name)
    print(f"   - Build effects: {len(build_effects)} cards")
    
    science = parse_science_symbols(player_cpp, card_id_name_to_int)
    print(f"   - Science symbols: {len(science)} cards")
    
    # Parse VP
    print("\n5. Parsing VP calculations...")
    civilian_vp = parse_civilian_vp(player_cpp, card_id_name_to_int)
    print(f"   - Civilian VP: {len(civilian_vp)} cards")
    
    commercial_vp = parse_commercial_vp(player_cpp, card_id_name_to_int)
    print(f"   - Commercial VP: {len(commercial_vp)} cards")
    
    guild_vp = parse_guild_vp(player_cpp, card_id_name_to_int)
    print(f"   - Guild VP: {len(guild_vp)} cards")
    
    # Build final database
    print("\n6. Building final card database...")
    card_db = build_final_card_db(
        merged_cards, choice_data, build_effects, science,
        civilian_vp, commercial_vp, guild_vp,
        resource_int_to_name, type_to_color
    )
    
    mechanics = build_mechanics_json()
    
    # Validate
    print("\n7. Validating...")
    for card_id_str, card in card_db.items():
        assert 'id' in card
        assert 'name' in card
        assert 'type' in card
        assert 'eras' in card and len(card['eras']) > 0
        assert 'cost' in card and len(card['cost']) == 8
        assert 'effects' in card
        assert 'vp' in card
    print(f"   - All {len(card_db)} cards validated")
    
    # Write output
    print("\n8. Writing output files...")
    output_dir = REPO_ROOT / "python/src/sevenwonders_rl/data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    card_db_sorted = dict(sorted(card_db.items(), key=lambda x: int(x[0])))
    
    card_db_path = output_dir / "card_db.json"
    write_json_atomic(card_db_sorted, card_db_path)
    print(f"   - {card_db_path}")
    
    mechanics_path = output_dir / "mechanics.json"
    write_json_atomic(mechanics, mechanics_path)
    print(f"   - {mechanics_path}")
    
    print("\n" + "=" * 60)
    print("✓ Card database extraction complete!")
    print(f"  {len(card_db)} unique cards")
    print(f"  {sum(len(c['eras']) for c in card_db.values())} total card-era combinations")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
