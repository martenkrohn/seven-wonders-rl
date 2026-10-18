#!/usr/bin/env python3
"""Extract wonder database from C++ source files.

Parses wonder.h and wonder.cpp to build comprehensive wonder_db.json.
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


def _find_block_by_braces(content: str, brace_start: int) -> Tuple[str, int]:
    """Return block content inside braces and index after closing brace."""
    depth = 0
    i = brace_start
    n = len(content)
    while i < n:
        ch = content[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return content[brace_start + 1:i], i + 1
        i += 1
    raise ValueError("Unbalanced braces while parsing block")


def _find_matching_paren(content: str, paren_start: int) -> int:
    """Return index of matching closing parenthesis."""
    depth = 0
    i = paren_start
    n = len(content)
    while i < n:
        ch = content[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("Unbalanced parentheses while parsing")


def parse_wonder_classes(wonder_cpp: Path) -> List[Tuple[str, str, str]]:
    """Extract wonder class names and their constructor + AddStage blocks."""
    content = wonder_cpp.read_text()
    wonders = []

    constructor_pattern = re.compile(r'(\w+)::\1\s*\(\s*\)\s*{')
    for match in constructor_pattern.finditer(content):
        class_name = match.group(1)
        brace_start = content.find('{', match.end() - 1)
        constructor_body, _ = _find_block_by_braces(content, brace_start)

        addstage_pattern = re.compile(rf'bool\s+{class_name}::AddStage\s*\([^)]*\)\s*{{')
        addstage_match = addstage_pattern.search(content)
        if not addstage_match:
            continue
        addstage_brace_start = content.find('{', addstage_match.end() - 1)
        addstage_body, _ = _find_block_by_braces(content, addstage_brace_start)

        wonders.append((class_name, constructor_body, addstage_body))

    return wonders


def parse_wonder_constructor(constructor_body: str, 
                             wonder_id_map: Dict[str, int],
                             resource_map: Dict[int, str],
                             effect_map: Dict[int, str]) -> Dict[str, Any]:
    """Parse wonder constructor to extract basic info."""
    data = {}
    
    # Parse id: this->id = WONDER_ID::gizah_a;
    id_match = re.search(r'this->id\s*=\s*WONDER_ID::(\w+)', constructor_body)
    if id_match:
        id_name = id_match.group(1)
        data['id'] = wonder_id_map.get(id_name)
        data['id_name'] = id_name
    
    # Parse name: this->name = "Gizah A";
    name_match = re.search(r'this->name\s*=\s*"([^"]+)"', constructor_body)
    if name_match:
        data['name'] = name_match.group(1)
    
    # Parse production: this->production = RESOURCE::stone;
    prod_match = re.search(r'this->production\s*=\s*RESOURCE::(\w+)', constructor_body)
    if prod_match:
        res_name = prod_match.group(1)
        # Find resource int from name
        res_int = None
        for idx, name in resource_map.items():
            if name == res_name:
                res_int = idx
                break
        data['starting_resource'] = res_name
    
    # Parse effects vector: this->effects = {EFFECT::vp, EFFECT::choose_raw, ...};
    effects_match = re.search(r'this->effects\s*=\s*{([^}]+)}', constructor_body)
    if effects_match:
        effects_str = effects_match.group(1)
        effect_names = re.findall(r'EFFECT::(\w+)', effects_str)
        data['effects_raw'] = effect_names
    else:
        data['effects_raw'] = []
    
    # Parse initial cost: this->cost[RESOURCE::stone] = 2;
    cost_matches = re.finditer(r'this->cost\[RESOURCE::(\w+)\]\s*=\s*(\d+)', constructor_body)
    initial_cost = {}
    for match in cost_matches:
        resource = match.group(1)
        amount = int(match.group(2))
        initial_cost[resource] = amount
    data['stage_0_cost'] = initial_cost
    
    return data


def normalize_effect_name(effect_name: str) -> str:
    """Convert EFFECT enum name to normalized string."""
    mapping = {
        'vp': None,  # VP handled separately
        'hand_build_for_free': 'hand_build_for_free',
        'discard_build_for_free': 'discard_build_for_free',
        'add_shields': 'add_shields',
        'choose_raw': 'choose_raw_nontradable',
        'add_coins': 'add_coins',
        'add_science': 'add_science',
        'add_shield_vp_coin': 'add_shield_vp_coin',
        'choose_manuf': 'choose_manuf_nontradable',
        'add_vp_coin': 'add_vp_coin',
        'play_seventh_card': 'play_seventh_card',
        'buy_raw_cheap': 'buy_raw_cheap',
        'copy_guild': 'copy_guild',
        'add_vp_discard': 'add_vp_discard',
    }
    return mapping.get(effect_name)


def _extract_first_if_block(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (condition, body) for the first if-block in text."""
    if_match = re.search(r'\bif\s*\(', text)
    if not if_match:
        return None, None
    cond_start = if_match.end() - 1
    cond_end = _find_matching_paren(text, cond_start)
    condition = text[cond_start + 1:cond_end]
    brace_start = text.find('{', cond_end)
    if brace_start == -1:
        return condition, None
    body, _ = _find_block_by_braces(text, brace_start)
    return condition, body


def parse_addstage_method(addstage_body: str,
                          stage_0_cost: Dict[str, int],
                          effects_raw: List[str]) -> List[Dict[str, Any]]:
    """Parse AddStage method to extract stage costs, VP, and effects."""
    stages = []

    switch_match = re.search(r'switch\s*\(\s*this->stage\s*\)\s*{', addstage_body)
    if not switch_match:
        return stages

    switch_body, _ = _find_block_by_braces(addstage_body, switch_match.end() - 1)

    case_iter = list(re.finditer(r'case\s+(\d+)\s*:', switch_body))
    if not case_iter:
        return stages

    cost_map = stage_0_cost.copy()
    stage_index = 0

    for idx, case_match in enumerate(case_iter):
        start = case_match.end()
        end = case_iter[idx + 1].start() if idx + 1 < len(case_iter) else len(switch_body)
        case_body = switch_body[start:end]

        condition, body = _extract_first_if_block(case_body)
        if condition is None:
            continue

        required_resources = re.findall(r'ProduceResource\s*\(\s*RESOURCE::(\w+)', condition)
        stage_cost = {res: cost_map.get(res, 0) for res in required_resources}

        vp = 0
        if body:
            for vp_match in re.finditer(r'this->wonder_points\s*\+=\s*(\d+)', body):
                vp += int(vp_match.group(1))

        stage_effects = []
        if body:
            for shield_match in re.finditer(r'p->AddResource\s*\(\s*RESOURCE::shields\s*,\s*(\d+)\s*\)', body):
                stage_effects.append({'type': 'add_shields', 'amount': int(shield_match.group(1))})

            for coin_match in re.finditer(r'p->AddResource\s*\(\s*RESOURCE::coins\s*,\s*(\d+)\s*\)', body):
                stage_effects.append({'type': 'add_coins', 'amount': int(coin_match.group(1))})

            if 'p->CanPlaySeventh' in body:
                stage_effects.append({'type': 'play_seventh_card'})
            if 'p->CanCopyGuild' in body:
                stage_effects.append({'type': 'copy_guild'})
            if 'p->CanBuildFromDiscard' in body:
                stage_effects.append({'type': 'discard_build_for_free'})
            if 'p->CanBuildFromHand' in body:
                stage_effects.append({'type': 'hand_build_for_free'})

            for cost_update in re.finditer(r'this->cost\[RESOURCE::(\w+)\]\s*=\s*(\d+)', body):
                res = cost_update.group(1)
                amt = int(cost_update.group(2))
                cost_map[res] = amt

        if stage_index < len(effects_raw):
            normalized = normalize_effect_name(effects_raw[stage_index])
            if normalized and not any(e.get('type') == normalized for e in stage_effects):
                if normalized not in {'add_shields', 'add_coins'}:
                    stage_effects.append({'type': normalized})

        stages.append({
            'cost': stage_cost,
            'vp': vp,
            'effects': stage_effects,
        })

        stage_index += 1

    return stages


def build_wonder_db(wonders_data: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]) -> Dict[str, Dict[str, Any]]:
    """Build final wonder database."""
    wonder_db = {}
    
    for class_name, base_data, stages in wonders_data:
        wonder_id = base_data.get('id')
        if wonder_id is None:
            continue
        
        wonder_db[str(wonder_id)] = {
            'id': wonder_id,
            'id_name': base_data.get('id_name', ''),
            'name': base_data.get('name', ''),
            'starting_resource': base_data.get('starting_resource', ''),
            'stages': stages,
        }
    
    return wonder_db


def write_json_atomic(data: Any, file_path: Path):
    """Write JSON atomically using temp file."""
    temp_path = file_path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write('\n')
    temp_path.replace(file_path)


def main():
    print("=" * 60)
    print("Wonder Database Extraction")
    print("=" * 60)
    
    # Parse enums
    print("\n1. Parsing enums...")
    wonder_h = REPO_ROOT / "include/wonder.h"
    resources_h = REPO_ROOT / "include/resources.h"
    
    wonder_id_name_to_int, wonder_id_int_to_name = parse_enum(wonder_h, 'WONDER_ID')
    print(f"   - Found {len(wonder_id_name_to_int)} WONDER_ID entries")
    
    effect_name_to_int, effect_int_to_name = parse_enum(wonder_h, 'EFFECT')
    print(f"   - Found {len(effect_name_to_int)} EFFECT entries")
    
    resource_name_to_int, resource_int_to_name = parse_enum(resources_h, 'RESOURCE')
    print(f"   - Found {len(resource_int_to_name)} RESOURCE entries")
    
    # Parse wonder classes
    print("\n2. Parsing wonder class definitions from wonder.cpp...")
    wonder_cpp = REPO_ROOT / "src/wonder.cpp"
    wonder_classes = parse_wonder_classes(wonder_cpp)
    print(f"   - Found {len(wonder_classes)} wonder classes")
    
    # Parse each wonder
    print("\n3. Extracting wonder data...")
    wonders_data = []
    
    for class_name, constructor_body, addstage_body in wonder_classes:
        # Parse constructor
        base_data = parse_wonder_constructor(
            constructor_body,
            wonder_id_name_to_int,
            resource_int_to_name,
            effect_int_to_name
        )
        
        # Parse AddStage
        stages = parse_addstage_method(
            addstage_body,
            base_data.get('stage_0_cost', {}),
            base_data.get('effects_raw', [])
        )
        
        wonders_data.append((class_name, base_data, stages))
        print(f"   - {class_name}: {len(stages)} stages")
    
    # Build final database
    print("\n4. Building final wonder database...")
    wonder_db = build_wonder_db(wonders_data)
    
    # Validate
    print("\n5. Validating...")
    for wonder_id_str, wonder in wonder_db.items():
        assert 'id' in wonder
        assert 'name' in wonder
        assert 'starting_resource' in wonder
        assert 'stages' in wonder
        assert len(wonder['stages']) >= 2, f"Wonder {wonder['name']} has {len(wonder['stages'])} stages"
    print(f"   - All {len(wonder_db)} wonders validated")
    
    # Write output
    print("\n6. Writing output file...")
    output_dir = REPO_ROOT / "python/src/sevenwonders_rl/data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wonder_db_sorted = dict(sorted(wonder_db.items(), key=lambda x: int(x[0])))
    
    wonder_db_path = output_dir / "wonder_db.json"
    write_json_atomic(wonder_db_sorted, wonder_db_path)
    print(f"   - {wonder_db_path}")
    
    print("\n" + "=" * 60)
    print("✓ Wonder database extraction complete!")
    print(f"  {len(wonder_db)} wonders")
    print(f"  {sum(len(w['stages']) for w in wonder_db.values())} total stages")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
