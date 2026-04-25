#!/usr/bin/env python3
"""
Data Processing Pipeline: normalize > merge > deduplicate > clean > quality filter
Processes external data from datasets/advmodel and datasets/evalmodel
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import re

def normalize_text(text):
    """Normalize text: remove extra whitespace, normalize quotes, etc."""
    if not isinstance(text, str):
        return str(text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    # Remove excessive newlines
    text = re.sub(r'\n+', '\n', text)

    return text.strip()

def merge_datasets(data_list):
    """Merge multiple datasets into one"""
    merged = []
    for data in data_list:
        if isinstance(data, list):
            merged.extend(data)
        elif isinstance(data, dict):
            merged.append(data)
    return merged

def deduplicate_data(data):
    """Remove duplicate entries based on text content"""
    seen = set()
    deduplicated = []

    for item in data:
        if isinstance(item, dict) and 'text' in item:
            # Create a normalized key for deduplication
            text_key = normalize_text(item['text']).lower().strip()
            if text_key and text_key not in seen:
                seen.add(text_key)
                deduplicated.append(item)
        else:
            # Keep non-text items as-is
            deduplicated.append(item)

    return deduplicated

def clean_data(data, cleaning_steps=None):
    """Apply various cleaning steps"""
    if cleaning_steps is None:
        cleaning_steps = ['normalize_text', 'remove_short', 'remove_empty']

    cleaned = data.copy()

    if 'normalize_text' in cleaning_steps:
        for item in cleaned:
            if 'text' in item:
                item['text'] = normalize_text(item['text'])

    if 'remove_empty' in cleaning_steps:
        cleaned = [item for item in cleaned if item.get('text', '').strip()]

    if 'remove_short' in cleaning_steps:
        cleaned = [item for item in cleaned if len(item.get('text', '')) > 10]

    return cleaned

def quality_filter(data, min_score=3):
    """Filter by quality score"""
    if not any('score' in item for item in data):
        return data  # No scores to filter by

    filtered = []
    for item in data:
        score = item.get('score', 0)
        if isinstance(score, (int, float)) and score >= min_score:
            filtered.append(item)
        elif isinstance(score, str):
            # Try to parse string scores
            try:
                score_val = float(score)
                if score_val >= min_score:
                    filtered.append(item)
            except:
                filtered.append(item)  # Keep if can't parse

    return filtered

def process_external_data_pipeline():
    """Complete pipeline: normalize > merge > deduplicate > clean > quality filter"""

    datasets_dir = Path("datasets")
    advmodel_dir = datasets_dir / "advmodel"
    evalmodel_dir = datasets_dir / "evalmodel"

    print("🔧 Starting data processing pipeline...")
    print("📊 Processing external data from datasets/advmodel and datasets/evalmodel")

    # Step 1: Collect all data
    adv_data = []
    eval_data = []

    # Process advmodel data
    if advmodel_dir.exists():
        for json_file in advmodel_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        adv_data.extend(data)
                    else:
                        adv_data.append(data)
                print(f"✅ Loaded {len(data) if isinstance(data, list) else 1} samples from {json_file.name}")
            except Exception as e:
                print(f"❌ Error loading {json_file}: {e}")

    # Process evalmodel data
    if evalmodel_dir.exists():
        for json_file in evalmodel_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        eval_data.extend(data)
                    else:
                        eval_data.append(data)
                print(f"✅ Loaded {len(data) if isinstance(data, list) else 1} samples from {json_file.name}")
            except Exception as e:
                print(f"❌ Error loading {json_file}: {e}")

    print(f"📈 Raw data: {len(adv_data)} adv samples, {len(eval_data)} eval samples")

    # Step 2: Normalize
    print("🔧 Step 2: Normalizing text...")
    for item in adv_data + eval_data:
        if 'text' in item:
            item['text'] = normalize_text(item['text'])

    # Step 3: Merge (keep separate for now, but could merge if needed)
    print("🔧 Step 3: Data organized (adv vs eval kept separate)")

    # Step 4: Deduplicate
    print("🔧 Step 4: Deduplicating...")
    adv_data = deduplicate_data(adv_data)
    eval_data = deduplicate_data(eval_data)

    # Step 5: Clean
    print("🔧 Step 5: Cleaning...")
    cleaning_steps = ['normalize_text', 'remove_short', 'remove_empty']
    adv_data = clean_data(adv_data, cleaning_steps)
    eval_data = clean_data(eval_data, cleaning_steps)

    # Step 6: Quality filter
    print("🔧 Step 6: Quality filtering...")
    adv_data = quality_filter(adv_data, min_score=3)
    eval_data = quality_filter(eval_data, min_score=1)  # Lower threshold for eval data

    # Save processed data
    processed_dir = datasets_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    adv_output = processed_dir / "advmodel_processed.json"
    eval_output = processed_dir / "evalmodel_processed.json"

    with open(adv_output, 'w', encoding='utf-8') as f:
        json.dump(adv_data, f, indent=2, ensure_ascii=False)

    with open(eval_output, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    print("✅ Pipeline complete!")
    print(f"💾 Saved {len(adv_data)} processed adv samples to {adv_output}")
    print(f"💾 Saved {len(eval_data)} processed eval samples to {eval_output}")

    return {
        'adv_data': adv_data,
        'eval_data': eval_data,
        'adv_path': str(adv_output),
        'eval_path': str(eval_output)
    }

if __name__ == "__main__":
    result = process_external_data_pipeline()

    print("
🚀 Ready for training!"    print(f"Use {result['adv_path']} for attacker model training")
    print(f"Use {result['eval_path']} for evaluator model training")

    print("
💡 Training commands:"    print(f"python train_adam_efficient.py qwen-1.5b 3 adam_attacker_custom")
    print(f"python train_adam_efficient.py gemma-2b 3 adam_evaluator_custom")