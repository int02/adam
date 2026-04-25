import json
from datasets import load_dataset, get_dataset_config_names

all_prompts = []

def extract_prompt(entry):
    for key in ["prompt", "instruction", "text", "query"]:
        if key in entry and entry[key]:
            return entry[key]
    return None

# -------------------
# JailbreakBench
# -------------------
try:
    configs = get_dataset_config_names("JailbreakBench/JBB-Behaviors")
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", configs[0])

    split = list(jbb.keys())[0]
    for entry in jbb[split]:
        p = extract_prompt(entry)
        if p:
            all_prompts.append(p)

except Exception as e:
    print(f"Failed to load JailbreakBench: {e}")

# -------------------
# AdvBench (mirror-safe fallback)
# -------------------
adv_sources = [
    "walledai/AdvBench",
    "AmberYifan/AdvBench_safe",
    "kelly8tom/advbench_orig",
]

loaded = False

for src in adv_sources:
    try:
        adv = load_dataset(src)

        for split in adv:
            for entry in adv[split]:
                p = extract_prompt(entry)
                if p:
                    all_prompts.append(p)

        loaded = True
        break

    except Exception as e:
        print(f"Failed AdvBench source {src}: {e}")

if not loaded:
    print("AdvBench not available (skipping)")

# -------------------
# cleanup
# -------------------
all_prompts = list({p.strip() for p in all_prompts if p and p.strip()})

with open("datasets/jailbreaks.json", "w", encoding="utf-8") as f:
    json.dump(all_prompts, f, indent=2, ensure_ascii=False)

print(f"Saved {len(all_prompts)} prompts")