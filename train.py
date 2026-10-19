import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sugarscape import Sugarscape


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_training_data(sugarscape, timestep):
    records = []
    for agent in sugarscape.agents:
        if not agent.isAlive():
            continue

        cell = agent.cell
        state = {
            "id": agent.ID,
            "timestep": timestep,
            "age": agent.age,
            "sugar": round(agent.sugar, 2),
            "spice": round(agent.spice, 2),
            "sugarMetabolism": agent.findSugarMetabolism(),
            "spiceMetabolism": agent.findSpiceMetabolism(),
            "happiness": round(agent.happiness, 2),
            "vision": agent.vision,
            "movement": agent.movement,
            "x": cell.x if cell else None,
            "y": cell.y if cell else None,
            "tribe": agent.tribe if agent.tribe is not None else "None",
        }

        candidates_raw = agent.rankCellsInRange()
        candidates = [
            {
                "candidate": index,
                "x": record["cell"].x,
                "y": record["cell"].y,
                "wealth": round(record["wealth"], 2),
                "distance": record["range"],
            }
            for index, record in enumerate(candidates_raw)
        ]

        action = agent.lastMoveRank if hasattr(agent, "lastMoveRank") else None

        records.append({
            "state": state,
            "candidates": candidates,
            "action": action,
        })

    return records


def run_training(config_path, output_path, max_timesteps=None):
    config = load_config(config_path)
    options = config.get("sugarscapeOptions", config)

    options["simulationMode"] = "llm"
    options["headlessMode"] = True
    if max_timesteps is not None:
        options["timesteps"] = max_timesteps

    print(f"[train] Loading config: {config_path}")
    print(f"[train] simulationMode: {options.get('simulationMode')}")
    print(f"[train] llmBackend: {options.get('llmBackend')}")
    print(f"[train] model: {options.get('ollamaModel') or options.get('vllmModel') or options.get('geminiModel')}")
    print(f"[train] startingAgents: {options.get('startingAgents')}, timesteps: {options.get('timesteps')}")

    sugarscape = Sugarscape(options)

    all_records = []
    start_time = datetime.now()

    try:
        t = 1
        timesteps = sugarscape.maxTimestep if hasattr(sugarscape, "maxTimestep") and sugarscape.maxTimestep else options.get("timesteps", 250)
        timesteps = timesteps - sugarscape.timestep

        while t <= timesteps:
            if len(sugarscape.agents) == 0:
                print(f"[train] All agents died at timestep {sugarscape.timestep}. Stopping.")
                break

            sugarscape.doTimestep()
            records = collect_training_data(sugarscape, sugarscape.timestep)
            all_records.extend(records)

            if t % 50 == 0 or t == timesteps:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"[train] Timestep {sugarscape.timestep}/{timesteps} "
                      f"(living: {len(sugarscape.agents)}, records: {len(all_records)}, {elapsed:.1f}s)")
            t += 1

    except KeyboardInterrupt:
        print("\n[train] Interrupted by user.")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[train] Saved {len(all_records)} training records to {output_path} ({elapsed:.1f}s)")

    print_statistics(all_records)

    return all_records


def print_statistics(records):
    if not records:
        print("[train] No records collected.")
        return

    total = len(records)
    actions = [r["action"] for r in records if r["action"] is not None]
    action_counter = Counter(actions)

    print("\n[train] ---- Statistics ----")
    print(f"[train] Total records: {total}")
    print(f"[train] Records with valid action: {len(actions)}")
    if action_counter:
        print(f"[train] Action (chosen candidate index) distribution:")
        for idx in sorted(action_counter):
            print(f"    candidate {idx}: {action_counter[idx]}")
    print("[train] --------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train an LLM-based agent model by collecting (state, action) data from Sugarscape LLM simulation."
    )
    parser.add_argument("--config", default="config/llm.json",
                        help="Path to Sugarscape config JSON (default: config/llm.json)")
    parser.add_argument("--output", default="training_data.jsonl",
                        help="Output JSONL path for training records (default: training_data.jsonl)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override number of timesteps (default: use config value)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[train] Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    run_training(args.config, args.output, args.timesteps)


if __name__ == "__main__":
    main()
