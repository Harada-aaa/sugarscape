"""Log inspection and truncation helpers for resumable simulations."""

import json
import os
import re


def detectLogTimestep(filepath, logFormat="json"):
    if not filepath or not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return -1, None
    try:
        if logFormat == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.strip():
                return -1, None
            timesteps = [int(m) for m in re.findall(r'"timestep":\s*(\d+)', content)]
            if not timesteps:
                return -1, None
            seed_match = re.search(r'"seed":\s*(-?\d+)', content)
            seed = int(seed_match.group(1)) if seed_match else None
            return max(timesteps), seed
        elif logFormat == "csv":
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) <= 1:
                return -1, None
            headers = [h.strip() for h in lines[0].split(",")]
            if "timestep" not in headers:
                return -1, None
            ts_idx = headers.index("timestep")
            seed_idx = headers.index("seed") if "seed" in headers else None
            timesteps = []
            seed = None
            for line in lines[1:]:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) > ts_idx and parts[ts_idx].lstrip("-").isdigit():
                    timesteps.append(int(parts[ts_idx]))
                    if seed_idx is not None and len(parts) > seed_idx and seed is None:
                        try:
                            seed = int(parts[seed_idx])
                        except ValueError:
                            pass
            if not timesteps:
                return -1, None
            return max(timesteps), seed
    except Exception:
        return -1, None
    return -1, None


def prepareLogFileForResume(filepath, targetTimestep, logFormat="json"):
    if not filepath or not os.path.exists(filepath):
        return False
    try:
        if logFormat == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if not content:
                return False
            records = []
            try:
                loaded = json.loads(content)
                if isinstance(loaded, list):
                    for item in loaded:
                        if isinstance(item, dict) and item.get("timestep", -1) <= targetTimestep:
                            records.append(item)
            except json.JSONDecodeError:
                for line in content.splitlines():
                    stripped = line.strip().rstrip(",").rstrip("]").lstrip("[")
                    if stripped.startswith("{") and stripped.endswith("}"):
                        try:
                            record = json.loads(stripped)
                            if record.get("timestep", -1) <= targetTimestep:
                                records.append(record)
                        except json.JSONDecodeError:
                            pass
            if records:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("[\n")
                    for record in records:
                        f.write(f"\t{json.dumps(record)},\n")
                return True
        elif logFormat == "csv":
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) > 1:
                headers = [h.strip() for h in lines[0].split(",")]
                if "timestep" in headers:
                    ts_idx = headers.index("timestep")
                    kept_lines = [lines[0]]
                    for line in lines[1:]:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) > ts_idx and parts[ts_idx].lstrip("-").isdigit():
                            if int(parts[ts_idx]) <= targetTimestep:
                                kept_lines.append(line.rstrip("\r\n") + "\n")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.writelines(kept_lines)
                    return True
    except Exception:
        return False
    return False