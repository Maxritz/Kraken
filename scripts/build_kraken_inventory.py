#!/usr/bin/env python3
# Read-only inventory of GGUF files reachable from this checkout.
# Source of truth is each file's GGUF metadata and loader verdict, not its name.

import json
import pathlib
import re
import subprocess
import time

WORK = pathlib.Path(__file__).resolve().parents[1]
REPORT_DIR = WORK / "reports"
TOOLS_DIR = WORK / "tools"
LOG_FILE = TOOLS_DIR / "kraken_inventory.log"
INVENTORY_JSON = REPORT_DIR / "kraken_inventory.json"
INVENTORY_MD = REPORT_DIR / "kraken_inventory.md"
KNOWN_ABSENT = TOOLS_DIR / "known_absent.txt"

REQUESTED_FILENAMES = frozenset([
    "Bonsai-27B-Q1_0.gguf",
    "Ternary-Bonsai-2-27B-PQ2_0.gguf",
    "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
    "DeepSeek-V4-Flash-0731-K160-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf",
    "DeepSeek-V4-Flash-0731-K160-REAP-MTP-IQ2XXS-w2Q2K-AttnQ8-SExpQ8-ROCm.gguf",
    "gemma-4-E4B-it-Q4_K_M.gguf",
    "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
    "gemma4-coding-Q6_K.gguf",
    "mtp-gemma-4-26B-A4B-it-Q4_0.gguf",
    "GLM-4.7-Flash-Q4_K_M.gguf",
    "GLM-4.7-Flash-APEX-I-Quality.gguf",
    "K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf",
    "laguna-xs2-Q4_K_M.gguf",
    "laguna-xs21-dflash-q8.gguf",
    "laguna-s-2.1-DFlash-Q4_K_M.gguf",
    "Laguna-S-2.1-UD-Q4_K_M-00001-of-00003.gguf",
    "Laguna-S-2.1-UD-Q4_K_M-00002-of-00003.gguf",
    "Laguna-S-2.1-UD-Q4_K_M-00003-of-00003.gguf",
    "Laguna-S-2.1-UD-Q4_K_M-00001-of-00003.gguf.edge0-index.json",
    "Laguna-S-2.1-UD-Q4_K_M-00002-of-00003.gguf.edge0-index.json",
    "Laguna-S-2.1-UD-Q4_K_M-00003-of-00003.gguf.edge0-index.json",
    "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-noMTP.gguf",
    "nemotron-3-nano-4b-NVFP4.gguf",
    "mtp-Ornith-1.0-9B-head-Q8_0.gguf",
    "Ornith-1.0-9b-ROCmFPX-STRIX_LEAN.gguf",
    "Ornith-1.5-122B-Q4_K_M-UNCENSORED.gguf",
    "Unsloth-Ornith-1.5-35B-A3B-UD-Q4_K_XL.gguf",
    "Muse-Glimmer-30B-UD-Q8_K_XL.gguf",
    "Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf",
    "Tiel-Coder-35B-A3B-UD-Q5_K_XL.gguf",
    "Tiel-Coder-35B-A3B-MTP-APEX.gguf",
    "Underdog-Saluki-27B-1.0-IQ2-mix-MTP-abliterated.gguf",
    "ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf",
    "qwen3.5-flash-Q4.gguf",
    "Qwen3.5-9B-DeepSeek-V4-Flash-MTP-Q3_K_M.gguf",
    "qwen35-9b-instruct-nvfp4.gguf",
    "qwen3.5-9b-nvfp4.gguf",
    "Qwen3.8-9B-Q6_K.gguf",
    "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
    "Qwen3.8-27B-WebGGUF-Q4_0.gguf",
    "Qwen3-Coder-53B-A3B-Instruct-TOTAL-RECALL-v2-MASTER-CODER-L.i1-Q4_K_M.gguf",
    "Qwen2.5-Coder-32B-Instruct-3MPER0RR-abliterated.Q4_K_M.gguf",
    "qwen2.5-coder-3b-instruct-q8_0.gguf",
    "Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf",
    "Qwen3-4B-Q8_0.gguf",
    "smolcode-coder-cpp-1.5b-q4_k_m.gguf",
    "smolcode-coder-cpp-3b-q4_k_m.gguf",
    "qwen3vl-4b-q4_k_m.gguf",
    "qwable-v1-mxfp4_moe.gguf",
    "Spark-X2.5-4B-Q4_0_ROCMFP4_STRIX_LEAN.gguf",
    "Spark-X2.5-4B-Q8_0.gguf",
    "Spark_one.Q6_K.gguf",
    "Ternary-Bonsai-27B-dspark-Q4_1.gguf",
    "Qwen3-4B-Instruct-2507-Q6_K.gguf",
    "Qwen3.5-0.8B.Q4_K_M.gguf",
    "Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf",
    "SmolLM2-135M-Instruct.IQ4_XS",
    "SmolLM2-135M-Instruct.Q4_K_M.gguf",
    "L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-Q8_0.gguf",
    "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB_QAT_IQ2_S.gguf",
    "MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf",
    "qwen-agentworld-35b-a3b-nvfp4.gguf",
    "Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16-MTP-i1.Q5_K_M.gguf",
])

SCAN_ROOTS = [WORK / "models"]


def display_path(path):
    try:
        return path.resolve().relative_to(WORK).as_posix()
    except ValueError:
        return path.as_posix()


def discover_ggufs():
    found = {}
    roots_info = []
    for root in SCAN_ROOTS:
        if not root.exists():
            roots_info.append({"path": display_path(root), "exists": False, "gguf_files": 0})
            continue
        count = 0
        for path in root.rglob("*.gguf"):
            count += 1
            found.setdefault(path.name, []).append(path.resolve())
        roots_info.append({"path": display_path(root), "exists": True, "gguf_files": count})
    return found, roots_info


def inspector_path():
    candidates = [
        WORK / "build-hip" / "kraken-inspect.exe",
        WORK / "build-hip" / "kraken-inspect",
        WORK / "build-name-hpc" / "kraken-inspect.exe",
        WORK / "build-name-hpc" / "kraken-inspect",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def parse_inspect_output(output, duration):
    fields = {}
    metadata = {}
    in_metadata = False
    for line in output.splitlines():

        if line.strip() == "metadata:":
            in_metadata = True
            continue
        if in_metadata:
            match = re.match(r"^\s+([A-Za-z0-9_.-]+)\s+=\s+(.*)$", line)
            if match:
                key, value = match.groups()
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.isdigit():
                    value = int(value)
                metadata[key] = value
            continue
        match = re.match(r"^(file|tensors|arch|verdict)\s+(.+?)\s*$", line)
        if match:
            fields[match.group(1)] = match.group(2).strip()

    headline = fields.get("arch", "")
    arch_name = headline.split(" ", 1)[0]
    if not arch_name:
        raise ValueError("inspector output has no architecture headline")
    block_count = metadata.get(arch_name + ".block_count")
    expert_count = metadata.get(arch_name + ".expert_count", 0)
    expert_used = metadata.get(arch_name + ".expert_used_count")
    split_count = metadata.get("split.count", 1)
    tensor_match = re.search(r"\d+", fields.get("tensors", ""))
    tensor_count = int(tensor_match.group()) if tensor_match else 0
    return {
        "file": fields.get("file", ""),
        "general": {"architecture": metadata.get("general.architecture", arch_name)},
        "arch": {"architecture": arch_name, "block_count": block_count or 0},
        "layers": {"count": block_count or 0},
        "tensor_count": tensor_count,
        "experts": {
            "count": expert_count if isinstance(expert_count, int) else 0,
            "used_count": expert_used if isinstance(expert_used, int) else 0,
        },
        "kv": {},
        "rope": {},
        "shard_count": split_count if isinstance(split_count, int) and split_count > 0 else 1,
        "sidecars": [],
        "memory": {},
        "partition": [],
        "_verdict": fields.get("verdict", ""),
        "_headline": headline,
        "_inspect_sec": round(duration, 3),
    }


def run_inspect(path):
    executable = inspector_path()
    if not executable.exists():
        return None, "kraken-inspect missing (checked build-hip and build-name-hpc)"
    start = time.time()
    try:
        result = subprocess.run(
            [str(executable), "--meta", str(path)],
            cwd=str(WORK),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout after 120s"
    duration = time.time() - start
    if result.returncode != 0:
        return None, "rc=%d stderr=%s" % (result.returncode, result.stderr.strip())
    try:
        return parse_inspect_output(result.stdout, duration), None
    except (ValueError, TypeError) as exc:
        return None, "could not parse kraken-inspect output: %s" % exc


def normalize_arch_family(general):
    name = (general.get("architecture") or "").strip()
    if not name:
        return "", ""
    if name in {"qwen35", "qwen35moe", "qwen4exp", "qwen3", "qwen3moe",
                "qwen2", "qwen25"}:
        return name, name
    if name in {"gemma4", "gemma", "gemma2", "gemma3", "gpt-oss",
                "deepseek2", "deepseek4", "deepseek4-dspark"}:
        return name, name
    if name.startswith("qwen38") or name == "qwen38":
        return name, "qwen38-27b-family"
    return name, ""


def qwen_family_rules():
    return [
        {"family": "qwen4exp", "description": "Qwen3.8-Flash-Next REAP; distinguish by GGUF metadata, not filename."},
        {"family": "qwen35", "description": "Dense recurrent Qwen3.5 architecture."},
        {"family": "qwen35moe", "description": "Recurrent MoE Qwen3.5 architecture."},
        {"family": "qwen3moe", "description": "Dense-routed Qwen3 MoE; not recurrent."},
    ]


def write_reports(rows, requested, roots_info):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    present_names = {row["filename"] for row in rows}
    absent_requested = sorted(set(requested) - present_names)
    payload = {
        "generated_at": time.time(),
        "worktree": ".",
        "scan_roots": roots_info,
        "total_files": len(rows),
        "requested_count": len(requested),
        "requested_present_count": sum(row["filename"] in requested for row in rows),
        "requested_absent": absent_requested,
        "unrequested_discovered": sorted(present_names - set(requested)),
        "qwen_family_rules": qwen_family_rules(),
        "files": rows,
    }
    INVENTORY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Kraken GGUF inventory",
        "",
        "Generated from the project root.",
        "",
        "## Scan surface",
        "",
        "| root | exists | .gguf found |",
        "|------|:------:|------------:|",
    ]
    for root in roots_info:
        lines.append(f"| {root['path']} | {'yes' if root['exists'] else 'no'} | {root['gguf_files']} |")
    lines.extend([
        "",
        f"Total distinct filenames discovered: {len(rows)}",
        f"Requested files present: {payload['requested_present_count']}",
        f"Requested files absent: {len(absent_requested)}",
        f"Additional local files discovered: {len(payload['unrequested_discovered'])}",
        "",
    ])
    if absent_requested:
        lines.extend(["## Requested but not present on scanned roots", ""])
        lines.extend(f"- `{name}`" for name in absent_requested)
        lines.append("")
    lines.extend([
        "## File table",
        "",
        "| filename | bytes | arch | family | verdict | tensors | experts | shards |",
        "|---|---:|---|---|---|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['filename']} | {row['bytes']} | {row['arch_name']} | {row['family']} "
            f"| {row['verdict']} | {row['tensor_count']} | {row['expert_count']} | {row['shard_count']} |"
        )
    INVENTORY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    KNOWN_ABSENT.write_text("\n".join(absent_requested) + "\n", encoding="utf-8")


def main():
    print("discovering GGUF files on configured scan roots...", flush=True)
    found, roots_info = discover_ggufs()
    print("discovered distinct filenames:", len(found), flush=True)
    log_lines = []
    rows = []
    for name in sorted(found):
        paths = found[name]
        path = paths[0]
        size = path.stat().st_size
        print(f"inspecting {name} ({size} bytes) ...", flush=True)
        data, error = run_inspect(path)
        log_lines.append(f"FILE {name}\nPATH {display_path(path)}\nSIZE {size}\nRC {'ok' if data else 'FAIL'}\n")
        if error:
            log_lines.append(f"ERROR {error}\n")
            rows.append({
                "filename": name, "bytes": size,
                "paths": [display_path(p) for p in paths],
                "inspect_sec": None, "inspect_error": error,
                "general": {}, "arch_name": "", "family": "", "verdict": "",
                "tensor_count": 0, "expert_count": 0, "shard_count": 0,
            })
            continue
        general = data["general"]
        arch_name, family = normalize_arch_family(general)
        row = {
            "filename": name,
            "bytes": size,
            "paths": [display_path(p) for p in paths],
            "inspect_sec": data["_inspect_sec"],
            "loader_headline": data["_headline"],
            "arch_name": arch_name,
            "family": family,
            "verdict": data["_verdict"],
            "general": general,
            "arch": data["arch"],
            "layers": data["layers"],
            "tensor_count": data["tensor_count"],
            "experts": data["experts"],
            "shard_count": data["shard_count"],
        }
        rows.append(row)
        log_lines.append(json.dumps(row, ensure_ascii=False, indent=2))
        log_lines.append("")
        row["expert_count"] = data["experts"]["count"]
        print(f"  arch={arch_name!r} family={family!r} verdict={row['verdict']!r} "
              f"tensors={row['tensor_count']} experts={row['expert_count']} shards={row['shard_count']}",
              flush=True)
    LOG_FILE.write_text("\n".join(log_lines).rstrip() + "\n", encoding="utf-8")
    write_reports(rows, REQUESTED_FILENAMES, roots_info)
    print("wrote inventory JSON, Markdown, log and known-absent list")


if __name__ == "__main__":
    main()
