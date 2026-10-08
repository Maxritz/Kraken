#!/usr/bin/env python3
# kv_tier_geometry.py — does the KV residency the engine reports match the
# geometry the file declares, and can any configuration be short of it?
#
# Two independent sources, compared:
#
#   EXPECT  the model file's own metadata, via tools/gguf_scan.py: which layers
#           carry KV, each layer's width, and therefore the KV bytes that can
#           exist and the number of layers a KV tier has to be able to hold.
#   OBSERVE the engine's load report: the KV layer count, the charge, and the
#           HOT and WARM slot counts.
#
# The property under test is a correctness one and not a performance one: with
# no COLD directory a page that fits nowhere is DROPPED, its layer is
# zero-filled on the next step, and the run keeps producing fluent text that is
# wrong. So HOT + WARM must cover every KV-carrying layer, for every flag
# combination, and the WARM capacity is therefore derived from the geometry
# rather than from the byte budget.
#
# Usage:
#   python3 kv_tier_geometry.py check  <scan.json> <arm.json> [<arm.json> ...]
#   python3 kv_tier_geometry.py expect <scan.json>
#   python3 kv_tier_geometry.py selftest
#
# arm.json is written by scripts/kv_tier_geometry_check.sh:
#   {"name": "...", "cold_dir": bool, "asked_warm_mb": int|null,
#    "rc": int, "log": "path/to/engine.log"}
# Exit status: 0 every check passed or was not applicable, 1 something failed.
import json
import os
import sys

# The engine rounds the KV figure it prints (%.0f MiB), so a byte-exact
# comparison is not available and a tolerance is the honest test. One MiB of
# slack on a number that the file's geometry fixes to the byte.
MIB = 1048576.0
CHARGE_TOL_MIB = 1.0
ACT_SIZE = 2  # backend activation type: f16 both on the CPU and the device path


def expect(scan):
    """The file's own statement of its KV geometry, in the engine's units."""
    out = {"file": scan.get("file"), "arch": scan.get("arch"), "ok": False}
    if scan.get("error"):
        out["reason"] = scan["error"]
        return out
    n_layer = scan.get("n_layer")
    by_layer = scan.get("n_head_kv_by_layer")
    by_tensor = scan.get("kv_layers_by_tensor")
    key_len = scan.get("key_length")
    n_head = scan.get("n_head")
    n_embd = scan.get("n_embd")
    if key_len is None and n_head and n_embd and n_embd % n_head == 0:
        key_len = n_embd // n_head  # inferred, as the loader does
    out["n_layer"] = n_layer
    out["kv_head_dim"] = key_len

    # Which layers carry KV, and how wide each one is.
    #
    # The head count and the layer COUNT are two different statements and only
    # one of them is per-layer. A scalar head count says how wide the layers
    # that carry KV are, never how many there are: qwen35 declares that scalar
    # on a 24-layer model where 18 layers are recurrent and only 6 have a query
    # projection. So the layer set comes from the attn_q tensors (kraken's own
    # layer_has_kv predicate) when the naming convention applies, else from the
    # non-zero entries of the array, else from n_layer for a uniform model.
    # MTP heads are the skipped tail and are not part of the stack the engine
    # charges: `<arch>.nextn_predict_layers` says how many of the last blocks
    # never load (verdict.cpp). A uniform model with a draft tail is therefore
    # charged n_layer - nextn layers, not n_layer.
    nextn = int(scan.get("nextn_predict_layers") or 0)
    loaded = (n_layer - nextn) if n_layer else 0
    at = scan.get("kv_layers_at") or []
    array_idx = [i for i, v in enumerate(by_layer) if v] if isinstance(by_layer, list) else None
    if isinstance(array_idx, list) and loaded:
        array_idx = [i for i in array_idx if i < loaded]
    if at:
        kv_idx = sorted(at)
        src = "attn_q tensors"
    elif array_idx:
        kv_idx = array_idx
        src = "attention.head_count_kv[]"
    elif isinstance(scan.get("n_head_kv"), int):
        kv_idx = list(range(loaded))
        src = "attention.head_count_kv (uniform%s)" % (
            ", %d MTP block(s) skipped" % nextn if nextn else "")
    else:
        out["reason"] = "no KV head count in the file (neither scalar nor array)"
        return out

    if isinstance(by_layer, list) and array_idx is not None and at \
            and array_idx != sorted(at):
        out["head_count_mismatch"] = (
            "attention.head_count_kv[] says KV on %s, the attn_q tensors say %s"
            % (array_idx[:8], sorted(at)[:8]))

    # Widths, paired by layer index. A KV layer whose array entry is absent or
    # zero is a real inconsistency: it has a query projection but no KV heads.
    per_layer_heads = [0] * max(loaded, (max(kv_idx) + 1) if kv_idx else 0)
    for i in kv_idx:
        if isinstance(by_layer, list) and i < len(by_layer):
            per_layer_heads[i] = by_layer[i]
        elif isinstance(scan.get("n_head_kv"), int):
            per_layer_heads[i] = scan["n_head_kv"]
    kv_layers = sum(1 for i in kv_idx if per_layer_heads[i])
    if kv_layers != len(kv_idx):
        out["head_count_mismatch"] = (
            "%d layer(s) carry a query projection but only %d have KV heads"
            % (len(kv_idx), kv_layers))

    out["kv_layers"] = kv_layers
    out["kv_layer_indices"] = kv_idx
    out["heads_source"] = src
    out["kv_layers_by_tensor"] = by_tensor
    if not key_len:
        out["reason"] = "no KV width (key_length and n_embd/n_head both absent)"
        return out
    out["ok"] = True
    out["per_layer_heads"] = per_layer_heads
    return out


def expected_kv_mib(exp, ctx):
    """KV bytes that can exist, both planes: sum over KV-carrying layers."""
    if not exp.get("ok"):
        return None
    total = 0
    for h in exp["per_layer_heads"]:
        if not h:
            continue
        total += int(h) * int(exp["kv_head_dim"]) * int(ctx) * ACT_SIZE
    return total * 2.0 / MIB


def observe(log):
    """What the engine's own load report says it did."""
    obs = {"mode": "none", "kv_layers": None, "hot": None, "warm": None,
           "slot_mib": None, "kv_mib": None, "reason": None,
           "floor_warn": False, "cold_warn": False, "dropped": None,
           "one_pass_warn": False}
    for raw in log.splitlines():
        line = raw.replace("\r", "")
        if "kv tiered:" in line:
            # [info ] kv tiered: 30 KV layer(s) of 11 MiB, 2 HOT + 30 WARM
            # slot(s) of 0.19 MiB (largest layer), 0.38 MiB VRAM for HOT
            obs["mode"] = "tiered"
            try:
                head = line.split("kv tiered:")[1]
                obs["kv_layers"] = int(head.split("KV layer(s)")[0].strip())
                tail = head.split(",")[1]
                obs["hot"] = int(tail.split("HOT")[0].strip())
                obs["warm"] = int(tail.split("+")[1].split("WARM")[0].strip())
                obs["slot_mib"] = float(head.split("of ")[-1].split(" MiB")[0])
            except (IndexError, ValueError) as e:
                obs["reason"] = "could not parse the kv tiered line (%s)" % e
        elif "workspaces ready:" in line and "KV=" in line:
            obs["mode"] = "flat"
            try:
                obs["kv_mib"] = float(line.split("KV=")[1].split(" MiB")[0])
            except (IndexError, ValueError) as e:
                obs["reason"] = "could not parse the workspaces line (%s)" % e
        if "cannot run this file" in line:
            obs["mode"] = "refused"
            obs["reason"] = line.split("cannot run this file")[1].strip()
        elif line.startswith("kraken: ") and obs["mode"] == "none":
            obs["reason"] = line[len("kraken: "):].strip()
        if "WARM was raised to" in line or "--kv-warm-mb allows" in line:
            obs["floor_warn"] = True
        if "is not writable, so it was ignored" in line:
            obs["cold_warn"] = True
        if "cannot hold one pass over the model" in line:
            obs["one_pass_warn"] = True
        # The stats line only. The engine's own warning text also says
        # "DROPPED" (it explains what happens to a page), so a bare substring
        # match here reads every raised-budget run as a corrupt one.
        if "[stats ]" in line and "DROPPED" in line and "page(s)" in line:
            try:
                obs["dropped"] = int(line.split("DROPPED")[1].split("page")[0].strip())
            except (IndexError, ValueError):
                obs["dropped"] = -1
    return obs


def check_arm(exp, arm, ctx, label):
    """One arm's verdict. Returns (status, detail) with status in
    PASS / FAIL / SKIP."""
    log = ""
    if arm.get("log") and os.path.exists(arm["log"]):
        log = open(arm["log"], "r", encoding="utf-8", errors="replace").read()
    obs = observe(log)
    rc = arm.get("rc", 0)

    if obs["mode"] == "refused":
        return "SKIP", "refused to load: %s" % (obs["reason"] or "")[:90]
    if obs["mode"] == "none":
        if rc != 0:
            return "SKIP", "rc=%d %s" % (rc, (obs["reason"] or "no report")[:80])
        return "FAIL", "no load report in the log (format changed?)"
    if obs.get("reason") and obs["mode"] == "none":
        return "FAIL", obs["reason"]

    warm = obs["warm"] if obs["warm"] is not None else 0
    hot = obs["hot"] if obs["hot"] is not None else 0
    cold = bool(arm.get("cold_dir"))
    asked = arm.get("asked_warm_mb")
    tiered = obs["mode"] == "tiered"

    # 1. The charge is the KV that exists. This is what makes the tier
    #    arithmetic meaningful at all: a padded charge also pads the flat-or-
    #    tiered decision and the auto WARM budget.
    if tiered:
        if obs["kv_layers"] is None or obs["hot"] is None or obs["slot_mib"] is None:
            return "FAIL", "kv tiered report did not parse (%s)" % (obs.get("reason") or "?")
        if obs["kv_layers"] < 1 or hot < 1 or obs["slot_mib"] <= 0:
            return "FAIL", "tier report says nothing (zero geometry?)"
        if exp.get("ok") and obs["kv_layers"] != exp["kv_layers"]:
            return "FAIL", ("charge counts %d KV layer(s); the file declares %d"
                            % (obs["kv_layers"], exp["kv_layers"]))
    else:
        # Flat: the whole cache is resident, so the invariant holds by
        # construction and what is left to check is the charge itself. The MiB
        # figure encodes the per-layer sum -- a hybrid charged n_layer x widest
        # reports 208 MiB where its geometry allows 24 -- so this is the same
        # check by a different number.
        if obs["kv_mib"] is None:
            return "FAIL", "no KV figure in the flat report"
        if exp.get("ok"):
            want = expected_kv_mib(exp, ctx)
            if want is not None and abs(want - obs["kv_mib"]) > CHARGE_TOL_MIB:
                return "FAIL", ("flat charge %.0f MiB, the file's geometry says "
                                "%.0f MiB" % (obs["kv_mib"], want))

    # 2. The invariant, where there is nowhere to spill.
    if tiered and not cold:
        if hot + warm < obs["kv_layers"]:
            return "FAIL", ("HOT %d + WARM %d < %d KV layer(s): a page has "
                            "nowhere to go and will be dropped"
                            % (hot, warm, obs["kv_layers"]))
        if obs["one_pass_warn"]:
            return "FAIL", "the engine warned that the tiers cannot cover one pass"

    # 3. The other half of the policy: a spill directory means the budget is a
    #    budget. If the floor engaged here, it is overriding a choice it should
    #    not be overriding.
    if cold and obs["floor_warn"]:
        return "FAIL", ("the geometry floor engaged even though a spill "
                        "directory exists: WARM %d slot(s)" % warm)
    if cold and asked == 0 and warm != 0:
        return "FAIL", ("a cold dir exists and --kv-warm-mb 0 was asked for, "
                        "but WARM has %d slot(s)" % warm)
    if cold and obs["cold_warn"]:
        return "FAIL", "the cold dir was reported unwritable though it was created"

    # 4. A raised budget is a deviation from the caller's flag: it has to be
    #    reported, or the host-RAM behaviour changed silently. What the budget
    #    alone allowed is `asked/2 MiB per plane / slot MiB` floored and capped
    #    at the layer count -- the same arithmetic init does -- so the harness
    #    derives it rather than trusting the engine's own warning text.
    if tiered and not cold and asked is not None and obs["slot_mib"]:
        per_plane = min(obs["kv_layers"],
                        int((asked / 2.0) / obs["slot_mib"]))
        if warm > per_plane and not obs["floor_warn"]:
            return "FAIL", ("WARM %d slot(s) exceeds the %d the --kv-warm-mb %d "
                            "budget allows, with no report of the raise"
                            % (warm, per_plane, asked))
        if asked == 0 and warm < obs["kv_layers"]:
            return "FAIL", ("WARM %d slot(s) for %d KV layer(s) at "
                            "--kv-warm-mb 0: the floor did not engage"
                            % (warm, obs["kv_layers"]))

    # 5. Nothing may have been dropped.
    if obs["dropped"]:
        return "FAIL", "%d page(s) DROPPED (KV history lost)" % obs["dropped"]

    if tiered:
        detail = "tiered: %d KV layer(s), %d HOT + %d WARM slots" % (
            obs["kv_layers"], hot, warm)
    else:
        detail = "flat: KV=%.0f MiB, whole cache resident" % obs["kv_mib"]
    return "PASS", detail


def cmd_check(argv):
    if len(argv) < 2:
        sys.stderr.write("check needs <scan.json> <arm.json> [...]\n")
        return 1
    scan = load_scan(argv[0])
    exp = expect(scan)
    ctx = int(os.environ.get("KRK_CTX", "512"))
    worst = 0
    line = []
    for path in argv[1:]:
        arm = json.load(open(path, "r", encoding="utf-8"))
        status, detail = check_arm(exp, arm, ctx, arm.get("name", path))
        if status == "FAIL":
            worst = 1
        line.append("%s %s [%s] %s" % (status, scan.get("file", "?"),
                                       arm.get("name", "?"), detail))
    for l in line:
        print(l)
    if exp.get("head_count_mismatch"):
        print("note %s: %s" % (scan.get("file", "?"), exp["head_count_mismatch"]))
    return worst


def load_scan(path):
    """A scan JSON, or a model file to scan (the same call the shell makes)."""
    if path.lower().endswith(".gguf"):
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run([sys.executable, os.path.join(here, "gguf_scan.py"), path],
                             capture_output=True)
        return json.loads(out.stdout.decode("utf-8", "replace").splitlines()[0])
    return json.load(open(path, "r", encoding="utf-8"))


def cmd_expect(argv):
    scan = load_scan(argv[0])
    exp = expect(scan)
    ctx = int(os.environ.get("KRK_CTX", "512"))
    print(json.dumps({"expect": exp, "kv_mib_at_ctx": expected_kv_mib(exp, ctx)}))
    return 0


def cmd_selftest(argv):
    """The harness has to be able to fail, or a green run proves nothing. These
    are the two defects it exists to catch, replayed as logs, plus the shapes it
    must not flag."""
    base = ("[info ] kv tiered: 30 KV layer(s) of 11 MiB, 2 HOT + 30 WARM "
            "slot(s) of 0.19 MiB (largest layer), 0.38 MiB VRAM for HOT\n")
    scan = {"file": "t.gguf", "arch": "llama", "n_layer": 30, "n_head": 9,
            "n_embd": 576, "n_head_kv": 3, "key_length": 64,
            "kv_layers_by_tensor": 30}
    raised = ("[warn ] kv: --kv-warm-mb allows 0.0 MiB of WARM for 30 KV "
              "layer(s) and one pass needs all of them, and there is no "
              "--kv-cold-dir to spill to, so WARM was raised to 30 slot(s) = "
              "11.2 MiB.")
    cases = [
        # The two shapes it exists to catch, and then the policy's other half.
        ("short of the working set", base.replace("2 HOT + 30", "2 HOT + 26"),
         {"name": "B", "cold_dir": False, "asked_warm_mb": None}, "FAIL"),
        ("charge disagrees with the file",
         base.replace("30 KV layer(s)", "52 KV layer(s)"),
         {"name": "B", "cold_dir": False, "asked_warm_mb": None}, "FAIL"),
        ("a dropped page",
         base + "[stats ] kv DROPPED 4 page(s) with nowhere to put them\n",
         {"name": "B", "cold_dir": False, "asked_warm_mb": None}, "FAIL"),
        ("floor silent at --kv-warm-mb 0", base,
         {"name": "B", "cold_dir": False, "asked_warm_mb": 0}, "FAIL"),
        ("floor engaged but short",
         base.replace("2 HOT + 30", "2 HOT + 2") + raised + "\n",
         {"name": "B", "cold_dir": False, "asked_warm_mb": 0}, "FAIL"),
        ("floor engaged and reported", base + raised + "\n",
         {"name": "B", "cold_dir": False, "asked_warm_mb": 0}, "PASS"),
        ("auto WARM, nothing raised", base,
         {"name": "A", "cold_dir": False, "asked_warm_mb": None}, "PASS"),
        ("budget above the working set, nothing raised", base,
         {"name": "A", "cold_dir": False, "asked_warm_mb": 64}, "PASS"),
        ("cold dir, budget kept", base.replace("2 HOT + 30", "2 HOT + 0"),
         {"name": "C", "cold_dir": True, "asked_warm_mb": 0}, "PASS"),
        ("cold dir, floor engaged anyway",
         base.replace("2 HOT + 30", "2 HOT + 0") + raised + "\n",
         {"name": "C", "cold_dir": True, "asked_warm_mb": 0}, "FAIL"),
        ("flat and exactly right",
         "[info ] workspaces ready: chunk=256 ctx=512 KV=11 MiB (load 1 ms)\n",
         {"name": "F", "cold_dir": False, "asked_warm_mb": None}, "PASS"),
        ("flat but padded (the 8.67x charge)",
         "[info ] workspaces ready: chunk=256 ctx=512 KV=208 MiB (load 1 ms)\n",
         {"name": "F", "cold_dir": False, "asked_warm_mb": None}, "FAIL"),
        ("refused file is a skip, not a failure",
         "kraken: cannot run this file ('gemma4'). per-layer input embeddings\n",
         {"name": "A", "cold_dir": False, "asked_warm_mb": None}, "SKIP"),
    ]
    import tempfile
    bad = 0
    for name, log, arm, want in cases:
        d = tempfile.mkdtemp()
        lp = os.path.join(d, "e.log")
        open(lp, "w", encoding="utf-8").write(log)
        arm = dict(arm)
        arm["log"] = lp
        arm["rc"] = 0
        got, detail = check_arm(expect(scan), arm, 512, arm["name"])
        ok = got == want
        if not ok:
            bad += 1
        print("%-4s %-32s want %-4s got %-4s  %s"
              % ("ok" if ok else "BAD", name, want, got, detail[:70]))
    print("%d/%d selftest cases as expected" % (len(cases) - bad, len(cases)))
    return 1 if bad else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    args = sys.argv[2:]
    if cmd == "check":
        sys.exit(cmd_check(args))
    elif cmd == "expect":
        sys.exit(cmd_expect(args))
    elif cmd == "selftest":
        sys.exit(cmd_selftest(args))
    else:
        sys.stderr.write(__doc__ or "usage: kv_tier_geometry.py check|expect|selftest\n")
        sys.exit(2)
