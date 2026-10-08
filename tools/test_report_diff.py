#!/usr/bin/env python3
"""Compare a kraken-tests report against the recorded green baseline.

    test_report_diff.py BASELINE CURRENT          # verdict on what changed
    test_report_diff.py --record BASELINE CURRENT # write the baseline from a run
    test_report_diff.py --selftest                # constructed pairs, in memory

Exit status: 0 when nothing regressed, 1 when something did or the suite's shape
changed, 2 for a usage or IO error.

A group is NEWLY BROKEN when the baseline has it passing and this run does not.
That is the only finding that makes the comparison worth running: it names the
regression instead of printing 53 lines and leaving the reader to diff them.

Everything else is reported but not necessarily fatal:

  count changed   both runs pass, the group's check count moved. The watermark
                  moves whenever a check is added; that is a change, not a
                  regression.
  fixed           the baseline had it failing (a forced red baseline) and it
                  passes now.
  still broken    failing in both. Not a regression, but not news either.
  new / removed   a change to the suite's shape. Fatal unless --allow-new,
                  because the baseline has to be re-recorded deliberately
                  rather than drifting out from under the comparison.

A run whose own exit status is non-zero while every group reads as passing is a
failure of the runner itself (a report that could not be written, say), and it is
reported as such rather than compared.
"""
import json
import subprocess
import sys

PASS = "pass"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if d.get("suite") != "kraken-tests" or not isinstance(d.get("groups"), list):
        raise ValueError("%s is not a kraken-tests report" % path)
    return d


def by_name(report):
    return {g["name"]: g for g in report["groups"]}


def classify(base, cur):
    """Returns the findings, keyed by kind, each a list of group dicts."""
    b, c = by_name(base), by_name(cur)
    out = {k: [] for k in ("newly_broken", "still_broken", "fixed", "changed",
                           "new", "removed")}
    for name in [g["name"] for g in cur["groups"]]:
        if name not in b:
            out["new"].append(c[name])
            continue
        was, now = b[name], c[name]
        if was["status"] == PASS and now["status"] != PASS:
            out["newly_broken"].append(now)
        elif was["status"] != PASS and now["status"] == PASS:
            out["fixed"].append(now)
        elif was["status"] != PASS and now["status"] != PASS:
            out["still_broken"].append(now)
        elif (was["passed"], was["run"]) != (now["passed"], now["run"]):
            out["changed"].append(now)
    for name in [g["name"] for g in base["groups"]]:
        if name not in c:
            out["removed"].append(b[name])
    return out


def base_counts(base, cur, name):
    """The baseline's own numbers for a group, for a 'was -> is' line."""
    b = by_name(base)
    if name not in b:
        return "not in the baseline"
    g = b[name]
    return "%d/%d (%s)" % (g["passed"], g["run"], g["status"])


def counts(g):
    return "%d/%d (%s)" % (g["passed"], g["run"], g["status"])


def render(base, cur, findings, fail_lines_shown=5):
    lines = []
    meta = base.get("recorded_utc", "unknown time")
    commit = base.get("recorded_commit", "unknown commit")
    lines.append("baseline   recorded %s at %s, %d groups"
                 % (meta, commit, len(base["groups"])))
    t = cur.get("totals", {})
    lines.append("current    %d groups, %d/%d checks, exit %s%s"
                 % (len(cur["groups"]), t.get("passed", 0), t.get("run", 0),
                    t.get("exit", "?"),
                    "" if t.get("workers") is None else ", %s workers" % t["workers"]))

    def section(title, groups, show_reason=False):
        if not groups:
            return
        lines.append("%s (%d):" % (title, len(groups)))
        for g in groups:
            lines.append("  %-44s %s   was %s"
                         % (g["name"], counts(g), base_counts(base, cur, g["name"])))
            if show_reason:
                for ln in g.get("fail_lines", [])[:fail_lines_shown]:
                    lines.append("      " + ln)
                extra = (g["run"] - g["passed"]) - len(g.get("fail_lines", [])[:fail_lines_shown])
                if extra > 0:
                    lines.append("      ... and %d more failing check(s)" % extra)

    section("newly broken", findings["newly_broken"], show_reason=True)
    section("still broken", findings["still_broken"])
    section("fixed", findings["fixed"])
    section("count changed", findings["changed"])
    section("new groups", findings["new"])
    section("removed groups", findings["removed"])

    n = len(findings["newly_broken"])
    shape = len(findings["new"]) + len(findings["removed"])
    verdict = "no regression" if n == 0 else "%d newly broken" % n
    lines.append("verdict: %s, %d fixed, %d changed, %d new, %d removed"
                 % (verdict, len(findings["fixed"]), len(findings["changed"]),
                    len(findings["new"]), len(findings["removed"])))
    return lines


def runner_failed_without_a_failing_group(cur):
    t = cur.get("totals", {})
    if t.get("exit", 0) == 0:
        return False
    return all(g["status"] == PASS for g in cur["groups"])


def record(base_path, cur, force=False):
    t = cur.get("totals", {})
    bad = [g for g in cur["groups"] if g["status"] != PASS]
    if (t.get("exit", 0) != 0 or bad) and not force:
        sys.stderr.write("refusing to record a baseline from a run that is not green: ")
        if t.get("exit", 0) != 0:
            sys.stderr.write("the run exited %s" % t["exit"])
        if bad:
            sys.stderr.write("%s%s" % (", " if t.get("exit", 0) != 0 else "",
                                       ", ".join(g["name"] for g in bad)))
        sys.stderr.write("\n(use --force to record the failures as known)\n")
        return 1
    commit = "unknown"
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        pass
    utc = subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                         capture_output=True, text=True).stdout.strip()
    out = ["{", '  "format": 1,', '  "suite": "kraken-tests",',
           '  "recorded_utc": "%s",' % utc,
           '  "recorded_commit": "%s",' % commit,
           '  "totals": %s,' % json.dumps({"groups": len(cur["groups"]),
                                           "passed": t.get("passed"), "run": t.get("run"),
                                           "exit": t.get("exit")}),
           '  "groups": [']
    groups = []
    for g in cur["groups"]:
        groups.append("    " + json.dumps({"name": g["name"], "status": g["status"],
                                           "passed": g["passed"], "run": g["run"]}))
    out.append(",\n".join(groups))
    out += ["  ]", "}"]
    with open(base_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print("recorded %d groups from %s at %s into %s"
          % (len(cur["groups"]), cur.get("suite"), utc, base_path))
    return 0


def group(name, status, passed, run, fail_lines=None):
    return {"name": name, "status": status, "passed": passed, "run": run,
            "fail_lines": fail_lines or [], "ms": 1, "index": 0}


def report(groups, exit_code=0, workers=8):
    return {"format": 1, "suite": "kraken-tests", "workers": workers, "wall_ms": 1,
            "totals": {"groups": len(groups),
                       "passed": sum(g["passed"] for g in groups),
                       "run": sum(g["run"] for g in groups),
                       "bad_groups": sum(1 for g in groups if g["status"] != PASS),
                       "exit": exit_code},
            "groups": groups}


def selftest():
    cases = []

    def case(name, base, cur, want_findings, want_exit, allow_new=False):
        f = classify(base, cur)
        got = {k: len(v) for k, v in f.items()}
        ok = all(got.get(k, 0) == v for k, v in want_findings.items())
        exit_code = 1 if (f["newly_broken"] or
                          ((f["new"] or f["removed"]) and not allow_new)) else 0
        cases.append((name, ok and exit_code == want_exit,
                      "%s exit=%d want %s exit=%d" % (got, exit_code, want_findings, want_exit)))
        return f

    green = report([group("a", PASS, 5, 5), group("b", PASS, 3, 3)])
    f = case("identical green run", green, green, {}, 0)
    assert render(green, green, f)[-1].endswith("no regression, 0 fixed, 0 changed, 0 new, 0 removed")

    red = report([group("a", PASS, 5, 5), group("b", "failed", 0, 3,
                                                ["FAIL x.cpp:1  b broke"])], exit_code=1)
    f = case("one group newly failed", green, red, {"newly_broken": 1}, 1)
    assert any("FAIL x.cpp:1  b broke" in l for l in render(green, red, f))

    case("count changed while passing", green,
         report([group("a", PASS, 6, 6), group("b", PASS, 3, 3)]), {"changed": 1}, 0)
    case("new group", green, report([group("a", PASS, 5, 5), group("b", PASS, 3, 3),
                                     group("c", PASS, 1, 1)]), {"new": 1}, 1)
    case("new group, allowed", green, report([group("a", PASS, 5, 5), group("b", PASS, 3, 3),
                                              group("c", PASS, 1, 1)]), {"new": 1}, 0, True)
    case("removed group", green, report([group("a", PASS, 5, 5)]), {"removed": 1}, 1)
    case("crashed group", green,
         report([group("a", PASS, 5, 5), group("b", "crashed", 2, 3)], exit_code=1),
         {"newly_broken": 1}, 1)
    case("empty group", green,
         report([group("a", PASS, 5, 5), group("b", "no-checks", 0, 0)], exit_code=1),
         {"newly_broken": 1}, 1)
    case("fixed against a forced red baseline",
         report([group("a", PASS, 5, 5), group("b", "failed", 0, 3)], exit_code=1),
         green, {"fixed": 1}, 0)
    case("still broken is not news",
         report([group("a", PASS, 5, 5), group("b", "failed", 0, 3)], exit_code=1),
         red, {"still_broken": 1}, 0)
    # The runner failing with every group passing: nothing to compare, and the
    # tool has to say so rather than call it green.
    weird = report(green["groups"], exit_code=1)
    cases.append(("runner failed with no failing group",
                  runner_failed_without_a_failing_group(weird),
                  "want the run's own non-zero exit to be seen"))
    cases.append(("green run is not 'runner failed'",
                  not runner_failed_without_a_failing_group(green), ""))

    bad = [c for c in cases if not c[1]]
    for name, ok, detail in cases:
        print("%-4s %s%s" % ("ok" if ok else "FAIL", name, "" if ok else "  -- " + detail))
    print("\n%d/%d selftest cases passed" % (len(cases) - len(bad), len(cases)))
    return 1 if bad else 0


def main(argv):
    args = [a for a in argv[1:]]
    if "--selftest" in args:
        return selftest()
    do_record = "--record" in args
    allow_new = "--allow-new" in args
    force = "--force" in args
    paths = [a for a in args if not a.startswith("--")]
    if len(paths) != 2:
        sys.stderr.write(__doc__)
        return 2
    base_path, cur_path = paths
    try:
        cur = load(cur_path)
    except (OSError, ValueError) as e:
        sys.stderr.write("cannot read the current report: %s\n" % e)
        return 2
    if do_record:
        return record(base_path, cur, force)
    try:
        base = load(base_path)
    except OSError as e:
        sys.stderr.write("no baseline at %s (%s)\n"
                         "record one from a green run: "
                         "KRK_TEST_BASELINE_RECORD=1 sh scripts/kraken_tests_baseline_check.sh\n"
                         % (base_path, e))
        return 2
    except ValueError as e:
        sys.stderr.write("%s\n" % e)
        return 2

    if runner_failed_without_a_failing_group(cur):
        print("current    exit %s with every group passing: the runner itself failed "
              "(was the report written?)" % cur["totals"]["exit"])
        print("verdict: the run is not usable as a comparison")
        return 1

    findings = classify(base, cur)
    for line in render(base, cur, findings):
        print(line)
    if findings["newly_broken"]:
        return 1
    if (findings["new"] or findings["removed"]) and not allow_new:
        print("the suite's shape changed: re-record the baseline deliberately "
              "(KRK_TEST_BASELINE_RECORD=1), or --allow-new to compare anyway")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
