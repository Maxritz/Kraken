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

A --selfcheck report (kind "selfcheck") carries each group's arm verdicts beside
its counts. The group comparison above still applies -- a group whose arm broke is
not passing -- and the arms are compared arm by arm, so a regression names the arm
and the problem it reported, even in a report where the group status did not move.
An arm the baseline has as ok and this run does not is a regression; an arm added
or removed is a shape change, re-recorded deliberately like a new group.
"""
import json
import subprocess
import sys

PASS = "pass"
OK = "ok"


def kind(report):
    """'run' for a normal suite report, 'selfcheck' for an arm sweep."""
    return report.get("kind", "run")


def arm_map(group):
    """The group's arm verdicts, keyed by arm name; {} when it has none."""
    arms = group.get("arms")
    return arms if isinstance(arms, dict) else {}


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
                           "new", "removed", "newly_broken_arms", "still_broken_arms",
                           "fixed_arms", "new_arms", "removed_arms")}
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
        # A selfcheck report's groups also carry arm verdicts. The group status
        # already moves when an arm breaks, so this is what names the arm -- and
        # it holds in a report where the status did NOT move.
        was_arms, now_arms = arm_map(was), arm_map(now)
        if was_arms or now_arms:
            for arm in now_arms:
                if arm not in was_arms:
                    out["new_arms"].append((name, arm))
                elif was_arms[arm] == OK and now_arms[arm] != OK:
                    out["newly_broken_arms"].append((name, arm, now_arms[arm]))
                elif was_arms[arm] != OK and now_arms[arm] != OK:
                    out["still_broken_arms"].append((name, arm))
                elif was_arms[arm] != OK and now_arms[arm] == OK:
                    out["fixed_arms"].append((name, arm))
            for arm in was_arms:
                if arm not in now_arms:
                    out["removed_arms"].append((name, arm))
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

    def arm_section(title, arms):
        if not arms:
            return
        lines.append("%s (%d):" % (title, len(arms)))
        for entry in arms:
            name, arm = entry[0], entry[1]
            if len(entry) > 2:
                lines.append("  %-44s arm %s: %s" % (name, arm, entry[2]))
            else:
                lines.append("  %-44s arm %s" % (name, arm))

    arm_section("newly broken arms", findings["newly_broken_arms"])
    arm_section("still broken arms", findings["still_broken_arms"])
    arm_section("fixed arms", findings["fixed_arms"])
    arm_section("new arms", findings["new_arms"])
    arm_section("removed arms", findings["removed_arms"])

    n = len(findings["newly_broken"])
    na = len(findings["newly_broken_arms"])
    if n == 0 and na == 0:
        verdict = "no regression"
    elif na == 0:
        verdict = "%d newly broken" % n
    elif n == 0:
        verdict = "%d newly broken arm(s)" % na
    else:
        verdict = "%d newly broken, %d arm(s)" % (n, na)
    lines.append("verdict: %s, %d fixed, %d changed, %d new, %d removed, "
                 "%d arm(s) newly broken, %d arm(s) fixed, %d arm(s) new, "
                 "%d arm(s) removed"
                 % (verdict, len(findings["fixed"]), len(findings["changed"]),
                    len(findings["new"]), len(findings["removed"]), na,
                    len(findings["fixed_arms"]), len(findings["new_arms"]),
                    len(findings["removed_arms"])))
    return lines


def fatal(findings, allow_new=False):
    """Whether these findings make the comparison fail. One predicate, so the
    selftest exercises the same rule main() decides with."""
    if findings["newly_broken"] or findings["newly_broken_arms"]:
        return True
    shape = (findings["new"] or findings["removed"] or findings["new_arms"] or
             findings["removed_arms"])
    return bool(shape) and not allow_new


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
    out = ["{", '  "format": 1,', '  "suite": "kraken-tests",']
    if cur.get("kind"):
        out.append('  "kind": "%s",' % cur["kind"])
    out += ['  "recorded_utc": "%s",' % utc,
           '  "recorded_commit": "%s",' % commit,
           '  "totals": %s,' % json.dumps({"groups": len(cur["groups"]),
                                           "passed": t.get("passed"), "run": t.get("run"),
                                           "exit": t.get("exit")}),
           '  "groups": [']
    groups = []
    for g in cur["groups"]:
        entry = {"name": g["name"], "status": g["status"], "passed": g["passed"],
                 "run": g["run"]}
        # A selfcheck baseline records the arm verdicts as well: they are what the
        # comparison names when the reporting proof breaks.
        if arm_map(g):
            entry["arms"] = dict(arm_map(g))
        groups.append("    " + json.dumps(entry))
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
        exit_code = 1 if fatal(f, allow_new) else 0
        cases.append((name, ok and exit_code == want_exit,
                      "%s exit=%d want %s exit=%d" % (got, exit_code, want_findings, want_exit)))
        return f

    green = report([group("a", PASS, 5, 5), group("b", PASS, 3, 3)])
    f = case("identical green run", green, green, {}, 0)
    assert render(green, green, f)[-1].endswith(
        "no regression, 0 fixed, 0 changed, 0 new, 0 removed, "
        "0 arm(s) newly broken, 0 arm(s) fixed, 0 arm(s) new, 0 arm(s) removed")

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

    # A --selfcheck report: the arm verdicts are the reporting proof, so a broken
    # arm is a regression whose reason names the arm and what it missed.
    def sc_group(name, status, passed, run, arms):
        g = group(name, status, passed, run)
        g["arms"] = arms
        return g

    def sc_report(groups, exit_code=0):
        return {"format": 1, "suite": "kraken-tests", "kind": "selfcheck", "arms": 6,
                "totals": {"groups": len(groups),
                           "passed": sum(g["passed"] for g in groups),
                           "run": sum(g["run"] for g in groups),
                           "bad_groups": sum(1 for g in groups if g["status"] != PASS),
                           "exit": exit_code},
                "groups": groups}

    arms_ok = {"crash": OK, "fuzz": OK}
    sc_green = sc_report([sc_group("a", PASS, 5, 5, arms_ok),
                          sc_group("b", PASS, 3, 3, arms_ok)])
    sc_broken = sc_report([sc_group("a", PASS, 5, 5, arms_ok),
                           sc_group("b", "failed", 3, 3,
                                    {"crash": OK,
                                     "fuzz": "the report records 1/1, want 2/2"})],
                          exit_code=1)
    f = case("a selfcheck arm newly broke", sc_green, sc_broken,
             {"newly_broken": 1, "newly_broken_arms": 1}, 1)
    named = [l for l in render(sc_green, sc_broken, f)
             if "arm fuzz: the report records 1/1" in l]
    cases.append(("the render names the broken arm and its problem", bool(named),
                  "want the arm and its problem in the findings, got %r"
                  % render(sc_green, sc_broken, f)))
    case("an arm broken with the group status unmoved", sc_green,
         sc_report([sc_group("a", PASS, 5, 5, arms_ok),
                    sc_group("b", PASS, 3, 3, {"crash": OK, "fuzz": "broke"})]),
         {"newly_broken_arms": 1}, 1)
    sc_nofuzz = sc_report([sc_group("a", PASS, 5, 5, {"crash": OK}),
                           sc_group("b", PASS, 3, 3, {"crash": OK})])
    case("an arm added is a shape change", sc_nofuzz, sc_green, {"new_arms": 2}, 1)
    case("an arm added, allowed", sc_nofuzz, sc_green, {"new_arms": 2}, 0, True)
    case("an arm fixed", sc_broken, sc_green, {"fixed_arms": 1, "fixed": 1}, 0)

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
    if findings["newly_broken"] or findings["newly_broken_arms"]:
        return 1
    if (findings["new"] or findings["removed"] or findings["new_arms"] or
            findings["removed_arms"]) and not allow_new:
        print("the suite's shape changed: re-record the baseline deliberately "
              "(KRK_TEST_BASELINE_RECORD=1), or --allow-new to compare anyway")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
