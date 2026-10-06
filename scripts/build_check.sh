#!/usr/bin/env bash
# Build honesty check: is the binary on disk the one the configure claimed?
#
# This does not exercise the engine. It tests the build, because every failure
# below has actually happened here and none of them is visible to the test
# suite:
#
#   * A bare `cmake -Bbuild` under a Windows Clang toolchain came out Debug.
#     CMake's platform modules write Debug into CMAKE_BUILD_TYPE while
#     project() is running, so the Release default further down matched a
#     non-empty value and left it alone. The suite passes in Debug; every
#     benchmark number taken from it is meaningless.
#   * `kraken: HIP SDK not found - building the CPU backend only` is a status
#     line, not an error. A configure can "succeed" into a binary with no GPU
#     path in it, and the only symptom is a missing device line at run time.
#   * CMake's HIP probe leaves its own default in CMAKE_HIP_ARCHITECTURES
#     (gfx906 on this box) while KRK_GPU_TARGETS says something else. Only one
#     of the two reaches --offload-arch.
#   * A failed ninja leaves the previous .exe in place, so the next run
#     silently executes the previous binary and can read as a pass.
#
# That last one is why freshness is asked of ninja rather than of mtimes:
# `ninja -n` re-derives pending work from the graph, so a build that died
# halfway reports as pending work instead of as up to date.
#
# Verdicts:
#   ok    consistent
#   WARN  a real difference that a rebuild cannot fix, printed not failed
#   FAIL  the binary is not what it claims to be
#
# Exit status is non-zero when anything failed, 2 when there is nothing to
# check (no build dir, no cache, no ninja file).
#
#   scripts/build_check.sh                  # build-hip
#   scripts/build_check.sh build-rel
#   KRK_TARGETS="kraken" scripts/build_check.sh build-cpu
#
# Which binaries are required to be current: KRK_TARGETS, or by default
# `kraken`, `kraken-inspect` and `kraken-tests` plus `kraken-bench` if this
# configuration builds it (it links krk_hip, so a CPU-only tree has no such
# target, and demanding one would be a false alarm). kraken-inspect is in the
# list because it answers loader verdicts from the same tables the loader
# uses: one built before a new dequantizer landed refused formats the engine
# could already run. A target named explicitly must exist.
#
# Environment: KRK_BUILD_DIR (or $1), KRK_TARGETS, KRK_NO_DEVICE=1 to skip the
# run-time device probe, NINJA to pick a different ninja,
# KRK_REFRESH_FINGERPRINT=1 to re-record the artifact hashes after an
# intentional manual staging (see section 6).
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR=${KRK_BUILD_DIR:-${1:-$ROOT/build-hip}}
NINJA=${NINJA:-ninja}

fails=0
warns=0
ok()   { printf 'ok    %-12s %s\n' "$1" "$2"; }
warn() { printf 'WARN  %-12s %s\n' "$1" "$2"; warns=$((warns + 1)); }
bad()  { printf 'FAIL  %-12s %s\n' "$1" "$2"; fails=$((fails + 1)); }

# Everything the engine prints is CRLF on Windows; a stray \r turns an exact
# match into a silent miss, which would read as a passing check.
strip() { tr -d '\r'; }

cache() {
    grep -m1 "^$1:" "$DIR/CMakeCache.txt" 2>/dev/null | cut -d= -f2- | strip
}

# Detect a linker-qualified target name (e.g. CXX_EXECUTABLE_LINKER__kraken).
# Ninja writes the link rule as "CXX_EXECUTABLE_LINKER__kraken.exe: ...",
# so "has_target" and "link_rule" must strip the toolchain prefix.
has_target() { printf '%s\n' "$graph" | grep -qE "^(.*__)?$1\\.exe:"; }
link_rule() {
    printf '%s\n' "$graph" | grep -m1E "^(.*__)?$1\\.exe:" | sed 's/^[^:]*: //; s/__.*//'
}

[ -d "$DIR" ] || { echo "no build directory: $DIR"; exit 2; }
[ -f "$DIR/CMakeCache.txt" ] || { echo "no CMakeCache.txt in $DIR - not a configured build"; exit 2; }
[ -f "$DIR/build.ninja" ] || { echo "no build.ninja in $DIR - not a Ninja build"; exit 2; }

# The build graph is the authority on what this configuration produces, and on
# which toolchain produced it: CMake names the link rule for a HIP executable
# HIP_EXECUTABLE_LINKER and a host-only one CXX_EXECUTABLE_LINKER.
graph=$("$NINJA" -C "$DIR" -t targets all 2>/dev/null | strip || true)
has_target() { printf '%s\n' "$graph" | grep -q "^$1\.exe:"; }
link_rule() {
    printf '%s\n' "$graph" | grep -m1 "^$1\.exe:" | sed 's/^[^:]*: //; s/__.*//'
}

if [ -n "${KRK_TARGETS:-}" ]; then
    TARGETS=$KRK_TARGETS
    for t in $TARGETS; do
        has_target "$t" || {
            echo "FAIL  targets      '$t' is not a target of $DIR"
            echo "          (KRK_TARGETS was set explicitly, so it is required)"
            exit 1
        }
    done
else
    TARGETS="kraken kraken-tests"
    has_target kraken-inspect && TARGETS="$TARGETS kraken-inspect"
    has_target kraken-bench && TARGETS="$TARGETS kraken-bench"
fi

echo "build dir: $DIR"
echo "targets  : $TARGETS"
echo

# ---------------------------------------------------------------------------
# 1. Is the binary on disk the build of the source on disk?
# ---------------------------------------------------------------------------
# Section 6 needs to know whether ninja considers this build finished, because
# it only records a fresh fingerprint when there is nothing left to compile.
up_to_date=0
missing=""
for t in $TARGETS; do
    [ -f "$DIR/$t.exe" ] || [ -f "$DIR/$t" ] || missing="$missing $t"
done
if [ -n "$missing" ]; then
    bad "artifacts" "not built:$missing"
else
    ninja_out=$("$NINJA" -C "$DIR" -n $TARGETS 2>&1)
    ninja_rc=$?
    if [ "$ninja_rc" -ne 0 ]; then
        bad "up-to-date" "ninja could not even plan the build (rc=$ninja_rc); the graph is broken:"
        printf '%s\n' "$ninja_out" | strip | sed 's/^/          /' | head -5
    elif printf '%s\n' "$ninja_out" | strip | grep -q "no work to do"; then
        up_to_date=1
        ok "up-to-date" "ninja has no pending work for: $TARGETS"
    else
        pending=$(printf '%s\n' "$ninja_out" | strip | grep '^\[[0-9]' || true)
        n_pending=$(printf '%s\n' "$pending" | grep -c . || true)
        n_gen=$(printf '%s\n' "$pending" | grep -c 'Re-running CMake' || true)
        if [ "$n_pending" -gt 0 ] && [ "$n_pending" -eq "$n_gen" ]; then
            # A pending reconfigure is not a stale binary: ninja re-derives the
            # build before it compiles anything. It still means something moved
            # underneath, so it is reported rather than passed over.
            warn "up-to-date" "ninja will re-run CMake first ($n_pending step(s)); the binaries"
            printf '%s\n' "          themselves are up to date. CMakeCache.txt or a CMakeLists.txt"
            printf '%s\n' "          changed since the last build."
        else
            bad "up-to-date" "$n_pending step(s) still pending for {$TARGETS}: the binaries are"
            printf '%s\n' "          older than the sources, or the last build failed and left the"
            printf '%s\n' "          previous .exe in place. A run right now executes that binary."
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 2. Optimised, and the flags say so.
# ---------------------------------------------------------------------------
build_type=$(cache CMAKE_BUILD_TYPE)
ndebug=$(grep -c -- '-DNDEBUG' "$DIR/build.ninja" 2>/dev/null || true)
o0=$(grep -c -- '-O0' "$DIR/build.ninja" 2>/dev/null || true)
ndebug=${ndebug:-0}
o0=${o0:-0}

# Per-translation-unit proof. A grep of build.ninja counts a flag wherever it
# appears, including in rules this configuration never runs, so it can pass
# while a TU is compiled without it. `ninja -t commands` expands the rule
# variables and prints the command line each TU is actually handed, so this
# counts compiles: filter to lines carrying ' -c ' (the ar/archive step also
# mentions .cpp, as .obj arguments) and to ones naming a .cpp.
n_tu=0
n_tu_ndebug=0
for t in $TARGETS; do
    tu_cmds=$("$NINJA" -C "$DIR" -t commands "$t.exe" 2>/dev/null | strip || true)
    [ -n "$tu_cmds" ] || tu_cmds=$("$NINJA" -C "$DIR" -t commands "$t" 2>/dev/null | strip || true)
    [ -n "$tu_cmds" ] || continue
    tus=$(printf '%s
' "$tu_cmds" | grep -- ' -c ' | grep -- '\.cpp' || true)
    [ -n "$tus" ] || continue
    c=$(printf '%s
' "$tus" | grep -c . || true)
    m=$(printf '%s
' "$tus" | grep -c -- '-DNDEBUG' || true)
    n_tu=$(( n_tu + ${c:-0} ))
    n_tu_ndebug=$(( n_tu_ndebug + ${m:-0} ))
done

if [ -z "$build_type" ]; then
    bad "build-type" "CMAKE_BUILD_TYPE is empty; the configure never picked one"
elif [ "$build_type" = "Debug" ]; then
    bad "build-type" "CMAKE_BUILD_TYPE=Debug. Pass -DCMAKE_BUILD_TYPE=Release (a bare"
    printf '%s\n' "          configure can land here: CMake's Windows platform modules default"
    printf '%s\n' "          it to Debug before this project's own default is consulted)."
elif [ "$ndebug" -eq 0 ]; then
    bad "build-type" "$build_type, but no -DNDEBUG in build.ninja: assertions are live"
elif [ "$o0" -ne 0 ]; then
    bad "build-type" "$build_type, but build.ninja passes -O0"
elif [ "$n_tu" -gt 0 ] && [ "$n_tu_ndebug" -ne "$n_tu" ]; then
    bad "build-type" "$build_type, but only $n_tu_ndebug of $n_tu compiled .cpp files"
    printf '%s
' "          are handed -DNDEBUG. At least one is built with assertions live:"
    for t in $TARGETS; do
        tu_cmds=$("$NINJA" -C "$DIR" -t commands "$t.exe" 2>/dev/null | strip || true)
        [ -n "$tu_cmds" ] || continue
        printf '%s
' "$tu_cmds" | grep -- ' -c ' | grep -- '\.cpp' |
            grep -v -- '-DNDEBUG' | sed 's/^/            /' | cut -c1-160
    done
else
    ok "build-type" "$build_type, -DNDEBUG in all $n_tu compiled .cpp TU(s), no -O0"
fi

# ---------------------------------------------------------------------------
# 3. Is the HIP backend in the binary, or only in the cache?
# ---------------------------------------------------------------------------
hip_wanted=$(cache KRK_ENABLE_HIP)
gpu_targets=$(cache KRK_GPU_TARGETS)
hip_bin="$DIR/kraken.exe"
[ -f "$hip_bin" ] || hip_bin="$DIR/kraken"

# The HIP language was enabled only if the compiler detection ran and wrote its
# result, and the ROCm root in that file must carry forward slashes: a
# backslash there is the "Invalid character escape '\R'" failure, one level
# below where it bites.
hipc=$(ls "$DIR"/CMakeFiles/*/CMakeHIPCompiler.cmake 2>/dev/null | head -1)
rocm_root=""
if [ -n "$hipc" ]; then
    rocm_root=$(grep -m1 'CMAKE_HIP_COMPILER_ROCM_ROOT' "$hipc" | strip |
                    sed 's/.*"\(.*\)".*/\1/')
    if printf '%s' "$rocm_root" | grep -q '\\'; then
        bad "hip-detected" "ROCm root in $(basename "$hipc") contains a backslash:" 
        printf '%s\n' "          $rocm_root"
        printf '%s\n' "          That string is written verbatim into the generated file and will"
        printf '%s\n' "          not parse. Reconfigure with -DKRK_ROCM_ROOT=<forward/slashes>."
    fi
fi

simple_bin=""
for t in $TARGETS; do
    [ "$t" = "kraken" ] && simple_bin=1
done

if [ "$hip_wanted" = "OFF" ]; then
    if [ "$simple_bin" = "1" ] && [ -f "$hip_bin" ] && grep -aq amdhip64 "$hip_bin"; then
        bad "hip-backend" "KRK_ENABLE_HIP=OFF but $(basename "$hip_bin") links the HIP runtime"
    else
        ok "hip-backend" "KRK_ENABLE_HIP=OFF: CPU reference build, as asked"
    fi
elif [ -z "$hipc" ]; then
    bad "hip-backend" "KRK_ENABLE_HIP=$hip_wanted but no CMakeHIPCompiler.cmake: HIP was"
    printf '%s\n' "          never enabled, so this binary has no GPU path in it."
elif [ "$simple_bin" = "1" ] && [ -f "$hip_bin" ] && ! grep -aq amdhip64 "$hip_bin"; then
    bad "hip-backend" "KRK_ENABLE_HIP=$hip_wanted and HIP was enabled, but $(basename "$hip_bin")"
    printf '%s\n' "          does not link the HIP runtime. This is the CPU-only fallback."
else
    rule=$(link_rule kraken)
    case "$rule" in
        HIP_*) ok "hip-backend" "KRK_ENABLE_HIP=$hip_wanted, HIP enabled, kraken.exe links"
               printf '%s\n' "          amdhip64 and its build rule is $rule" ;;
        "")    ok "hip-backend" "KRK_ENABLE_HIP=$hip_wanted, HIP enabled, kraken.exe links amdhip64" ;;
        *)     bad "hip-backend" "HIP was enabled but kraken.exe's build rule is $rule, so it was"
               printf '%s\n' "          compiled by the host toolchain, not the HIP one." ;;
    esac
fi

# Negative control: the CPU-only suite links krk_core alone, so it must NOT
# carry the runtime. If it does, this probe is not measuring what it claims
# and every verdict above it is worthless.
ct="$DIR/kraken-tests.exe"
[ -f "$ct" ] || ct="$DIR/kraken-tests"
if [ -f "$ct" ] && grep -aq amdhip64 "$ct"; then
    warn "probe" "kraken-tests links amdhip64, so it is no longer the CPU-only control"
fi

# ---------------------------------------------------------------------------
# 4. Does the offload arch match the request, and the card in the machine?
# ---------------------------------------------------------------------------
requested=$(printf '%s' "$gpu_targets" | tr ';' ' ' | tr -s ' ')
actual=$(grep -o -- '--offload-arch=[a-z0-9]*' "$DIR/build.ninja" 2>/dev/null |
             sed 's/.*=//' | sort -u | tr '\n' ' ' | sed 's/ *$//')
cached_arch=$(cache CMAKE_HIP_ARCHITECTURES)

if [ "$hip_wanted" = "OFF" ]; then
    ok "gpu-target" "not applicable (KRK_ENABLE_HIP=OFF)"
elif [ -z "$actual" ]; then
    bad "gpu-target" "KRK_GPU_TARGETS='$gpu_targets' but no --offload-arch in build.ninja"
elif [ "$actual" != "$(printf '%s' "$requested" | tr -s ' ')" ]; then
    bad "gpu-target" "KRK_GPU_TARGETS='$requested' but the compiler was given '$actual'"
else
    ok "gpu-target" "KRK_GPU_TARGETS=$actual"
fi

if [ -n "$cached_arch" ] && [ "$cached_arch" != "$gpu_targets" ] && [ "$hip_wanted" != "OFF" ]; then
    warn "gpu-target" "CMAKE_HIP_ARCHITECTURES=$cached_arch disagrees with KRK_GPU_TARGETS="
    printf '%s\n' "          '$gpu_targets'. CMakeLists.txt overrides it, so '$actual' is what"
    printf '%s\n' "          was compiled, but the cache is stale - reconfigure if in doubt."
fi

# ---------------------------------------------------------------------------
# 5. What the machine actually presents, and is it one we built for?
# ---------------------------------------------------------------------------
if [ "${KRK_NO_DEVICE:-0}" = "1" ]; then
    ok "device" "skipped (KRK_NO_DEVICE=1)"
elif [ "$hip_wanted" = "OFF" ] || [ ! -f "$hip_bin" ]; then
    ok "device" "skipped (no HIP build to probe)"
else
    # The binary can answer two separate questions on one invocation. The
    # HIP device line is produced before any model is opened, so an absent
    # model path is a cheap way to get it without downloading.
    #
    # A run can also hit a linker-approved-but-unsupported target (for example a
    # flat-rocm configure that cached CMAKE_HIP_ARCHITECTURES=gfx906, so the
    # binary literally has no kernels for the card in the machine). That is a
    # HIPAA device and cannot be detected from the cache alone, so keep the
    # runtime probe: a device line with a gfx arch not in KRK_GPU_TARGETS fails
    # here rather than pretending the build was correct.
    probe=$("$hip_bin" -m "$DIR/__build_check_has_no_model__.gguf" --info 2>&1 | strip)
    line=$(printf '%s\n' "$probe" | grep -m1 '^\[info \] HIP' || true)
    dev_arch=$(printf '%s' "$line" | grep -o 'gfx[0-9a-z]*' | head -1)
    dev_name=$(printf '%s' "$line" | awk -F'|' '{ print $2 }' | sed 's/^ *//;s/ *$//')
    if [ -z "$line" ]; then
        bad "device" "the binary did not announce a HIP device. Either no HIP runtime is"
        printf '%s\n' "          loadable next to it, or it is the CPU-only build. Output was:"
        printf '%s\n' "$probe" | head -3 | sed 's/^/          /'
    elif [ -z "$dev_arch" ]; then
        bad "device" "device line has no gfx arch: $line"
    else
        case " $requested " in
            *" $dev_arch "*)
                ok "device" "$dev_name ($dev_arch) is in KRK_GPU_TARGETS" ;;
            *)
                bad "device" "$dev_name reports $dev_arch, which is NOT in KRK_GPU_TARGETS="
                printf '%s\n' "          '$requested'. This binary has no kernels for the card it is"
                printf '%s\n' "          running on. Rebuild with -DKRK_GPU_TARGETS=$dev_arch." ;;
        esac
    fi
    if printf '%s\n' "$probe" | grep -q "not the SDK this binary was built against"; then
        warn "runtime" "running a different HIP runtime than this build was linked to:"
        printf '%s\n' "          $(printf '%s\n' "$probe" | grep -m1 'not the SDK')"
        printf '%s\n' "          Stage the SDK's amdhip64_*.dll, amd_comgr.dll and rocm_kpack.dll"
        printf '%s\n' "          next to the binary to pin it. Measurement made under a different"
        printf '%s\n' "          runtime is not a measurement of this build."
    fi
fi

# ---------------------------------------------------------------------------
# 6. Was the binary on disk produced by this build, or swapped in from outside?
# ---------------------------------------------------------------------------
# Every check above compares the build GRAPH, so a binary replaced outside ninja
# -- an older .exe copied over this one, a release zip extracted on top, a build
# from a different source tree -- passes all of them, and a run then executes
# code that was never built here. This is the only check that looks at the
# artifact's bytes rather than at what the build intended to produce.
#
# Cost: one hash per required binary plus its size. These are tens of MB, so
# this is milliseconds on the reference box; the size is recorded beside the
# hash so a mismatch is readable without rehashing. If the targets ever grow to
# gigabytes this needs revisiting, and that is the reason it is stated here.
#
# Re-record deliberately (after staging a binary by hand) with
# KRK_REFRESH_FINGERPRINT=1.
FP="$DIR/.artifact-sha256"
if command -v sha256sum >/dev/null 2>&1; then
    HASHKIND=sha256
    hashof() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
elif command -v md5sum >/dev/null 2>&1; then
    HASHKIND=md5
    hashof() { md5sum "$1" 2>/dev/null | cut -d' ' -f1; }
else
    HASHKIND=""
    hashof() { echo ""; }
fi

# The binary a target name produces: `kraken` -> kraken.exe (or plain kraken).
fp_bin() { b="$DIR/$1.exe"; [ -f "$b" ] || b="$DIR/$1"; printf '%s' "$b"; }

fp_tmp="$DIR/.artifact-sha256.tmp"
: > "$fp_tmp"
for t in $TARGETS; do
    b=$(fp_bin "$t")
    [ -f "$b" ] || continue
    printf '%s %s:%s %s\n' "$t" "$HASHKIND" "$(hashof "$b")" \
        "$(wc -c < "$b" | tr -d ' ')" >> "$fp_tmp"
done
n_fp=$(grep -c . "$fp_tmp" 2>/dev/null || true)
n_fp=${n_fp:-0}

# Which algorithm the existing record used. Comparing a sha256 against an md5
# record would read as every binary having been swapped, which is exactly the
# kind of false alarm that trains a reader to ignore the check.
rec_kind=""
[ -f "$FP" ] && rec_kind=$(awk 'NR == 1 { n = split($2, a, ":"); if (n > 1) print a[1] }' \
                                   "$FP" 2>/dev/null)

if [ -n "$missing" ]; then
    ok "fingerprint" "skipped: an artifact is missing (reported above)"
elif [ -z "$HASHKIND" ]; then
    warn "fingerprint" "neither sha256sum nor md5sum is available, so the binaries"
    printf '%s\n' "          cannot be tied to this build"
elif [ -n "${KRK_REFRESH_FINGERPRINT:-}" ] && [ "${KRK_REFRESH_FINGERPRINT}" != "0" ]; then
    mv -f "$fp_tmp" "$FP"
    ok "fingerprint" "re-recorded $n_fp artifact(s) (KRK_REFRESH_FINGERPRINT)"
elif [ ! -f "$FP" ]; then
    if [ "$up_to_date" = "1" ]; then
        mv -f "$fp_tmp" "$FP"
        ok "fingerprint" "no record existed; recorded $n_fp artifact(s) for the next run"
    else
        warn "fingerprint" "no record exists and ninja still has pending work, so"
        printf '%s\n' "          nothing is recorded yet; re-run after a clean build"
    fi
elif [ -n "$rec_kind" ] && [ "$rec_kind" != "$HASHKIND" ]; then
    mv -f "$fp_tmp" "$FP"
    ok "fingerprint" "record was $rec_kind and this run uses $HASHKIND; re-recorded"
else
    # A target the record does not cover is not a swap - it is a stale record
    # (KRK_TARGETS changed, or a new binary was added). Reported and refreshed
    # rather than skipped, because a silent skip is a hole in the only check
    # that looks at the bytes.
    swapped=""
    not_rec=""
    for t in $TARGETS; do
        b=$(fp_bin "$t")
        [ -f "$b" ] || continue
        want=$(awk -v t="$t" '$1 == t { print $2; exit }' "$FP" 2>/dev/null)
        if [ -z "$want" ]; then
            not_rec="$not_rec $t"
            continue
        fi
        [ "$want" = "$HASHKIND:$(hashof "$b")" ] || swapped="$swapped $t"
    done
    if [ -n "$not_rec" ]; then
        mv -f "$fp_tmp" "$FP"
        ok "fingerprint" "record did not cover:$not_rec (target list changed?)"
        printf '%s\n' "          re-recorded $n_fp artifact(s); run again to verify them"
    elif [ -n "$swapped" ]; then
        bad "fingerprint" "the binary on disk was NOT produced by this build:$swapped"
        for t in $swapped; do
            b=$(fp_bin "$t")
            want=$(awk -v t="$t" '$1 == t { print $2 }' "$FP" 2>/dev/null)
            got="$HASHKIND:$(hashof "$b")"
            printf '%s\n' "          $t: recorded ${want#*:}"
            printf '%s\n' "          $t: on disk  ${got#*:}"
        done
        printf '%s\n' "          Something replaced it outside ninja - an older build, an extracted"
        printf '%s\n' "          zip, a copy from another tree. A run right now executes that"
        printf '%s\n' "          binary, not this source. Rebuild it, or re-record deliberately"
        printf '%s\n' "          with KRK_REFRESH_FINGERPRINT=1 if the replacement was intended."
    else
        ok "fingerprint" "all $n_fp artifact(s) match the recorded build ($HASHKIND)"
    fi
fi
rm -f "$fp_tmp"

echo
printf '%d failed, %d warning(s)\n' "$fails" "$warns"
[ "$fails" -eq 0 ]
