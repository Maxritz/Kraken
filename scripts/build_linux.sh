#!/bin/sh
# ============================================================================
#  build_linux.sh — build kraken on Linux (and macOS, for the CPU backend).
#
#  Hardware priority: CPU oracle first, then RX 6700 XT (gfx1031, RDNA2 —
#  the primary GPU target), then RX 9070 XT (gfx1201, RDNA4).
#
#    sh ./scripts/build_linux.sh                       # auto targets
#    sh ./scripts/build_linux.sh gfx1031               # RX 6700 XT only
#    KRK_TARGETS="gfx1031 gfx1201" sh ./scripts/build_linux.sh
#    sh ./scripts/build_linux.sh --cpu-only
#
#  Uses CMake when it is installed, and falls back to a direct hipcc build so
#  the engine can be built on a machine with only the ROCm runtime.
# ============================================================================
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

targets="${KRK_TARGETS:-gfx1031 gfx1201}"
cpu_only=0
for arg in "$@"; do
    case "$arg" in
        --cpu-only) cpu_only=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) targets="$arg" ;;
    esac
done

if command -v hipcc >/dev/null 2>&1; then
    have_hip=1
else
    have_hip=0
fi

if [ "$cpu_only" -eq 0 ] && [ "$have_hip" -eq 0 ]; then
    echo "kraken: hipcc not found; building the CPU reference backend only."
    echo "        (install ROCm, or pass --cpu-only to silence this)"
    cpu_only=1
fi

arch_flags=""
for t in $targets; do arch_flags="$arch_flags --offload-arch=$t"; done

if [ "$cpu_only" -eq 1 ]; then
    echo "kraken: CPU-only build"
    cxx="${CXX:-g++}"
    mkdir -p build-cpu
    api="src/json.cpp src/http.cpp src/server.cpp"
    $cxx -std=c++17 -O3 -Wall -Wextra -Iinclude \
        src/common.cpp src/quant.cpp src/gguf.cpp src/tokenizer.cpp \
        src/sampler.cpp src/model.cpp src/expert_cache.cpp src/engine.cpp \
        src/backend_cpu.cpp $api \
        src/hip/hip_stub.cpp src/main_cli.cpp \
        -o build-cpu/kraken
    $cxx -std=c++17 -O3 -Wall -Wextra -Iinclude \
        src/common.cpp src/quant.cpp src/gguf.cpp src/tokenizer.cpp \
        src/sampler.cpp src/model.cpp src/expert_cache.cpp src/engine.cpp \
        src/backend_cpu.cpp $api \
        src/hip/hip_stub.cpp src/main_server.cpp \
        -o build-cpu/kraken-server
    $cxx -std=c++17 -O3 -Iinclude tools/inspect_gguf.cpp \
        src/common.cpp src/quant.cpp src/gguf.cpp \
        -o build-cpu/kraken-inspect
    $cxx -std=c++17 -O3 -Iinclude tests/test_kraken.cpp \
        src/common.cpp src/quant.cpp src/gguf.cpp src/tokenizer.cpp \
        src/sampler.cpp src/model.cpp src/expert_cache.cpp src/engine.cpp \
        src/backend_cpu.cpp $api \
        src/hip/hip_stub.cpp -o build-cpu/kraken-tests
    echo "kraken: build-cpu/{kraken,kraken-server,kraken-inspect,kraken-tests} built"
    exit 0
fi

echo "kraken: HIP build for [$targets]"
mkdir -p build-hip/obj

# Two flag sets, because there are two kinds of code here. The scalar/core
# translation units are the reference backend -- the oracle every coherence
# check is measured against -- and they get the same flags CMake gives
# krk_core: -O3 and warnings, and deliberately NOT -ffast-math. -ffast-math
# implies -fftz, which flushes denormals to zero, and the MXFP4 denormal
# exponent ladder asserts a 2^-128 scale survives (test_kraken.cpp:529);
# building it with -ffast-math fails that check for reasons that have nothing
# to do with any model. Only the device offload TUs get the fast-math
# relaxation, which is where the speed actually comes from.
#
# -march=native is not a tuning choice, it is a correctness one: CMake puts it
# on krk_core for the same reason, so the scalar reference and its pooled
# variant compile with identical flags and stay bit-identical. Without it the
# MXFP4 denormal ladder (test_kraken.cpp:529) fails, because the denormal scale
# it checks does not survive the baseline codegen.
core_flags="-std=c++17 -O3 -march=native"
flags="$core_flags -ffast-math -fno-finite-math-only"
objs=""

# arch/verdict carry the architecture table and the runnable-model verdict;
# dflash carries the drafter. engine.cpp references all three, so omitting
# any of them from this list fails at link with undefined symbols rather
# than at configure, which is the expensive way to find out.
for src in common quant gguf tokenizer sampler arch verdict dflash model expert_cache engine backend_cpu json http server; do
    hipcc $arch_flags $core_flags -Iinclude -c "src/$src.cpp" -o "build-hip/obj/$src.o"
    objs="$objs build-hip/obj/$src.o"
done

hipcc $arch_flags $flags -Iinclude -Isrc -DKRK_ENABLE_HIP=1 \
      -c src/hip/backend_hip.hip -o build-hip/obj/backend_hip.o
hipcc $arch_flags $flags -Iinclude -Isrc -DKRK_ENABLE_HIP=1 \
      -c src/main_cli.cpp -o build-hip/obj/main_cli.o
hipcc $arch_flags $flags $objs build-hip/obj/backend_hip.o build-hip/obj/main_cli.o \
      -o build-hip/kraken
hipcc $arch_flags $flags -Iinclude -Isrc -DKRK_ENABLE_HIP=1 \
      -c src/main_server.cpp -o build-hip/obj/main_server.o
hipcc $arch_flags $flags $objs build-hip/obj/backend_hip.o build-hip/obj/main_server.o \
      -o build-hip/kraken-server

hipcc $arch_flags $flags -Iinclude -c tools/inspect_gguf.cpp -o build-hip/obj/inspect.o
# kraken-tests links krk_core only -- no backend_hip.o -- because that is what
# CMakeLists.txt:339 does for this target and the two have to agree. Linking the
# HIP backend into the test process changes the host scalar reference: the same
# quant.cpp passes the MXFP4 denormal ladder (test_kraken.cpp:529) in the CMake
# build and fails it here, 2152/2154 against 2154/2154. The mechanism is not
# known -- it is not FTZ/DAZ, which stay clear after a HIP init -- but a test
# binary that disagrees with the project's own test target is measuring
# something other than the engine, so link what CMake links. hip_stub stands in
# for the backend here, as the --cpu-only branch already does. The test is also
# compiled to an object first: handing hipcc the .cpp and the .o files together
# does not work, because hipcc injects `-x hip`, which applies to every following
# argument and so reads the object files as source.
hipcc $arch_flags $core_flags -Iinclude -c tests/test_kraken.cpp -o build-hip/obj/test_kraken.o
hipcc $arch_flags $core_flags -Iinclude $objs build-hip/obj/test_kraken.o \
      src/hip/hip_stub.cpp -o build-hip/kraken-tests
hipcc $arch_flags $flags $objs build-hip/obj/inspect.o -o build-hip/kraken-inspect

echo "kraken: build-hip/{kraken,kraken-server,kraken-inspect,kraken-tests} built"
