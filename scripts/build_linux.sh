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

flags="-std=c++17 -O3 -ffast-math -fno-finite-math-only"
objs=""

for src in common quant gguf tokenizer sampler model expert_cache engine backend_cpu json http server; do
    hipcc $arch_flags $flags -Iinclude -c "src/$src.cpp" -o "build-hip/obj/$src.o"
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
hipcc $arch_flags $flags -Iinclude tests/test_kraken.cpp $objs \
      build-hip/obj/backend_hip.o -o build-hip/kraken-tests
hipcc $arch_flags $flags $objs build-hip/obj/inspect.o -o build-hip/kraken-inspect

echo "kraken: build-hip/{kraken,kraken-server,kraken-inspect,kraken-tests} built"
