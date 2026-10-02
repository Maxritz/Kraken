# ============================================================================
#  KRAKEN — RDNA-native GGUF inference engine
#
#  Two configurations:
#    * KRK_ENABLE_HIP=ON  (default) — HIP backend + scalar reference backend.
#      Requires the ROCm HIP SDK (Linux) or the HIP SDK for Windows.
#    * KRK_ENABLE_HIP=OFF          — CPU-only reference build. Useful for CI and
#      for validating the host pipeline without a GPU.
#
#  GPU targets are selected with KRK_GPU_TARGETS. Build a fat binary covering
#  both cards in the fleet with:
#      cmake -DKRK_GPU_TARGETS="gfx1031;gfx1201" ..
#  One --offload-arch pass per target; each pass compiles the kernel headers
#  with that target's __gfx*__ macros, so no runtime dispatch is needed.
#
#  Hardware priority (see README): CPU oracle first, then RX 6700 XT (gfx1031,
#  RDNA2 — the primary GPU target), then RX 9070 XT (gfx1201, RDNA4). The
#  default target list is ordered accordingly.
# ============================================================================
cmake_minimum_required(VERSION 3.21)
project(kraken VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(KRK_ENABLE_HIP "Build the HIP/ROCm backend" ON)
option(KRK_BUILD_TESTS "Build the test suite" ON)
set(KRK_GPU_TARGETS "gfx1031;gfx1201"
    CACHE STRING "Semicolon-separated GPU targets, priority order (gfx1031 = RX 6700 XT RDNA2 primary, gfx1201 = RX 9070 XT RDNA4)")

# ---------------------------------------------------------------------------
# host-side core (always built, no HIP dependency)
# ---------------------------------------------------------------------------

set(KRK_CORE_SOURCES
    src/common.cpp
    src/arch.cpp
    src/verdict.cpp
    src/quant.cpp
    src/gguf.cpp
    src/tokenizer.cpp
    src/sampler.cpp
    src/model.cpp
    src/expert_cache.cpp
    src/engine.cpp
    src/backend_cpu.cpp
    src/json.cpp
    src/http.cpp
    src/server.cpp
)

add_library(krk_core STATIC ${KRK_CORE_SOURCES})
target_include_directories(krk_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(WIN32)
    target_link_libraries(krk_core PUBLIC ws2_32)
endif()

if(MSVC)
    target_compile_options(krk_core PRIVATE /W4 /permissive-)
else()
    # -march=native lets the scalar dequant/dot kernels use the
    # host's SIMD (AVX2/FMA on the dev box); the quantized GEMVs are
    # memory- and unpack-bound and gain several fold from it. Results
    # stay bit-identical between the serial and pooled paths because
    # both compile with the same flags; disable with
    # -DKRK_NATIVE_CPU=OFF for portable binaries.
    option(KRK_NATIVE_CPU "Build krk_core with -march=native" ON)
    target_compile_options(krk_core PRIVATE -O3 -Wall -Wextra -Wshadow
                                             -Wno-unused-parameter)
    if(KRK_NATIVE_CPU)
        target_compile_options(krk_core PRIVATE -march=native)
    endif()
endif()

# ---------------------------------------------------------------------------
# HIP backend
# ---------------------------------------------------------------------------

set(KRK_HIP_AVAILABLE OFF)
if(KRK_ENABLE_HIP)
    find_package(hip QUIET CONFIG)
    if(NOT hip_FOUND)
        find_package(HIP QUIET)
    endif()

    if(hip_FOUND OR HIP_FOUND)
        set(KRK_HIP_AVAILABLE ON)
        enable_language(HIP)

        # Normalize the target list into the form CMake wants.
        set(CMAKE_HIP_ARCHITECTURES "${KRK_GPU_TARGETS}")
        if(NOT CMAKE_HIP_COMPILER_ID MATCHES "Clang")
            message(WARNING "HIP compiler is ${CMAKE_HIP_COMPILER_ID}; the WMMA "
                            "builtins require amdclang++")
        endif()

        add_library(krk_hip STATIC src/hip/backend_hip.hip)
        target_include_directories(krk_hip PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(krk_hip PUBLIC krk_core)
        target_compile_definitions(krk_hip PUBLIC KRK_ENABLE_HIP=1)

        # amdclang defines the __gfx*__ macros only in the device pass
        # of an offload compile. The host pass must still compile the
        # kernel launch sites — a kernel that is #if'ed out of the host
        # pass never enters the fat binary, and launching it silently
        # does nothing. Mirror every target into a -DKRK_OFFLOAD_GFX*
        # definition so both passes make the same arch decision.
        foreach(tgt IN LISTS CMAKE_HIP_ARCHITECTURES)
            string(TOUPPER "${tgt}" tgt_uc)
            target_compile_definitions(krk_hip PRIVATE
                                       "KRK_OFFLOAD_${tgt_uc}=1")
        endforeach()
        target_compile_options(krk_hip PRIVATE -O3 -ffast-math
                                               -fno-finite-math-only)
        if(hip_FOUND AND TARGET hip::device)
            target_link_libraries(krk_hip PUBLIC hip::device)
        elseif(HIP_FOUND)
            target_include_directories(krk_hip PUBLIC ${HIP_INCLUDE_DIRS})
            target_link_libraries(krk_hip PUBLIC ${HIP_LIBRARIES})
        endif()
        message(STATUS "kraken: HIP backend enabled, targets = ${KRK_GPU_TARGETS}")
    else()
        message(STATUS "kraken: HIP SDK not found — building the CPU backend only")
    endif()
endif()

if(NOT KRK_HIP_AVAILABLE)
    target_sources(krk_core PRIVATE src/hip/hip_stub.cpp)
endif()

# ---------------------------------------------------------------------------
# executables
# ---------------------------------------------------------------------------

add_executable(kraken src/main_cli.cpp)
target_link_libraries(kraken PRIVATE krk_core)
if(KRK_HIP_AVAILABLE)
    target_link_libraries(kraken PRIVATE krk_hip)
    target_compile_definitions(kraken PRIVATE KRK_ENABLE_HIP=1)
endif()
if(WIN32)
    # `kraken --bench` reports which GPU engine actually ran the work by
    # sampling the per-process "GPU Engine" performance counters (PDH).
    target_link_libraries(kraken PRIVATE pdh)
endif()

add_executable(kraken-server src/main_server.cpp)
target_link_libraries(kraken-server PRIVATE krk_core)
if(KRK_HIP_AVAILABLE)
    target_link_libraries(kraken-server PRIVATE krk_hip)
    target_compile_definitions(kraken-server PRIVATE KRK_ENABLE_HIP=1)
endif()

add_executable(kraken-inspect tools/inspect_gguf.cpp)
target_link_libraries(kraken-inspect PRIVATE krk_core)

if(KRK_HIP_AVAILABLE)
    add_executable(kraken-oracle tools/oracle_probe.cpp)
    target_link_libraries(kraken-oracle PRIVATE krk_core krk_hip)
    # Checks the int8 DP4A weight staging (Q8_0/Q4_K/Q5_K/Q6_K) against the
    # host reference decoder, code position by code position. Two
    # implementations of the same block layout is where a silent mis-read
    # hides, so it is a test rather than a comment.
    add_executable(kraken-i8k tools/gemm_i8k_verify.cpp)
    target_link_libraries(kraken-i8k PRIVATE krk_core krk_hip)
    # The kernel headers are included as the backend includes them
    # ("kernels/..."), so the tool compiles the same file the engine does.
    target_include_directories(kraken-i8k PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/hip)
endif()

# ---------------------------------------------------------------------------
# diagnostic probes (opt-in)
#
# Standalone microbenchmarks of individual kernels, built through the same
# HIP toolchain and with the same flags as the engine, so their numbers are
# directly comparable to `kraken --bench`. Off by default; enable with
# -DKRK_BUILD_PROBES=ON.
# ---------------------------------------------------------------------------

option(KRK_BUILD_PROBES "Build the standalone kernel probes" OFF)
if(KRK_BUILD_PROBES AND KRK_HIP_AVAILABLE)
    foreach(probe probe_min probe_decode probe_engine graph_event_probe)
        add_executable(${probe} ${CMAKE_CURRENT_SOURCE_DIR}/${probe}.hip)
        target_include_directories(${probe} PRIVATE
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include
                                   ${CMAKE_CURRENT_SOURCE_DIR}/src
                                   ${CMAKE_CURRENT_SOURCE_DIR}/src/hip)
        # Probes are standalone: they include the kernel headers directly, so
        # linking krk_hip would duplicate every kernel stub.
        if(TARGET hip::device)
            target_link_libraries(${probe} PRIVATE hip::device)
        elseif(HIP_FOUND)
            target_link_libraries(${probe} PRIVATE ${HIP_LIBRARIES})
        endif()
        target_compile_options(${probe} PRIVATE -O3 -ffast-math -fno-finite-math-only)
        foreach(tgt IN LISTS CMAKE_HIP_ARCHITECTURES)
            string(TOUPPER "${tgt}" tgt_uc)
            target_compile_definitions(${probe} PRIVATE "KRK_OFFLOAD_${tgt_uc}=1")
        endforeach()
    endforeach()
endif()

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

if(KRK_BUILD_TESTS)
    enable_testing()
    add_executable(kraken-tests tests/test_kraken.cpp)
    target_link_libraries(kraken-tests PRIVATE krk_core)
    add_test(NAME kraken-tests COMMAND kraken-tests)
endif()

message(STATUS "kraken: build type ${CMAKE_BUILD_TYPE}")
