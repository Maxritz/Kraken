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
    src/dflash.cpp
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

        # ROCm device bitcode location. The compiler only searches
        # <rocm>/amdgcn/bitcode when ROCM_PATH (or --rocm-path) names the
        # install root. A relocated runtime pack on neither the default
        # prefix nor PATH fails every device compile with
        #   "cannot find ROCm device library; ... '-nogpulib'"
        # which is a hard error, not a warning.
        #
        # This must be applied globally, not just to krk_hip: CMake
        # compiles the executable targets' .cpp files with amdclang++ and
        # an injected `-x hip`, so they are device compiles too and they
        # fail identically. Compile *and* link options are needed because
        # the device-link step resolves the same libraries.
        set(_krk_devlib "")
        get_filename_component(_hip_bin "${CMAKE_HIP_COMPILER}" DIRECTORY)
        get_filename_component(_llvm_root "${_hip_bin}" DIRECTORY)   # .../lib/llvm
        get_filename_component(_rocm_root "${_llvm_root}" DIRECTORY) # .../lib
        foreach(_cand "${_rocm_root}/llvm/amdgcn/bitcode"
                    "${_llvm_root}/amdgcn/bitcode"
                    "${_rocm_root}/amdgcn/bitcode")
            if(EXISTS "${_cand}/oclc_abi_version_600.bc")
                set(_krk_devlib "${_cand}")
                break()
            endif()
        endforeach()
        # Second tier: the compiler's own pack can be present but hollow.
        # MEASURED on this host: G:/ROCM10RT-gfx1201/lib/llvm/amdgcn/bitcode
        # was emptied at 17:16 on 2025-10-05 while the tree above it stayed
        # intact, and every device compile then failed. The arch libraries
        # are not per-pack -- oclc_isa_version_1201.bc is identical in every
        # ROCm 10 pack -- so a sibling pack of the same product line is a
        # valid source. Prefer the same product prefix (everything up to the
        # first '-'), which also keeps the compiler's ROCm major version
        # matched to the bitcode it has to load.
        if(NOT _krk_devlib)
            get_filename_component(_pack_dir "${_rocm_root}" DIRECTORY)
            get_filename_component(_pack_base "${_pack_dir}" NAME)
            string(REGEX REPLACE "-.*$" "" _pack_prefix "${_pack_base}")
            file(GLOB _siblings "${_pack_dir}/../${_pack_prefix}*/lib/llvm/amdgcn/bitcode")
            foreach(_cand IN LISTS _siblings)
                if(EXISTS "${_cand}/oclc_isa_version_600.bc")
                    set(_krk_devlib "${_cand}")
                    message(WARNING "kraken: ${_pack_dir} has no device "
                                    "bitcode; using sibling pack ${_cand}")
                    break()
                endif()
            endforeach()
        endif()
        if(_krk_devlib)
            add_compile_options(--rocm-device-lib-path=${_krk_devlib})
            add_link_options(--rocm-device-lib-path=${_krk_devlib})
            message(STATUS "kraken: ROCm device bitcode = ${_krk_devlib}")
        else()
            message(WARNING "kraken: ROCm device bitcode not found next to "
                            "${CMAKE_HIP_COMPILER}; device compiles may fail")
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

# Sizes the CPU expert fallback (spec section 22 / Fiddler) before it is
# built: how fast can host threads compute one cold expert from its WARM
# copy, against the measured cost of moving it. CPU-only, links krk_core
# alone, so it can be run without the GPU being busy.
add_executable(kraken-cpu-expert-ceiling tools/cpu_expert_ceiling.cpp)
target_link_libraries(kraken-cpu-expert-ceiling PRIVATE krk_core)

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
    # Per-shape raw kernel speed with the launch cost amortized out of the
    # measurement: `(T(N) - T(1)) / (N - 1)`. This is the number the perf plan's
    # acceptance criteria are written against, because the engine's own per-op
    # timer cannot see below the ~10-27 µs an event record costs.
    add_executable(kraken-bench tools/bench_ops.cpp)
    target_link_libraries(kraken-bench PRIVATE krk_core krk_hip)

    # The numeric gate. Exits non-zero on NaN, on inf, on a value outside the
    # dtype's tolerance, or on a logits_topk mismatch, so `ninja -C build-hip
    # gate` fails rather than printing a reassuring number. It exists because
    # this harness previously scored a NaN-producing Q6_K kernel as a perfect
    # chk=0 and a hand-written v_perm_b32 helper's garbage went unnoticed:
    # a fast wrong kernel is the failure mode a code review does not catch.
    add_custom_target(gate
        COMMAND $<TARGET_FILE:kraken-bench> --gate
        DEPENDS kraken-bench
        COMMENT "numeric gate: dtype tolerances + logits_topk differential")
    if(KRK_BUILD_TESTS)
        add_test(NAME kraken-gate COMMAND kraken-bench --gate)
    endif()
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

    # probe_attn_qtile lives in tools/ rather than the repo root because it is
    # the one probe that has to stay version-controlled: it is the A/B that
    # reproduces the attention_qtile_kernel correctness question, and .gitignore
    # drops everything at the root. Built the same way, from a different path.
    add_executable(probe_attn_qtile
                   ${CMAKE_CURRENT_SOURCE_DIR}/tools/probe_attn_qtile.hip)
    target_include_directories(probe_attn_qtile PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR}/include
                               ${CMAKE_CURRENT_SOURCE_DIR}/src
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/hip)
    if(TARGET hip::device)
        target_link_libraries(probe_attn_qtile PRIVATE hip::device)
    elseif(HIP_FOUND)
        target_link_libraries(probe_attn_qtile PRIVATE ${HIP_LIBRARIES})
    endif()
    target_compile_options(probe_attn_qtile PRIVATE
                           -O3 -ffast-math -fno-finite-math-only)
    foreach(tgt IN LISTS CMAKE_HIP_ARCHITECTURES)
        string(TOUPPER "${tgt}" tgt_uc)
        target_compile_definitions(probe_attn_qtile PRIVATE
                                   "KRK_OFFLOAD_${tgt_uc}=1")
    endforeach()

    # probe_gemv answers a question the decode trace cannot: the trace's device
    # span includes launch-to-arrival idle, so 35 GB/s there does not separate
    # "the kernel is slow" from "the op is starved". This runs the same kernel
    # back to back at the decode shapes and at oversized ones, which separates
    # kernel quality from shape/geometry.
    add_executable(probe_gemv
                   ${CMAKE_CURRENT_SOURCE_DIR}/tools/probe_gemv.hip)
    target_include_directories(probe_gemv PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR}/include
                               ${CMAKE_CURRENT_SOURCE_DIR}/src
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/hip)
    if(TARGET hip::device)
        target_link_libraries(probe_gemv PRIVATE hip::device)
    elseif(HIP_FOUND)
        target_link_libraries(probe_gemv PRIVATE ${HIP_LIBRARIES})
    endif()
    target_compile_options(probe_gemv PRIVATE
                           -O3 -ffast-math -fno-finite-math-only)
    foreach(tgt IN LISTS CMAKE_HIP_ARCHITECTURES)
        string(TOUPPER "${tgt}" tgt_uc)
        target_compile_definitions(probe_gemv PRIVATE
                                   "KRK_OFFLOAD_${tgt_uc}=1")
    endforeach()

    # probe_sync prices the host-side operations a decode token performs 40
    # times over: a launch, a device barrier, and a 512 B D2H copy of the
    # router row. probe_gemv says the arithmetic is cheap; this says what the
    # round trip costs, at the queue depths the engine actually has.
    add_executable(probe_sync
                   ${CMAKE_CURRENT_SOURCE_DIR}/tools/probe_sync.hip)
    target_include_directories(probe_sync PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR}/include
                               ${CMAKE_CURRENT_SOURCE_DIR}/src
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/hip)
    if(TARGET hip::device)
        target_link_libraries(probe_sync PRIVATE hip::device)
    elseif(HIP_FOUND)
        target_link_libraries(probe_sync PRIVATE ${HIP_LIBRARIES})
    endif()
    target_compile_options(probe_sync PRIVATE
                           -O3 -ffast-math -fno-finite-math-only)
    foreach(tgt IN LISTS CMAKE_HIP_ARCHITECTURES)
        string(TOUPPER "${tgt}" tgt_uc)
        target_compile_definitions(probe_sync PRIVATE
                                   "KRK_OFFLOAD_${tgt_uc}=1")
    endforeach()

    # probe_promote measures the copy a decode token cannot avoid: 347 MiB of
    # cold expert weights, pageable host source to VRAM. The shipped path keeps
    # exactly one of those copies in flight and host-waits for each, which is
    # where the ~89 ms/token of expert time goes; this prices the in-flight
    # depths that would recover it.
    add_executable(probe_promote
                   ${CMAKE_CURRENT_SOURCE_DIR}/tools/probe_promote.hip)
    target_include_directories(probe_promote PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR}/include
                               ${CMAKE_CURRENT_SOURCE_DIR}/src
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/hip)
    if(TARGET hip::device)
        target_link_libraries(probe_promote PRIVATE hip::device)
    elseif(HIP_FOUND)
        target_link_libraries(probe_promote PRIVATE ${HIP_LIBRARIES})
    endif()
    target_compile_options(probe_promote PRIVATE
                           -O3 -ffast-math -fno-finite-math-only)
    foreach(tgt IN LISTS CMAKE_HIP_ARCHITECTURES)
        string(TOUPPER "${tgt}" tgt_uc)
        target_compile_definitions(probe_promote PRIVATE
                                   "KRK_OFFLOAD_${tgt_uc}=1")
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
