#!/usr/bin/env python3
"""Author and compile a ROCm kernel in-session, and hand back a loadable code object.

This is the in-session proof of the tilelang ROCm kernel-authoring path, plus the
artifact contract an offline kernel swap needs. It answers, in one run:

  1. is the interpreter we were started with the one the dev build supports, and
     which tvm / tvm_ffi / native lib does the import chain resolve?
  2. can `tilelang.rocm.language.kernel.Kernel` and
     `tilelang.contrib.hipcc.compile_hip` compile a kernel for THIS card to an
     HSACO? (arch from KRK_GPU_TARGETS, else tilelang's detected target)
  3. is the result directly loadable by hipModuleLoadData -- i.e. a raw ELF code
     object with EM_AMDGPU and the kernel symbol in it?

Two facts this tool exists to encode, both measured (see sherlock_report.md):

  * **The interpreter, not the package, was the blocker.** `python3` on this box
    is 3.14 (pythoncore-3.14-64) whose usersite has no `tvm_ffi`, while the
    tilelang dev build ships `cp312` extension modules -- so no 3.14 interpreter
    can load it even with tvm_ffi installed. `python` / `py -3.12` is 3.12.10
    (Program Files) and has `apache-tvm-ffi 0.1.12`; that is the one that works.
    This tool therefore checks the pyd tag against the running interpreter and, if
    they disagree, re-execs itself under an interpreter that fits (`KRK_PYTHON`
    overrides the search).
  * **`--genco` output is a clang offload bundle, not a loadable code object.** On
    this ROCm (clang 23) `hipcc --genco` and `compile_hip` both emit
    `__CLANG_OFFLOAD_BUNDLE__` holding a `hipv4-amdgcn-amd-amdhsa--gfx1201` device
    payload plus a host stub. A HIP loader cannot take that file, so the device
    payload is extracted here with clang-offload-bundler and then verified.

  A third, cheaper trap: the source MUST include <hip/hip_runtime.h> and
  <hip/hip_fp16.h>. Without them clang ignores `__global__` as an unknown
  attribute and `__half` / `threadIdx` / `blockIdx` are undeclared -- which is the
  exact failure the previous version of this file produced. kraken's own
  src/hip/krk_hip.hpp:24 includes exactly this pair.

Usage:
    python tools/tilelang_hipcc_repro.py [--out DIR] [--arch gfx1201] [--keep-src]
    python tools/tilelang_hipcc_repro.py --no-tilelang    # direct hipcc, same checks

Exit status: 0 = every check passed, 2 = a check failed, 3 = setup missing.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import struct
import subprocess
import sys
import tempfile

KERNEL_NAME = "kraken_probe_gemv_q4"

# The prelude is the whole lesson: kraken's device code is written against these
# two headers, and clang needs them before it will accept __global__/__half at all.
SRC_TEXT = f"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Decode-side GEMV probe: one warp per output row, lane-strided K walk, warp
// shuffle reduce -- the same geometry as kraken's gemv_kernel<256>, so what this
// compiles stands in for the body a swap would replace.
extern "C" __global__ void __launch_bounds__(256) {KERNEL_NAME}(
    const unsigned char* __restrict__ w,
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int n_out)
{{
    const int lane = (int)threadIdx.x & 31;
    const int row  = (int)blockIdx.x * 8 + ((int)threadIdx.x >> 5);
    if (row >= n_out) return;
    float acc = 0.0f;
    for (int c = lane; c < 64; c += 32) acc += __half2float(x[c]);
    for (int off = 16; off; off >>= 1) acc += __shfl_xor(acc, off);
    if (lane == 0) out[row] = __float2half(acc);
}}
"""


def say(ok: bool, name: str, detail: str) -> bool:
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name:24s} {detail}")
    return ok


def elf_facts(path: str) -> dict:
    """Minimal ELF reader: magic, machine, class, type. No llvm tools required."""
    with open(path, "rb") as f:
        head = f.read(64)
    if len(head) < 64 or head[:4] != b"\x7fELF":
        return {"elf": False}
    e_type, e_machine = struct.unpack_from("<HH", head, 16)
    return {
        "elf": True,
        "class64": head[4] == 2,
        "machine": e_machine,
        "machine_name": {224: "EM_AMDGPU"}.get(e_machine, f"EM_{e_machine}"),
        "type": {1: "REL", 2: "EXEC", 3: "DYN"}.get(e_type, str(e_type)),
        "osabi": head[7],
    }


def pyd_tag(root: str) -> str | None:
    """Extension-module tag the dev build was linked against, e.g. 'cp312'."""
    hits = glob.glob(os.path.join(root, "build", "lib", "*.cp3*-win_amd64.pyd"))
    if not hits:
        hits = glob.glob(os.path.join(root, "build", "lib", "*.cp3*"))
    tags = set()
    for h in hits:
        m = re.search(r"(cp3\d+)", os.path.basename(h))
        if m:
            tags.add(m.group(1))
    return tags.pop() if len(tags) == 1 else None


def interpreter_state(exe: str, args: list[str] | None = None) -> tuple[str, bool]:
    """(tag, has_tvm_ffi) for an interpreter, or ('?', False) if it will not run."""
    cmd = (args or [exe]) + [
        "-c",
        "import sys;"
        "t='cp%d%d'%sys.version_info[:2];"
        "import importlib.util;"
        "print(t, importlib.util.find_spec('tvm_ffi') is not None)",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except OSError:
        return ("?", False)
    if p.returncode != 0:
        return ("?", False)
    parts = p.stdout.split()
    return (parts[0], parts[1] == "True") if len(parts) == 2 else ("?", False)


def resolve_interpreter(root: str) -> int | None:
    """Re-exec under an interpreter that can load the dev build. Returns exit code, or None."""
    tag = pyd_tag(root)
    ver = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    mine = interpreter_state(sys.executable)
    if tag is None or (mine[0] == tag and mine[1]):
        return None  # nothing to fix: no build tag found, or we already fit

    print(f"[0] interpreter: {sys.executable} is {ver} but the dev build is {tag}"
          f"{'' if mine[1] else ' (and has no tvm_ffi)'} -- looking for one that fits")

    cands: list[tuple[str, list[str]]] = []
    if os.environ.get("KRK_PYTHON"):
        cands.append((os.environ["KRK_PYTHON"], [os.environ["KRK_PYTHON"]]))
    for tag_ver in (f"{tag[2]}.{tag[3:]}",):
        cands.append((f"py -{tag_ver}", ["py", f"-{tag_ver}"]))
    cands += [
        (r"C:\Program Files\Python312\python.exe", [r"C:\Program Files\Python312\python.exe"]),
        (os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python312\python.exe"),
         [os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python312\python.exe")]),
    ]

    good = None
    for label, cmd in cands:
        if label == sys.executable:
            continue
        state = interpreter_state(cmd[0], cmd)
        if state[0] != tag or not state[1]:
            continue
        # Resolve the real exe (the `py -3.12` launcher needs a second step) so we
        # can re-exec without the launcher in the way.
        try:
            p = subprocess.run(cmd + ["-c", "import sys;print(sys.executable)"],
                               capture_output=True, text=True, timeout=120)
            exe = p.stdout.strip().splitlines()[-1] if p.returncode == 0 else ""
        except OSError:
            exe = ""
        if exe and os.path.exists(exe):
            good = (label, exe)
            break

    if not good:
        say(False, "interpreter", f"no {tag} interpreter with tvm_ffi found "
                                  f"(set KRK_PYTHON to one)")
        return 3

    label, exe = good
    say(True, "re-exec", f"{label} -> {exe}")
    if os.environ.get("KRK_PROBE_REEXEC") == "1":
        say(False, "re-exec", "already re-exec'd once; refusing to loop")
        return 3
    os.environ["KRK_PROBE_REEXEC"] = "1"
    return subprocess.call([exe, os.path.abspath(__file__)] + sys.argv[1:])


def find_rocm_root() -> str | None:
    """A ROCm install that carries the LLVM bin dir, without needing tilelang.

    The direct-hipcc path has no tilelang to ask for ROCm_PATH, and on this box the
    `hipcc` on PATH is a pip shim whose parent-of-parent is a Python root, so the
    search has to recognise a real install by its `lib/llvm/bin` rather than trust a
    path-derived prefix.
    """
    for cand in (os.environ.get("ROCM_PATH"), os.environ.get("ROCM_HOME")):
        if cand and os.path.isdir(os.path.join(cand, "lib", "llvm", "bin")):
            return cand
    pats = ["G:/ROCM*", "C:/Program Files/AMD/ROCm/*", "C:/ROCM*", "/opt/rocm*"]
    for pat in pats:
        for d in sorted(glob.glob(pat), reverse=True):
            if os.path.isdir(os.path.join(d, "lib", "llvm", "bin")):
                return d
    return None


def find_bundler(rocm: str | None) -> str | None:
    import shutil

    rocm = rocm or find_rocm_root()
    if rocm:
        cand = os.path.join(rocm, "lib", "llvm", "bin", "clang-offload-bundler.exe")
        if os.path.exists(cand):
            return cand
    return shutil.which("clang-offload-bundler")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output directory (default: a fresh temp dir)")
    ap.add_argument("--arch", default=None, help="target arch (default: KRK_GPU_TARGETS or detected)")
    ap.add_argument("--no-tilelang", action="store_true", help="call hipcc directly instead of tilelang")
    ap.add_argument("--keep-src", action="store_true", help="keep the generated .cc")
    args = ap.parse_args()

    root = os.environ.get("KRK_TILELANG_ROOT", "G:/tilelang-rocm")
    outdir = args.out or tempfile.mkdtemp(prefix="krk-tilelang-probe-")
    os.makedirs(outdir, exist_ok=True)
    src_path = os.path.join(outdir, "probe_gemv.cc")
    bundle_path = os.path.join(outdir, f"{KERNEL_NAME}.bundle.hsaco")
    elf_path = os.path.join(outdir, f"{KERNEL_NAME}.hsaco")

    print(f"tilelang dev root : {root}")
    print(f"output dir        : {outdir}")
    print(f"python            : {sys.executable}")

    if not args.no_tilelang:
        rc = resolve_interpreter(root)
        if rc is not None:
            return rc

    # --- 1. the import chain -------------------------------------------------
    print("\n[1] import chain")
    arch = args.arch
    rocm_path = None
    if args.no_tilelang:
        say(True, "tilelang", "skipped (--no-tilelang)")
        rocm_path = find_rocm_root()
        tmp = os.environ.get("KRK_GPU_TARGETS", "gfx1201").split(";")[0].strip()
        arch = arch or tmp or "gfx1201"
        say(True, "arch", f"{arch}  (rocm: {rocm_path})")
    else:
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            import tilelang  # noqa: F401  (must precede tvm/tvm_ffi imports)
        except Exception as e:  # noqa: BLE001
            say(False, "import tilelang", f"{type(e).__name__}: {e}")
            print("\nverdict: SETUP MISSING -- set KRK_TILELANG_ROOT / KRK_PYTHON")
            return 3
        import tvm
        import tvm_ffi

        say(True, "tilelang", f"{tilelang.__version__} ({tilelang.__file__})")
        say(True, "tvm", f"{tvm.__version__}")
        say(True, "tvm_ffi", f"{tvm_ffi.__version__} ({tvm_ffi.__file__})")
        native = None
        try:
            native = tvm_ffi.libinfo.find_libtvm_ffi()
        except Exception as e:  # noqa: BLE001
            say(False, "native tvm_ffi", str(e))
        if native:
            say(True, "native tvm_ffi", native)

        try:
            from tilelang.contrib.hipcc import compile_hip
            from tilelang.contrib.rocm import find_rocm_path, get_rocm_arch
        except Exception as e:  # noqa: BLE001
            say(False, "ROCm entry points", f"{type(e).__name__}: {e}")
            return 3
        try:
            from tilelang.rocm.language.kernel import Kernel  # noqa: F401

            say(True, "Kernel + compile_hip", "tilelang.rocm.language.kernel / contrib.hipcc")
        except Exception as e:  # noqa: BLE001
            say(False, "Kernel", f"{type(e).__name__}: {e}")
        try:
            rocm_path = find_rocm_path()
        except Exception:  # noqa: BLE001
            rocm_path = None
        arch = arch or os.environ.get("KRK_GPU_TARGETS", "").split(";")[0].strip()
        if not arch:
            arch = get_rocm_arch(rocm_path) if rocm_path else "gfx1201"
        say(True, "arch", f"{arch}  (rocm: {rocm_path})")

    # --- 2. author + compile -------------------------------------------------
    print("\n[2] compile")
    with open(src_path, "w", newline="\n") as f:
        f.write(SRC_TEXT)
    say(True, "source", f"{len(SRC_TEXT)} bytes, kernel {KERNEL_NAME} -> {src_path}")

    if args.no_tilelang:
        hipcc = None
        if rocm_path:
            cand = os.path.join(rocm_path, "bin", "hipcc.exe")
            if os.path.exists(cand):
                hipcc = cand
        cmd = [hipcc or "hipcc", "-O3", "-c", "-gline-tables-only",
               f"--offload-arch={arch}", "--genco", "-o", bundle_path, src_path]
        p = subprocess.run(cmd, capture_output=True, text=True)
        compiled = p.returncode == 0 and os.path.exists(bundle_path)
        say(compiled, "hipcc --genco", f"rc={p.returncode} ({cmd[0]})")
        if not compiled:
            print(p.stdout[-2000:] + p.stderr[-2000:])
    else:
        try:
            data = compile_hip(SRC_TEXT, target_format="hsaco", arch=arch,
                               path_target=bundle_path, verbose=False)
            compiled = bool(data)
            say(compiled, "compile_hip", f"{len(data)} bytes -> {bundle_path}")
        except Exception as e:  # noqa: BLE001
            say(False, "compile_hip", f"{type(e).__name__}: {str(e)[:1500]}")
            return 2
    if not compiled:
        return 2

    # --- 3. loader-ready artifact -------------------------------------------
    print("\n[3] loader-ready artifact")
    with open(bundle_path, "rb") as f:
        magic = f.read(64)  # the bundle prefix is 24 bytes; do not read short
    is_bundle = magic.startswith(b"__CLANG_OFFLOAD_BUNDLE__")
    say(True, "raw output", f"{'clang offload bundle' if is_bundle else 'not a bundle'}"
                            f" ({os.path.getsize(bundle_path)} bytes)")

    produced = bundle_path
    if is_bundle:
        # --genco wraps the device payload in an offload bundle; a HIP loader needs
        # the ELF inside it, so unwrap by the bundle's own target id.
        delta = f"hipv4-amdgcn-amd-amdhsa--{arch}"
        bundler = find_bundler(rocm_path)
        if not bundler:
            say(False, "bundler", "clang-offload-bundler not found (set ROCM_PATH)")
            return 2
        p = subprocess.run([bundler, "--unbundle", "--type=o", f"--targets={delta}",
                            f"--input={bundle_path}", f"--output={elf_path}"],
                           capture_output=True, text=True)
        ok = p.returncode == 0 and os.path.exists(elf_path)
        say(ok, "unbundle", f"{delta} -> {os.path.getsize(elf_path) if ok else 0} bytes")
        if not ok:
            print(p.stdout[-1500:] + p.stderr[-1500:])
            return 2
        produced = elf_path

    facts = elf_facts(produced)
    say(bool(facts.get("elf")), "ELF magic",
        "7f 45 4c 46" if facts.get("elf") else "NOT an ELF")
    say(facts.get("machine") == 224, "machine",
        f"{facts.get('machine_name')} (224 = EM_AMDGPU) type={facts.get('type')} osabi={facts.get('osabi')}")
    with open(produced, "rb") as f:
        blob = f.read()
    say(KERNEL_NAME.encode() in blob, "kernel symbol", f"{KERNEL_NAME} present in the code object")
    say(b"amdhsa.kernels" in blob, "kernel metadata", "NT_AMDGPU_METADATA lists amdhsa kernels")

    if not args.keep_src:
        try:
            os.remove(src_path)
        except OSError:
            pass

    print(f"\nartifact : {produced}")
    print(f"launch   : hipModuleLoadData(bytes of {os.path.basename(produced)}), "
          f"hipModuleGetFunction(\"{KERNEL_NAME}\")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
