#!/bin/sh
# ============================================================================
#  fetch_moe_model.sh — download a small, REAL Mixture-of-Experts GGUF.
#
#  The test suite proves the MoE path on synthetic checkpoints, but only a real
#  qwen3moe file exercises the loader against an actual producer's tensor
#  layout (3-D expert tensors, routing metadata, 150k+ BPE vocab). This fetches
#  a small MoE checkpoint whose quantization KRAKEN supports (Q4_K_M).
#
#    sh ./scripts/fetch_moe_model.sh                  # ~965 MB -> kraken/models/
#    sh ./scripts/fetch_moe_model.sh /some/other/path.gguf
#    KRK_MOE_REPO=owner/repo KRK_MOE_FILE=name.gguf sh ./scripts/fetch_moe_model.sh
#
#  The file lands INSIDE the project tree (kraken/models/) so it is visible in
#  the workspace file browser and downloadable from there.
#
#  Verified with:
#    kraken --model <file> --cpu --info
#  arch qwen3moe, 28 layers, 4 experts, top-2, QK-norm, byte-BPE vocab 151936.
#
#  Note the IQ* quants of the same checkpoint (IQ2/IQ3/IQ4_XS) are rejected by
#  the loader by design; only IQ4_NL of the IQ family is supported. Q4_K_M is
#  the recommended default because every KRAKEN device kernel has a Q4_K path.
# ============================================================================
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

repo="${KRK_MOE_REPO:-mradermacher/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-i1-GGUF}"
file="${KRK_MOE_FILE:-Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder.i1-Q4_K_M.gguf}"
default_name="Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf"
if [ $# -ge 1 ]; then
    out="$1"
else
    mkdir -p "$root/models"
    out="$root/models/$default_name"
fi
out="$(cd "$(dirname "$out")" && pwd)/$(basename "$out")"
url="${KRK_MOE_URL:-https://huggingface.co/$repo/resolve/main/$file}"

echo "kraken: fetching MoE GGUF"
echo "  repo: $repo"
echo "  file: $file"
echo "  dest: $out"

if [ -f "$out" ]; then
    echo "kraken: '$out' already exists; delete it first to re-fetch"
    exit 0
fi

if command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar -o "$out" "$url"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
else
    echo "kraken: need curl or wget" >&2
    exit 1
fi

# GGUF files begin with the magic 'GGUF' (little-endian 0x46554747).
if [ "$(dd if="$out" bs=1 count=4 2>/dev/null)" != "GGUF" ]; then
    echo "kraken: downloaded file is not a GGUF (bad magic)" >&2
    exit 1
fi

echo "kraken: OK — $(du -h "$out" | cut -f1)"
echo "  path:  $out"
echo "  inspect: kraken-inspect $out"
echo "  run:     kraken --model $out --cpu --info"
echo "  (a real multi-layer MoE forward is minutes on the CPU oracle; use a GPU"
echo "   for interactive decode — see docs/GPU_BRINGUP.md)"
