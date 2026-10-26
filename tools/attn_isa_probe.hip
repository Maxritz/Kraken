// Instantiates the prefill attention kernel on its own so its device ISA can be
// read (and diffed against another revision of attention.hpp) without building
// the whole engine. Compile with the engine's HIP flags, which matter: `-O3
// -ffast-math -fno-finite-math-only` is what decides contraction and reassociation.
//
//   hipcc -O3 -ffast-math -fno-finite-math-only -std=c++17 -DNDEBUG \
//         --offload-arch=gfx1201 -I <header dirs> -c tools/attn_isa_probe.hip \
//         -o /tmp/probe.o --save-temps
//
// hd 128 / QT 8 is the instantiation the 8B and the 4B run (VPT 4), hd 256 /
// QT 4 the other dense shape (VPT 8). Taking the address of a __global__
// template from host code is what forces both halves to be emitted.
#include "attention.hpp"

void *probe_kernels_vpt4() {
    return reinterpret_cast<void *>(&krk::attention_bexact_kernel<4, 8>);
}

void *probe_kernels_vpt8() {
    return reinterpret_cast<void *>(&krk::attention_bexact_kernel<8, 4>);
}
