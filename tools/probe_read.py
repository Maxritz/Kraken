#!/usr/bin/env python3
"""Measure how fast the model file can be read, 1 thread vs N.

The expert cold path reads ~394 MiB per decode token out of this file, one
2.06 MiB expert at a time, through a single handle. Whether that read is the
decode bottleneck -- and whether parallelising it is worth anything -- is a
property of the file, not of the engine, so it is measured here rather than
inferred from a decode timing.

  usage: probe_read.py <path> [chunk_mib] [nthreads]
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

path = sys.argv[1]
chunk_mib = int(sys.argv[2]) if len(sys.argv) > 2 else 2
nthreads = int(sys.argv[3]) if len(sys.argv) > 3 else 4

chunk = chunk_mib << 20
size = os.path.getsize(path)
# A contiguous slice in the middle of the file: past the header, and the same
# kind of bytes the expert tensors are.
total = 2 << 30
start = min(size // 8, size - total)

def read_slice(args):
    fh, off, nbytes = args
    fh.seek(off)
    got = 0
    while got < nbytes:
        b = fh.read(min(chunk, nbytes - got))
        if not b:
            break
        got += len(b)
    return got


def run(n):
    per = total // n
    # One handle per thread, opened fresh for this run. On Windows a file HANDLE
    # serialises concurrent ReadFile, so a shared handle measures the OS and not
    # the filesystem -- which is exactly the shape ExpertCache::read_host has.
    handles = [open(path, "rb", buffering=0) for _ in range(n)]
    jobs = [(handles[i], start + i * per, per) for i in range(n)]
    t0 = time.perf_counter()
    got = sum(read_slice(j) for j in jobs)
    dt = time.perf_counter() - t0
    for fh in handles:
        fh.close()
    return got, dt


def run_scattered(n, slice_bytes, count):
    """The expert tier's real pattern: many small reads at scattered offsets.

    An expert is three slices of ~0.69 MiB, and consecutive experts are a whole
    expert-width apart, so the file is walked in 0.69 MiB steps with nothing
    sequential about it. A device can be fast at both of those and still be slow
    at this one, so the sequential number above does not answer the question the
    engine actually asks.
    """
    per = count // n
    handles = [open(path, "rb", buffering=0) for _ in range(n)]
    # Deterministic but spread out: a prime stride coprime with the slice size
    # walks the data section without clustering.
    stride = 104729
    buf = bytearray(slice_bytes)

    def job(i):
        fh = handles[i]
        got = 0
        for j in range(per):
            off = start + ((i * per + j) * stride) % max(1, size - slice_bytes)
            fh.seek(off)
            n_read = 0
            while n_read < slice_bytes:
                b = fh.readinto(memoryview(buf)[n_read:])
                if not b:
                    break
                n_read += b
            got += n_read
        return got

    t0 = time.perf_counter()
    got = sum(job(i) for i in range(n))
    dt = time.perf_counter() - t0
    for fh in handles:
        fh.close()
    return got, dt


print(f"file   {path}")
print(f"size   {size / (1 << 30):.2f} GiB   reading {total / (1 << 30):.2f} GiB "
      f"from offset {start} in {chunk_mib} MiB sequential reads\n")
print("threads   GiB/s   GB/s   s")
for n in (1, 2, 4, 8, 16, 32):
    got, dt = run(n)
    gb = got / (1 << 30)
    print(f"{n:>7}   {gb / dt:7.2f}   {got / dt / 1e9:6.2f}   {dt:6.2f}")

# Scattered 0.69 MiB slices: what ExpertCache::read_host actually asks for.
slice_bytes = 723 << 10  # ~0.69 MiB, one of laguna's three expert slices
count = 8192            # ~5.4 GiB of slices, the scale of one 24-token run
print(f"\nscattered: {count} reads of {slice_bytes / 1048576.0:.2f} MiB "
      f"({count * slice_bytes / (1 << 30):.1f} GiB), prime-strided")
print("threads   GiB/s   GB/s   s   us/read")
for n in (1, 2, 4, 8):
    got, dt = run_scattered(n, slice_bytes, count)
    print(f"{n:>7}   {got / (1 << 30) / dt:7.2f}   {got / dt / 1e9:6.2f}   "
          f"{dt:6.2f}   {dt / count * 1e6:8.1f}")