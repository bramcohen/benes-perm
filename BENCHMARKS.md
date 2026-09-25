# Benchmarks

Results below were collected on 2026-09-25 on an Apple M5 Max:

- 40-core Apple GPU, Metal 4
- 18 CPU cores (12 performance, 6 efficiency)
- macOS 26.6
- Apple Clang 21.0
- Release build

Commands are shown so the same workload can be repeated on CUDA and AVX-512
machines. Times are medians of five runs unless stated otherwise. Compression
includes validation, allocation, GPU submission, synchronization, and canonical
blob construction. `device_seconds` is Metal command-buffer GPU time.

## Correctness

```sh
ctest --test-dir build --output-on-failure
./build/benes-tests --extended
```

Both passed. Coverage includes all permutations through size 8; identity,
reverse, structured, and seeded-random inputs around powers of two; forward and
inverse lookup; 1/2/4/8/12-thread canonical CPU output; byte equality between
portable CPU and Metal; golden blobs; malformed data; file round trips; mmap;
cached explicit file reads; custom middle group sizes, cluster compositions,
and middle placements; byte-identical 32-bit/64-bit source encodings; checked
layout arithmetic around `2^32`; sampled lookups in a sparse encoded file with
`2^33` entries; and large non-power-of-two inputs. The final version-1
suite also passed ThreadSanitizer and combined
AddressSanitizer/UndefinedBehaviorSanitizer.

## Compression size

The variable header is excluded from `payload`. `packed` is a tightly packed
array using `ceil(log2(n))` bits per entry. `minimum` is
`ceil(log2(n!)) / 8`.

| n | network | payload | uint32 | packed | minimum |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 31 | 32 | 15 | 124 | 20 | 15 |
| 32 | 32 | 15 | 128 | 20 | 15 |
| 33 | 64 | 16 | 132 | 25 | 16 |
| 63 | 64 | 37 | 252 | 48 | 37 |
| 64 | 64 | 38 | 256 | 48 | 37 |
| 65 | 128 | 39 | 260 | 57 | 38 |
| 96 | 128 | 64 | 384 | 84 | 63 |
| 127 | 128 | 91 | 508 | 112 | 89 |
| 128 | 128 | 91 | 512 | 112 | 90 |
| 129 | 256 | 92 | 516 | 129 | 91 |
| 1,048,575 | 1,048,576 | 2,449,406 | 4,194,300 | 2,621,438 | 2,432,342 |
| 1,048,576 | 1,048,576 | 2,449,408 | 4,194,304 | 2,621,440 | 2,432,345 |

Version 1 stores exactly `ceil(log2(c!))` bits for each populated middle group
of `c` entries and rounds only the three complete logical streams. It reports the
logical power-of-two network width, but unreachable padded subtrees and odd
unmatched switches occupy no payload bytes. Crossing a power-of-two boundary
therefore has no padding cliff.

Reproduce:

```sh
for n in 31 32 33 63 64 65 96 127 128 129 1048575 1048576; do
  ./build/benes-benchmark cpu "$n" 18 3
done
```

## Portable CPU thread scaling

One seeded random permutation of 1,048,575 entries using the compact recursive
encoding:

| threads | seconds | million entries/s | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.2575 | 4.07 | 1.00x |
| 2 | 0.1672 | 6.27 | 1.54x |
| 4 | 0.1026 | 10.22 | 2.51x |
| 8 | 0.0781 | 13.43 | 3.30x |
| 18 | 0.0672 | 15.60 | 3.83x |

The portable encoder recursively divides independent child networks among
threads. It continues improving through all 18 cores. Root routing work and
memory bandwidth limit ideal linear scaling, but this deliberately simple
backend provides the canonical byte-for-byte reference for accelerated paths.

```sh
for t in 1 2 4 8 18; do
  ./build/benes-benchmark cpu 1048575 "$t" 5
done
```

### 64-bit source overhead

The optimized 32-bit route remains separate from the true 64-bit route. A
nine-run median comparison at 1,048,575 entries and 18 threads measured:

| build/path | seconds | million entries/s | relative to current 32-bit |
| --- | ---: | ---: | ---: |
| prior root commit, 32-bit | 0.074232 | 14.13 | 1.00x |
| current, 32-bit | 0.074113 | 14.15 | 1.00x |
| current, 64-bit | 0.102672 | 10.21 | 0.72x |

The common 32-bit path showed no measurable regression. The 64-bit path was
38.5% slower on this bounded workload, principally because its permutation,
inverse, routing-neighbor, and child arrays use twice the memory bandwidth.
It emits the same 2,449,430-byte blob.

```sh
./build/benes-benchmark cpu 1048575 18 9
./build/benes-benchmark cpu64 1048575 18 9
```

## Metal scaling

The recursive Metal path batches all nodes at each depth and performs
inverse construction, parity union-find, odd-tail orientation, switch packing,
routing, and exact-width middle-permutation coding on the GPU. Median results
from five runs:

| n | backend | wall seconds | device seconds | million entries/s |
| ---: | --- | ---: | ---: | ---: |
| 43,690 | Metal | 0.003750 | 0.002740 | 11.65 |
| 1,048,575 | portable CPU, 18 threads | 0.067202 | — | 15.60 |
| 1,048,575 | Metal | 0.012948 | 0.005558 | 80.99 |
| 1,048,576 | Metal | 0.014568 | 0.005486 | 71.98 |

At 1,048,575 entries Metal is 5.19x faster by wall time than all 18 CPU cores
and 19.9x faster than one CPU thread while producing byte-identical output.
The neighboring power-of-two case has essentially identical device time, so
non-power-of-two routing does not lose GPU parallelism or pay for padded
entries.

```sh
for n in 43690 1048575 1048576; do
  ./build/benes-benchmark cpu "$n" 18 5
  ./build/benes-benchmark metal "$n" 18 5
done
```

GPU generation uses 96 bytes of routing metadata per internal node and final
group.
At one million entries that flattened metadata is about 6 MiB. A measured
64-byte alternative saved about 2 MiB but increased device time from 5.5 ms to
7.1 ms, so the precomputed packet offsets were retained. A future streaming or
formula-only descriptor scheme could reduce large-input memory without paying
that repeated address-calculation cost.

Only Command Line Tools are installed on this machine, so Xcode's Metal System
Trace occupancy counters were unavailable. Device timestamps, equal
power/non-power behavior, and the 19.9x single-core speedup establish broad
parallel use; an occupancy-counter capture remains desirable with full Xcode.

The true 64-bit Metal source/index path was also measured with nine-run
medians at 1,048,575 entries:

| path | wall seconds | device seconds | million entries/s |
| --- | ---: | ---: | ---: |
| Metal, 32-bit | 0.012152 | 0.005213 | 86.29 |
| Metal, 64-bit | 0.016224 | 0.009304 | 64.63 |
| Metal, 64-bit with bit-63 parents forced | 0.021428 | 0.012072 | 48.93 |

The 64-bit descriptor and permutation path is 1.34x slower by wall time. It
normally uses parallel 32-bit parent words at routing levels whose node-local
indexes fit below bit 31, while retaining 64-bit global offsets. Levels that
genuinely exceed that range use lock-backed bit-63 parent words, still with one
pair per GPU work item. `BENES_METAL_FORCE_PARENT64=1` exercises that fallback
on bounded inputs; the full test suite passes in both modes.

```sh
./build/benes-benchmark metal64 1048575 18 9
BENES_METAL_FORCE_PARENT64=1 \
  ./build/benes-benchmark metal64 1048575 18 9
```

`/usr/bin/time -l` reported 79.5 MB maximum resident set size for the
18-thread portable compressor at 1,048,575 entries and 50.1 MB host RSS for
Metal. The latter also reported a 179 MB peak footprint including driver/GPU
resources. These commands include the benchmark's full forward/inverse
verification pass; the timed compression measurements above do not.

## Lookup

1,048,575 entries and 100,000 seeded random lookups:

| middle placement | memory forward | memory inverse | cached file forward | cached file inverse |
| --- | ---: | ---: | ---: | ---: |
| separate | 3,145 ns | 3,142 ns | 9,459 ns | 9,553 ns |
| embedded input | 3,171 ns | 3,211 ns | 8,877 ns | 8,913 ns |
| embedded output | 3,215 ns | 3,185 ns | 8,818 ns | 8,902 ns |

```sh
for layout in separate input output; do
  ./build/benes-lookup-benchmark 1048575 100000 "$layout"
done
```

The explicit file reader avoids loading or mapping the entire blob, but each
lookup follows data-dependent packets. mmap remains preferable for high-rate
random lookup. Embedded placement improved this cached-file microbenchmark,
but it is a logical-locality option, not a guarantee about physical page reads.

## Required CUDA and AVX-512 runs

CUDA and AVX-512 have not been executed on this Apple system. CUDA's separate
32-bit and 64-bit descriptors and buffers, bit-63 atomic parents, grid-stride
large-work dispatch, clustered atomic switch packing, exact-bit middle
offsets, and host serialization were reviewed for parity with the tested Metal
path, but no CUDA compiler or hardware was available. On the appropriate
machines:

1. Build with the backend enabled.
2. Run `ctest` and `benes-tests --extended`.
3. Run the exact size, non-power-of-two, and lookup commands above.
4. Require both 32-bit and 64-bit source paths to match the single-thread
   portable bytes. The test suite includes backend-gated compact boundary
   cases through 4,097 entries; on AVX-512 it also exhausts all permutations
   through size 8.
5. For CUDA, sweep threadblock sizes and capture occupancy, memory throughput,
   launch gaps, and kernel/device time on the selected NVIDIA GPU.
6. For AVX-512, collect vector instruction mix, cycles per element, memory
   bandwidth, and the same 1/2/4/... thread sweep.

Do not treat source review or cross-compilation as runtime validation.

The x86-64 cross-compile and `otool` inspection command in `README.md` is the
maintenance check available on Apple Silicon. It must be followed by the
runtime workload above on an AVX-512 host.
