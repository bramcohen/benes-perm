# Benes permutation compression

This library stores permutations as a single, very compact data structure. It
supports both forward and backward lookups and performs well even when the
data structure is too large for memory and must remain on disk.

The implementation originated from Chia Network's pos2 research and was
written by Rohan Ridenour. This repository packages that work behind one format
and API, with portable CPU, Metal, CUDA, and AVX-512 compression backends.

The general construction and its disk-locality motivation are described in
[How to Store a Permutation Compactly](https://hackmd.io/@dabo/rkP8Pcf9t) by
Bram Cohen and Dan Boneh.

## API

The input is an ordinary permutation array: `permutation[i]` is the forward
image of `i`, and each value in `[0, permutation.size())` occurs exactly once.

```cpp
#include <benes/benes.hpp>

std::vector<std::uint64_t> permutation = {2, 0, 3, 1};
std::vector<std::byte> encoded = benes::compress(permutation);

benes::CompressedPermutationView view(encoded);
assert(view.forward(0) == 2);
assert(view.inverse(2) == 0);
```

Both `uint32_t` and `uint64_t` arrays are accepted. The 32-bit overload retains
the smaller, faster common path; the 64-bit overload supports entry IDs and
counts beyond `2^32`. Both produce exactly the same bytes when given the same
permutation.

`benes::compress_file` accepts the same input as raw little-endian entries and
writes the opaque binary. Raw entries are 64-bit by default; pass
`InputValueWidth::bits32` explicitly for a compact 32-bit source.
`CompressedPermutationView` is non-owning,
so it can point at a vector, bytes read from disk, or a read-only memory mapping.
`CompressedPermutationFile` performs explicit random-access file reads through
a bounded page cache when mapping the whole file is undesirable.

Lookup is CPU-based and backend-independent. Compression backend selection is
automatic, or can be requested with `CompressOptions`:

- `portable_cpu`: scalar reference plus deterministic multithreaded union-find.
- `metal`: Apple GPU atomic union-find, SIMD-group packing, fused threadgroup
  inner layers, and SIMD-group factorial coding.
- `cuda`: the equivalent NVIDIA implementation.
- `avx512`: portable routing with AVX-512 middle-permutation coding.

The single-threaded portable encoder defines the canonical bytes. Every backend
and both input widths must match it byte for byte. Accelerated encoders retain
separate 32-bit and 64-bit routing paths, while middle groups remain at most 32
entries and use the same local factoradic code.

The portable encoder writes clustered switch and rank bits directly into the
final blob. It does not materialize a second full encoded representation for
reordering, and it releases node-coloring scratch before descending into child
permutations.

Format options select the middle enumerated
permutation group size, independent input/output cluster-size lists, and where
the middle data is placed:

```cpp
benes::CompressOptions options{
    .format = {
        .middle_group_log2 = 5,  // 32-entry groups
        .input_clusters = {8, 7},  // for ceil_log2(size) == 20
        .output_clusters = {7, 8},
        .middle_placement = benes::MiddlePlacement::embedded_input,
    },
};
```

Empty cluster lists select a balanced one- or two-cluster default. Each list
contains cluster sizes in base-2 logarithmic form: a value of `8` groups eight
routing layers, covering `2^8` positions. The values must be positive and sum
to `ceil_log2(size) - middle_group_log2`. Final groups on pruned non-power-of-two
paths may terminate before consuming every cluster.

## Size

For a power-of-two size `N = 2^k` and default 32-entry middle groups:

- Input switch layers: `N * (k - 5) / 2` bits.
- Output switch layers: `N * (k - 5) / 2` bits.
- One exact 118-bit factorial rank per 32 entries.

The payload is `N * (k - 1.3125) / 8` bytes before the three section-end
roundings. At `N = 2^20`, it is about 2.336 MiB. Tightly packed 20-bit indexes use
2.5 MiB, a `uint32_t` array uses 4 MiB, and `ceil(log2(N!))` is about 2.32 MiB.
The file adds a small variable header.

Other sizes use the identity-padding idea from the original `benes` branch,
but prune the implied padded paths from the persistent representation. Each
remaining recursive node stores one bit per paired input and output, then
recurses into children of
`ceil(n / 2)` and `floor(n / 2)` entries. Final groups of at most 32 entries
use an exact-width enumerated-permutation code. Ranks are concatenated at
`ceil(log2(c!))` bits for
the actual final-group population `c`; they are not individually byte-aligned. No
dummy entries are stored, so crossing a power of two does not double the
payload.

## Persistent format

Version 1 is little-endian:

1. A 20-byte fixed prefix followed by the cluster-depth bytes:
   - bytes 0–7: `BENES01\0`
   - bytes 8–15: permutation size
   - byte 16: middle placement (`0` separate, `1` input, `2` output)
   - byte 17: base-2 logarithm of the middle permutation group size
   - bytes 18–19: input/output cluster counts
   - bytes 20 onward: input cluster depths, then output cluster depths
2. Input and output switch streams. A `d`-level cluster groups the switches
   reachable from each `2^d`-entry neighborhood into one packet, with
   immediate-neighbor switches first. Frontier subnetwork packets then restart
   the same ordering for the next configured depth. Thus one lookup consumes
   one contiguous packet per configured cluster.
3. Enumerated middle-permutation ranks in left-to-right group order, either as
   a separate stream or densely embedded into the selected innermost cluster
   packets.

Bits are LSB-first. Input switches, output switches, and all middle ranks each
receive exactly one whole-stream byte-boundary rounding. There is no per-rank
or per-packet padding. Embedded placement preserves the same three logical
rounding budgets while changing physical proximity.

The magic identifies the format version. Header length, section offsets, and
exact file size are derived from the stored parameters rather than encoded
again. The decoder accepts only exact canonical section bounds, and every
unused section-tail bit must be zero. Truncated, malformed, or unknown versions
are rejected. See `src/internal.hpp` for the field-level specification and
`tests/golden` for canonical examples.

Entry counts and lookup IDs are unsigned 64-bit values. Counts are limited by
the representable encoded size (and, for in-memory compression or views, by
the platform address space), not by `UINT32_MAX`. File-backed lookup can address
valid sparse or stored encodings larger than memory. Counts whose recursive
bit counts, section offsets, file offsets, or logical network width overflow
are rejected with checked arithmetic.

Lookups read one input and output switch bit per recursive level plus one
enumerated middle permutation. With two clusters on each side, storing the
middle separately requires five logical disk regions per lookup: two input
clusters, the middle, and two output clusters. Embedding the middle into the
innermost packets on one side is specifically intended to eliminate the
separate middle-table seek by making the middle rank and adjacent inner
switches one locality region. Choosing the input or output side allows the
layout to be tuned for the expected forward/inverse access pattern. This does
not promise a particular number of physical page reads, because the operating
system and storage device still control caching and read granularity. Explicit
file lookup uses the same derived offsets through a page cache. Memory mapping
is normally faster for random workloads; measured results are in
`BENCHMARKS.md`.

Each cluster puts immediate-neighbor switches before progressively deeper,
farther switches. The unpaired tail of an odd-sized node is an implied no-flip,
so its nonexistent switch bit and unreachable padded descendants are omitted.

The classic Waksman normalization can omit one additional redundant bit per
internal routing node—`N / q - 1` bits for power-of-two `N` and middle group
size `q`.
Version 1 deliberately stores those bits to keep packet sizing, parallel
generation, and random addressing simple. This small additional saving remains
a possible future format change.

## Possible future work: many-to-many relations

A many-to-many relation could use this format by expanding each logical input
and output into one permutation slot per relationship. Additional input and
output fanout tables would map each logical entry to its range of expanded
slots. Those tables should use whichever compression scheme best matches the
application's fanout distribution; they need not be part of the Benes encoding
itself. The total input and output fanouts must match, leaving a permutation
between the two expanded slot sets.

The order of slots within one logical entry's fanout is unobservable. Routing
bits that only permute those slots can therefore be omitted and inferred by
the decoder. Compression could improve further by reordering logical inputs
and outputs according to their fanouts before assigning expanded slots, so
equal-size groups align with the switch structure. For example, a fanout-two
entry aligned with a corresponding Waksman switch makes that switch bit
entirely redundant; the same entry at an unaligned position provides no such
saving.

This would require a new format version, a canonical endpoint-ordering rule,
and application-specific fanout-table codecs. It is not implemented by version
1.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/benes-tests --extended
```

Metal is enabled by default on Apple platforms and compiles its shader through
the Metal runtime, so Command Line Tools are sufficient. CUDA is opt-in:

```sh
cmake -S . -B build-cuda \
  -DBENES_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

AVX-512 is built on x86-64 and guarded by runtime feature detection. Backends
can be disabled with `BENES_ENABLE_METAL`, `BENES_ENABLE_CUDA`, and
`BENES_ENABLE_AVX512`.

On an Apple Silicon development machine, the AVX-512 translation unit can be
cross-compiled without running it:

```sh
cmake -S . -B build-avx512-check \
  -DBENES_ENABLE_METAL=OFF -DBENES_ENABLE_CUDA=OFF \
  -DBENES_ENABLE_AVX512=ON -DBENES_BUILD_TOOLS=OFF \
  -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
  -DCMAKE_OSX_ARCHITECTURES=x86_64
cmake --build build-avx512-check --verbose
otool -tvV build-avx512-check/CMakeFiles/benes.dir/src/avx512/benes_avx512.cpp.o
```

This checks x86-64 syntax and emitted vector instructions, but does not provide
AVX-512 runtime validation.

The benchmark tools are:

```sh
./build/benes-benchmark metal 1048576 1 5
./build/benes-benchmark metal64 1048576 1 5
./build/benes-lookup-benchmark 1048576 100000
```

Appending `64` to a compression backend name (`cpu64`, `metal64`, `cuda64`, or
`avx51264`) benchmarks the distinct 64-bit source/index path on a bounded input.

CUDA and AVX-512 compile paths are maintained, but their full runtime test and
benchmark matrices still need to be run on NVIDIA and AVX-512 machines. The
exact required workload and current Metal/portable results are documented in
`BENCHMARKS.md`.

The optional tools exercise the public API for compression timing, lookup
timing, golden generation, and persistence checks. The obsolete standalone
prototype implementations are not duplicated in the package.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
