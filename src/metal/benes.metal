// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;

struct U128 {
    ulong low;
    ulong high;
};

inline uint root_index(device atomic_uint* parents, uint position) {
    while (true) {
        const uint parent =
            atomic_load_explicit(&parents[position / 2], memory_order_relaxed);
        if ((position & ~1u) == (parent & ~1u)) {
            return position;
        }
        position = parent ^ (position & 1u);
    }
}

inline uint root_index(threadgroup atomic_uint* parents, uint position) {
    while (true) {
        const uint parent =
            atomic_load_explicit(&parents[position / 2], memory_order_relaxed);
        if ((position & ~1u) == (parent & ~1u)) {
            return position;
        }
        position = parent ^ (position & 1u);
    }
}

kernel void initialize_parents(
    device uint* parents [[buffer(0)]],
    constant uint& block_size [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    parents[index] = (index & (block_size - 1)) * 2;
}

kernel void merge_cycles(
    device const uint* permutation [[buffer(0)]],
    device atomic_uint* parents [[buffer(1)]],
    constant uint& block_size [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint base = index & ~(block_size - 1);
    const uint local = index - base;
    device atomic_uint* block = parents + base;
    uint left = local + (local & ~1u);
    const uint value = permutation[index];
    uint right = (value + (value & ~1u)) | 2u;
    while (true) {
        left = root_index(block, left);
        right = root_index(block, right);
        if (left == right) {
            return;
        }
        if (left > right) {
            const uint temporary = left;
            left = right;
            right = temporary;
        }
        uint expected = right & ~1u;
        const uint replacement = left ^ (right & 1u);
        if (atomic_compare_exchange_weak_explicit(
                &block[right / 2], &expected, replacement,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
        right = expected ^ (right & 1u);
    }
}

kernel void output_switches(
    device atomic_uint* parents [[buffer(0)]],
    device uint* input_words [[buffer(1)]],
    device uint* output_words [[buffer(2)]],
    constant uint& block_size [[buffer(3)]],
    constant uint& words_per_layer [[buffer(4)]],
    constant uint& layer_index [[buffer(5)]],
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint base = index & ~(block_size - 1);
    const uint local = index - base;
    device atomic_uint* block = parents + base;
    const uint mapped = (local & ~63u) | ((local >> 5) & 1u) |
        ((local << 1) & 62u);
    const uint bit = root_index(block, mapped * 2) & 1u;
    uint word = 0;
    for (uint source_lane = 0; source_lane < 32; ++source_lane) {
        word |= simd_shuffle(bit, source_lane) << source_lane;
    }
    if (lane == 0) {
        const uint word_index = layer_index * words_per_layer +
            (base + mapped) / 64;
        if ((mapped & 1u) == 0) {
            input_words[word_index] = word;
        } else {
            output_words[word_index] = word;
        }
    }
}

kernel void route_layer(
    device const uint* permutation [[buffer(0)]],
    device uint* routed [[buffer(1)]],
    device const uint* input_words [[buffer(2)]],
    device const uint* output_words [[buffer(3)]],
    constant uint& block_size [[buffer(4)]],
    constant uint& words_per_layer [[buffer(5)]],
    constant uint& layer_index [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint base = index & ~(block_size - 1);
    const uint local = index - base;
    const uint input_switch = index / 2;
    const uint input_word =
        input_words[layer_index * words_per_layer + input_switch / 32];
    const uint input_bit = (input_word >> (input_switch & 31)) & 1u;
    uint output = permutation[index ^ input_bit];
    const uint output_switch = (base + output) / 2;
    const uint output_word =
        output_words[layer_index * words_per_layer + output_switch / 32];
    output ^= (output_word >> (output_switch & 31)) & 1u;
    const uint half_size = block_size / 2;
    routed[base + (local >> 1) + ((local & 1u) ? half_size : 0)] = output / 2;
}

kernel void compress_small_layers(
    device const uint* permutation [[buffer(0)]],
    device uint* compressed_permutation [[buffer(1)]],
    device uint* input_words [[buffer(2)]],
    device uint* output_words [[buffer(3)]],
    constant uint& log_size [[buffer(4)]],
    constant uint& words_per_layer [[buffer(5)]],
    uint global_index [[thread_position_in_grid]],
    uint local [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint permutation_a[512];
    threadgroup uint permutation_b[512];
    threadgroup atomic_uint parents[512];
    permutation_a[local] = permutation[global_index];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint* current = permutation_a;
    threadgroup uint* routed = permutation_b;
    const uint global_base = global_index - local;
    for (uint layer = 9; layer > 5; --layer) {
        const uint block_size = 1u << layer;
        const uint block_base = local & ~(block_size - 1);
        const uint in_block = local - block_base;
        atomic_store_explicit(
            &parents[local], in_block * 2, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup atomic_uint* block_parents = parents + block_base;
        uint left = in_block + (in_block & ~1u);
        const uint value = current[local];
        uint right = (value + (value & ~1u)) | 2u;
        while (true) {
            left = root_index(block_parents, left);
            right = root_index(block_parents, right);
            if (left == right) {
                break;
            }
            if (left > right) {
                const uint temporary = left;
                left = right;
                right = temporary;
            }
            uint expected = right & ~1u;
            const uint replacement = left ^ (right & 1u);
            if (atomic_compare_exchange_weak_explicit(
                    &block_parents[right / 2], &expected, replacement,
                    memory_order_relaxed, memory_order_relaxed)) {
                break;
            }
            right = expected ^ (right & 1u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint mapped = (in_block & ~63u) | ((in_block >> 5) & 1u) |
            ((in_block << 1) & 62u);
        const uint switch_bit =
            root_index(block_parents, mapped * 2) & 1u;
        uint word = 0;
        for (uint source_lane = 0; source_lane < 32; ++source_lane) {
            word |= simd_shuffle(switch_bit, source_lane) << source_lane;
        }
        const uint layer_index = log_size - layer;
        if (lane == 0) {
            const uint word_index = layer_index * words_per_layer +
                global_base / 64 + (block_base + mapped) / 64;
            if ((mapped & 1u) == 0) {
                input_words[word_index] = word;
            } else {
                output_words[word_index] = word;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        const uint input_switch = global_base / 2 + local / 2;
        const uint input_word =
            input_words[layer_index * words_per_layer + input_switch / 32];
        const uint input_bit = (input_word >> (input_switch & 31)) & 1u;
        uint output = current[local ^ input_bit];
        const uint output_switch =
            global_base / 2 + (block_base + output) / 2;
        const uint output_word =
            output_words[layer_index * words_per_layer + output_switch / 32];
        output ^= (output_word >> (output_switch & 31)) & 1u;
        const uint half_size = block_size / 2;
        routed[block_base + (in_block >> 1) +
            ((in_block & 1u) ? half_size : 0)] = output / 2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup uint* swap = current;
        current = routed;
        routed = swap;
    }
    compressed_permutation[global_index] = current[local];
}

inline void multiply_add(thread U128& value, uint multiplier, uint addend) {
    const ulong low_half =
        (value.low & 0xfffffffful) * multiplier + addend;
    const ulong high_half =
        (value.low >> 32) * multiplier + (low_half >> 32);
    value.low = (high_half << 32) | (low_half & 0xfffffffful);
    value.high = value.high * multiplier + (high_half >> 32);
}

kernel void encode_middles(
    device const uint* permutation [[buffer(0)]],
    device U128* middles [[buffer(1)]],
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint value = permutation[index];
    for (uint i = 32; i-- > 1;) {
        const uint other = simd_shuffle(value, i);
        if (lane < i && value > other) {
            --value;
        }
    }
    U128 encoded{0, 0};
    for (uint i = 0; i < 32; ++i) {
        multiply_add(encoded, i + 1, simd_shuffle(value, i));
    }
    if (lane == 0) {
        middles[index / 32] = encoded;
    }
}

struct CompactNode {
    uint count;
    uint permutation_offset;
    uint left_offset;
    uint right_offset;
    ulong input_packet_base;
    ulong output_packet_base;
    ulong middle_bit_offset;
    uint input_root_count;
    uint input_path;
    uint output_root_count;
    uint output_path;
    uchar input_group_depth;
    uchar input_relative_level;
    uchar output_group_depth;
    uchar output_relative_level;
    uint reserved;
    ulong input_full_packet_bits;
    ulong input_full_packet_prefix;
    ulong output_full_packet_bits;
    ulong output_full_packet_prefix;
};

inline ulong compact_packet_prefix(
    uint root_count,
    uint group_depth,
    uint level,
    uint packet,
    uint path_count,
    uint leaf_size) {
    const uint nodes = 1u << level;
    const uint large_nodes = root_count % nodes;
    uint included_large = 0;
    uint remaining_paths = path_count;
    uint remaining_large = large_nodes;
    for (uint bits = level; bits != 0; --bits) {
        const uint midpoint = 1u << (bits - 1);
        if (remaining_large <= midpoint) {
            remaining_paths = (remaining_paths + 1) / 2;
        } else {
            included_large += (remaining_paths + 1) / 2;
            remaining_paths /= 2;
            remaining_large -= midpoint;
        }
    }
    if (remaining_paths != 0 && remaining_large != 0) {
        ++included_large;
    }
    const uint small_count = root_count / nodes;
    const ulong span = 1ul << (group_depth - level - 1);
    const ulong first = ulong(packet) * span;
    const auto contribution = [&](uint count) {
        if (count <= leaf_size || first >= count / 2) {
            return 0ul;
        }
        return min(span, ulong(count / 2) - first);
    };
    return ulong(included_large) * contribution(small_count + 1) +
        ulong(path_count - included_large) * contribution(small_count);
}

inline ulong compact_switch_bit(
    ulong packet_base,
    uint node_count,
    uint root_count,
    uint path,
    uint group_depth,
    uint relative_level,
    uint pair,
    uint leaf_size,
    ulong full_packet_bits,
    ulong full_packet_prefix) {
    const uint remaining = group_depth - relative_level;
    const uint span = 1u << (remaining - 1);
    const uint packet = pair / span;
    ulong packet_prefix = full_packet_prefix;
    const uint tail_packet = (node_count / 2 - 1) / span;
    if (packet == tail_packet) {
        packet_prefix = 0;
        for (uint level = 0; level < relative_level; ++level) {
            packet_prefix += compact_packet_prefix(root_count,
                group_depth, level, packet, 1u << level, leaf_size);
        }
        packet_prefix += compact_packet_prefix(root_count, group_depth,
            relative_level, packet, path, leaf_size);
    }
    return packet_base + ulong(packet) * full_packet_bits +
        packet_prefix +
        pair % span;
}

inline uint2 compact_find(
    device atomic_uint* parents,
    uint position) {
    uint parity = 0;
    while (true) {
        const uint packed = atomic_load_explicit(
            &parents[position], memory_order_relaxed);
        parity ^= packed >> 31;
        const uint next = packed & 0x7fffffffu;
        if (next == position) {
            return uint2(position, parity);
        }
        position = next;
    }
}

inline void compact_join(
    device atomic_uint* parents,
    uint left,
    uint right) {
    while (true) {
        const uint2 left_root = compact_find(parents, left);
        const uint2 right_root = compact_find(parents, right);
        if (left_root.x == right_root.x) {
            return;
        }
        const uint high = max(left_root.x, right_root.x);
        const uint low = min(left_root.x, right_root.x);
        const uint relation = left_root.y ^ right_root.y ^ 1u;
        uint expected = high;
        if (atomic_compare_exchange_weak_explicit(
                &parents[high], &expected, low | (relation << 31),
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

kernel void compact_inverse(
    device const uint* permutation [[buffer(0)]],
    device uint* inverse [[buffer(1)]],
    device const CompactNode* nodes [[buffer(2)]],
    constant uint& node_base [[buffer(3)]],
    uint2 index [[thread_position_in_grid]]) {
    const CompactNode node = nodes[node_base + index.y];
    if (index.x < node.count) {
        const uint offset = node.permutation_offset;
        inverse[offset + permutation[offset + index.x]] = index.x;
    }
}

kernel void compact_initialize(
    device atomic_uint* parents [[buffer(0)]],
    device atomic_uint* flips [[buffer(1)]],
    device const CompactNode* nodes [[buffer(2)]],
    constant uint& node_base [[buffer(3)]],
    uint2 index [[thread_position_in_grid]]) {
    const CompactNode node = nodes[node_base + index.y];
    if (index.x < node.count) {
        const uint position = node.permutation_offset + index.x;
        atomic_store_explicit(&parents[position], position, memory_order_relaxed);
        atomic_store_explicit(&flips[position], 0, memory_order_relaxed);
    }
}

kernel void compact_join_pairs(
    device const uint* inverse [[buffer(0)]],
    device atomic_uint* parents [[buffer(1)]],
    device const CompactNode* nodes [[buffer(2)]],
    constant uint& node_base [[buffer(3)]],
    uint2 index [[thread_position_in_grid]]) {
    const CompactNode node = nodes[node_base + index.y];
    if (index.x >= node.count / 2) {
        return;
    }
    const uint offset = node.permutation_offset;
    compact_join(parents, offset + index.x * 2, offset + index.x * 2 + 1);
    compact_join(parents, offset + inverse[offset + index.x * 2],
        offset + inverse[offset + index.x * 2 + 1]);
}

kernel void compact_orient_odd(
    device const uint* inverse [[buffer(0)]],
    device atomic_uint* parents [[buffer(1)]],
    device atomic_uint* flips [[buffer(2)]],
    device const CompactNode* nodes [[buffer(3)]],
    constant uint& node_base [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const CompactNode node = nodes[node_base + index];
    if ((node.count & 1u) == 0) {
        return;
    }
    const uint offset = node.permutation_offset;
    const uint tail = node.count - 1;
    const uint2 input_root = compact_find(parents, offset + tail);
    atomic_store_explicit(
        &flips[input_root.x], input_root.y, memory_order_relaxed);
    const uint2 output_root =
        compact_find(parents, offset + inverse[offset + tail]);
    atomic_store_explicit(
        &flips[output_root.x], output_root.y, memory_order_relaxed);
}

kernel void compact_route(
    device const uint* permutation [[buffer(0)]],
    device uint* routed [[buffer(1)]],
    device atomic_uint* parents [[buffer(2)]],
    device atomic_uint* flips [[buffer(3)]],
    device uint* colors [[buffer(4)]],
    device const CompactNode* nodes [[buffer(5)]],
    constant uint& node_base [[buffer(6)]],
    uint2 index [[thread_position_in_grid]]) {
    const CompactNode node = nodes[node_base + index.y];
    if (index.x >= node.count) {
        return;
    }
    const uint offset = node.permutation_offset;
    const uint2 root = compact_find(parents, offset + index.x);
    const uint branch = root.y ^ atomic_load_explicit(
        &flips[root.x], memory_order_relaxed);
    colors[offset + index.x] = branch;
    const uint destination =
        branch != 0 ? node.right_offset : node.left_offset;
    routed[destination + index.x / 2] =
        permutation[offset + index.x] / 2;
}

kernel void compact_pack_switches(
    device const uint* inverse [[buffer(0)]],
    device const uint* colors [[buffer(1)]],
    device atomic_uint* input_bits [[buffer(2)]],
    device atomic_uint* output_bits [[buffer(3)]],
    device const CompactNode* nodes [[buffer(4)]],
    constant uint& node_base [[buffer(5)]],
    constant uint& leaf_size [[buffer(6)]],
    uint2 index [[thread_position_in_grid]]) {
    const CompactNode node = nodes[node_base + index.y];
    const uint pairs = node.count / 2;
    if (index.x >= pairs) {
        return;
    }
    const uint offset = node.permutation_offset;
    const ulong input_bit = compact_switch_bit(node.input_packet_base,
        node.count,
        node.input_root_count, node.input_path, node.input_group_depth,
        node.input_relative_level, index.x, leaf_size,
        node.input_full_packet_bits, node.input_full_packet_prefix);
    const ulong output_bit = compact_switch_bit(node.output_packet_base,
        node.count,
        node.output_root_count, node.output_path, node.output_group_depth,
        node.output_relative_level, index.x, leaf_size,
        node.output_full_packet_bits, node.output_full_packet_prefix);
    if (colors[offset + index.x * 2] != 0) {
        atomic_fetch_or_explicit(
            &input_bits[input_bit / 32], 1u << (input_bit & 31),
            memory_order_relaxed);
    }
    if (colors[offset + inverse[offset + index.x * 2]] != 0) {
        atomic_fetch_or_explicit(
            &output_bits[output_bit / 32], 1u << (output_bit & 31),
            memory_order_relaxed);
    }
}

kernel void compact_encode_leaves(
    device const uint* permutation [[buffer(0)]],
    device atomic_uint* middle_bits [[buffer(1)]],
    device const CompactNode* leaves [[buffer(2)]],
    constant uint& leaf_base [[buffer(3)]],
    uint global_index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const CompactNode leaf = leaves[leaf_base + global_index / 32];
    if (leaf.count <= 1) {
        return;
    }
    uint value = lane < leaf.count
        ? permutation[leaf.permutation_offset + lane]
        : 0;
    for (uint i = leaf.count; i-- > 1;) {
        const uint other = simd_shuffle(value, i);
        if (lane < i && value > other) {
            --value;
        }
    }
    U128 encoded{0, 0};
    for (uint i = 0; i < leaf.count; ++i) {
        multiply_add(encoded, i + 1, simd_shuffle(value, i));
    }
    if (lane == 0) {
        U128 factorial{1, 0};
        for (uint i = 2; i <= leaf.count; ++i) {
            multiply_add(factorial, i, 0);
        }
        if (factorial.low-- == 0) {
            --factorial.high;
        }
        const uint bits = factorial.high != 0
            ? 64 + 64 - clz(factorial.high)
            : 64 - clz(factorial.low);
        for (uint i = 0; i < bits; ++i) {
            const ulong word = i < 64 ? encoded.low : encoded.high;
            if (((word >> (i & 63)) & 1ul) != 0) {
                const ulong bit = leaf.middle_bit_offset + i;
                atomic_fetch_or_explicit(
                    &middle_bits[bit / 32], 1u << (bit & 31),
                    memory_order_relaxed);
            }
        }
    }
}

struct CompactNode64 {
    ulong count;
    ulong permutation_offset;
    ulong left_offset;
    ulong right_offset;
    ulong input_packet_base;
    ulong output_packet_base;
    ulong middle_bit_offset;
    ulong input_root_count;
    ulong input_path;
    ulong output_root_count;
    ulong output_path;
    uchar input_group_depth;
    uchar input_relative_level;
    uchar output_group_depth;
    uchar output_relative_level;
    uint reserved;
    ulong input_full_packet_bits;
    ulong input_full_packet_prefix;
    ulong output_full_packet_bits;
    ulong output_full_packet_prefix;
};

inline ulong compact64_packet_prefix(
    ulong root_count,
    uint group_depth,
    uint level,
    ulong packet,
    ulong path_count,
    uint leaf_size) {
    const ulong nodes = 1ul << level;
    const ulong large_nodes = root_count % nodes;
    ulong included_large = 0;
    ulong remaining_paths = path_count;
    ulong remaining_large = large_nodes;
    for (uint bits = level; bits != 0; --bits) {
        const ulong midpoint = 1ul << (bits - 1);
        if (remaining_large <= midpoint) {
            remaining_paths = (remaining_paths + 1) / 2;
        } else {
            included_large += (remaining_paths + 1) / 2;
            remaining_paths /= 2;
            remaining_large -= midpoint;
        }
    }
    if (remaining_paths != 0 && remaining_large != 0) {
        ++included_large;
    }
    const ulong small_count = root_count / nodes;
    const ulong span = 1ul << (group_depth - level - 1);
    const ulong first = packet * span;
    const auto contribution = [&](ulong count) {
        if (count <= leaf_size || first >= count / 2) {
            return 0ul;
        }
        return min(span, count / 2 - first);
    };
    return included_large * contribution(small_count + 1) +
        (path_count - included_large) * contribution(small_count);
}

inline ulong compact64_switch_bit(
    ulong packet_base,
    ulong node_count,
    ulong root_count,
    ulong path,
    uint group_depth,
    uint relative_level,
    ulong pair,
    uint leaf_size,
    ulong full_packet_bits,
    ulong full_packet_prefix) {
    const uint remaining = group_depth - relative_level;
    const ulong span = 1ul << (remaining - 1);
    const ulong packet = pair / span;
    ulong packet_prefix = full_packet_prefix;
    const ulong tail_packet = (node_count / 2 - 1) / span;
    if (packet == tail_packet) {
        packet_prefix = 0;
        for (uint level = 0; level < relative_level; ++level) {
            packet_prefix += compact64_packet_prefix(root_count,
                group_depth, level, packet, 1ul << level, leaf_size);
        }
        packet_prefix += compact64_packet_prefix(root_count, group_depth,
            relative_level, packet, path, leaf_size);
    }
    return packet_base + packet * full_packet_bits +
        packet_prefix + pair % span;
}

inline ulong compact64_load(
    device ulong* parents,
    device atomic_uint* locks,
    ulong position) {
    device atomic_uint* parent_words =
        reinterpret_cast<device atomic_uint*>(parents);
    while (true) {
        const bool acquired = atomic_exchange_explicit(
            &locks[position], 1u, memory_order_relaxed) == 0;
        ulong value = 0;
        if (acquired) {
            atomic_thread_fence(
                mem_flags::mem_device, memory_order_seq_cst);
            const ulong low = atomic_load_explicit(
                &parent_words[position * 2], memory_order_relaxed);
            const ulong high = atomic_load_explicit(
                &parent_words[position * 2 + 1], memory_order_relaxed);
            value = low | (high << 32);
            atomic_thread_fence(
                mem_flags::mem_device, memory_order_seq_cst);
            atomic_store_explicit(
                &locks[position], 0u, memory_order_relaxed);
        }
        if (acquired) {
            return value;
        }
    }
}

inline bool compact64_compare_exchange(
    device ulong* parents,
    device atomic_uint* locks,
    ulong position,
    thread ulong& expected,
    ulong replacement) {
    device atomic_uint* parent_words =
        reinterpret_cast<device atomic_uint*>(parents);
    while (true) {
        const bool acquired = atomic_exchange_explicit(
            &locks[position], 1u, memory_order_relaxed) == 0;
        ulong actual = 0;
        bool equal = false;
        if (acquired) {
            atomic_thread_fence(
                mem_flags::mem_device, memory_order_seq_cst);
            const ulong low = atomic_load_explicit(
                &parent_words[position * 2], memory_order_relaxed);
            const ulong high = atomic_load_explicit(
                &parent_words[position * 2 + 1], memory_order_relaxed);
            actual = low | (high << 32);
            equal = actual == expected;
            if (equal) {
                atomic_store_explicit(
                    &parent_words[position * 2], uint(replacement),
                    memory_order_relaxed);
                atomic_store_explicit(
                    &parent_words[position * 2 + 1],
                    uint(replacement >> 32), memory_order_relaxed);
            }
            atomic_thread_fence(
                mem_flags::mem_device, memory_order_seq_cst);
            atomic_store_explicit(
                &locks[position], 0u, memory_order_relaxed);
        }
        if (acquired) {
            if (!equal) {
                expected = actual;
            }
            return equal;
        }
    }
}

inline ulong2 compact64_find(
    device ulong* parents,
    device atomic_uint* locks,
    ulong position) {
    ulong parity = 0;
    while (true) {
        const ulong packed = compact64_load(parents, locks, position);
        parity ^= packed >> 63;
        const ulong next = packed & 0x7ffffffffffffffful;
        if (next == position) {
            return ulong2(position, parity);
        }
        position = next;
    }
}

inline ulong2 compact64_find_readonly(
    device const ulong* parents,
    ulong position) {
    ulong parity = 0;
    while (true) {
        const ulong packed = parents[position];
        parity ^= packed >> 63;
        const ulong next = packed & 0x7ffffffffffffffful;
        if (next == position) {
            return ulong2(position, parity);
        }
        position = next;
    }
}

inline void compact64_join(
    device ulong* parents,
    device atomic_uint* locks,
    ulong left,
    ulong right) {
    while (true) {
        const ulong2 left_root = compact64_find(parents, locks, left);
        const ulong2 right_root = compact64_find(parents, locks, right);
        if (left_root.x == right_root.x) {
            return;
        }
        const ulong high = max(left_root.x, right_root.x);
        const ulong low = min(left_root.x, right_root.x);
        const ulong relation = left_root.y ^ right_root.y ^ 1ul;
        ulong expected = high;
        if (compact64_compare_exchange(parents, locks, high,
                expected, low | (relation << 63))) {
            return;
        }
    }
}

inline uint2 compact32_local_find(
    device atomic_uint* parents,
    ulong base,
    uint position) {
    uint parity = 0;
    while (true) {
        const uint packed = atomic_load_explicit(
            &parents[base + position], memory_order_relaxed);
        parity ^= packed >> 31;
        const uint next = packed & 0x7fffffffu;
        if (next == position) {
            return uint2(position, parity);
        }
        position = next;
    }
}

inline void compact32_local_join(
    device atomic_uint* parents,
    ulong base,
    uint left,
    uint right) {
    while (true) {
        const uint2 left_root =
            compact32_local_find(parents, base, left);
        const uint2 right_root =
            compact32_local_find(parents, base, right);
        if (left_root.x == right_root.x) {
            return;
        }
        const uint high = max(left_root.x, right_root.x);
        const uint low = min(left_root.x, right_root.x);
        const uint relation = left_root.y ^ right_root.y ^ 1u;
        uint expected = high;
        if (atomic_compare_exchange_weak_explicit(
                &parents[base + high], &expected,
                low | (relation << 31),
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

kernel void compact64_inverse(
    device const ulong* permutation [[buffer(0)]],
    device ulong* inverse [[buffer(1)]],
    device const CompactNode64* nodes [[buffer(2)]],
    constant ulong& node_base [[buffer(3)]],
    constant ulong& width [[buffer(4)]],
    constant ulong& work_base [[buffer(5)]],
    constant ulong& work_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong flat = work_base + gid;
    if (flat >= work_count) {
        return;
    }
    const CompactNode64 node = nodes[node_base + flat / width];
    const ulong index = flat % width;
    if (index < node.count) {
        const ulong offset = node.permutation_offset;
        inverse[offset + permutation[offset + index]] = index;
    }
}

kernel void compact64_32_initialize(
    device atomic_uint* parents [[buffer(0)]],
    device atomic_uint* flips [[buffer(1)]],
    device const CompactNode64* nodes [[buffer(2)]],
    constant ulong& node_base [[buffer(3)]],
    constant ulong& width [[buffer(4)]],
    constant ulong& work_base [[buffer(5)]],
    constant ulong& work_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong flat = work_base + gid;
    if (flat >= work_count) {
        return;
    }
    const CompactNode64 node = nodes[node_base + flat / width];
    const ulong index = flat % width;
    if (index < node.count) {
        const ulong position = node.permutation_offset + index;
        atomic_store_explicit(
            &parents[position], uint(index), memory_order_relaxed);
        atomic_store_explicit(&flips[position], 0, memory_order_relaxed);
    }
}

kernel void compact64_32_join_pairs(
    device const ulong* inverse [[buffer(0)]],
    device atomic_uint* parents [[buffer(1)]],
    device const CompactNode64* nodes [[buffer(2)]],
    constant ulong& node_base [[buffer(3)]],
    constant ulong& width [[buffer(4)]],
    constant ulong& work_base [[buffer(5)]],
    constant ulong& work_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong flat = work_base + gid;
    if (flat >= work_count) {
        return;
    }
    const CompactNode64 node = nodes[node_base + flat / width];
    const ulong pair = flat % width;
    if (pair >= node.count / 2) {
        return;
    }
    const ulong offset = node.permutation_offset;
    const uint local_pair = uint(pair);
    compact32_local_join(
        parents, offset, local_pair * 2, local_pair * 2 + 1);
    compact32_local_join(parents, offset,
        uint(inverse[offset + local_pair * 2]),
        uint(inverse[offset + local_pair * 2 + 1]));
}

kernel void compact64_32_orient_odd(
    device const ulong* inverse [[buffer(0)]],
    device atomic_uint* parents [[buffer(1)]],
    device atomic_uint* flips [[buffer(2)]],
    device const CompactNode64* nodes [[buffer(3)]],
    constant ulong& node_base [[buffer(4)]],
    constant ulong& work_base [[buffer(5)]],
    constant ulong& work_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong index = work_base + gid;
    if (index >= work_count) {
        return;
    }
    const CompactNode64 node = nodes[node_base + index];
    if ((node.count & 1ul) == 0) {
        return;
    }
    const ulong offset = node.permutation_offset;
    const uint tail = uint(node.count - 1);
    const uint2 input_root = compact32_local_find(parents, offset, tail);
    atomic_store_explicit(
        &flips[offset + input_root.x], input_root.y, memory_order_relaxed);
    const uint2 output_root =
        compact32_local_find(
            parents, offset, uint(inverse[offset + tail]));
    atomic_store_explicit(
        &flips[offset + output_root.x], output_root.y, memory_order_relaxed);
}

kernel void compact64_32_route(
    device const ulong* permutation [[buffer(0)]],
    device ulong* routed [[buffer(1)]],
    device atomic_uint* parents [[buffer(2)]],
    device atomic_uint* flips [[buffer(3)]],
    device uint* colors [[buffer(4)]],
    device const CompactNode64* nodes [[buffer(5)]],
    constant ulong& node_base [[buffer(6)]],
    constant ulong& width [[buffer(7)]],
    constant ulong& work_base [[buffer(8)]],
    constant ulong& work_count [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong flat = work_base + gid;
    if (flat >= work_count) {
        return;
    }
    const CompactNode64 node = nodes[node_base + flat / width];
    const ulong index = flat % width;
    if (index >= node.count) {
        return;
    }
    const ulong offset = node.permutation_offset;
    const uint local_index = uint(index);
    const uint2 root =
        compact32_local_find(parents, offset, local_index);
    const uint branch = root.y ^ atomic_load_explicit(
        &flips[offset + root.x], memory_order_relaxed);
    colors[offset + local_index] = branch;
    const ulong destination =
        branch != 0 ? node.right_offset : node.left_offset;
    routed[destination + index / 2] =
        permutation[offset + local_index] / 2;
}

kernel void compact64_initialize(
    device ulong* parents [[buffer(0)]],
    device atomic_uint* locks [[buffer(1)]],
    device atomic_uint* flips [[buffer(2)]],
    device const CompactNode64* nodes [[buffer(3)]],
    constant ulong& node_base [[buffer(4)]],
    constant ulong& width [[buffer(5)]],
    constant ulong& work_base [[buffer(6)]],
    constant ulong& work_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong flat = work_base + gid;
    if (flat >= work_count) {
        return;
    }
    const CompactNode64 node = nodes[node_base + flat / width];
    const ulong index = flat % width;
    if (index < node.count) {
        const ulong position = node.permutation_offset + index;
        device atomic_uint* parent_words =
            reinterpret_cast<device atomic_uint*>(parents);
        atomic_store_explicit(
            &parent_words[position * 2], uint(position),
            memory_order_relaxed);
        atomic_store_explicit(
            &parent_words[position * 2 + 1], uint(position >> 32),
            memory_order_relaxed);
        atomic_thread_fence(
            mem_flags::mem_device, memory_order_seq_cst);
        atomic_store_explicit(&locks[position], 0, memory_order_relaxed);
        atomic_store_explicit(&flips[position], 0, memory_order_relaxed);
    }
}

kernel void compact64_join_pairs(
    device const ulong* inverse [[buffer(0)]],
    device ulong* parents [[buffer(1)]],
    device atomic_uint* locks [[buffer(2)]],
    device const CompactNode64* nodes [[buffer(3)]],
    constant ulong& node_base [[buffer(4)]],
    constant ulong& width [[buffer(5)]],
    constant ulong& work_base [[buffer(6)]],
    constant ulong& work_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong flat = work_base + gid;
    if (flat >= work_count) {
        return;
    }
    const CompactNode64 node = nodes[node_base + flat / width];
    const ulong pair = flat % width;
    if (pair >= node.count / 2) {
        return;
    }
    const ulong offset = node.permutation_offset;
    compact64_join(
        parents, locks, offset + pair * 2, offset + pair * 2 + 1);
    compact64_join(parents, locks, offset + inverse[offset + pair * 2],
        offset + inverse[offset + pair * 2 + 1]);
}

kernel void compact64_orient_odd(
    device const ulong* inverse [[buffer(0)]],
    device ulong* parents [[buffer(1)]],
    device atomic_uint* locks [[buffer(2)]],
    device atomic_uint* flips [[buffer(3)]],
    device const CompactNode64* nodes [[buffer(4)]],
    constant ulong& node_base [[buffer(5)]],
    constant ulong& work_base [[buffer(6)]],
    constant ulong& work_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong index = work_base + gid;
    if (index >= work_count) {
        return;
    }
    const CompactNode64 node = nodes[node_base + index];
    if ((node.count & 1ul) == 0) {
        return;
    }
    const ulong offset = node.permutation_offset;
    const ulong tail = node.count - 1;
    const ulong2 input_root =
        compact64_find_readonly(parents, offset + tail);
    atomic_store_explicit(
        &flips[input_root.x], uint(input_root.y), memory_order_relaxed);
    const ulong2 output_root =
        compact64_find_readonly(parents, offset + inverse[offset + tail]);
    atomic_store_explicit(
        &flips[output_root.x], uint(output_root.y), memory_order_relaxed);
}

kernel void compact64_route(
    device const ulong* permutation [[buffer(0)]],
    device ulong* routed [[buffer(1)]],
    device ulong* parents [[buffer(2)]],
    device atomic_uint* locks [[buffer(3)]],
    device atomic_uint* flips [[buffer(4)]],
    device uint* colors [[buffer(5)]],
    device const CompactNode64* nodes [[buffer(6)]],
    constant ulong& node_base [[buffer(7)]],
    constant ulong& width [[buffer(8)]],
    constant ulong& work_base [[buffer(9)]],
    constant ulong& work_count [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong flat = work_base + gid;
    if (flat >= work_count) {
        return;
    }
    const CompactNode64 node = nodes[node_base + flat / width];
    const ulong index = flat % width;
    if (index >= node.count) {
        return;
    }
    const ulong offset = node.permutation_offset;
    (void)locks;
    const ulong2 root = compact64_find_readonly(parents, offset + index);
    const uint branch = uint(root.y) ^ atomic_load_explicit(
        &flips[root.x], memory_order_relaxed);
    colors[offset + index] = branch;
    const ulong destination =
        branch != 0 ? node.right_offset : node.left_offset;
    routed[destination + index / 2] =
        permutation[offset + index] / 2;
}

kernel void compact64_pack_switches(
    device const ulong* inverse [[buffer(0)]],
    device const uint* colors [[buffer(1)]],
    device atomic_uint* input_bits [[buffer(2)]],
    device atomic_uint* output_bits [[buffer(3)]],
    device const CompactNode64* nodes [[buffer(4)]],
    constant ulong& node_base [[buffer(5)]],
    constant uint& leaf_size [[buffer(6)]],
    constant ulong& width [[buffer(7)]],
    constant ulong& work_base [[buffer(8)]],
    constant ulong& work_count [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong flat = work_base + gid;
    if (flat >= work_count) {
        return;
    }
    const CompactNode64 node = nodes[node_base + flat / width];
    const ulong pair = flat % width;
    if (pair >= node.count / 2) {
        return;
    }
    const ulong offset = node.permutation_offset;
    const ulong input_bit = compact64_switch_bit(node.input_packet_base,
        node.count, node.input_root_count, node.input_path,
        node.input_group_depth, node.input_relative_level, pair, leaf_size,
        node.input_full_packet_bits, node.input_full_packet_prefix);
    const ulong output_bit = compact64_switch_bit(node.output_packet_base,
        node.count, node.output_root_count, node.output_path,
        node.output_group_depth, node.output_relative_level, pair, leaf_size,
        node.output_full_packet_bits, node.output_full_packet_prefix);
    if (colors[offset + pair * 2] != 0) {
        atomic_fetch_or_explicit(
            &input_bits[input_bit / 32], 1u << (input_bit & 31),
            memory_order_relaxed);
    }
    if (colors[offset + inverse[offset + pair * 2]] != 0) {
        atomic_fetch_or_explicit(
            &output_bits[output_bit / 32], 1u << (output_bit & 31),
            memory_order_relaxed);
    }
}

kernel void compact64_encode_leaves(
    device const ulong* permutation [[buffer(0)]],
    device atomic_uint* middle_bits [[buffer(1)]],
    device const CompactNode64* leaves [[buffer(2)]],
    constant ulong& leaf_base [[buffer(3)]],
    constant ulong& work_base [[buffer(4)]],
    constant ulong& work_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong global_index = work_base + gid;
    if (global_index >= work_count) {
        return;
    }
    const CompactNode64 leaf = leaves[leaf_base + global_index / 32];
    if (leaf.count <= 1) {
        return;
    }
    const uint count = uint(leaf.count);
    uint value = lane < count
        ? uint(permutation[leaf.permutation_offset + lane])
        : 0;
    for (uint i = count; i-- > 1;) {
        const uint other = simd_shuffle(value, i);
        if (lane < i && value > other) {
            --value;
        }
    }
    U128 encoded{0, 0};
    for (uint i = 0; i < count; ++i) {
        multiply_add(encoded, i + 1, simd_shuffle(value, i));
    }
    if (lane == 0) {
        U128 factorial{1, 0};
        for (uint i = 2; i <= count; ++i) {
            multiply_add(factorial, i, 0);
        }
        if (factorial.low-- == 0) {
            --factorial.high;
        }
        const uint bits = factorial.high != 0
            ? 64 + 64 - clz(factorial.high)
            : 64 - clz(factorial.low);
        for (uint i = 0; i < bits; ++i) {
            const ulong word = i < 64 ? encoded.low : encoded.high;
            if (((word >> (i & 63)) & 1ul) != 0) {
                const ulong bit = leaf.middle_bit_offset + i;
                atomic_fetch_or_explicit(
                    &middle_bits[bit / 32], 1u << (bit & 31),
                    memory_order_relaxed);
            }
        }
    }
}
