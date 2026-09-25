// SPDX-License-Identifier: Apache-2.0
#include "../internal.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace benes::detail {
namespace {

constexpr std::uint32_t threads_per_block = 256;

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

template<typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer_),
            std::max<std::size_t>(
                sizeof(T), checked_size_bytes(count, sizeof(T)))),
            "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return pointer_; }
    std::size_t size() const { return count_; }

private:
    T* pointer_ = nullptr;
    std::size_t count_;
};

struct DeviceU128 {
    std::uint64_t low;
    std::uint64_t high;
};

__device__ std::uint32_t root_index(
    std::uint32_t* parents,
    std::uint32_t position) {
    while (true) {
        const std::uint32_t parent = parents[position / 2];
        if ((position & ~std::uint32_t{1}) ==
            (parent & ~std::uint32_t{1})) {
            return position;
        }
        position = parent ^ (position & 1);
    }
}

__device__ std::uint32_t root_index_shared(
    std::uint32_t* parents,
    std::uint32_t position) {
    while (true) {
        const std::uint32_t parent = parents[position / 2];
        if ((position & ~std::uint32_t{1}) ==
            (parent & ~std::uint32_t{1})) {
            return position;
        }
        position = parent ^ (position & 1);
    }
}

__global__ void initialize_parents(
    std::uint32_t* parents,
    std::uint32_t block_size,
    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        parents[index] = (index & (block_size - 1)) * 2;
    }
}

__global__ void merge_cycles(
    const std::uint32_t* permutation,
    std::uint32_t* parents,
    std::uint32_t block_size,
    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::uint32_t base = index & ~(block_size - 1);
    const std::uint32_t local = index - base;
    std::uint32_t* block = parents + base;
    std::uint32_t left = local + (local & ~std::uint32_t{1});
    const std::uint32_t value = permutation[index];
    std::uint32_t right =
        (value + (value & ~std::uint32_t{1})) | std::uint32_t{2};
    while (true) {
        left = root_index(block, left);
        right = root_index(block, right);
        if (left == right) {
            return;
        }
        if (left > right) {
            const std::uint32_t temporary = left;
            left = right;
            right = temporary;
        }
        const std::uint32_t expected = right & ~std::uint32_t{1};
        const std::uint32_t replacement = left ^ (right & 1);
        if (atomicCAS(&block[right / 2], expected, replacement) == expected) {
            return;
        }
    }
}

__global__ void output_switches(
    std::uint32_t* parents,
    std::uint32_t* input_words,
    std::uint32_t* output_words,
    std::uint32_t block_size,
    std::uint32_t words_per_layer,
    std::uint32_t layer_index,
    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::uint32_t base = index & ~(block_size - 1);
    const std::uint32_t local = index - base;
    std::uint32_t* block = parents + base;
    const std::uint32_t mapped = (local & ~std::uint32_t{63}) |
        ((local >> 5) & 1) | ((local << 1) & 62);
    const std::uint32_t bit = root_index(block, mapped * 2) & 1;
    const std::uint32_t word = __ballot_sync(0xffffffffU, bit != 0);
    if ((threadIdx.x & 31) == 0) {
        const std::uint32_t word_index = layer_index * words_per_layer +
            (base + mapped) / 64;
        if ((mapped & 1) == 0) {
            input_words[word_index] = word;
        } else {
            output_words[word_index] = word;
        }
    }
}

__global__ void route_layer(
    const std::uint32_t* permutation,
    std::uint32_t* routed,
    const std::uint32_t* input_words,
    const std::uint32_t* output_words,
    std::uint32_t block_size,
    std::uint32_t words_per_layer,
    std::uint32_t layer_index,
    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::uint32_t base = index & ~(block_size - 1);
    const std::uint32_t local = index - base;
    const std::uint32_t input_switch = index / 2;
    const std::uint32_t input_word =
        input_words[layer_index * words_per_layer + input_switch / 32];
    const std::uint32_t input_bit =
        (input_word >> (input_switch & 31)) & 1;
    std::uint32_t output = permutation[index ^ input_bit];
    const std::uint32_t output_switch = (base + output) / 2;
    const std::uint32_t output_word =
        output_words[layer_index * words_per_layer + output_switch / 32];
    output ^= (output_word >> (output_switch & 31)) & 1;
    const std::uint32_t half = block_size / 2;
    routed[base + (local >> 1) + ((local & 1) ? half : 0)] = output / 2;
}

__global__ void compress_small_layers(
    const std::uint32_t* permutation,
    std::uint32_t* compressed_permutation,
    std::uint32_t* input_words,
    std::uint32_t* output_words,
    std::uint32_t log_size,
    std::uint32_t words_per_layer,
    std::uint32_t count) {
    const std::uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_index >= count) {
        return;
    }
    __shared__ std::uint32_t permutation_a[512];
    __shared__ std::uint32_t permutation_b[512];
    __shared__ std::uint32_t parents[512];
    const std::uint32_t local = threadIdx.x;
    const std::uint32_t global_base = global_index - local;
    permutation_a[local] = permutation[global_index];
    __syncthreads();

    std::uint32_t* current = permutation_a;
    std::uint32_t* routed = permutation_b;
    for (std::uint32_t layer = 9; layer > 5; --layer) {
        const std::uint32_t block_size = std::uint32_t{1} << layer;
        const std::uint32_t block_base = local & ~(block_size - 1);
        const std::uint32_t in_block = local - block_base;
        parents[local] = in_block * 2;
        __syncthreads();

        std::uint32_t* block_parents = parents + block_base;
        std::uint32_t left = in_block + (in_block & ~std::uint32_t{1});
        const std::uint32_t value = current[local];
        std::uint32_t right =
            (value + (value & ~std::uint32_t{1})) | std::uint32_t{2};
        while (true) {
            left = root_index_shared(block_parents, left);
            right = root_index_shared(block_parents, right);
            if (left == right) {
                break;
            }
            if (left > right) {
                const std::uint32_t temporary = left;
                left = right;
                right = temporary;
            }
            const std::uint32_t expected = right & ~std::uint32_t{1};
            const std::uint32_t replacement = left ^ (right & 1);
            if (atomicCAS(&block_parents[right / 2], expected, replacement) ==
                expected) {
                break;
            }
        }
        __syncthreads();

        const std::uint32_t mapped =
            (in_block & ~std::uint32_t{63}) |
            ((in_block >> 5) & 1) | ((in_block << 1) & 62);
        const std::uint32_t switch_bit =
            root_index_shared(block_parents, mapped * 2) & 1;
        const std::uint32_t word =
            __ballot_sync(0xffffffffU, switch_bit != 0);
        const std::uint32_t layer_index = log_size - layer;
        if ((threadIdx.x & 31) == 0) {
            const std::uint32_t word_index =
                layer_index * words_per_layer + global_base / 64 +
                (block_base + mapped) / 64;
            if ((mapped & 1) == 0) {
                input_words[word_index] = word;
            } else {
                output_words[word_index] = word;
            }
        }
        __syncthreads();

        const std::uint32_t input_switch = global_base / 2 + local / 2;
        const std::uint32_t input_word =
            input_words[layer_index * words_per_layer + input_switch / 32];
        const std::uint32_t input_bit =
            (input_word >> (input_switch & 31)) & 1;
        std::uint32_t output = current[local ^ input_bit];
        const std::uint32_t output_switch =
            global_base / 2 + (block_base + output) / 2;
        const std::uint32_t output_word =
            output_words[layer_index * words_per_layer + output_switch / 32];
        output ^= (output_word >> (output_switch & 31)) & 1;
        const std::uint32_t half = block_size / 2;
        routed[block_base + (in_block >> 1) +
            ((in_block & 1) ? half : 0)] = output / 2;
        __syncthreads();
        std::uint32_t* swap = current;
        current = routed;
        routed = swap;
    }
    compressed_permutation[global_index] = current[local];
}

__device__ void multiply_add(
    DeviceU128& value,
    std::uint32_t multiplier,
    std::uint32_t addend) {
    const std::uint64_t low_half =
        (value.low & 0xffffffffULL) * multiplier + addend;
    const std::uint64_t high_half =
        (value.low >> 32) * multiplier + (low_half >> 32);
    value.low = (high_half << 32) | (low_half & 0xffffffffULL);
    value.high = value.high * multiplier + (high_half >> 32);
}

__global__ void encode_middles(
    const std::uint32_t* permutation,
    DeviceU128* middles,
    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::uint32_t lane = threadIdx.x & 31;
    std::uint32_t value = permutation[index];
    for (std::uint32_t i = 32; i-- > 1;) {
        const std::uint32_t other = __shfl_sync(0xffffffffU, value, i);
        if (lane < i && value > other) {
            --value;
        }
    }
    DeviceU128 encoded{0, 0};
    for (std::uint32_t i = 0; i < 32; ++i) {
        multiply_add(encoded, i + 1,
            __shfl_sync(0xffffffffU, value, i));
    }
    if (lane == 0) {
        middles[index / 32] = encoded;
    }
}

__device__ uint2 compact_find(
    std::uint32_t* parents,
    std::uint32_t position) {
    std::uint32_t parity = 0;
    while (true) {
        const std::uint32_t packed = atomicAdd(&parents[position], 0U);
        parity ^= packed >> 31;
        const std::uint32_t next = packed & 0x7fffffffU;
        if (next == position) {
            return make_uint2(position, parity);
        }
        position = next;
    }
}

__device__ void compact_join(
    std::uint32_t* parents,
    std::uint32_t left,
    std::uint32_t right) {
    while (true) {
        const uint2 left_root = compact_find(parents, left);
        const uint2 right_root = compact_find(parents, right);
        if (left_root.x == right_root.x) {
            return;
        }
        const std::uint32_t high =
            left_root.x > right_root.x ? left_root.x : right_root.x;
        const std::uint32_t low =
            left_root.x < right_root.x ? left_root.x : right_root.x;
        const std::uint32_t relation = left_root.y ^ right_root.y ^ 1U;
        if (atomicCAS(&parents[high], high, low | (relation << 31)) == high) {
            return;
        }
    }
}

__global__ void compact_inverse(
    const std::uint32_t* permutation,
    std::uint32_t* inverse,
    const CompactNode* nodes,
    std::uint32_t node_base,
    std::uint32_t width,
    std::uint32_t node_count) {
    const std::uint64_t flat =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total =
        static_cast<std::uint64_t>(width) * node_count;
    if (flat >= total) {
        return;
    }
    const std::uint32_t node_index = static_cast<std::uint32_t>(flat / width);
    const std::uint32_t index = static_cast<std::uint32_t>(flat % width);
    const CompactNode node = nodes[node_base + node_index];
    if (index < node.count) {
        const std::uint32_t offset = node.permutation_offset;
        inverse[offset + permutation[offset + index]] = index;
    }
}

__global__ void compact_initialize(
    std::uint32_t* parents,
    std::uint32_t* flips,
    const CompactNode* nodes,
    std::uint32_t node_base,
    std::uint32_t width,
    std::uint32_t node_count) {
    const std::uint64_t flat =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total =
        static_cast<std::uint64_t>(width) * node_count;
    if (flat >= total) {
        return;
    }
    const std::uint32_t node_index = static_cast<std::uint32_t>(flat / width);
    const std::uint32_t index = static_cast<std::uint32_t>(flat % width);
    const CompactNode node = nodes[node_base + node_index];
    if (index < node.count) {
        const std::uint32_t position = node.permutation_offset + index;
        atomicExch(&parents[position], position);
        atomicExch(&flips[position], 0U);
    }
}

__global__ void compact_join_pairs(
    const std::uint32_t* inverse,
    std::uint32_t* parents,
    const CompactNode* nodes,
    std::uint32_t node_base,
    std::uint32_t width,
    std::uint32_t node_count) {
    const std::uint64_t flat =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total =
        static_cast<std::uint64_t>(width) * node_count;
    if (flat >= total) {
        return;
    }
    const std::uint32_t node_index = static_cast<std::uint32_t>(flat / width);
    const std::uint32_t pair = static_cast<std::uint32_t>(flat % width);
    const CompactNode node = nodes[node_base + node_index];
    if (pair >= node.count / 2) {
        return;
    }
    const std::uint32_t offset = node.permutation_offset;
    compact_join(parents, offset + pair * 2, offset + pair * 2 + 1);
    compact_join(parents, offset + inverse[offset + pair * 2],
        offset + inverse[offset + pair * 2 + 1]);
}

__global__ void compact_orient_odd(
    const std::uint32_t* inverse,
    std::uint32_t* parents,
    std::uint32_t* flips,
    const CompactNode* nodes,
    std::uint32_t node_base,
    std::uint32_t node_count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= node_count) {
        return;
    }
    const CompactNode node = nodes[node_base + index];
    if ((node.count & 1U) == 0) {
        return;
    }
    const std::uint32_t offset = node.permutation_offset;
    const std::uint32_t tail = node.count - 1;
    const uint2 input_root = compact_find(parents, offset + tail);
    atomicExch(&flips[input_root.x], input_root.y);
    const uint2 output_root =
        compact_find(parents, offset + inverse[offset + tail]);
    atomicExch(&flips[output_root.x], output_root.y);
}

__global__ void compact_route(
    const std::uint32_t* permutation,
    std::uint32_t* routed,
    std::uint32_t* parents,
    std::uint32_t* flips,
    std::uint32_t* colors,
    const CompactNode* nodes,
    std::uint32_t node_base,
    std::uint32_t width,
    std::uint32_t node_count) {
    const std::uint64_t flat =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total =
        static_cast<std::uint64_t>(width) * node_count;
    if (flat >= total) {
        return;
    }
    const std::uint32_t node_index = static_cast<std::uint32_t>(flat / width);
    const std::uint32_t index = static_cast<std::uint32_t>(flat % width);
    const CompactNode node = nodes[node_base + node_index];
    if (index >= node.count) {
        return;
    }
    const std::uint32_t offset = node.permutation_offset;
    const uint2 root = compact_find(parents, offset + index);
    const std::uint32_t branch =
        root.y ^ atomicAdd(&flips[root.x], 0U);
    colors[offset + index] = branch;
    const std::uint32_t destination =
        branch != 0 ? node.right_offset : node.left_offset;
    routed[destination + index / 2] =
        permutation[offset + index] / 2;
}

__device__ std::uint64_t compact_packet_contribution(
    std::uint32_t count,
    std::uint32_t leaf_size,
    std::uint64_t first,
    std::uint64_t span) {
    if (count <= leaf_size || first >= count / 2) {
        return 0;
    }
    const std::uint64_t available = count / 2 - first;
    return span < available ? span : available;
}

__device__ std::uint64_t compact_packet_prefix(
    std::uint32_t root_count,
    std::uint32_t group_depth,
    std::uint32_t level,
    std::uint32_t packet,
    std::uint32_t path_count,
    std::uint32_t leaf_size) {
    const std::uint32_t nodes = 1U << level;
    const std::uint32_t large_nodes = root_count % nodes;
    std::uint32_t included_large = 0;
    std::uint32_t remaining_paths = path_count;
    std::uint32_t remaining_large = large_nodes;
    for (std::uint32_t bits = level; bits != 0; --bits) {
        const std::uint32_t midpoint = 1U << (bits - 1);
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
    const std::uint32_t small_count = root_count / nodes;
    const std::uint64_t span =
        std::uint64_t{1} << (group_depth - level - 1);
    const std::uint64_t first =
        static_cast<std::uint64_t>(packet) * span;
    return static_cast<std::uint64_t>(included_large) *
            compact_packet_contribution(
                small_count + 1, leaf_size, first, span) +
        static_cast<std::uint64_t>(path_count - included_large) *
            compact_packet_contribution(
                small_count, leaf_size, first, span);
}

__device__ std::uint64_t compact_switch_bit(
    std::uint64_t packet_base,
    std::uint32_t node_count,
    std::uint32_t root_count,
    std::uint32_t path,
    std::uint32_t group_depth,
    std::uint32_t relative_level,
    std::uint32_t pair,
    std::uint32_t leaf_size,
    std::uint64_t full_packet_bits,
    std::uint64_t full_packet_prefix) {
    const std::uint32_t remaining = group_depth - relative_level;
    const std::uint32_t span = 1U << (remaining - 1);
    const std::uint32_t packet = pair / span;
    std::uint64_t packet_prefix = full_packet_prefix;
    const std::uint32_t tail_packet = (node_count / 2 - 1) / span;
    if (packet == tail_packet) {
        packet_prefix = 0;
        for (std::uint32_t level = 0; level < relative_level; ++level) {
            packet_prefix += compact_packet_prefix(root_count,
                group_depth, level, packet, 1U << level, leaf_size);
        }
        packet_prefix += compact_packet_prefix(root_count, group_depth,
            relative_level, packet, path, leaf_size);
    }
    return packet_base +
        static_cast<std::uint64_t>(packet) * full_packet_bits +
        packet_prefix +
        pair % span;
}

__global__ void compact_pack_switches(
    const std::uint32_t* inverse,
    const std::uint32_t* colors,
    std::uint32_t* input_bits,
    std::uint32_t* output_bits,
    const CompactNode* nodes,
    std::uint32_t node_base,
    std::uint32_t width,
    std::uint32_t node_count,
    std::uint32_t leaf_size) {
    const std::uint64_t flat =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total =
        static_cast<std::uint64_t>(width) * node_count;
    if (flat >= total) {
        return;
    }
    const std::uint32_t node_index = static_cast<std::uint32_t>(flat / width);
    const std::uint32_t pair = static_cast<std::uint32_t>(flat % width);
    const CompactNode node = nodes[node_base + node_index];
    const std::uint32_t pairs = node.count / 2;
    if (pair >= pairs) {
        return;
    }
    const std::uint32_t offset = node.permutation_offset;
    const std::uint64_t input_bit = compact_switch_bit(
        node.input_packet_base, node.count,
        node.input_root_count, node.input_path,
        node.input_group_depth, node.input_relative_level, pair, leaf_size,
        node.input_full_packet_bits, node.input_full_packet_prefix);
    const std::uint64_t output_bit = compact_switch_bit(
        node.output_packet_base, node.count,
        node.output_root_count, node.output_path,
        node.output_group_depth, node.output_relative_level, pair, leaf_size,
        node.output_full_packet_bits, node.output_full_packet_prefix);
    if (colors[offset + pair * 2] != 0) {
        atomicOr(&input_bits[input_bit / 32],
            1U << (input_bit & 31));
    }
    if (colors[offset + inverse[offset + pair * 2]] != 0) {
        atomicOr(&output_bits[output_bit / 32],
            1U << (output_bit & 31));
    }
}

__global__ void compact_encode_leaves(
    const std::uint32_t* permutation,
    std::uint32_t* middle_bits,
    const CompactNode* leaves,
    std::uint32_t leaf_base,
    std::uint32_t leaf_count) {
    const std::uint64_t global_index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (global_index >= static_cast<std::uint64_t>(leaf_count) * 32) {
        return;
    }
    const std::uint32_t lane = threadIdx.x & 31;
    const CompactNode leaf =
        leaves[leaf_base + static_cast<std::uint32_t>(global_index / 32)];
    if (leaf.count <= 1) {
        return;
    }
    std::uint32_t value = lane < leaf.count
        ? permutation[leaf.permutation_offset + lane]
        : 0;
    for (std::uint32_t i = leaf.count; i-- > 1;) {
        const std::uint32_t other =
            __shfl_sync(0xffffffffU, value, static_cast<int>(i));
        if (lane < i && value > other) {
            --value;
        }
    }
    DeviceU128 encoded{0, 0};
    for (std::uint32_t i = 0; i < leaf.count; ++i) {
        multiply_add(encoded, i + 1,
            __shfl_sync(0xffffffffU, value, static_cast<int>(i)));
    }
    if (lane == 0) {
        DeviceU128 factorial{1, 0};
        for (std::uint32_t i = 2; i <= leaf.count; ++i) {
            multiply_add(factorial, i, 0);
        }
        if (factorial.low-- == 0) {
            --factorial.high;
        }
        const std::uint32_t bit_count = factorial.high != 0
            ? 64 + 64 - __clzll(factorial.high)
            : 64 - __clzll(factorial.low);
        for (std::uint32_t bit = 0; bit < bit_count; ++bit) {
            const std::uint64_t word =
                bit < 64 ? encoded.low : encoded.high;
            if (((word >> (bit & 63)) & 1) != 0) {
                const std::uint64_t destination =
                    leaf.middle_bit_offset + bit;
                atomicOr(&middle_bits[destination / 32],
                    1U << (destination & 31));
            }
        }
    }
}

__device__ ulonglong2 compact_find64(
    std::uint64_t* parents,
    std::uint64_t position) {
    std::uint64_t parity = 0;
    while (true) {
        const auto* address = reinterpret_cast<unsigned long long*>(
            &parents[position]);
        const std::uint64_t packed = atomicCAS(address, 0, 0);
        parity ^= packed >> 63;
        const std::uint64_t next = packed & 0x7fffffffffffffffULL;
        if (next == position) {
            return make_ulonglong2(position, parity);
        }
        position = next;
    }
}

__device__ void compact_join64(
    std::uint64_t* parents,
    std::uint64_t left,
    std::uint64_t right) {
    while (true) {
        const ulonglong2 left_root = compact_find64(parents, left);
        const ulonglong2 right_root = compact_find64(parents, right);
        if (left_root.x == right_root.x) {
            return;
        }
        const std::uint64_t high =
            left_root.x > right_root.x ? left_root.x : right_root.x;
        const std::uint64_t low =
            left_root.x < right_root.x ? left_root.x : right_root.x;
        const std::uint64_t relation = left_root.y ^ right_root.y ^ 1ULL;
        auto* address = reinterpret_cast<unsigned long long*>(&parents[high]);
        if (atomicCAS(address, high, low | (relation << 63)) == high) {
            return;
        }
    }
}

__device__ std::uint64_t compact64_packet_contribution(
    std::uint64_t count,
    std::uint32_t leaf_size,
    std::uint64_t first,
    std::uint64_t span) {
    if (count <= leaf_size || first >= count / 2) {
        return 0;
    }
    const std::uint64_t available = count / 2 - first;
    return span < available ? span : available;
}

__device__ std::uint64_t compact64_packet_prefix(
    std::uint64_t root_count,
    std::uint32_t group_depth,
    std::uint32_t level,
    std::uint64_t packet,
    std::uint64_t path_count,
    std::uint32_t leaf_size) {
    const std::uint64_t nodes = std::uint64_t{1} << level;
    const std::uint64_t large_nodes = root_count % nodes;
    std::uint64_t included_large = 0;
    std::uint64_t remaining_paths = path_count;
    std::uint64_t remaining_large = large_nodes;
    for (std::uint32_t bits = level; bits != 0; --bits) {
        const std::uint64_t midpoint = std::uint64_t{1} << (bits - 1);
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
    const std::uint64_t small_count = root_count / nodes;
    const std::uint64_t span =
        std::uint64_t{1} << (group_depth - level - 1);
    const std::uint64_t first = packet * span;
    return included_large * compact64_packet_contribution(
            small_count + 1, leaf_size, first, span) +
        (path_count - included_large) * compact64_packet_contribution(
            small_count, leaf_size, first, span);
}

__device__ std::uint64_t compact64_switch_bit(
    const CompactNode64& node,
    std::uint64_t pair,
    std::uint32_t leaf_size,
    bool output) {
    const std::uint32_t group_depth =
        output ? node.output_group_depth : node.input_group_depth;
    const std::uint32_t relative_level =
        output ? node.output_relative_level : node.input_relative_level;
    const std::uint64_t root_count =
        output ? node.output_root_count : node.input_root_count;
    const std::uint64_t path =
        output ? node.output_path : node.input_path;
    const std::uint64_t remaining = group_depth - relative_level;
    const std::uint64_t span = std::uint64_t{1} << (remaining - 1);
    const std::uint64_t packet = pair / span;
    std::uint64_t packet_prefix = output
        ? node.output_full_packet_prefix
        : node.input_full_packet_prefix;
    const std::uint64_t tail_packet = (node.count / 2 - 1) / span;
    if (packet == tail_packet) {
        packet_prefix = 0;
        for (std::uint32_t level = 0; level < relative_level; ++level) {
            packet_prefix += compact64_packet_prefix(root_count,
                group_depth, level, packet,
                std::uint64_t{1} << level, leaf_size);
        }
        packet_prefix += compact64_packet_prefix(root_count, group_depth,
            relative_level, packet, path, leaf_size);
    }
    const std::uint64_t packet_base =
        output ? node.output_packet_base : node.input_packet_base;
    const std::uint64_t full_packet_bits =
        output ? node.output_full_packet_bits : node.input_full_packet_bits;
    return packet_base + packet * full_packet_bits +
        packet_prefix + pair % span;
}

__global__ void compact64_inverse(
    const std::uint64_t* permutation,
    std::uint64_t* inverse,
    const CompactNode64* nodes,
    std::uint64_t node_base,
    std::uint64_t width,
    std::uint64_t work) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t flat =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < work; flat += stride) {
        const CompactNode64 node = nodes[node_base + flat / width];
        const std::uint64_t index = flat % width;
        if (index < node.count) {
            const std::uint64_t offset = node.permutation_offset;
            inverse[offset + permutation[offset + index]] = index;
        }
    }
}

__global__ void compact64_initialize(
    std::uint64_t* parents,
    std::uint32_t* flips,
    const CompactNode64* nodes,
    std::uint64_t node_base,
    std::uint64_t width,
    std::uint64_t work) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t flat =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < work; flat += stride) {
        const CompactNode64 node = nodes[node_base + flat / width];
        const std::uint64_t index = flat % width;
        if (index < node.count) {
            const std::uint64_t position = node.permutation_offset + index;
            atomicExch(reinterpret_cast<unsigned long long*>(&parents[position]),
                position);
            atomicExch(&flips[position], 0U);
        }
    }
}

__global__ void compact64_join_pairs(
    const std::uint64_t* inverse,
    std::uint64_t* parents,
    const CompactNode64* nodes,
    std::uint64_t node_base,
    std::uint64_t width,
    std::uint64_t work) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t flat =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < work; flat += stride) {
        const CompactNode64 node = nodes[node_base + flat / width];
        const std::uint64_t pair = flat % width;
        if (pair >= node.count / 2) {
            continue;
        }
        const std::uint64_t offset = node.permutation_offset;
        compact_join64(
            parents, offset + pair * 2, offset + pair * 2 + 1);
        compact_join64(parents, offset + inverse[offset + pair * 2],
            offset + inverse[offset + pair * 2 + 1]);
    }
}

__global__ void compact64_orient_odd(
    const std::uint64_t* inverse,
    std::uint64_t* parents,
    std::uint32_t* flips,
    const CompactNode64* nodes,
    std::uint64_t node_base,
    std::uint64_t work) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < work; index += stride) {
        const CompactNode64 node = nodes[node_base + index];
        if ((node.count & 1ULL) == 0) {
            continue;
        }
        const std::uint64_t offset = node.permutation_offset;
        const std::uint64_t tail = node.count - 1;
        const ulonglong2 input_root = compact_find64(parents, offset + tail);
        atomicExch(&flips[input_root.x],
            static_cast<std::uint32_t>(input_root.y));
        const ulonglong2 output_root =
            compact_find64(parents, offset + inverse[offset + tail]);
        atomicExch(&flips[output_root.x],
            static_cast<std::uint32_t>(output_root.y));
    }
}

__global__ void compact64_route(
    const std::uint64_t* permutation,
    std::uint64_t* routed,
    std::uint64_t* parents,
    std::uint32_t* flips,
    std::uint32_t* colors,
    const CompactNode64* nodes,
    std::uint64_t node_base,
    std::uint64_t width,
    std::uint64_t work) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t flat =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < work; flat += stride) {
        const CompactNode64 node = nodes[node_base + flat / width];
        const std::uint64_t index = flat % width;
        if (index >= node.count) {
            continue;
        }
        const std::uint64_t offset = node.permutation_offset;
        const ulonglong2 root = compact_find64(parents, offset + index);
        const std::uint32_t branch =
            static_cast<std::uint32_t>(root.y) ^ atomicAdd(&flips[root.x], 0U);
        colors[offset + index] = branch;
        const std::uint64_t destination =
            branch != 0 ? node.right_offset : node.left_offset;
        routed[destination + index / 2] =
            permutation[offset + index] / 2;
    }
}

__global__ void compact64_pack_switches(
    const std::uint64_t* inverse,
    const std::uint32_t* colors,
    std::uint32_t* input_bits,
    std::uint32_t* output_bits,
    const CompactNode64* nodes,
    std::uint64_t node_base,
    std::uint64_t width,
    std::uint64_t work,
    std::uint32_t leaf_size) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t flat =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < work; flat += stride) {
        const CompactNode64 node = nodes[node_base + flat / width];
        const std::uint64_t pair = flat % width;
        if (pair >= node.count / 2) {
            continue;
        }
        const std::uint64_t offset = node.permutation_offset;
        const std::uint64_t input_bit =
            compact64_switch_bit(node, pair, leaf_size, false);
        const std::uint64_t output_bit =
            compact64_switch_bit(node, pair, leaf_size, true);
        if (colors[offset + pair * 2] != 0) {
            atomicOr(&input_bits[input_bit / 32],
                1U << (input_bit & 31));
        }
        if (colors[offset + inverse[offset + pair * 2]] != 0) {
            atomicOr(&output_bits[output_bit / 32],
                1U << (output_bit & 31));
        }
    }
}

__global__ void compact64_encode_leaves(
    const std::uint64_t* permutation,
    std::uint32_t* middle_bits,
    const CompactNode64* leaves,
    std::uint64_t leaf_base,
    std::uint64_t work) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t global_index =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         global_index < work; global_index += stride) {
        const std::uint32_t lane = threadIdx.x & 31;
        const CompactNode64 leaf = leaves[leaf_base + global_index / 32];
        if (leaf.count <= 1) {
            continue;
        }
        const std::uint32_t count =
            static_cast<std::uint32_t>(leaf.count);
        std::uint32_t value = lane < count
            ? static_cast<std::uint32_t>(
                  permutation[leaf.permutation_offset + lane])
            : 0;
        for (std::uint32_t i = count; i-- > 1;) {
            const std::uint32_t other =
                __shfl_sync(0xffffffffU, value, static_cast<int>(i));
            if (lane < i && value > other) {
                --value;
            }
        }
        DeviceU128 encoded{0, 0};
        for (std::uint32_t i = 0; i < count; ++i) {
            multiply_add(encoded, i + 1,
                __shfl_sync(0xffffffffU, value, static_cast<int>(i)));
        }
        if (lane == 0) {
            DeviceU128 factorial{1, 0};
            for (std::uint32_t i = 2; i <= count; ++i) {
                multiply_add(factorial, i, 0);
            }
            if (factorial.low-- == 0) {
                --factorial.high;
            }
            const std::uint32_t bit_count = factorial.high != 0
                ? 64 + 64 - __clzll(factorial.high)
                : 64 - __clzll(factorial.low);
            for (std::uint32_t bit = 0; bit < bit_count; ++bit) {
                const std::uint64_t word =
                    bit < 64 ? encoded.low : encoded.high;
                if (((word >> (bit & 63)) & 1) != 0) {
                    const std::uint64_t destination =
                        leaf.middle_bit_offset + bit;
                    atomicOr(&middle_bits[destination / 32],
                        1U << (destination & 31));
                }
            }
        }
    }
}

void validate(std::span<const std::uint32_t> permutation) {
    if (permutation.empty()) {
        throw std::invalid_argument("permutation must not be empty");
    }
    std::vector<bool> seen(permutation.size());
    for (const std::uint32_t value : permutation) {
        if (value >= permutation.size() || seen[value]) {
            throw std::invalid_argument("input is not a permutation");
        }
        seen[value] = true;
    }
}

void validate(std::span<const std::uint64_t> permutation) {
    if (permutation.empty()) {
        throw std::invalid_argument("permutation must not be empty");
    }
    std::vector<bool> seen(permutation.size());
    for (const std::uint64_t value : permutation) {
        if (value >= permutation.size() ||
            seen[static_cast<std::size_t>(value)]) {
            throw std::invalid_argument("input is not a permutation");
        }
        seen[static_cast<std::size_t>(value)] = true;
    }
}

} // namespace

bool cuda_available() noexcept {
    int count = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

std::vector<std::byte> compress_cuda(
    std::span<const std::uint32_t> source) {
    validate(source);
    const std::uint64_t network64 =
        std::bit_ceil(std::max<std::uint64_t>(32, source.size()));
    if (network64 > (std::uint64_t{1} << 31)) {
        throw std::invalid_argument("permutation is too large");
    }
    const auto network_size = static_cast<std::uint32_t>(network64);
    const auto log_size =
        static_cast<std::uint32_t>(std::bit_width(network_size) - 1);
    const std::uint32_t words_per_layer = network_size / 64;
    const std::size_t word_count =
        static_cast<std::size_t>(log_size - 5) * words_per_layer;
    std::vector<std::uint32_t> padded(network_size);
    std::copy(source.begin(), source.end(), padded.begin());
    std::iota(padded.begin() + source.size(), padded.end(),
        static_cast<std::uint32_t>(source.size()));

    DeviceBuffer<std::uint32_t> permutation(network_size);
    DeviceBuffer<std::uint32_t> temporary(network_size);
    DeviceBuffer<std::uint32_t> parents(network_size);
    DeviceBuffer<std::uint32_t> input_words(word_count);
    DeviceBuffer<std::uint32_t> output_words(word_count);
    DeviceBuffer<DeviceU128> middles(network_size / 32);
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");
    try {
        cuda_check(cudaMemcpyAsync(permutation.get(), padded.data(),
            padded.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync permutation");
        const std::uint32_t blocks =
            (network_size + threads_per_block - 1) / threads_per_block;
        std::uint32_t* current = permutation.get();
        std::uint32_t* next = temporary.get();
        cudaEvent_t compute_start = nullptr;
        cudaEvent_t compute_end = nullptr;
        cuda_check(cudaEventCreate(&compute_start), "cudaEventCreate");
        cuda_check(cudaEventCreate(&compute_end), "cudaEventCreate");
        cuda_check(cudaEventRecord(compute_start, stream), "cudaEventRecord");
        const std::uint32_t last_separate_layer =
            network_size >= 512 ? 9 : 5;
        for (std::uint32_t layer = log_size;
            layer > last_separate_layer; --layer) {
            const std::uint32_t block_size = std::uint32_t{1} << layer;
            const std::uint32_t layer_index = log_size - layer;
            initialize_parents<<<blocks, threads_per_block, 0, stream>>>(
                parents.get(), block_size, network_size);
            merge_cycles<<<blocks, threads_per_block, 0, stream>>>(
                current, parents.get(), block_size, network_size);
            output_switches<<<blocks, threads_per_block, 0, stream>>>(
                parents.get(), input_words.get(), output_words.get(),
                block_size, words_per_layer, layer_index, network_size);
            route_layer<<<blocks, threads_per_block, 0, stream>>>(
                current, next, input_words.get(), output_words.get(),
                block_size, words_per_layer, layer_index, network_size);
            cuda_check(cudaGetLastError(), "CUDA layer launch");
            std::swap(current, next);
        }
        if (network_size >= 512) {
            const std::uint32_t fused_blocks = network_size / 512;
            compress_small_layers<<<fused_blocks, 512, 0, stream>>>(
                current, next, input_words.get(), output_words.get(),
                log_size, words_per_layer, network_size);
            cuda_check(cudaGetLastError(), "CUDA fused layer launch");
            std::swap(current, next);
        }
        encode_middles<<<blocks, threads_per_block, 0, stream>>>(
            current, middles.get(), network_size);
        cuda_check(cudaGetLastError(), "CUDA middle launch");
        cuda_check(cudaEventRecord(compute_end, stream), "cudaEventRecord");

        std::vector<std::uint32_t> input_host(word_count);
        std::vector<std::uint32_t> output_host(word_count);
        std::vector<U128> middle_host(network_size / 32);
        if (word_count != 0) {
            cuda_check(cudaMemcpyAsync(input_host.data(), input_words.get(),
                word_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync input words");
            cuda_check(cudaMemcpyAsync(output_host.data(), output_words.get(),
                word_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync output words");
        }
        static_assert(sizeof(U128) == sizeof(DeviceU128));
        cuda_check(cudaMemcpyAsync(middle_host.data(), middles.get(),
            middle_host.size() * sizeof(U128), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync middles");
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        float compute_milliseconds = 0;
        cuda_check(cudaEventElapsedTime(
            &compute_milliseconds, compute_start, compute_end),
            "cudaEventElapsedTime");
        set_last_device_seconds(compute_milliseconds / 1000.0);
        cudaEventDestroy(compute_start);
        cudaEventDestroy(compute_end);
        cudaStreamDestroy(stream);
        return make_blob(static_cast<std::uint32_t>(source.size()), network_size,
            input_host, output_host, middle_host);
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }
}

std::vector<std::byte> compress_compact_cuda(
    std::span<const std::uint32_t> source,
    FormatOptions format) {
    validate(source);
    if (source.size() > 0x7fffffffU) {
        throw std::invalid_argument("permutation is too large for CUDA");
    }
    const auto count = static_cast<std::uint32_t>(source.size());
    const FormatOptions normalized_format =
        normalize_format(count, std::move(format));
    const Layout layout = make_layout(count, normalized_format);
    const CompactTree tree =
        make_compact_tree(count, normalized_format);
    DeviceBuffer<std::uint32_t> permutation(count);
    DeviceBuffer<std::uint32_t> routed(count);
    DeviceBuffer<std::uint32_t> inverse(count);
    DeviceBuffer<std::uint32_t> parents(count);
    DeviceBuffer<std::uint32_t> flips(count);
    DeviceBuffer<std::uint32_t> colors(count);
    DeviceBuffer<CompactNode> nodes(tree.nodes.size());
    DeviceBuffer<CompactNode> leaves(tree.leaves.size());
    DeviceBuffer<std::uint32_t> input_bits(
        static_cast<std::size_t>((tree.input_section_bits + 31) / 32));
    DeviceBuffer<std::uint32_t> output_bits(
        static_cast<std::size_t>((tree.output_section_bits + 31) / 32));
    DeviceBuffer<std::uint32_t> middle_bits(
        static_cast<std::size_t>((tree.middle_section_bits + 31) / 32));
    std::uint32_t* middle_destination = middle_bits.get();
    if (normalized_format.middle_placement ==
        MiddlePlacement::embedded_input) {
        middle_destination = input_bits.get();
    } else if (
        normalized_format.middle_placement ==
        MiddlePlacement::embedded_output) {
        middle_destination = output_bits.get();
    }

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");
    cudaEvent_t compute_start = nullptr;
    cudaEvent_t compute_end = nullptr;
    try {
        cuda_check(cudaMemcpyAsync(permutation.get(), source.data(),
            source.size_bytes(), cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync compact permutation");
        if (!tree.nodes.empty()) {
            cuda_check(cudaMemcpyAsync(nodes.get(), tree.nodes.data(),
                tree.nodes.size() * sizeof(CompactNode),
                cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync compact nodes");
        }
        cuda_check(cudaMemcpyAsync(leaves.get(), tree.leaves.data(),
            tree.leaves.size() * sizeof(CompactNode),
            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync compact leaves");
        cuda_check(cudaMemsetAsync(input_bits.get(), 0,
            input_bits.size() * sizeof(std::uint32_t), stream),
            "cudaMemsetAsync compact input bits");
        cuda_check(cudaMemsetAsync(output_bits.get(), 0,
            output_bits.size() * sizeof(std::uint32_t), stream),
            "cudaMemsetAsync compact output bits");
        cuda_check(cudaMemsetAsync(middle_bits.get(), 0,
            middle_bits.size() * sizeof(std::uint32_t), stream),
            "cudaMemsetAsync compact middle bits");
        cuda_check(cudaEventCreate(&compute_start), "cudaEventCreate");
        cuda_check(cudaEventCreate(&compute_end), "cudaEventCreate");
        cuda_check(cudaEventRecord(compute_start, stream), "cudaEventRecord");

        const auto block_count = [](std::uint64_t work) {
            return static_cast<std::uint32_t>(
                (work + threads_per_block - 1) / threads_per_block);
        };
        std::uint32_t* current = permutation.get();
        std::uint32_t* next = routed.get();
        for (const CompactLevel& level : tree.levels) {
            if (level.leaf_count != 0) {
                const std::uint64_t work =
                    static_cast<std::uint64_t>(level.leaf_count) * 32;
                compact_encode_leaves<<<block_count(work),
                    threads_per_block, 0, stream>>>(current, middle_destination,
                    leaves.get(), level.leaf_base, level.leaf_count);
                cuda_check(cudaGetLastError(),
                    "CUDA compact leaf launch");
            }
            if (level.node_count == 0) {
                continue;
            }
            const std::uint64_t element_work =
                static_cast<std::uint64_t>(level.maximum_count) *
                level.node_count;
            const std::uint32_t element_blocks = block_count(element_work);
            compact_inverse<<<element_blocks, threads_per_block, 0, stream>>>(
                current, inverse.get(), nodes.get(), level.node_base,
                level.maximum_count, level.node_count);
            compact_initialize<<<element_blocks,
                threads_per_block, 0, stream>>>(parents.get(), flips.get(),
                nodes.get(), level.node_base, level.maximum_count,
                level.node_count);

            const std::uint32_t pair_width = level.maximum_count / 2;
            const std::uint64_t pair_work =
                static_cast<std::uint64_t>(pair_width) * level.node_count;
            compact_join_pairs<<<block_count(pair_work),
                threads_per_block, 0, stream>>>(inverse.get(), parents.get(),
                nodes.get(), level.node_base, pair_width, level.node_count);
            compact_orient_odd<<<block_count(level.node_count),
                threads_per_block, 0, stream>>>(inverse.get(), parents.get(),
                flips.get(), nodes.get(), level.node_base, level.node_count);
            compact_route<<<element_blocks, threads_per_block, 0, stream>>>(
                current, next, parents.get(), flips.get(), colors.get(),
                nodes.get(), level.node_base, level.maximum_count,
                level.node_count);

            const std::uint64_t bit_work =
                static_cast<std::uint64_t>(level.maximum_pairs) *
                level.node_count;
            compact_pack_switches<<<block_count(bit_work),
                threads_per_block, 0, stream>>>(inverse.get(), colors.get(),
                input_bits.get(), output_bits.get(), nodes.get(),
                level.node_base, level.maximum_pairs, level.node_count,
                std::uint32_t{1} <<
                    normalized_format.middle_group_log2);
            cuda_check(cudaGetLastError(), "CUDA compact level launch");
            std::swap(current, next);
        }
        cuda_check(cudaEventRecord(compute_end, stream), "cudaEventRecord");

        const std::size_t input_bytes = static_cast<std::size_t>(
            layout.output_offset - layout.input_offset);
        const std::size_t output_bytes = static_cast<std::size_t>(
            layout.middle_offset - layout.output_offset);
        const std::size_t middle_bytes = static_cast<std::size_t>(
            layout.end_offset - layout.middle_offset);
        std::vector<std::byte> bytes(
            static_cast<std::size_t>(layout.end_offset));
        if (input_bytes != 0) {
            cuda_check(cudaMemcpyAsync(
                bytes.data() + layout.input_offset, input_bits.get(),
                input_bytes, cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync compact input bits");
        }
        if (output_bytes != 0) {
            cuda_check(cudaMemcpyAsync(
                bytes.data() + layout.output_offset, output_bits.get(),
                output_bytes, cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync compact output bits");
        }
        if (middle_bytes != 0) {
            cuda_check(cudaMemcpyAsync(
                bytes.data() + layout.middle_offset, middle_bits.get(),
                middle_bytes, cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync compact middle bits");
        }
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        float compute_milliseconds = 0;
        cuda_check(cudaEventElapsedTime(
            &compute_milliseconds, compute_start, compute_end),
            "cudaEventElapsedTime");
        set_last_device_seconds(compute_milliseconds / 1000.0);

        write_header(bytes, layout);

        cudaEventDestroy(compute_start);
        cudaEventDestroy(compute_end);
        cudaStreamDestroy(stream);
        return bytes;
    } catch (...) {
        if (compute_start != nullptr) {
            cudaEventDestroy(compute_start);
        }
        if (compute_end != nullptr) {
            cudaEventDestroy(compute_end);
        }
        cudaStreamDestroy(stream);
        throw;
    }
}

std::vector<std::byte> compress_compact_cuda(
    std::span<const std::uint64_t> source,
    FormatOptions format) {
    validate(source);
    if (source.size() > 0x7fffffffffffffffULL) {
        throw std::invalid_argument("permutation is too large for CUDA");
    }
    const std::uint64_t count = source.size();
    const FormatOptions normalized_format =
        normalize_format(count, std::move(format));
    const Layout layout = make_layout(count, normalized_format);
    const CompactTree64 tree =
        make_compact_tree64(count, normalized_format);
    DeviceBuffer<std::uint64_t> permutation(source.size());
    DeviceBuffer<std::uint64_t> routed(source.size());
    DeviceBuffer<std::uint64_t> inverse(source.size());
    DeviceBuffer<std::uint64_t> parents(source.size());
    DeviceBuffer<std::uint32_t> flips(source.size());
    DeviceBuffer<std::uint32_t> colors(source.size());
    DeviceBuffer<CompactNode64> nodes(tree.nodes.size());
    DeviceBuffer<CompactNode64> leaves(tree.leaves.size());
    DeviceBuffer<std::uint32_t> input_bits(
        static_cast<std::size_t>(
            tree.input_section_bits / 32 +
            (tree.input_section_bits % 32 != 0)));
    DeviceBuffer<std::uint32_t> output_bits(
        static_cast<std::size_t>(
            tree.output_section_bits / 32 +
            (tree.output_section_bits % 32 != 0)));
    DeviceBuffer<std::uint32_t> middle_bits(
        static_cast<std::size_t>(
            tree.middle_section_bits / 32 +
            (tree.middle_section_bits % 32 != 0)));
    std::uint32_t* middle_destination = middle_bits.get();
    if (normalized_format.middle_placement ==
        MiddlePlacement::embedded_input) {
        middle_destination = input_bits.get();
    } else if (
        normalized_format.middle_placement ==
        MiddlePlacement::embedded_output) {
        middle_destination = output_bits.get();
    }
    const auto multiply = [](std::uint64_t left, std::uint64_t right) {
        if (left != 0 &&
            right > std::numeric_limits<std::uint64_t>::max() / left) {
            throw std::overflow_error("CUDA work count overflow");
        }
        return left * right;
    };
    const auto block_count = [](std::uint64_t work) {
        const std::uint64_t required =
            work / threads_per_block +
            (work % threads_per_block != 0);
        return static_cast<std::uint32_t>(
            std::min<std::uint64_t>(required, 65535));
    };

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");
    cudaEvent_t compute_start = nullptr;
    cudaEvent_t compute_end = nullptr;
    try {
        cuda_check(cudaMemcpyAsync(permutation.get(), source.data(),
            source.size_bytes(), cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync compact64 permutation");
        if (!tree.nodes.empty()) {
            cuda_check(cudaMemcpyAsync(nodes.get(), tree.nodes.data(),
                tree.nodes.size() * sizeof(CompactNode64),
                cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync compact64 nodes");
        }
        cuda_check(cudaMemcpyAsync(leaves.get(), tree.leaves.data(),
            tree.leaves.size() * sizeof(CompactNode64),
            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync compact64 leaves");
        cuda_check(cudaMemsetAsync(input_bits.get(), 0,
            input_bits.size() * sizeof(std::uint32_t), stream),
            "cudaMemsetAsync compact64 input bits");
        cuda_check(cudaMemsetAsync(output_bits.get(), 0,
            output_bits.size() * sizeof(std::uint32_t), stream),
            "cudaMemsetAsync compact64 output bits");
        cuda_check(cudaMemsetAsync(middle_bits.get(), 0,
            middle_bits.size() * sizeof(std::uint32_t), stream),
            "cudaMemsetAsync compact64 middle bits");
        cuda_check(cudaEventCreate(&compute_start), "cudaEventCreate");
        cuda_check(cudaEventCreate(&compute_end), "cudaEventCreate");
        cuda_check(cudaEventRecord(compute_start, stream), "cudaEventRecord");

        std::uint64_t* current = permutation.get();
        std::uint64_t* next = routed.get();
        for (const CompactLevel64& level : tree.levels) {
            if (level.leaf_count != 0) {
                const std::uint64_t work =
                    multiply(level.leaf_count, 32);
                compact64_encode_leaves<<<block_count(work),
                    threads_per_block, 0, stream>>>(current,
                    middle_destination, leaves.get(), level.leaf_base, work);
                cuda_check(cudaGetLastError(),
                    "CUDA compact64 leaf launch");
            }
            if (level.node_count == 0) {
                continue;
            }
            const std::uint64_t element_work =
                multiply(level.maximum_count, level.node_count);
            const std::uint32_t element_blocks =
                block_count(element_work);
            compact64_inverse<<<element_blocks,
                threads_per_block, 0, stream>>>(current, inverse.get(),
                nodes.get(), level.node_base, level.maximum_count,
                element_work);
            compact64_initialize<<<element_blocks,
                threads_per_block, 0, stream>>>(parents.get(), flips.get(),
                nodes.get(), level.node_base, level.maximum_count,
                element_work);

            const std::uint64_t pair_work =
                multiply(level.maximum_pairs, level.node_count);
            compact64_join_pairs<<<block_count(pair_work),
                threads_per_block, 0, stream>>>(inverse.get(), parents.get(),
                nodes.get(), level.node_base, level.maximum_pairs, pair_work);
            compact64_orient_odd<<<block_count(level.node_count),
                threads_per_block, 0, stream>>>(inverse.get(), parents.get(),
                flips.get(), nodes.get(), level.node_base, level.node_count);
            compact64_route<<<element_blocks,
                threads_per_block, 0, stream>>>(current, next, parents.get(),
                flips.get(), colors.get(), nodes.get(), level.node_base,
                level.maximum_count, element_work);
            compact64_pack_switches<<<block_count(pair_work),
                threads_per_block, 0, stream>>>(inverse.get(), colors.get(),
                input_bits.get(), output_bits.get(), nodes.get(),
                level.node_base, level.maximum_pairs, pair_work,
                std::uint32_t{1} <<
                    normalized_format.middle_group_log2);
            cuda_check(cudaGetLastError(), "CUDA compact64 level launch");
            std::swap(current, next);
        }
        cuda_check(cudaEventRecord(compute_end, stream), "cudaEventRecord");

        const std::size_t input_bytes = static_cast<std::size_t>(
            layout.output_offset - layout.input_offset);
        const std::size_t output_bytes = static_cast<std::size_t>(
            layout.middle_offset - layout.output_offset);
        const std::size_t middle_bytes = static_cast<std::size_t>(
            layout.end_offset - layout.middle_offset);
        std::vector<std::byte> bytes(
            static_cast<std::size_t>(layout.end_offset));
        if (input_bytes != 0) {
            cuda_check(cudaMemcpyAsync(
                bytes.data() + layout.input_offset, input_bits.get(),
                input_bytes, cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync compact64 input bits");
        }
        if (output_bytes != 0) {
            cuda_check(cudaMemcpyAsync(
                bytes.data() + layout.output_offset, output_bits.get(),
                output_bytes, cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync compact64 output bits");
        }
        if (middle_bytes != 0) {
            cuda_check(cudaMemcpyAsync(
                bytes.data() + layout.middle_offset, middle_bits.get(),
                middle_bytes, cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync compact64 middle bits");
        }
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        float compute_milliseconds = 0;
        cuda_check(cudaEventElapsedTime(
            &compute_milliseconds, compute_start, compute_end),
            "cudaEventElapsedTime");
        set_last_device_seconds(compute_milliseconds / 1000.0);
        write_header(bytes, layout);

        cudaEventDestroy(compute_start);
        cudaEventDestroy(compute_end);
        cudaStreamDestroy(stream);
        return bytes;
    } catch (...) {
        if (compute_start != nullptr) {
            cudaEventDestroy(compute_start);
        }
        if (compute_end != nullptr) {
            cudaEventDestroy(compute_end);
        }
        cudaStreamDestroy(stream);
        throw;
    }
}

} // namespace benes::detail
