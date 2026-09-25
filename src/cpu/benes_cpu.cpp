// SPDX-License-Identifier: Apache-2.0
#include "../internal.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <limits>
#include <numeric>
#include <thread>

namespace benes::detail {
namespace {

template<typename Function>
void parallel_for(
    std::size_t count,
    std::size_t thread_count,
    Function&& function) {
    thread_count = std::min(thread_count, count);
    if (thread_count <= 1 || count < 4096) {
        for (std::size_t i = 0; i < count; ++i) {
            function(i);
        }
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (std::size_t thread = 0; thread < thread_count; ++thread) {
        const std::size_t begin = count * thread / thread_count;
        const std::size_t end = count * (thread + 1) / thread_count;
        workers.emplace_back([begin, end, &function] {
            for (std::size_t i = begin; i < end; ++i) {
                function(i);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
}

std::uint32_t root_index(
    const std::uint32_t* parents,
    std::uint32_t position) {
    while ((position & ~std::uint32_t{1}) !=
        (parents[position / 2] & ~std::uint32_t{1})) {
        position = parents[position / 2] ^ (position & 1);
    }
    return position;
}

std::uint32_t root_index_atomic(
    std::uint32_t* parents,
    std::uint32_t position) {
    while (true) {
        std::atomic_ref<std::uint32_t> slot(parents[position / 2]);
        const std::uint32_t parent = slot.load(std::memory_order_relaxed);
        if ((position & ~std::uint32_t{1}) ==
            (parent & ~std::uint32_t{1})) {
            return position;
        }
        position = parent ^ (position & 1);
    }
}

void find_cycles_serial(
    const std::uint32_t* permutation,
    std::uint32_t* parents,
    std::uint32_t size) {
    for (std::uint32_t i = 0; i < size; ++i) {
        parents[i] = i * 2;
    }
    for (std::uint32_t i = 0; i < size; ++i) {
        std::uint32_t left = root_index(parents, i + (i & ~std::uint32_t{1}));
        const std::uint32_t value = permutation[i];
        std::uint32_t right = root_index(
            parents, (value + (value & ~std::uint32_t{1})) | 2);
        if (left == right) {
            continue;
        }
        if (left > right) {
            std::swap(left, right);
        }
        parents[right / 2] = left ^ (right & 1);
    }
    for (std::uint32_t i = 0; i < size; ++i) {
        parents[i] = root_index(parents, i * 2);
    }
}

void find_cycles_parallel(
    std::span<const std::uint32_t> permutation,
    std::span<std::uint32_t> parents,
    std::uint32_t block_size,
    std::size_t thread_count) {
    parallel_for(parents.size(), thread_count, [&](std::size_t index) {
        parents[index] = static_cast<std::uint32_t>(index % block_size) * 2;
    });
    parallel_for(permutation.size(), thread_count, [&](std::size_t index) {
        const std::size_t base = index & ~(static_cast<std::size_t>(block_size) - 1);
        const std::uint32_t local = static_cast<std::uint32_t>(index - base);
        std::uint32_t left = local + (local & ~std::uint32_t{1});
        std::uint32_t right = (permutation[index] +
            (permutation[index] & ~std::uint32_t{1})) | 2;
        std::uint32_t* block = parents.data() + base;
        while (true) {
            left = root_index_atomic(block, left);
            right = root_index_atomic(block, right);
            if (left == right) {
                break;
            }
            if (left > right) {
                std::swap(left, right);
            }
            std::atomic_ref<std::uint32_t> destination(block[right / 2]);
            std::uint32_t expected = right & ~std::uint32_t{1};
            const std::uint32_t replacement = left ^ (right & 1);
            if (destination.compare_exchange_weak(expected, replacement,
                    std::memory_order_relaxed)) {
                break;
            }
            right = expected ^ (right & 1);
        }
    });
    parallel_for(parents.size(), thread_count, [&](std::size_t index) {
        const std::size_t base = index & ~(static_cast<std::size_t>(block_size) - 1);
        const std::uint32_t local = static_cast<std::uint32_t>(index - base);
        const std::uint32_t root =
            root_index_atomic(parents.data() + base, local * 2);
        std::atomic_ref<std::uint32_t>(parents[index]).store(
            root, std::memory_order_relaxed);
    });
}

void validate_permutation(std::span<const std::uint32_t> permutation) {
    if (permutation.empty()) {
        throw std::invalid_argument("permutation must not be empty");
    }
    if (permutation.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("permutation is too large for uint32 indexes");
    }
    std::vector<bool> seen(permutation.size());
    for (const std::uint32_t value : permutation) {
        if (value >= permutation.size()) {
            throw std::invalid_argument("permutation value is out of range");
        }
        if (seen[value]) {
            throw std::invalid_argument("permutation contains a duplicate value");
        }
        seen[value] = true;
    }
}

} // namespace

std::vector<std::byte> compress_cpu_with_encoder(
    std::span<const std::uint32_t> source,
    std::size_t threads,
    MiddleEncoder encoder) {
    set_last_device_seconds(0);
    validate_permutation(source);
    const std::uint64_t minimum = std::max<std::uint64_t>(32, source.size());
    const std::uint64_t network64 = std::bit_ceil(minimum);
    if (network64 > (std::uint64_t{1} << 31)) {
        throw std::invalid_argument("permutation is too large for this Benes implementation");
    }
    const auto network_size = static_cast<std::uint32_t>(network64);
    const std::uint32_t log_size = static_cast<std::uint32_t>(
        std::bit_width(network_size) - 1);
    if (threads == 0) {
        threads = std::max(1U, std::thread::hardware_concurrency());
    }

    std::vector<std::uint32_t> permutation(network_size);
    std::copy(source.begin(), source.end(), permutation.begin());
    std::iota(permutation.begin() +
            static_cast<std::ptrdiff_t>(source.size()),
        permutation.end(),
        static_cast<std::uint32_t>(source.size()));
    std::vector<std::uint32_t> temporary(network_size);
    std::vector<std::uint32_t> parents(network_size);

    const std::size_t words_per_layer = network_size / 64;
    const std::size_t layer_count = log_size - 5;
    std::vector<std::uint32_t> input_words(words_per_layer * layer_count);
    std::vector<std::uint32_t> output_words(words_per_layer * layer_count);

    for (std::uint32_t layer = log_size; layer > 5; --layer) {
        const std::uint32_t block_size = std::uint32_t{1} << layer;
        const std::size_t layer_index = log_size - layer;
        if (threads == 1) {
            for (std::size_t base = 0; base < network_size; base += block_size) {
                find_cycles_serial(
                    permutation.data() + base, parents.data() + base, block_size);
            }
        } else {
            find_cycles_parallel(permutation, parents, block_size, threads);
        }

        parallel_for(words_per_layer, threads, [&](std::size_t word_index) {
            std::uint32_t input_word = 0;
            std::uint32_t output_word = 0;
            const std::size_t first_switch = word_index * 32;
            for (std::size_t bit = 0; bit < 32; ++bit) {
                const std::size_t element = (first_switch + bit) * 2;
                input_word |= (parents[element] & 1) << bit;
                output_word |= (parents[element + 1] & 1) << bit;
            }
            input_words[layer_index * words_per_layer + word_index] = input_word;
            output_words[layer_index * words_per_layer + word_index] = output_word;
        });

        parallel_for(network_size, threads, [&](std::size_t index) {
            const std::size_t base = index &
                ~(static_cast<std::size_t>(block_size) - 1);
            const std::uint32_t local = static_cast<std::uint32_t>(index - base);
            const std::size_t switch_index = index / 2;
            const std::uint32_t input_word =
                input_words[layer_index * words_per_layer + switch_index / 32];
            const std::uint32_t input_bit =
                (input_word >> (switch_index & 31)) & 1;
            std::uint32_t output = permutation[index ^ input_bit];
            const std::size_t output_switch = (base + output) / 2;
            const std::uint32_t output_word =
                output_words[layer_index * words_per_layer + output_switch / 32];
            output ^= (output_word >> (output_switch & 31)) & 1;
            const std::uint32_t half = block_size / 2;
            temporary[base + (local >> 1) + ((local & 1) ? half : 0)] = output / 2;
        });
        permutation.swap(temporary);
    }

    std::vector<U128> middles(network_size / 32);
    parallel_for(middles.size(), threads, [&](std::size_t middle) {
        middles[middle] = encoder(permutation.data() + middle * 32);
    });
    return make_blob(static_cast<std::uint32_t>(source.size()), network_size,
        input_words, output_words, middles);
}

namespace {

U128 encode_middle_portable(const std::uint32_t* permutation) {
    return encode_middle(std::span<const std::uint32_t, 32>(permutation, 32));
}

} // namespace

std::vector<std::byte> compress_portable(
    std::span<const std::uint32_t> source,
    std::size_t threads) {
    return compress_cpu_with_encoder(source, threads, encode_middle_portable);
}

} // namespace benes::detail
