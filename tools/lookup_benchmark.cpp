// SPDX-License-Identifier: Apache-2.0
#include <benes/benes.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace {

template<typename Lookup>
double measure(
    std::span<const std::uint32_t> queries,
    Lookup&& lookup,
    std::uint64_t& checksum) {
    const auto start = std::chrono::steady_clock::now();
    for (const std::uint32_t query : queries) {
        checksum += lookup(query);
    }
    const auto finish = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(finish - start).count() /
        queries.size();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t size = argc > 1 ? std::stoull(argv[1]) : 1U << 20;
        const std::size_t query_count =
            argc > 2 ? std::stoull(argv[2]) : 100000;
        benes::MiddlePlacement middle_placement =
            benes::MiddlePlacement::separate;
        if (argc > 3) {
            const std::string layout = argv[3];
            if (layout == "input") {
                middle_placement = benes::MiddlePlacement::embedded_input;
            } else if (layout == "output") {
                middle_placement = benes::MiddlePlacement::embedded_output;
            } else if (layout != "separate") {
                throw std::invalid_argument(
                    "middle layout must be separate, input, or output");
            }
        }
        std::vector<std::uint32_t> permutation(size);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::mt19937_64 random(0x1006b00c);
        std::shuffle(permutation.begin(), permutation.end(), random);
        const auto bytes = benes::compress(permutation,
            {.backend = benes::Backend::automatic,
                .format = {.middle_placement = middle_placement}});
        std::vector<std::uint32_t> queries(query_count);
        for (auto& query : queries) {
            query = static_cast<std::uint32_t>(random() % size);
        }
        std::vector<std::uint32_t> outputs(query_count);
        std::transform(queries.begin(), queries.end(), outputs.begin(),
            [&](std::uint32_t input) { return permutation[input]; });

        const auto path = std::filesystem::temp_directory_path() /
            ("benes-lookup-benchmark-" + std::to_string(::getpid()) + ".bin");
        {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            output.write(
                reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        const int descriptor = ::open(path.c_str(), O_RDONLY);
        if (descriptor < 0) {
            throw std::runtime_error("open failed");
        }
        void* mapping = ::mmap(nullptr, bytes.size(), PROT_READ, MAP_PRIVATE,
            descriptor, 0);
        if (mapping == MAP_FAILED) {
            throw std::runtime_error("mmap failed");
        }
        const benes::CompressedPermutationView memory(bytes);
        const benes::CompressedPermutationView mapped(std::span(
            static_cast<const std::byte*>(mapping), bytes.size()));
        benes::CompressedPermutationFile forward_file(path, 64);
        benes::CompressedPermutationFile inverse_file(path, 64);
        for (std::size_t i = 0; i < queries.size(); ++i) {
            if (memory.forward(queries[i]) != outputs[i] ||
                memory.inverse(outputs[i]) != queries[i]) {
                throw std::runtime_error(
                    "sampled lookup verification failed");
            }
        }

        std::uint64_t checksum = 0;
        std::cout << "mode,direction,nanoseconds_per_lookup\n";
        std::cout << "memory,forward,"
                  << measure(queries,
                         [&](std::uint32_t value) { return memory.forward(value); },
                         checksum)
                  << '\n';
        std::cout << "memory,inverse,"
                  << measure(outputs,
                         [&](std::uint32_t value) { return memory.inverse(value); },
                         checksum)
                  << '\n';
        std::cout << "mmap,forward,"
                  << measure(queries,
                         [&](std::uint32_t value) { return mapped.forward(value); },
                         checksum)
                  << '\n';
        std::cout << "mmap,inverse,"
                  << measure(outputs,
                         [&](std::uint32_t value) { return mapped.inverse(value); },
                         checksum)
                  << '\n';
        std::cout << "file-cold,forward,"
                  << measure(queries,
                         [&](std::uint32_t value) {
                             return forward_file.forward(value);
                         },
                         checksum)
                  << '\n';
        std::cout << "file-warm,forward,"
                  << measure(queries,
                         [&](std::uint32_t value) {
                             return forward_file.forward(value);
                         },
                         checksum)
                  << '\n';
        std::cout << "file-cold,inverse,"
                  << measure(outputs,
                         [&](std::uint32_t value) {
                             return inverse_file.inverse(value);
                         },
                         checksum)
                  << '\n';
        std::cout << "file-warm,inverse,"
                  << measure(outputs,
                         [&](std::uint32_t value) {
                             return inverse_file.inverse(value);
                         },
                         checksum)
                  << '\n';
        std::cout << "checksum," << checksum << '\n';

        ::munmap(mapping, bytes.size());
        ::close(descriptor);
        std::filesystem::remove(path);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "lookup benchmark failure: " << error.what() << '\n';
        return 1;
    }
}
