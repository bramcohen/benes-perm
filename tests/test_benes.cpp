// SPDX-License-Identifier: Apache-2.0
#include <benes/benes.hpp>

#include "../src/internal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace {

std::vector<std::byte> read_file(const std::filesystem::path& path);

std::uint64_t load_u64(std::span<const std::byte> bytes) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t{std::to_integer<std::uint8_t>(bytes[i])} <<
            (8 * i);
    }
    return value;
}

std::size_t encoded_header_size(std::span<const std::byte> bytes) {
    return 20 + std::to_integer<std::uint8_t>(bytes[18]) +
        std::to_integer<std::uint8_t>(bytes[19]);
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<typename Function>
void require_throws(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

std::vector<std::uint32_t> random_permutation(
    std::size_t size,
    std::uint64_t seed) {
    std::vector<std::uint32_t> permutation(size);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::mt19937_64 random(seed);
    std::shuffle(permutation.begin(), permutation.end(), random);
    return permutation;
}

std::vector<std::uint64_t> widen(
    std::span<const std::uint32_t> permutation) {
    return {permutation.begin(), permutation.end()};
}

void verify(
    const std::vector<std::uint32_t>& permutation,
    std::size_t threads) {
    const auto bytes = benes::compress(permutation,
        {.backend = benes::Backend::portable_cpu, .threads = threads});
    const benes::CompressedPermutationView view(bytes);
    require(view.size() == permutation.size(), "wrong original size");
    require(view.network_size() ==
            std::bit_ceil(std::max<std::size_t>(32, permutation.size())),
        "wrong logical padded network width");
    require(view.encoded_size() == bytes.size(), "wrong encoded size");
    require(load_u64(std::span(bytes).subspan(8, 8)) == permutation.size(),
        "header has wrong permutation size");
    require(encoded_header_size(bytes) <= bytes.size(),
        "header exceeds the encoded blob");
    for (std::uint32_t input = 0; input < permutation.size(); ++input) {
        const std::uint32_t output = permutation[input];
        require(view.forward(input) == output, "forward lookup mismatch");
        require(view.inverse(output) == input, "inverse lookup mismatch");
    }
    require_throws([&] { (void)view.forward(view.size()); },
        "out-of-range forward lookup was accepted");
    require_throws([&] { (void)view.inverse(view.size()); },
        "out-of-range inverse lookup was accepted");
}

void test_compact_sizes() {
    std::size_t previous = 0;
    for (std::uint32_t size = 31; size <= 65; ++size) {
        const auto bytes = benes::compress(random_permutation(size, size),
            {.backend = benes::Backend::portable_cpu, .threads = 1});
        require(benes::CompressedPermutationView(bytes).network_size() ==
                std::bit_ceil(std::max(32U, size)),
            "wrong logical padded network width");
        const std::size_t header = encoded_header_size(bytes);
        const std::size_t payload = bytes.size() - header;
        require(size == 31 || payload <= previous + 4 ||
                std::has_single_bit(size),
            "compact payload has a size cliff");
        previous = payload;
    }
    for (const auto [size, expected_payload] :
        std::array<std::pair<std::uint32_t, std::size_t>, 10>{{
            {31, 15}, {32, 15}, {33, 16}, {63, 37}, {64, 38},
            {65, 39}, {96, 64}, {127, 91}, {128, 91}, {129, 92}}}) {
        const auto bytes = benes::compress(random_permutation(size, size),
            {.backend = benes::Backend::portable_cpu, .threads = 1});
        const std::size_t header = encoded_header_size(bytes);
        require(bytes.size() == header + expected_payload,
            "v1 payload has the wrong exact size");
    }
}

void test_permutations() {
    for (const std::size_t size :
        {1U, 2U, 3U, 7U, 16U, 31U, 32U, 33U, 47U, 63U, 64U, 65U,
            96U, 127U, 128U, 129U, 255U}) {
        std::vector<std::uint32_t> identity(size);
        std::iota(identity.begin(), identity.end(), 0);
        verify(identity, 1);
        std::reverse(identity.begin(), identity.end());
        verify(identity, 1);
        verify(random_permutation(size, 0xbebe5000ULL + size), 1);
    }
}

void test_exhaustive_small() {
    const bool has_metal = benes::backend_available(benes::Backend::metal);
    const bool has_cuda = benes::backend_available(benes::Backend::cuda);
    const bool has_avx512 = benes::backend_available(benes::Backend::avx512);
    for (std::uint32_t size = 1; size <= 8; ++size) {
        std::vector<std::uint32_t> permutation(size);
        std::iota(permutation.begin(), permutation.end(), 0);
        do {
            verify(permutation, 1);
            if (has_metal) {
                const auto reference = benes::compress(permutation,
                    {.backend = benes::Backend::portable_cpu, .threads = 1});
                const auto metal = benes::compress(permutation,
                    {.backend = benes::Backend::metal});
                require(metal == reference,
                    "Metal small compact output is not canonical");
            }
            if (has_cuda) {
                const auto reference = benes::compress(permutation,
                    {.backend = benes::Backend::portable_cpu, .threads = 1});
                const auto cuda = benes::compress(permutation,
                    {.backend = benes::Backend::cuda});
                require(cuda == reference,
                    "CUDA small compact output is not canonical");
            }
            if (has_avx512) {
                const auto reference = benes::compress(permutation,
                    {.backend = benes::Backend::portable_cpu, .threads = 1});
                const auto avx512 = benes::compress(permutation,
                    {.backend = benes::Backend::avx512});
                require(avx512 == reference,
                    "AVX-512 small compact output is not canonical");
            }
        } while (std::next_permutation(permutation.begin(), permutation.end()));
    }
}

void test_accelerated_compact_boundaries(benes::Backend backend) {
    if (!benes::backend_available(backend)) {
        return;
    }
    for (const std::size_t size :
        {9U, 15U, 17U, 30U, 31U, 33U, 34U, 47U, 62U, 63U, 65U, 66U,
            95U, 96U, 97U, 126U, 127U, 129U, 130U, 255U, 257U, 511U,
            513U, 1023U, 1025U, 4095U, 4097U}) {
        const auto permutation =
            random_permutation(size, 0xc04fac700000ULL + size);
        const auto reference = benes::compress(permutation,
            {.backend = benes::Backend::portable_cpu, .threads = 1});
        const auto accelerated = benes::compress(permutation,
            {.backend = backend});
        require(accelerated == reference,
            "accelerated compact boundary output is not canonical");
    }
}

void test_thread_determinism() {
    for (const std::size_t size : {64U, 127U, 256U, 1024U, 4096U}) {
        const auto permutation = random_permutation(size, 0x12340000ULL + size);
        const auto reference = benes::compress(permutation,
            {.backend = benes::Backend::portable_cpu, .threads = 1});
        for (const std::size_t threads : {2U, 4U, 8U}) {
            const auto parallel = benes::compress(permutation,
                {.backend = benes::Backend::portable_cpu, .threads = threads});
            require(parallel == reference,
                "multithreaded encoder is not byte-for-byte canonical");
        }
        verify(permutation, 8);
        if (benes::backend_available(benes::Backend::metal)) {
            const auto metal = benes::compress(permutation,
                {.backend = benes::Backend::metal});
            require(metal == reference,
                "Metal encoder is not byte-for-byte canonical");
        }
        if (benes::backend_available(benes::Backend::cuda)) {
            const auto cuda = benes::compress(permutation,
                {.backend = benes::Backend::cuda});
            require(cuda == reference,
                "CUDA encoder is not byte-for-byte canonical");
        }
        if (benes::backend_available(benes::Backend::avx512)) {
            const auto avx512 = benes::compress(permutation,
                {.backend = benes::Backend::avx512, .threads = 8});
            require(avx512 == reference,
                "AVX-512 encoder is not byte-for-byte canonical");
        }
    }
}

void test_uint64_canonical_parity() {
    for (const std::size_t size :
        {1U, 2U, 3U, 31U, 32U, 33U, 63U, 65U, 127U, 129U, 4097U}) {
        const auto permutation =
            random_permutation(size, 0x640000000000ULL + size);
        const auto wide = widen(permutation);
        const std::uint8_t depth = benes::detail::routing_depth(size, 4);
        const benes::FormatOptions format{
            .middle_group_log2 = 4,
            .input_clusters = std::vector<std::uint8_t>(depth, 1),
            .output_clusters = depth == 0
                ? std::vector<std::uint8_t>{}
                : std::vector<std::uint8_t>{depth},
            .middle_placement = size % 2 == 0
                ? benes::MiddlePlacement::embedded_input
                : benes::MiddlePlacement::embedded_output,
        };
        const auto reference = benes::compress(permutation,
            {.backend = benes::Backend::portable_cpu,
                .threads = 1, .format = format});
        const auto wide_portable = benes::compress(wide,
            {.backend = benes::Backend::portable_cpu,
                .threads = 4, .format = format});
        require(wide_portable == reference,
            "uint64 portable output is not byte-for-byte canonical");
        for (const benes::Backend backend : {benes::Backend::metal,
                 benes::Backend::cuda, benes::Backend::avx512}) {
            if (!benes::backend_available(backend)) {
                continue;
            }
            require(benes::compress(wide,
                {.backend = backend, .format = format}) == reference,
                std::string(benes::backend_name(backend)) +
                    " uint64 output is not canonical");
        }
    }
}

void test_format_options() {
    for (const std::uint32_t size :
        {3U, 7U, 15U, 17U, 31U, 33U, 63U, 65U, 127U, 129U, 257U}) {
        const auto permutation =
            random_permutation(size, 0xf04a7000 + size);
        for (const std::uint8_t middle_group_log2 : {2U, 3U, 4U, 5U}) {
            std::uint32_t remaining = size;
            std::uint8_t depth = 0;
            while (remaining >
                (std::uint32_t{1} << middle_group_log2)) {
                remaining = (remaining + 1) / 2;
                ++depth;
            }
            std::vector<std::uint8_t> input_clusters(depth, 1);
            std::vector<std::uint8_t> output_clusters;
            if (depth != 0) {
                output_clusters.push_back(depth);
            }
            std::size_t encoded_size = 0;
            for (const benes::MiddlePlacement middle_placement : {
                     benes::MiddlePlacement::separate,
                     benes::MiddlePlacement::embedded_input,
                     benes::MiddlePlacement::embedded_output}) {
                benes::FormatOptions format{
                    .middle_group_log2 = middle_group_log2,
                    .input_clusters = input_clusters,
                    .output_clusters = output_clusters,
                    .middle_placement = middle_placement,
                };
                const auto reference = benes::compress(permutation,
                    {.backend = benes::Backend::portable_cpu,
                        .threads = 1,
                        .format = format});
                if (encoded_size == 0) {
                    encoded_size = reference.size();
                } else {
                    require(reference.size() == encoded_size,
                        "middle placement changed three-stream rounding size");
                }
                const benes::CompressedPermutationView view(reference);
                for (std::uint32_t input = 0;
                    input < permutation.size(); ++input) {
                    require(view.forward(input) == permutation[input],
                        "configured forward lookup mismatch");
                    require(view.inverse(permutation[input]) == input,
                        "configured inverse lookup mismatch");
                }
            }
        }
    }
    const auto permutation = random_permutation(4097, 0xf04a7fff);
    const benes::FormatOptions format{
        .middle_group_log2 = 4,
        .input_clusters = {3, 6},
        .output_clusters = {5, 4},
        .middle_placement = benes::MiddlePlacement::embedded_input,
    };
    const auto reference = benes::compress(permutation,
        {.backend = benes::Backend::portable_cpu,
            .threads = 1,
            .format = format});
    const auto parallel = benes::compress(permutation,
        {.backend = benes::Backend::portable_cpu,
            .threads = 8,
            .format = format});
    require(parallel == reference,
        "configured multithreaded output is not canonical");
    for (const benes::Backend backend : {benes::Backend::metal,
             benes::Backend::cuda, benes::Backend::avx512}) {
        if (benes::backend_available(backend)) {
            const auto accelerated = benes::compress(permutation,
                {.backend = backend, .format = format});
            require(accelerated == reference,
                "configured accelerated output is not canonical");
        }
    }
    require_throws([&] {
        (void)benes::compress(permutation,
            {.backend = benes::Backend::portable_cpu,
                .format = {.input_clusters = {1, 1}}});
    }, "cluster depths with the wrong sum were accepted");
    require_throws([&] {
        (void)benes::compress(permutation,
            {.backend = benes::Backend::portable_cpu,
                .format = {.middle_group_log2 = 6}});
    }, "unsupported middle permutation group size was accepted");
}

void test_invalid_data() {
    require_throws([] {
        (void)benes::compress(std::span<const std::uint32_t>{},
            {.backend = benes::Backend::portable_cpu});
    }, "empty permutation was accepted");
    const std::array<std::uint32_t, 3> duplicate = {0, 1, 1};
    require_throws([&] {
        (void)benes::compress(duplicate,
            {.backend = benes::Backend::portable_cpu});
    }, "duplicate permutation value was accepted");
    const std::array<std::uint32_t, 3> out_of_range = {0, 1, 3};
    require_throws([&] {
        (void)benes::compress(out_of_range,
            {.backend = benes::Backend::portable_cpu});
    }, "out-of-range permutation value was accepted");
    const std::array<std::uint64_t, 3> wide_duplicate = {0, 1, 1};
    require_throws([&] {
        (void)benes::compress(wide_duplicate,
            {.backend = benes::Backend::portable_cpu});
    }, "duplicate uint64 permutation value was accepted");
    const std::array<std::uint64_t, 3> wide_out_of_range = {0, 1, 3};
    require_throws([&] {
        (void)benes::compress(wide_out_of_range,
            {.backend = benes::Backend::portable_cpu});
    }, "out-of-range uint64 permutation value was accepted");

    const auto good = benes::compress(random_permutation(63, 9),
        {.backend = benes::Backend::portable_cpu});
    for (const std::size_t length :
        std::array<std::size_t, 4>{0, 1, 19, good.size() - 1}) {
        require_throws([&] {
            (void)benes::CompressedPermutationView(
                std::span(good).first(length));
        }, "truncated compressed data was accepted");
    }
    auto corrupt = good;
    corrupt[0] ^= std::byte{0xff};
    require_throws([&] { (void)benes::CompressedPermutationView(corrupt); },
        "bad magic was accepted");
    corrupt = good;
    std::fill(corrupt.begin() + 8, corrupt.begin() + 16, std::byte{0});
    require_throws([&] { (void)benes::CompressedPermutationView(corrupt); },
        "zero permutation size was accepted");
    corrupt = good;
    corrupt[12] = std::byte{1};
    require_throws([&] { (void)benes::CompressedPermutationView(corrupt); },
        "size/header mismatch was accepted");
    corrupt = good;
    corrupt[17] = std::byte{0};
    require_throws([&] { (void)benes::CompressedPermutationView(corrupt); },
        "invalid middle permutation group size was accepted");
    corrupt = good;
    corrupt[16] = std::byte{3};
    require_throws([&] { (void)benes::CompressedPermutationView(corrupt); },
        "invalid middle placement was accepted");
    corrupt = good;
    corrupt[18] = std::byte{65};
    require_throws([&] { (void)benes::CompressedPermutationView(corrupt); },
        "oversized cluster descriptor was accepted");
    corrupt = good;
    corrupt[20] = std::byte{2};
    require_throws([&] { (void)benes::CompressedPermutationView(corrupt); },
        "invalid cluster composition was accepted");
    corrupt = good;
    const std::size_t output_offset = encoded_header_size(corrupt) + 4;
    corrupt[output_offset + 3] |= std::byte{0x80};
    require_throws([&] { (void)benes::CompressedPermutationView(corrupt); },
        "nonzero high switch padding bits were accepted");
    corrupt = good;
    corrupt.back() |= std::byte{0x80};
    require_throws([&] { (void)benes::CompressedPermutationView(corrupt); },
        "nonzero high middle padding bits were accepted");
    corrupt = good;
    corrupt[6] = std::byte{2};
    require_throws([&] { (void)benes::CompressedPermutationView(corrupt); },
        "unsupported format version was accepted");

    corrupt = benes::compress(random_permutation(63, 10),
        {.backend = benes::Backend::portable_cpu,
            .format = {
                .middle_placement =
                    benes::MiddlePlacement::embedded_input}});
    const std::size_t embedded_output_offset = corrupt.size() - 4;
    corrupt[embedded_output_offset - 1] |= std::byte{0x80};
    require_throws([&] { (void)benes::CompressedPermutationView(corrupt); },
        "embedded section accepted nonzero logical-stream padding");
}

void test_uint64_layout_and_sparse_file() {
    for (const std::uint64_t count : {
             (std::uint64_t{1} << 32) - 1,
             std::uint64_t{1} << 32,
             (std::uint64_t{1} << 32) + 1,
             (std::uint64_t{1} << 32) + 3}) {
        const benes::detail::Layout layout =
            benes::detail::make_layout(count, {});
        require(layout.size == count, "64-bit layout truncated entry count");
        require(layout.input_offset < layout.output_offset &&
                layout.output_offset < layout.end_offset,
            "64-bit layout produced invalid section offsets");
    }
    require_throws([] {
        (void)benes::detail::make_layout(
            std::numeric_limits<std::uint64_t>::max(), {});
    }, "overflowing 64-bit layout was accepted");
    require_throws([] {
        (void)benes::detail::checked_size_bytes(
            std::numeric_limits<std::size_t>::max(), 2);
    }, "overflowing allocation size was accepted");

    const std::uint64_t count = std::uint64_t{1} << 33;
    const benes::detail::Layout layout =
        benes::detail::make_layout(count, {});
    std::vector<std::byte> header(
        static_cast<std::size_t>(layout.input_offset));
    benes::detail::write_header(header, layout);
    const auto path = std::filesystem::temp_directory_path() /
        "benes-perm-sparse-64.benes";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(header.data()),
            static_cast<std::streamsize>(header.size()));
    }
    std::filesystem::resize_file(path, layout.end_offset);
    const benes::CompressedPermutationFile file(path, 4);
    require(file.size() == count, "sparse file truncated 64-bit entry count");
    for (const std::uint64_t value : {
             std::uint64_t{0}, std::uint64_t{1},
             (std::uint64_t{1} << 32) - 1,
             std::uint64_t{1} << 32,
             (std::uint64_t{1} << 32) + 1,
             count - 1}) {
        const std::uint64_t output = file.forward(value);
        require(output < count,
            "sparse zero-blob forward lookup returned an invalid index");
        require(file.inverse(output) == value,
            "sparse zero-blob 64-bit lookup did not round trip at " +
                std::to_string(value));
    }
    std::filesystem::remove(path);
}

void test_file_lookup() {
    const auto permutation = random_permutation(127, 0x515151);
    const auto bytes = benes::compress(permutation,
        {.backend = benes::Backend::portable_cpu});
    const auto path = std::filesystem::temp_directory_path() /
        "benes-perm-test.bin";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    const benes::CompressedPermutationFile file(path, 2);
    require(file.size() == permutation.size(), "file view has wrong size");
    for (std::uint32_t input = 0; input < permutation.size(); ++input) {
        require(file.forward(input) == permutation[input],
            "file forward lookup mismatch");
        require(file.inverse(permutation[input]) == input,
            "file inverse lookup mismatch");
    }

    const int descriptor = ::open(path.c_str(), O_RDONLY);
    require(descriptor >= 0, "failed to open file for mmap");
    void* mapping = ::mmap(nullptr, bytes.size(), PROT_READ, MAP_PRIVATE,
        descriptor, 0);
    require(mapping != MAP_FAILED, "failed to mmap compressed permutation");
    const benes::CompressedPermutationView mapped(std::span(
        static_cast<const std::byte*>(mapping), bytes.size()));
    for (std::uint32_t input = 0; input < permutation.size(); ++input) {
        require(mapped.forward(input) == permutation[input],
            "mmap forward lookup mismatch");
        require(mapped.inverse(permutation[input]) == input,
            "mmap inverse lookup mismatch");
    }
    ::munmap(mapping, bytes.size());
    ::close(descriptor);

    const auto raw_path = std::filesystem::temp_directory_path() /
        "benes-perm-raw.bin";
    const auto encoded_path = std::filesystem::temp_directory_path() /
        "benes-perm-encoded.bin";
    {
        std::ofstream raw(raw_path, std::ios::binary | std::ios::trunc);
        for (const std::uint32_t value : permutation) {
            const std::array<char, 4> encoded = {
                static_cast<char>(value),
                static_cast<char>(value >> 8),
                static_cast<char>(value >> 16),
                static_cast<char>(value >> 24)};
            raw.write(encoded.data(), encoded.size());
        }
    }
    benes::compress_file(raw_path, encoded_path,
        benes::InputValueWidth::bits32,
        {.backend = benes::Backend::portable_cpu, .threads = 1});
    require(read_file(encoded_path) == bytes,
        "raw uint32 file compression differs from memory compression");
    {
        std::ofstream raw(raw_path, std::ios::binary | std::ios::trunc);
        for (const std::uint64_t value : widen(permutation)) {
            for (unsigned byte = 0; byte < 8; ++byte) {
                raw.put(static_cast<char>(value >> (8 * byte)));
            }
        }
    }
    benes::compress_file(raw_path, encoded_path,
        benes::InputValueWidth::bits64,
        {.backend = benes::Backend::portable_cpu, .threads = 1});
    require(read_file(encoded_path) == bytes,
        "raw uint64 file compression differs from memory compression");

    auto corrupt = bytes;
    corrupt.back() |= std::byte{0x80};
    const auto corrupt_path = std::filesystem::temp_directory_path() /
        "benes-perm-corrupt.bin";
    {
        std::ofstream output(
            corrupt_path, std::ios::binary | std::ios::trunc);
        output.write(
            reinterpret_cast<const char*>(corrupt.data()), corrupt.size());
    }
    require_throws(
        [&] { (void)benes::CompressedPermutationFile(corrupt_path, 2); },
        "file view accepted nonzero high middle padding bits");

    std::filesystem::remove(raw_path);
    std::filesystem::remove(encoded_path);
    std::filesystem::remove(corrupt_path);
    std::filesystem::remove(path);
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(input), "golden blob is missing");
    const auto size = input.tellg();
    input.seekg(0);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    require(static_cast<bool>(input), "failed to read golden blob");
    return bytes;
}

void test_golden_blobs() {
    const std::filesystem::path directory =
        std::filesystem::path(BENES_TEST_SOURCE_DIR) / "golden";
    std::vector<std::uint32_t> identity(32);
    std::iota(identity.begin(), identity.end(), 0);
    std::vector<std::uint32_t> reverse(33);
    std::iota(reverse.rbegin(), reverse.rend(), 0);
    std::vector<std::uint32_t> random(47);
    std::iota(random.begin(), random.end(), 0);
    std::mt19937_64 generator(0x474f4c44454eULL);
    std::shuffle(random.begin(), random.end(), generator);

    for (const auto& [name, permutation] :
        std::array<std::pair<std::string, std::vector<std::uint32_t>>, 3>{{
            {"identity-32.benes", identity},
            {"reverse-33.benes", reverse},
            {"random-47.benes", random}}}) {
        const auto expected = read_file(directory / name);
        const auto actual = benes::compress(permutation,
            {.backend = benes::Backend::portable_cpu, .threads = 1});
        require(actual == expected, "canonical golden blob changed: " + name);
        require(benes::compress(permutation) == expected,
            "automatic backend output differs from golden blob: " + name);
        if (benes::backend_available(benes::Backend::metal)) {
            require(benes::compress(permutation,
                {.backend = benes::Backend::metal}) == expected,
                "Metal output differs from golden blob: " + name);
        }
        if (benes::backend_available(benes::Backend::cuda)) {
            require(benes::compress(permutation,
                {.backend = benes::Backend::cuda}) == expected,
                "CUDA output differs from golden blob: " + name);
        }
        if (benes::backend_available(benes::Backend::avx512)) {
            require(benes::compress(permutation,
                {.backend = benes::Backend::avx512}) == expected,
                "AVX-512 output differs from golden blob: " + name);
        }
    }
}

void test_extended() {
    for (const std::size_t size :
        {32767U, 32768U, 32769U, 43690U, 49152U, 57344U,
            65535U, 65536U, 65537U}) {
        const auto permutation =
            random_permutation(size, 0xe57e0000ULL + size);
        const auto reference = benes::compress(permutation,
            {.backend = benes::Backend::portable_cpu, .threads = 1});
        const auto parallel = benes::compress(permutation,
            {.backend = benes::Backend::portable_cpu, .threads = 12});
        require(parallel == reference,
            "extended multithreaded output differs from reference");
        if (benes::backend_available(benes::Backend::metal)) {
            const auto metal = benes::compress(permutation,
                {.backend = benes::Backend::metal});
            require(metal == reference,
                "extended Metal output differs from reference");
        }
        if (benes::backend_available(benes::Backend::cuda)) {
            const auto cuda = benes::compress(permutation,
                {.backend = benes::Backend::cuda});
            require(cuda == reference,
                "extended CUDA output differs from reference");
        }
        if (benes::backend_available(benes::Backend::avx512)) {
            const auto avx512 = benes::compress(permutation,
                {.backend = benes::Backend::avx512});
            require(avx512 == reference,
                "extended AVX-512 output differs from reference");
        }
        const benes::CompressedPermutationView view(reference);
        for (std::uint32_t input = 0; input < permutation.size(); ++input) {
            require(view.forward(input) == permutation[input],
                "extended forward lookup mismatch");
            require(view.inverse(permutation[input]) == input,
                "extended inverse lookup mismatch");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        test_permutations();
        test_exhaustive_small();
        test_compact_sizes();
        test_accelerated_compact_boundaries(benes::Backend::metal);
        test_accelerated_compact_boundaries(benes::Backend::cuda);
        test_accelerated_compact_boundaries(benes::Backend::avx512);
        test_thread_determinism();
        test_uint64_canonical_parity();
        test_format_options();
        test_invalid_data();
        test_uint64_layout_and_sparse_file();
        test_file_lookup();
        test_golden_blobs();
        if (argc > 1 && std::string_view(argv[1]) == "--extended") {
            test_extended();
        }
        std::cout << "all tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
