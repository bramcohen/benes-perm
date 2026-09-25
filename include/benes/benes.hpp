// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace benes {

enum class Backend {
    automatic,
    portable_cpu,
    metal,
    cuda,
    avx512,
};

enum class MiddlePlacement : std::uint8_t {
    separate,
    embedded_input,
    embedded_output,
};

enum class InputValueWidth : std::uint8_t {
    bits32 = 4,
    bits64 = 8,
};

struct FormatOptions {
    // Middle groups contain up to 2^middle_group_log2 entries.
    std::uint8_t middle_group_log2 = 5;
    // Each value is a base-2 logarithmic cluster size.
    std::vector<std::uint8_t> input_clusters;
    std::vector<std::uint8_t> output_clusters;
    MiddlePlacement middle_placement = MiddlePlacement::separate;
};

struct CompressOptions {
    Backend backend = Backend::automatic;
    std::size_t threads = 0;
    FormatOptions format;
};

[[nodiscard]] std::vector<std::byte> compress(
    std::span<const std::uint32_t> permutation,
    CompressOptions options = {});
[[nodiscard]] std::vector<std::byte> compress(
    std::span<const std::uint64_t> permutation,
    CompressOptions options = {});

void compress_file(
    const std::filesystem::path& input,
    const std::filesystem::path& output,
    InputValueWidth input_width = InputValueWidth::bits64,
    CompressOptions options = {});

[[nodiscard]] bool backend_available(Backend backend) noexcept;
[[nodiscard]] const char* backend_name(Backend backend) noexcept;

class CompressedPermutationView {
public:
    explicit CompressedPermutationView(std::span<const std::byte> bytes);

    [[nodiscard]] std::uint64_t forward(std::uint64_t input) const;
    [[nodiscard]] std::uint64_t inverse(std::uint64_t output) const;
    [[nodiscard]] std::uint64_t size() const noexcept;
    [[nodiscard]] std::uint64_t network_size() const noexcept;
    [[nodiscard]] std::uint64_t encoded_size() const noexcept;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;

private:
    std::span<const std::byte> bytes_;
    std::uint64_t size_;
    std::uint64_t network_size_;
    std::uint8_t middle_group_log2_;
    MiddlePlacement middle_placement_;
    std::vector<std::uint8_t> input_clusters_;
    std::vector<std::uint8_t> output_clusters_;
    std::uint64_t input_offset_;
    std::uint64_t output_offset_;
    std::uint64_t middle_offset_;
};

class CompressedPermutationFile {
public:
    explicit CompressedPermutationFile(
        const std::filesystem::path& path,
        std::size_t cache_pages = 64);
    ~CompressedPermutationFile();

    CompressedPermutationFile(CompressedPermutationFile&&) noexcept;
    CompressedPermutationFile& operator=(CompressedPermutationFile&&) noexcept;
    CompressedPermutationFile(const CompressedPermutationFile&) = delete;
    CompressedPermutationFile& operator=(const CompressedPermutationFile&) = delete;

    [[nodiscard]] std::uint64_t forward(std::uint64_t input) const;
    [[nodiscard]] std::uint64_t inverse(std::uint64_t output) const;
    [[nodiscard]] std::uint64_t size() const noexcept;
    [[nodiscard]] std::uint64_t network_size() const noexcept;
    [[nodiscard]] std::uint64_t encoded_size() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace benes
