// SPDX-License-Identifier: Apache-2.0
#include "internal.hpp"

#include <stdexcept>

namespace benes::detail {

#if !defined(BENES_HAS_METAL)
bool metal_available() noexcept { return false; }
std::vector<std::byte> compress_metal(std::span<const std::uint32_t>) {
    throw std::runtime_error("Metal backend was not built");
}
std::vector<std::byte> compress_compact_metal(
    std::span<const std::uint32_t>,
    FormatOptions) {
    throw std::runtime_error("Metal backend was not built");
}
std::vector<std::byte> compress_compact_metal(
    std::span<const std::uint64_t>,
    FormatOptions) {
    throw std::runtime_error("Metal backend was not built");
}
#endif

#if !defined(BENES_HAS_CUDA)
bool cuda_available() noexcept { return false; }
std::vector<std::byte> compress_cuda(std::span<const std::uint32_t>) {
    throw std::runtime_error("CUDA backend was not built");
}
std::vector<std::byte> compress_compact_cuda(
    std::span<const std::uint32_t>,
    FormatOptions) {
    throw std::runtime_error("CUDA backend was not built");
}
std::vector<std::byte> compress_compact_cuda(
    std::span<const std::uint64_t>,
    FormatOptions) {
    throw std::runtime_error("CUDA backend was not built");
}
#endif

#if !defined(BENES_HAS_AVX512)
bool avx512_available() noexcept { return false; }
std::vector<std::byte> compress_avx512(
    std::span<const std::uint32_t>,
    std::size_t) {
    throw std::runtime_error("AVX-512 backend was not built");
}
std::vector<std::byte> compress_compact_avx512(
    std::span<const std::uint32_t>,
    std::size_t,
    FormatOptions) {
    throw std::runtime_error("AVX-512 backend was not built");
}
std::vector<std::byte> compress_compact_avx512(
    std::span<const std::uint64_t>,
    std::size_t,
    FormatOptions) {
    throw std::runtime_error("AVX-512 backend was not built");
}
#endif

} // namespace benes::detail
