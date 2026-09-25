// SPDX-License-Identifier: Apache-2.0
#include "../internal.hpp"

#include <immintrin.h>

namespace benes::detail {
namespace {

__mmask16 lane_mask(std::uint32_t count) {
    return static_cast<__mmask16>(
        count == 16 ? 0xffffU : (1U << count) - 1);
}

U128 encode_factoradic_avx512(
    std::span<const std::uint32_t> permutation) {
    const std::uint32_t count =
        static_cast<std::uint32_t>(permutation.size());
    const std::uint32_t low_count = std::min(count, 16U);
    const std::uint32_t high_count = count - low_count;
    const __m512i low = _mm512_maskz_loadu_epi32(
        lane_mask(low_count), permutation.data());
    const __m512i high = _mm512_maskz_loadu_epi32(
        lane_mask(high_count), permutation.data() + low_count);
    __m512i low_decrements = _mm512_setzero_si512();
    __m512i high_decrements = _mm512_setzero_si512();
    const __m512i one = _mm512_set1_epi32(1);
    for (std::uint32_t i = count; i-- > 1;) {
        const __m512i pivot = _mm512_set1_epi32(
            std::bit_cast<std::int32_t>(permutation[i]));
        const __mmask16 low_positions =
            lane_mask(std::min(i, 16U));
        const __mmask16 high_positions = i <= 16
            ? 0
            : lane_mask(i - 16);
        const __mmask16 low_mask =
            _mm512_cmp_epu32_mask(low, pivot, _MM_CMPINT_GT) & low_positions;
        const __mmask16 high_mask =
            _mm512_cmp_epu32_mask(high, pivot, _MM_CMPINT_GT) & high_positions;
        low_decrements = _mm512_mask_sub_epi32(
            low_decrements, low_mask, low_decrements, one);
        high_decrements = _mm512_mask_sub_epi32(
            high_decrements, high_mask, high_decrements, one);
    }
    alignas(64) std::uint32_t digits[32];
    _mm512_store_si512(digits, _mm512_add_epi32(low, low_decrements));
    _mm512_store_si512(digits + 16, _mm512_add_epi32(high, high_decrements));
    U128 encoded;
    for (std::uint32_t i = 0; i < count; ++i) {
        encoded.multiply_add(i + 1, digits[i]);
    }
    return encoded;
}

U128 encode_middle_avx512(const std::uint32_t* permutation) {
    return encode_factoradic_avx512(
        std::span<const std::uint32_t>(permutation, 32));
}

} // namespace

bool avx512_available() noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

std::vector<std::byte> compress_avx512(
    std::span<const std::uint32_t> permutation,
    std::size_t threads) {
    if (!avx512_available()) {
        throw std::runtime_error("AVX-512F is not supported by this CPU");
    }
    return compress_cpu_with_encoder(permutation, threads, encode_middle_avx512);
}

std::vector<std::byte> compress_compact_avx512(
    std::span<const std::uint32_t> permutation,
    std::size_t threads,
    FormatOptions format) {
    if (!avx512_available()) {
        throw std::runtime_error("AVX-512F is not supported by this CPU");
    }
    return compress_compact_with_encoder(
        permutation, threads, encode_factoradic_avx512, std::move(format));
}

std::vector<std::byte> compress_compact_avx512(
    std::span<const std::uint64_t> permutation,
    std::size_t threads,
    FormatOptions format) {
    if (!avx512_available()) {
        throw std::runtime_error("AVX-512F is not supported by this CPU");
    }
    return compress_compact_with_encoder(
        permutation, threads, encode_factoradic_avx512, std::move(format));
}

} // namespace benes::detail
