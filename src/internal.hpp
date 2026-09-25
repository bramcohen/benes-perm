// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <benes/benes.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace benes::detail {

inline constexpr std::array<std::byte, 8> magic = {
    std::byte{'B'}, std::byte{'E'}, std::byte{'N'}, std::byte{'E'},
    std::byte{'S'}, std::byte{'0'}, std::byte{'1'}, std::byte{0}};
inline constexpr std::uint32_t header_size = 20;

struct Layout {
    std::uint64_t size;
    std::uint64_t network_size;
    std::uint8_t middle_group_log2;
    MiddlePlacement middle_placement;
    std::vector<std::uint8_t> input_clusters;
    std::vector<std::uint8_t> output_clusters;
    std::uint64_t input_offset;
    std::uint64_t output_offset;
    std::uint64_t middle_offset;
    std::uint64_t end_offset;
};

struct LookupLayout {
    std::uint64_t size;
    std::uint8_t middle_group_log2;
    MiddlePlacement middle_placement;
    std::span<const std::uint8_t> input_clusters;
    std::span<const std::uint8_t> output_clusters;
    std::uint64_t input_offset;
    std::uint64_t output_offset;
    std::uint64_t middle_offset;
};

inline LookupLayout lookup_layout(const Layout& layout) {
    return {layout.size, layout.middle_group_log2, layout.middle_placement,
        layout.input_clusters, layout.output_clusters,
        layout.input_offset, layout.output_offset, layout.middle_offset};
}

struct U128 {
    std::uint64_t low = 0;
    std::uint64_t high = 0;

    void multiply_add(std::uint32_t multiplier, std::uint32_t addend) {
#if defined(__SIZEOF_INT128__)
        const auto low_product =
            static_cast<unsigned __int128>(low) * multiplier + addend;
        const auto high_product =
            static_cast<unsigned __int128>(high) * multiplier +
            static_cast<std::uint64_t>(low_product >> 64);
        low = static_cast<std::uint64_t>(low_product);
        high = static_cast<std::uint64_t>(high_product);
#else
        const std::uint64_t low_half = (low & 0xffffffffU) * multiplier + addend;
        const std::uint64_t high_half =
            (low >> 32) * multiplier + (low_half >> 32);
        low = (high_half << 32) | (low_half & 0xffffffffU);
        high = high * multiplier + (high_half >> 32);
#endif
    }

    std::uint32_t divide(std::uint32_t divisor) {
        std::array<std::uint32_t, 4> limbs = {
            static_cast<std::uint32_t>(high >> 32),
            static_cast<std::uint32_t>(high),
            static_cast<std::uint32_t>(low >> 32),
            static_cast<std::uint32_t>(low)};
        std::uint64_t remainder = 0;
        for (auto& limb : limbs) {
            const std::uint64_t value = (remainder << 32) | limb;
            limb = static_cast<std::uint32_t>(value / divisor);
            remainder = value % divisor;
        }
        high = (static_cast<std::uint64_t>(limbs[0]) << 32) | limbs[1];
        low = (static_cast<std::uint64_t>(limbs[2]) << 32) | limbs[3];
        return static_cast<std::uint32_t>(remainder);
    }

    void decrement() {
        if (low-- == 0) {
            --high;
        }
    }
};

inline std::uint32_t load_u32(std::span<const std::byte> bytes) {
    if (bytes.size() < 4) {
        throw std::invalid_argument("truncated uint32");
    }
    return std::to_integer<std::uint32_t>(bytes[0]) |
        (std::to_integer<std::uint32_t>(bytes[1]) << 8) |
        (std::to_integer<std::uint32_t>(bytes[2]) << 16) |
        (std::to_integer<std::uint32_t>(bytes[3]) << 24);
}

inline std::uint64_t load_u64(std::span<const std::byte> bytes) {
    if (bytes.size() < 8) {
        throw std::invalid_argument("truncated uint64");
    }
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t{std::to_integer<std::uint8_t>(bytes[i])} << (8 * i);
    }
    return value;
}

inline void store_u32(std::span<std::byte> bytes, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        bytes[i] = static_cast<std::byte>(value >> (8 * i));
    }
}

inline void store_u64(std::span<std::byte> bytes, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
        bytes[i] = static_cast<std::byte>(value >> (8 * i));
    }
}

inline std::array<std::uint32_t, 32> decode_middle(U128 value) {
    std::array<std::uint32_t, 32> output{};
    for (std::uint32_t i = 32; i-- > 0;) {
        output[i] = value.divide(i + 1);
    }
    for (std::uint32_t i = 0; i < 32; ++i) {
        for (std::uint32_t j = 0; j < i; ++j) {
            if (output[j] >= output[i]) {
                ++output[j];
            }
        }
    }
    return output;
}

inline U128 encode_factoradic(std::span<const std::uint32_t> permutation) {
    if (permutation.size() > 32) {
        throw std::logic_error("factoradic block is too large");
    }
    std::array<std::uint32_t, 32> digits{};
    std::copy(permutation.begin(), permutation.end(), digits.begin());
    for (std::uint32_t i = static_cast<std::uint32_t>(permutation.size());
        i-- > 1;) {
        for (std::uint32_t j = 0; j < i; ++j) {
            if (digits[j] > digits[i]) {
                --digits[j];
            }
        }
    }
    U128 result;
    for (std::uint32_t i = 0; i < permutation.size(); ++i) {
        result.multiply_add(i + 1, digits[i]);
    }
    return result;
}

inline U128 encode_middle(std::span<const std::uint32_t, 32> permutation) {
    return encode_factoradic(permutation);
}

inline std::uint32_t factoradic_bit_count(std::uint32_t size) {
    U128 factorial{1, 0};
    for (std::uint32_t i = 2; i <= size; ++i) {
        factorial.multiply_add(i, 0);
    }
    factorial.decrement();
    const std::uint32_t bits = factorial.high != 0
        ? 64 + static_cast<std::uint32_t>(std::bit_width(factorial.high))
        : static_cast<std::uint32_t>(std::bit_width(factorial.low));
    return bits;
}

inline std::uint32_t factoradic_byte_count(std::uint32_t size) {
    return (factoradic_bit_count(size) + 7) / 8;
}

inline std::array<std::uint32_t, 32> decode_factoradic(
    U128 value,
    std::uint32_t size) {
    std::array<std::uint32_t, 32> output{};
    for (std::uint32_t i = size; i-- > 0;) {
        output[i] = value.divide(i + 1);
    }
    for (std::uint32_t i = 0; i < size; ++i) {
        for (std::uint32_t j = 0; j < i; ++j) {
            if (output[j] >= output[i]) {
                ++output[j];
            }
        }
    }
    return output;
}

inline std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("compressed permutation is too large");
    }
    return left + right;
}

inline std::size_t checked_size_bytes(
    std::size_t count,
    std::size_t element_size) {
    if (count != 0 &&
        element_size > std::numeric_limits<std::size_t>::max() / count) {
        throw std::length_error("allocation size is not representable");
    }
    return count * element_size;
}

class FormatSizeCache {
public:
    explicit FormatSizeCache(std::uint8_t middle_group_log2 = 5)
        : leaf_size_(std::uint32_t{1} << middle_group_log2) {}

    struct Metrics {
        std::uint64_t switch_bits;
        std::uint64_t middle_bits;
    };

    Metrics metrics(std::uint64_t count) {
        if (count <= leaf_size_) {
            return {0, factoradic_bit_count(
                static_cast<std::uint32_t>(count))};
        }
        for (std::size_t i = 0; i < metric_count_; ++i) {
            if (metrics_[i].count == count) {
                return metrics_[i].metrics;
            }
        }
        const Metrics left = metrics(count / 2 + count % 2);
        const Metrics right = metrics(count / 2);
        const Metrics result{
            checked_add(count / 2,
                checked_add(left.switch_bits, right.switch_bits)),
            checked_add(left.middle_bits, right.middle_bits)};
        if (metric_count_ >= metrics_.size()) {
            throw std::logic_error("format metric cache exhausted");
        }
        metrics_[metric_count_++] = {count, result};
        return result;
    }

    std::uint64_t switch_bits(std::uint64_t count) {
        return metrics(count).switch_bits;
    }

    std::uint64_t middle_bits(std::uint64_t count) {
        return metrics(count).middle_bits;
    }

    std::uint64_t middle_bytes(std::uint64_t count) {
        const std::uint64_t bits = middle_bits(count);
        return bits / 8 + static_cast<std::uint64_t>(bits % 8 != 0);
    }

    std::uint64_t packet_switch_bits(
        std::uint64_t count,
        std::uint8_t depth) {
        if (count <= leaf_size_ || depth == 0) {
            return 0;
        }
        for (std::size_t i = 0; i < packet_count_; ++i) {
            if (packets_[i].count == count && packets_[i].depth == depth) {
                return packets_[i].bits;
            }
        }
        const std::uint64_t result = checked_add(count / 2,
            checked_add(
                packet_switch_bits(count / 2 + count % 2, depth - 1),
                packet_switch_bits(count / 2, depth - 1)));
        if (packet_count_ >= packets_.size()) {
            throw std::logic_error("packet metric cache exhausted");
        }
        packets_[packet_count_++] = {count, depth, result};
        return result;
    }

    [[nodiscard]] std::uint32_t leaf_size() const noexcept {
        return leaf_size_;
    }

private:
    struct MetricEntry {
        std::uint64_t count;
        Metrics metrics;
    };
    struct PacketEntry {
        std::uint64_t count;
        std::uint8_t depth;
        std::uint64_t bits;
    };

    std::uint32_t leaf_size_;
    std::array<MetricEntry, 128> metrics_;
    std::array<PacketEntry, 256> packets_;
    std::size_t metric_count_ = 0;
    std::size_t packet_count_ = 0;
};

inline std::uint8_t routing_depth(
    std::uint64_t count,
    std::uint8_t middle_group_log2) {
    const std::uint32_t leaf_size =
        std::uint32_t{1} << middle_group_log2;
    std::uint8_t depth = 0;
    while (count > leaf_size) {
        count = count / 2 + count % 2;
        ++depth;
    }
    return depth;
}

inline std::vector<std::uint8_t> default_clusters(std::uint8_t depth) {
    if (depth == 0) {
        return {};
    }
    if (depth == 1) {
        return {1};
    }
    return {
        static_cast<std::uint8_t>((depth + 1) / 2),
        static_cast<std::uint8_t>(depth / 2)};
}

inline void validate_clusters(
    std::span<const std::uint8_t> clusters,
    std::uint8_t depth) {
    std::uint32_t sum = 0;
    for (const std::uint8_t cluster : clusters) {
        if (cluster == 0) {
            throw std::invalid_argument(
                "cluster depths must be positive");
        }
        sum += cluster;
    }
    if (sum != depth) {
        throw std::invalid_argument(
            "cluster depths do not span every routing level");
    }
}

inline FormatOptions normalize_format(
    std::uint64_t count,
    FormatOptions format) {
    if (format.middle_group_log2 == 0 ||
        format.middle_group_log2 > 5) {
        throw std::invalid_argument(
            "middle_group_log2 must be between 1 and 5");
    }
    if (format.middle_placement > MiddlePlacement::embedded_output) {
        throw std::invalid_argument("invalid middle placement");
    }
    const std::uint8_t depth =
        routing_depth(count, format.middle_group_log2);
    if (format.input_clusters.empty()) {
        format.input_clusters = default_clusters(depth);
    }
    if (format.output_clusters.empty()) {
        format.output_clusters = default_clusters(depth);
    }
    validate_clusters(format.input_clusters, depth);
    validate_clusters(format.output_clusters, depth);
    if (depth == 0 &&
        format.middle_placement != MiddlePlacement::separate) {
        format.middle_placement = MiddlePlacement::separate;
    }
    return format;
}

inline std::uint64_t byte_count(std::uint64_t bits) {
    return bits / 8 + static_cast<std::uint64_t>(bits % 8 != 0);
}

inline Layout make_layout(
    std::uint64_t count,
    FormatOptions format) {
    format = normalize_format(count, std::move(format));
    FormatSizeCache sizes(format.middle_group_log2);
    const std::uint64_t switch_bits = sizes.switch_bits(count);
    const std::uint64_t middle_bits = sizes.middle_bits(count);
    const std::uint64_t header_bytes = checked_add(header_size,
        checked_add(format.input_clusters.size(),
            format.output_clusters.size()));
    const bool input_embeds =
        format.middle_placement == MiddlePlacement::embedded_input;
    const bool output_embeds =
        format.middle_placement == MiddlePlacement::embedded_output;
    const std::uint64_t input_bytes = checked_add(byte_count(switch_bits),
        input_embeds ? byte_count(middle_bits) : 0);
    const std::uint64_t output_bytes = checked_add(byte_count(switch_bits),
        output_embeds ? byte_count(middle_bits) : 0);
    const std::uint64_t middle_bytes =
        format.middle_placement == MiddlePlacement::separate
        ? byte_count(middle_bits)
        : 0;
    return {
        .size = count,
        .network_size = count > (std::uint64_t{1} << 63)
            ? throw std::overflow_error("logical network size is too large")
            : std::bit_ceil(std::max<std::uint64_t>(
                  std::uint64_t{1} << format.middle_group_log2, count)),
        .middle_group_log2 = format.middle_group_log2,
        .middle_placement = format.middle_placement,
        .input_clusters = std::move(format.input_clusters),
        .output_clusters = std::move(format.output_clusters),
        .input_offset = header_bytes,
        .output_offset = checked_add(header_bytes, input_bytes),
        .middle_offset =
            checked_add(checked_add(header_bytes, input_bytes), output_bytes),
        .end_offset = checked_add(
            checked_add(checked_add(header_bytes, input_bytes), output_bytes),
            middle_bytes),
    };
}

inline void write_header(
    std::span<std::byte> bytes,
    const Layout& layout) {
    if (bytes.size() < layout.input_offset) {
        throw std::logic_error("compressed permutation header is too small");
    }
    std::copy(magic.begin(), magic.end(), bytes.begin());
    store_u64(bytes.subspan(8, 8), layout.size);
    bytes[16] = static_cast<std::byte>(
        static_cast<std::uint8_t>(layout.middle_placement));
    bytes[17] = static_cast<std::byte>(layout.middle_group_log2);
    bytes[18] =
        static_cast<std::byte>(layout.input_clusters.size());
    bytes[19] =
        static_cast<std::byte>(layout.output_clusters.size());
    for (std::size_t i = 0; i < layout.input_clusters.size(); ++i) {
        bytes[header_size + i] =
            static_cast<std::byte>(layout.input_clusters[i]);
    }
    for (std::size_t i = 0; i < layout.output_clusters.size(); ++i) {
        bytes[header_size + layout.input_clusters.size() + i] =
            static_cast<std::byte>(layout.output_clusters[i]);
    }
}

struct SwitchBitAddress {
    std::uint64_t packet_base;
    std::uint64_t full_packet_bits;
    std::uint64_t full_packet_prefix;
    std::uint64_t tail_packet_prefix;
    std::uint64_t pair_span;
    std::uint64_t tail_packet;

    [[nodiscard]] std::uint64_t bit(std::uint64_t pair) const noexcept {
        const std::uint64_t packet = pair / pair_span;
        return packet_base +
            packet * full_packet_bits +
            (packet == tail_packet
                ? tail_packet_prefix
                : full_packet_prefix) +
            pair % pair_span;
    }
};

class SectionAddress {
public:
    SectionAddress() = default;

    SectionAddress(
        std::uint64_t count,
        std::span<const std::uint8_t> clusters,
        bool embeds_middle,
        FormatSizeCache& sizes)
        : clusters_(clusters),
          embeds_middle_(embeds_middle),
          sizes_(&sizes),
          cluster_root_count_(count),
          node_count_(count) {
        if (count > leaf_size()) {
            group_depth_ = clusters_[0];
            refresh_group_metrics();
            refresh_node_metrics();
        }
    }

    [[nodiscard]] std::uint64_t switch_bit(
        std::uint64_t pair) const {
        return bit_address().bit(pair);
    }

    [[nodiscard]] SwitchBitAddress bit_address() const noexcept {
        return {packet_base_, full_packet_bits_, full_packet_prefix_,
            tail_packet_prefix_, pair_span_, tail_packet_};
    }

    [[nodiscard]] std::uint64_t middle_bit() {
        if (!embeds_middle_) {
            throw std::logic_error(
                "middle bit requested from switch-only section");
        }
        if (cluster_root_count_ <= leaf_size()) {
            return packet_base_;
        }
        return packet_base_ +
            group_switch_bits_ +
            preceding_residual_;
    }

    [[nodiscard]] SectionAddress with_cache(FormatSizeCache& sizes) const {
        SectionAddress result = *this;
        result.sizes_ = &sizes;
        return result;
    }

    [[nodiscard]] std::uint64_t packet_base() const noexcept {
        return packet_base_;
    }

    [[nodiscard]] std::uint64_t cluster_root_count() const noexcept {
        return cluster_root_count_;
    }

    [[nodiscard]] std::uint64_t path() const noexcept {
        return path_;
    }

    [[nodiscard]] std::uint8_t group_depth() const noexcept {
        return group_depth_;
    }

    [[nodiscard]] std::uint8_t relative_level() const noexcept {
        return relative_level_;
    }

    [[nodiscard]] std::uint64_t full_packet_bits() const noexcept {
        return full_packet_bits_;
    }

    [[nodiscard]] std::uint64_t full_packet_prefix() const noexcept {
        return full_packet_prefix_;
    }

    [[nodiscard]] SectionAddress child(
        bool right,
        std::uint64_t count) const {
        SectionAddress result = *this;
        const std::uint64_t left_count = count / 2 + count % 2;
        const std::uint64_t child_count =
            right ? count / 2 : left_count;
        const std::uint8_t remaining =
            static_cast<std::uint8_t>(group_depth_ - relative_level_);
        if (remaining > 1) {
            if (right) {
                result.preceding_residual_ += residual_bits(
                    left_count, remaining - 1);
            }
            ++result.relative_level_;
            result.path_ = path_ * 2 + static_cast<std::uint64_t>(right);
            result.node_count_ = child_count;
            result.refresh_node_metrics();
            return result;
        }

        std::uint64_t preceding = preceding_residual_;
        if (right) {
            preceding += section_bits(left_count);
        }
        result.packet_base_ = packet_base_ +
            group_switch_bits_ +
            preceding;
        result.preceding_residual_ = 0;
        result.cluster_root_count_ = child_count;
        result.node_count_ = child_count;
        result.relative_level_ = 0;
        result.path_ = 0;
        if (child_count > leaf_size()) {
            ++result.cluster_index_;
            result.group_depth_ = clusters_[result.cluster_index_];
            result.refresh_group_metrics();
            result.refresh_node_metrics();
        } else {
            result.group_depth_ = 0;
            result.group_switch_bits_ = 0;
            result.full_packet_bits_ = 0;
            result.pair_span_ = 0;
            result.full_packet_prefix_ = 0;
            result.tail_packet_ = 0;
            result.tail_packet_prefix_ = 0;
        }
        return result;
    }

private:
    [[nodiscard]] std::uint64_t packet_level_bits(
        std::uint64_t packet,
        std::uint8_t level) const noexcept {
        const std::uint64_t nodes = std::uint64_t{1} << level;
        return packet_path_prefix_bits(packet, level, nodes);
    }

    [[nodiscard]] std::uint64_t packet_path_prefix_bits(
        std::uint64_t packet,
        std::uint8_t level,
        std::uint64_t path_count) const noexcept {
        const std::uint64_t nodes = std::uint64_t{1} << level;
        const std::uint64_t large_nodes = cluster_root_count_ % nodes;
        const auto contribution = [&](std::uint64_t count) {
            if (count <= leaf_size()) {
                return std::uint64_t{0};
            }
            const std::uint64_t span =
                std::uint64_t{1} << (group_depth_ - level - 1);
            const std::uint64_t first =
                static_cast<std::uint64_t>(packet) * span;
            const std::uint64_t pairs = count / 2;
            return first >= pairs
                ? std::uint64_t{0}
                : std::min(span, pairs - first);
        };
        const std::uint64_t small_count = cluster_root_count_ / nodes;
        std::uint64_t included_large = 0;
        std::uint64_t remaining_paths = path_count;
        std::uint64_t remaining_large = large_nodes;
        for (std::uint8_t bits = level; bits != 0; --bits) {
            const std::uint64_t half = std::uint64_t{1} << (bits - 1);
            if (remaining_large <= half) {
                remaining_paths = (remaining_paths + 1) / 2;
            } else {
                included_large += (remaining_paths + 1) / 2;
                remaining_paths /= 2;
                remaining_large -= half;
            }
        }
        if (remaining_paths != 0 && remaining_large != 0) {
            ++included_large;
        }
        return included_large * contribution(small_count + 1) +
            (path_count - included_large) * contribution(small_count);
    }

    [[nodiscard]] std::uint64_t calculate_full_packet_bits() const noexcept {
        std::uint64_t bits = 0;
        for (std::uint8_t level = 0; level < group_depth_; ++level) {
            bits += packet_level_bits(0, level);
        }
        return bits;
    }

    void refresh_group_metrics() {
        group_switch_bits_ =
            sizes_->packet_switch_bits(cluster_root_count_, group_depth_);
        full_packet_bits_ = calculate_full_packet_bits();
    }

    [[nodiscard]] std::uint64_t packet_prefix(
        std::uint64_t packet) const noexcept {
        std::uint64_t bits = 0;
        for (std::uint8_t level = 0;
            level < relative_level_; ++level) {
            bits += packet_level_bits(packet, level);
        }
        return bits + packet_path_prefix_bits(
            packet, relative_level_, path_);
    }

    void refresh_node_metrics() {
        pair_span_ = std::uint64_t{1} <<
            (group_depth_ - relative_level_ - 1);
        full_packet_prefix_ = packet_prefix(0);
        const std::uint64_t pairs = node_count_ / 2;
        tail_packet_ = pairs == 0 ? 0 : (pairs - 1) / pair_span_;
        tail_packet_prefix_ = packet_prefix(tail_packet_);
    }

    [[nodiscard]] std::uint32_t leaf_size() const noexcept {
        return sizes_->leaf_size();
    }

    [[nodiscard]] std::uint64_t section_bits(std::uint64_t count) const {
        return sizes_->switch_bits(count) +
            (embeds_middle_ ? sizes_->middle_bits(count) : 0);
    }

    [[nodiscard]] std::uint64_t residual_bits(
        std::uint64_t count,
        std::uint8_t consumed_depth) const {
        return section_bits(count) -
            sizes_->packet_switch_bits(count, consumed_depth);
    }

    std::span<const std::uint8_t> clusters_;
    bool embeds_middle_ = false;
    FormatSizeCache* sizes_ = nullptr;
    std::size_t cluster_index_ = 0;
    std::uint8_t group_depth_ = 0;
    std::uint8_t relative_level_ = 0;
    std::uint64_t path_ = 0;
    std::uint64_t cluster_root_count_;
    std::uint64_t node_count_;
    std::uint64_t packet_base_ = 0;
    std::uint64_t preceding_residual_ = 0;
    std::uint64_t group_switch_bits_ = 0;
    std::uint64_t full_packet_bits_ = 0;
    std::uint64_t pair_span_ = 0;
    std::uint64_t full_packet_prefix_ = 0;
    std::uint64_t tail_packet_ = 0;
    std::uint64_t tail_packet_prefix_ = 0;
};

struct CompactNode {
    std::uint32_t count;
    std::uint32_t permutation_offset;
    std::uint32_t left_offset;
    std::uint32_t right_offset;
    std::uint64_t input_packet_base;
    std::uint64_t output_packet_base;
    std::uint64_t middle_bit_offset;
    std::uint32_t input_root_count;
    std::uint32_t input_path;
    std::uint32_t output_root_count;
    std::uint32_t output_path;
    std::uint8_t input_group_depth;
    std::uint8_t input_relative_level;
    std::uint8_t output_group_depth;
    std::uint8_t output_relative_level;
    std::uint32_t reserved = 0;
    std::uint64_t input_full_packet_bits;
    std::uint64_t input_full_packet_prefix;
    std::uint64_t output_full_packet_bits;
    std::uint64_t output_full_packet_prefix;
};

static_assert(sizeof(CompactNode) == 96);

struct CompactLevel {
    std::uint32_t node_base = 0;
    std::uint32_t node_count = 0;
    std::uint32_t leaf_base = 0;
    std::uint32_t leaf_count = 0;
    std::uint32_t maximum_count = 0;
    std::uint32_t maximum_pairs = 0;
};

struct CompactTree {
    std::uint64_t switch_bits;
    std::uint64_t middle_bits;
    std::uint64_t input_section_bits;
    std::uint64_t output_section_bits;
    std::uint64_t middle_section_bits;
    std::vector<CompactNode> nodes;
    std::vector<CompactNode> leaves;
    std::vector<CompactLevel> levels;
};

struct CompactNode64 {
    std::uint64_t count;
    std::uint64_t permutation_offset;
    std::uint64_t left_offset;
    std::uint64_t right_offset;
    std::uint64_t input_packet_base;
    std::uint64_t output_packet_base;
    std::uint64_t middle_bit_offset;
    std::uint64_t input_root_count;
    std::uint64_t input_path;
    std::uint64_t output_root_count;
    std::uint64_t output_path;
    std::uint8_t input_group_depth;
    std::uint8_t input_relative_level;
    std::uint8_t output_group_depth;
    std::uint8_t output_relative_level;
    std::uint32_t reserved = 0;
    std::uint64_t input_full_packet_bits;
    std::uint64_t input_full_packet_prefix;
    std::uint64_t output_full_packet_bits;
    std::uint64_t output_full_packet_prefix;
};

static_assert(sizeof(CompactNode64) == 128);

struct CompactLevel64 {
    std::uint64_t node_base = 0;
    std::uint64_t node_count = 0;
    std::uint64_t leaf_base = 0;
    std::uint64_t leaf_count = 0;
    std::uint64_t maximum_count = 0;
    std::uint64_t maximum_pairs = 0;
};

struct CompactTree64 {
    std::uint64_t switch_bits;
    std::uint64_t middle_bits;
    std::uint64_t input_section_bits;
    std::uint64_t output_section_bits;
    std::uint64_t middle_section_bits;
    std::vector<CompactNode64> nodes;
    std::vector<CompactNode64> leaves;
    std::vector<CompactLevel64> levels;
};

inline CompactTree make_compact_tree(
    std::uint32_t count,
    const FormatOptions& format) {
    FormatSizeCache sizes(format.middle_group_log2);
    const std::uint64_t switch_bits = sizes.switch_bits(count);
    const std::uint64_t middle_bits = sizes.middle_bits(count);
    const bool input_embeds =
        format.middle_placement == MiddlePlacement::embedded_input;
    const bool output_embeds =
        format.middle_placement == MiddlePlacement::embedded_output;
    CompactTree tree{switch_bits, middle_bits,
        switch_bits + (input_embeds ? middle_bits : 0),
        switch_bits + (output_embeds ? middle_bits : 0),
        format.middle_placement == MiddlePlacement::separate ? middle_bits : 0,
        {}, {}, {}};
    std::vector<std::vector<CompactNode>> nodes_by_depth;
    std::vector<std::vector<CompactNode>> leaves_by_depth;
    std::vector<std::uint32_t> next_offsets(1);
    SectionAddress input_address(
        count, format.input_clusters, input_embeds, sizes);
    SectionAddress output_address(
        count, format.output_clusters, output_embeds, sizes);
    const auto build_nodes = [&](auto&& self,
                                 std::uint32_t node_count,
                                 std::uint32_t depth,
                                 std::uint32_t permutation_offset,
                                 SectionAddress input,
                                 SectionAddress output,
                                 std::uint64_t middle_bit_offset) -> void {
        if (leaves_by_depth.size() <= depth) {
            leaves_by_depth.resize(depth + 1);
            nodes_by_depth.resize(depth + 1);
        }
        if (node_count <= sizes.leaf_size()) {
            if (input_embeds) {
                middle_bit_offset = input.middle_bit();
            } else if (output_embeds) {
                middle_bit_offset = output.middle_bit();
            }
            leaves_by_depth[depth].push_back(CompactNode{
                .count = node_count,
                .permutation_offset = permutation_offset,
                .middle_bit_offset = middle_bit_offset,
            });
            return;
        }
        if (next_offsets.size() <= depth + 1) {
            next_offsets.resize(depth + 2);
        }
        const std::uint32_t left_count = (node_count + 1) / 2;
        const std::uint32_t right_count = node_count / 2;
        const std::uint32_t left_offset = next_offsets[depth + 1];
        next_offsets[depth + 1] += left_count;
        const std::uint32_t right_offset = next_offsets[depth + 1];
        next_offsets[depth + 1] += right_count;
        nodes_by_depth[depth].push_back({node_count, permutation_offset,
            left_offset, right_offset,
            input.packet_base(), output.packet_base(), middle_bit_offset,
            static_cast<std::uint32_t>(input.cluster_root_count()),
            static_cast<std::uint32_t>(input.path()),
            static_cast<std::uint32_t>(output.cluster_root_count()),
            static_cast<std::uint32_t>(output.path()),
            input.group_depth(), input.relative_level(),
            output.group_depth(), output.relative_level(), 0,
            input.full_packet_bits(), input.full_packet_prefix(),
            output.full_packet_bits(), output.full_packet_prefix()});
        self(self, left_count, depth + 1, left_offset,
            input.child(false, node_count),
            output.child(false, node_count), middle_bit_offset);
        self(self, right_count, depth + 1, right_offset,
            input.child(true, node_count),
            output.child(true, node_count),
            middle_bit_offset + sizes.middle_bits(left_count));
    };
    build_nodes(build_nodes, count, 0, 0,
        input_address, output_address, 0);

    tree.levels.resize(nodes_by_depth.size());
    for (std::size_t depth = 0; depth < tree.levels.size(); ++depth) {
        CompactLevel& level = tree.levels[depth];
        level.node_base = static_cast<std::uint32_t>(tree.nodes.size());
        level.node_count =
            static_cast<std::uint32_t>(nodes_by_depth[depth].size());
        for (const CompactNode& node : nodes_by_depth[depth]) {
            level.maximum_count = std::max(level.maximum_count, node.count);
            level.maximum_pairs = std::max(
                level.maximum_pairs, node.count / 2);
            tree.nodes.push_back(node);
        }
        level.leaf_base = static_cast<std::uint32_t>(tree.leaves.size());
        level.leaf_count =
            static_cast<std::uint32_t>(leaves_by_depth[depth].size());
        tree.leaves.insert(tree.leaves.end(), leaves_by_depth[depth].begin(),
            leaves_by_depth[depth].end());
    }
    return tree;
}

inline CompactTree64 make_compact_tree64(
    std::uint64_t count,
    const FormatOptions& format) {
    FormatSizeCache sizes(format.middle_group_log2);
    const std::uint64_t switch_bits = sizes.switch_bits(count);
    const std::uint64_t middle_bits = sizes.middle_bits(count);
    const bool input_embeds =
        format.middle_placement == MiddlePlacement::embedded_input;
    const bool output_embeds =
        format.middle_placement == MiddlePlacement::embedded_output;
    CompactTree64 tree{switch_bits, middle_bits,
        checked_add(switch_bits, input_embeds ? middle_bits : 0),
        checked_add(switch_bits, output_embeds ? middle_bits : 0),
        format.middle_placement == MiddlePlacement::separate ? middle_bits : 0,
        {}, {}, {}};
    std::vector<std::vector<CompactNode64>> nodes_by_depth;
    std::vector<std::vector<CompactNode64>> leaves_by_depth;
    std::vector<std::uint64_t> next_offsets(1);
    SectionAddress input_address(
        count, format.input_clusters, input_embeds, sizes);
    SectionAddress output_address(
        count, format.output_clusters, output_embeds, sizes);
    const auto build_nodes = [&](auto&& self,
                                 std::uint64_t node_count,
                                 std::size_t depth,
                                 std::uint64_t permutation_offset,
                                 SectionAddress input,
                                 SectionAddress output,
                                 std::uint64_t middle_bit_offset) -> void {
        if (leaves_by_depth.size() <= depth) {
            leaves_by_depth.resize(depth + 1);
            nodes_by_depth.resize(depth + 1);
        }
        if (node_count <= sizes.leaf_size()) {
            if (input_embeds) {
                middle_bit_offset = input.middle_bit();
            } else if (output_embeds) {
                middle_bit_offset = output.middle_bit();
            }
            leaves_by_depth[depth].push_back(CompactNode64{
                .count = node_count,
                .permutation_offset = permutation_offset,
                .middle_bit_offset = middle_bit_offset,
            });
            return;
        }
        if (next_offsets.size() <= depth + 1) {
            next_offsets.resize(depth + 2);
        }
        const std::uint64_t left_count =
            node_count / 2 + node_count % 2;
        const std::uint64_t right_count = node_count / 2;
        const std::uint64_t left_offset = next_offsets[depth + 1];
        next_offsets[depth + 1] =
            checked_add(next_offsets[depth + 1], left_count);
        const std::uint64_t right_offset = next_offsets[depth + 1];
        next_offsets[depth + 1] =
            checked_add(next_offsets[depth + 1], right_count);
        nodes_by_depth[depth].push_back({
            node_count, permutation_offset, left_offset, right_offset,
            input.packet_base(), output.packet_base(), middle_bit_offset,
            input.cluster_root_count(), input.path(),
            output.cluster_root_count(), output.path(),
            input.group_depth(), input.relative_level(),
            output.group_depth(), output.relative_level(), 0,
            input.full_packet_bits(), input.full_packet_prefix(),
            output.full_packet_bits(), output.full_packet_prefix()});
        self(self, left_count, depth + 1, left_offset,
            input.child(false, node_count),
            output.child(false, node_count), middle_bit_offset);
        self(self, right_count, depth + 1, right_offset,
            input.child(true, node_count),
            output.child(true, node_count),
            checked_add(middle_bit_offset, sizes.middle_bits(left_count)));
    };
    build_nodes(build_nodes, count, 0, 0,
        input_address, output_address, 0);

    tree.levels.resize(nodes_by_depth.size());
    for (std::size_t depth = 0; depth < tree.levels.size(); ++depth) {
        CompactLevel64& level = tree.levels[depth];
        level.node_base = tree.nodes.size();
        level.node_count = nodes_by_depth[depth].size();
        for (const CompactNode64& node : nodes_by_depth[depth]) {
            level.maximum_count = std::max(level.maximum_count, node.count);
            level.maximum_pairs =
                std::max(level.maximum_pairs, node.count / 2);
            tree.nodes.push_back(node);
        }
        level.leaf_base = tree.leaves.size();
        level.leaf_count = leaves_by_depth[depth].size();
        tree.leaves.insert(tree.leaves.end(), leaves_by_depth[depth].begin(),
            leaves_by_depth[depth].end());
    }
    return tree;
}

template<typename ReadByte>
bool read_bit(
    std::uint64_t section_offset,
    std::uint64_t bit,
    ReadByte&& read_byte) {
    const std::uint8_t byte = std::to_integer<std::uint8_t>(
        read_byte(section_offset + bit / 8));
    return ((byte >> (bit & 7)) & 1) != 0;
}

template<typename ReadByte>
U128 read_rank(
    std::uint64_t section_offset,
    std::uint64_t bit_offset,
    std::uint32_t bit_count,
    ReadByte&& read_byte) {
    U128 rank;
    for (std::uint32_t bit = 0; bit < bit_count; ++bit) {
        if (!read_bit(section_offset, bit_offset + bit, read_byte)) {
            continue;
        }
        if (bit < 64) {
            rank.low |= std::uint64_t{1} << bit;
        } else {
            rank.high |= std::uint64_t{1} << (bit - 64);
        }
    }
    return rank;
}

template<typename ReadByte>
std::uint64_t lookup(
    const LookupLayout& layout,
    std::uint64_t value,
    bool inverse,
    ReadByte&& read_byte) {
    if (value >= layout.size) {
        throw std::out_of_range("permutation lookup is out of range");
    }
    struct Frame {
        std::uint64_t size = 0;
        SwitchBitAddress input{};
        SwitchBitAddress output{};
        bool right = false;
    };
    std::array<Frame, 64> frames{};
    std::size_t depth = 0;
    std::uint64_t count = layout.size;
    std::uint64_t index = value;
    std::uint64_t middle_bit = 0;
    FormatSizeCache sizes(layout.middle_group_log2);
    SectionAddress input_address(count, layout.input_clusters,
        layout.middle_placement == MiddlePlacement::embedded_input, sizes);
    SectionAddress output_address(count, layout.output_clusters,
        layout.middle_placement == MiddlePlacement::embedded_output, sizes);
    while (count > sizes.leaf_size()) {
        const bool odd_tail = count % 2 == 1 && index == count - 1;
        bool right = false;
        if (!odd_tail) {
            const std::uint64_t bit =
                (inverse ? output_address : input_address)
                    .switch_bit(index / 2);
            const std::uint64_t section =
                inverse ? layout.output_offset : layout.input_offset;
            right = read_bit(section, bit, read_byte) !=
                static_cast<bool>(index & 1);
        }
        frames[depth++] = {count, input_address.bit_address(),
            output_address.bit_address(), right};
        const std::uint64_t left_count = count / 2 + count % 2;
        if (right) {
            middle_bit += sizes.middle_bits(left_count);
        }
        input_address = input_address.child(right, count);
        output_address = output_address.child(right, count);
        index /= 2;
        count = right ? count / 2 : left_count;
    }

    std::uint64_t rank_section = layout.middle_offset;
    std::uint64_t rank_bit = middle_bit;
    if (layout.middle_placement == MiddlePlacement::embedded_input) {
        rank_section = layout.input_offset;
        rank_bit = input_address.middle_bit();
    } else if (layout.middle_placement == MiddlePlacement::embedded_output) {
        rank_section = layout.output_offset;
        rank_bit = output_address.middle_bit();
    }
    const auto leaf = decode_factoradic(
        read_rank(rank_section, rank_bit,
            factoradic_bit_count(static_cast<std::uint32_t>(count)), read_byte),
        static_cast<std::uint32_t>(count));
    if (!inverse) {
        index = leaf[index];
    } else {
        const std::uint32_t target = static_cast<std::uint32_t>(index);
        for (index = 0; index < count && leaf[index] != target; ++index) {}
        if (index == count) {
            throw std::runtime_error("invalid compact factorial code");
        }
    }

    while (depth != 0) {
        const Frame frame = frames[--depth];
        const std::uint64_t pairs = frame.size / 2;
        if (!frame.right && frame.size % 2 == 1 && index == pairs) {
            index = frame.size - 1;
            continue;
        }
        const std::uint64_t section =
            inverse ? layout.input_offset : layout.output_offset;
        const std::uint64_t bit =
            (inverse ? frame.input : frame.output).bit(index);
        const std::uint64_t parity =
            static_cast<std::uint64_t>(frame.right) ^
            static_cast<std::uint64_t>(read_bit(section, bit, read_byte));
        index = index * 2 + parity;
    }
    return index;
}

template<typename ReadByte>
std::uint64_t forward_lookup(
    const LookupLayout& layout,
    std::uint64_t input,
    ReadByte&& read_byte) {
    return lookup(layout, input, false, std::forward<ReadByte>(read_byte));
}

template<typename ReadByte>
std::uint64_t inverse_lookup(
    const LookupLayout& layout,
    std::uint64_t output,
    ReadByte&& read_byte) {
    return lookup(layout, output, true, std::forward<ReadByte>(read_byte));
}

Layout parse_layout(std::span<const std::byte> bytes);
Layout parse_header_for_file(
    std::span<const std::byte> header,
    std::uint64_t file_size);
std::vector<std::byte> make_blob(
    std::uint32_t original_size,
    std::uint32_t network_size,
    std::span<const std::uint32_t> input_words,
    std::span<const std::uint32_t> output_words,
    std::span<const U128> middles);
void set_last_device_seconds(double seconds) noexcept;
double last_device_seconds() noexcept;

std::vector<std::byte> compress_portable(
    std::span<const std::uint32_t> permutation,
    std::size_t threads);
using MiddleEncoder = U128 (*)(const std::uint32_t*);
using FactoradicEncoder = U128 (*)(std::span<const std::uint32_t>);
std::vector<std::byte> compress_cpu_with_encoder(
    std::span<const std::uint32_t> permutation,
    std::size_t threads,
    MiddleEncoder encoder);
std::vector<std::byte> compress_compact(
    std::span<const std::uint32_t> permutation,
    std::size_t threads,
    FormatOptions format);
std::vector<std::byte> compress_compact(
    std::span<const std::uint64_t> permutation,
    std::size_t threads,
    FormatOptions format);
std::vector<std::byte> compress_compact_with_encoder(
    std::span<const std::uint32_t> permutation,
    std::size_t threads,
    FactoradicEncoder encoder,
    FormatOptions format);
std::vector<std::byte> compress_compact_with_encoder(
    std::span<const std::uint64_t> permutation,
    std::size_t threads,
    FactoradicEncoder encoder,
    FormatOptions format);

bool metal_available() noexcept;
std::vector<std::byte> compress_metal(std::span<const std::uint32_t> permutation);
std::vector<std::byte> compress_compact_metal(
    std::span<const std::uint32_t> permutation,
    FormatOptions format);
std::vector<std::byte> compress_compact_metal(
    std::span<const std::uint64_t> permutation,
    FormatOptions format);
bool cuda_available() noexcept;
std::vector<std::byte> compress_cuda(std::span<const std::uint32_t> permutation);
std::vector<std::byte> compress_compact_cuda(
    std::span<const std::uint32_t> permutation,
    FormatOptions format);
std::vector<std::byte> compress_compact_cuda(
    std::span<const std::uint64_t> permutation,
    FormatOptions format);
bool avx512_available() noexcept;
std::vector<std::byte> compress_avx512(
    std::span<const std::uint32_t> permutation,
    std::size_t threads);
std::vector<std::byte> compress_compact_avx512(
    std::span<const std::uint32_t> permutation,
    std::size_t threads,
    FormatOptions format);
std::vector<std::byte> compress_compact_avx512(
    std::span<const std::uint64_t> permutation,
    std::size_t threads,
    FormatOptions format);

} // namespace benes::detail
