// SPDX-License-Identifier: Apache-2.0
#include <benes/benes.hpp>

#include "internal.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>

namespace benes::detail {
namespace {

thread_local double device_seconds = 0;

Layout parse_header(std::span<const std::byte> bytes, std::uint64_t actual_size) {
    if (bytes.size() < header_size) {
        throw std::invalid_argument("compressed permutation header is truncated");
    }
    if (!std::equal(magic.begin(), magic.end(), bytes.begin())) {
        throw std::invalid_argument("compressed permutation has invalid magic");
    }
    const std::uint64_t size64 = load_u64(bytes.subspan(8, 8));
    const auto middle_placement = static_cast<MiddlePlacement>(
        std::to_integer<std::uint8_t>(bytes[16]));
    const std::uint8_t middle_group_log2 =
        std::to_integer<std::uint8_t>(bytes[17]);
    const std::size_t input_cluster_count =
        std::to_integer<std::uint8_t>(bytes[18]);
    const std::size_t output_cluster_count =
        std::to_integer<std::uint8_t>(bytes[19]);
    const std::size_t encoded_header_size =
        header_size + input_cluster_count + output_cluster_count;
    if (encoded_header_size > header_size + 64 ||
        encoded_header_size > actual_size ||
        bytes.size() < encoded_header_size) {
        throw std::invalid_argument("unsupported compressed permutation header size");
    }
    if (middle_placement > MiddlePlacement::embedded_output) {
        throw std::invalid_argument(
            "compressed permutation has invalid format parameters");
    }
    std::vector<std::uint8_t> input_clusters(input_cluster_count);
    std::vector<std::uint8_t> output_clusters(output_cluster_count);
    for (std::size_t i = 0; i < input_cluster_count; ++i) {
        input_clusters[i] =
            std::to_integer<std::uint8_t>(bytes[header_size + i]);
    }
    for (std::size_t i = 0; i < output_cluster_count; ++i) {
        output_clusters[i] = std::to_integer<std::uint8_t>(
            bytes[header_size + input_cluster_count + i]);
    }
    if (size64 == 0) {
        throw std::invalid_argument("compressed permutation has invalid dimensions");
    }
    const FormatOptions format = normalize_format(
        size64,
        {.middle_group_log2 = middle_group_log2,
            .input_clusters = std::move(input_clusters),
            .output_clusters = std::move(output_clusters),
            .middle_placement = middle_placement});
    Layout layout = make_layout(size64, format);
    if (layout.end_offset != actual_size) {
        throw std::invalid_argument("compressed permutation has invalid section bounds");
    }
    return layout;
}

template<typename ReadByte>
void validate_padding(const Layout& layout, ReadByte&& read_byte) {
    FormatSizeCache sizes(layout.middle_group_log2);
    const std::uint64_t switch_bits = sizes.switch_bits(layout.size);
    const std::uint64_t middle_bits = sizes.middle_bits(layout.size);
    const std::array<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>, 3>
        sections = {{
            {layout.input_offset, layout.output_offset,
                switch_bits +
                    (layout.middle_placement == MiddlePlacement::embedded_input
                        ? middle_bits : 0)},
            {layout.output_offset, layout.middle_offset,
                switch_bits +
                    (layout.middle_placement == MiddlePlacement::embedded_output
                        ? middle_bits : 0)},
            {layout.middle_offset, layout.end_offset,
                layout.middle_placement == MiddlePlacement::separate
                    ? middle_bits : 0},
        }};
    for (const auto [begin, end, used_bits] : sections) {
        const std::uint64_t section_bytes = end - begin;
        const std::uint64_t used_bytes = used_bits / 8;
        if (used_bytes > section_bytes ||
            (used_bytes == section_bytes && used_bits % 8 != 0)) {
            throw std::invalid_argument(
                "compressed permutation section is too small");
        }
        const std::uint64_t padding_bits =
            (section_bytes - used_bytes) * 8 - used_bits % 8;
        for (std::uint64_t bit = 0; bit < padding_bits; ++bit) {
            if (read_bit(begin, used_bits + bit, read_byte)) {
                throw std::invalid_argument(
                    "compressed permutation has nonzero section padding bits");
            }
        }
    }
}

} // namespace

void set_last_device_seconds(double seconds) noexcept {
    device_seconds = seconds;
}

double last_device_seconds() noexcept {
    return device_seconds;
}

Layout parse_layout(std::span<const std::byte> bytes) {
    const Layout layout = parse_header(bytes, bytes.size());
    validate_padding(
        layout, [&](std::uint64_t offset) { return bytes[offset]; });
    return layout;
}

Layout parse_header_for_file(
    std::span<const std::byte> header,
    std::uint64_t file_size) {
    return parse_header(header, file_size);
}

std::vector<std::byte> make_blob(
    std::uint32_t,
    std::uint32_t,
    std::span<const std::uint32_t>,
    std::span<const std::uint32_t>,
    std::span<const U128>) {
    throw std::logic_error(
        "legacy layer-major encoder cannot write the current format");
}

namespace {

template<typename Value>
void validate_permutation(std::span<const Value> permutation) {
    if (permutation.empty()) {
        throw std::invalid_argument("permutation size is out of range");
    }
    if constexpr (sizeof(Value) == sizeof(std::uint32_t)) {
        if (permutation.size() >
            std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                "permutation size requires 64-bit source values");
        }
    }
    std::vector<bool> seen(permutation.size());
    for (const Value value : permutation) {
        if (value >= permutation.size() ||
            seen[static_cast<std::size_t>(value)]) {
            throw std::invalid_argument(
                "input must contain every value from zero through size minus one");
        }
        seen[static_cast<std::size_t>(value)] = true;
    }
}

void set_encoded_bit(
    std::span<std::byte> bytes,
    std::uint64_t section_offset,
    std::uint64_t bit,
    bool value) {
    if (value) {
        auto& byte = reinterpret_cast<std::uint8_t&>(
            bytes[section_offset + bit / 8]);
        std::atomic_ref<std::uint8_t>(byte).fetch_or(
            static_cast<std::uint8_t>(1U << (bit & 7)),
            std::memory_order_relaxed);
    }
}

template<typename Value>
void encode_compact_node(
    std::span<const Value> permutation,
    std::span<std::byte> bytes,
    const Layout& layout,
    SectionAddress input_address,
    SectionAddress output_address,
    std::uint64_t middle_bit_offset,
    FormatSizeCache& sizes,
    std::size_t threads,
    FactoradicEncoder encoder) {
    const Value count = static_cast<Value>(permutation.size());
    if (count <= sizes.leaf_size()) {
        std::array<std::uint32_t, 32> local{};
        for (std::size_t i = 0; i < permutation.size(); ++i) {
            local[i] = static_cast<std::uint32_t>(permutation[i]);
        }
        const U128 rank = encoder(
            std::span(local).first(permutation.size()));
        std::uint64_t section_offset = layout.middle_offset;
        if (layout.middle_placement == MiddlePlacement::embedded_input) {
            section_offset = layout.input_offset;
            middle_bit_offset = input_address.middle_bit();
        } else if (
            layout.middle_placement == MiddlePlacement::embedded_output) {
            section_offset = layout.output_offset;
            middle_bit_offset = output_address.middle_bit();
        }
        const std::uint32_t bit_count =
            factoradic_bit_count(static_cast<std::uint32_t>(count));
        for (std::uint32_t bit = 0; bit < bit_count; ++bit) {
            const std::uint64_t word = bit < 64 ? rank.low : rank.high;
            set_encoded_bit(bytes, section_offset, middle_bit_offset + bit,
                ((word >> (bit & 63)) & 1) != 0);
        }
        return;
    }

    const Value pairs = count / 2;
    std::vector<Value> left(
        static_cast<std::size_t>(count / 2 + count % 2));
    std::vector<Value> right(static_cast<std::size_t>(count / 2));
    {
        std::vector<Value> inverse(static_cast<std::size_t>(count));
        for (Value input = 0; input < count; ++input) {
            inverse[static_cast<std::size_t>(
                permutation[static_cast<std::size_t>(input)])] =
                static_cast<Value>(input);
        }
        std::vector<std::array<Value, 2>> neighbors(
            static_cast<std::size_t>(count));
        std::vector<std::uint8_t> degree(static_cast<std::size_t>(count));
        const auto connect = [&](Value first, Value second) {
            neighbors[static_cast<std::size_t>(first)]
                [degree[static_cast<std::size_t>(first)]++] = second;
            neighbors[static_cast<std::size_t>(second)]
                [degree[static_cast<std::size_t>(second)]++] = first;
        };
        for (Value pair = 0; pair < pairs; ++pair) {
            connect(static_cast<Value>(pair * 2),
                static_cast<Value>(pair * 2 + 1));
            connect(inverse[static_cast<std::size_t>(pair * 2)],
                inverse[static_cast<std::size_t>(pair * 2 + 1)]);
        }
        std::vector<std::int8_t> color(static_cast<std::size_t>(count), -1);
        std::deque<Value> pending;
        const auto color_component = [&](Value first) {
            pending.push_back(first);
            while (!pending.empty()) {
                const Value edge = pending.front();
                pending.pop_front();
                const std::size_t edge_index =
                    static_cast<std::size_t>(edge);
                for (std::uint8_t i = 0; i < degree[edge_index]; ++i) {
                    const Value next = neighbors[edge_index][i];
                    const std::size_t next_index =
                        static_cast<std::size_t>(next);
                    const std::int8_t expected =
                        static_cast<std::int8_t>(color[edge_index] ^ 1);
                    if (color[next_index] < 0) {
                        color[next_index] = expected;
                        pending.push_back(next);
                    } else if (color[next_index] != expected) {
                        throw std::logic_error(
                            "compact network coloring failed");
                    }
                }
            }
        };
        if (count % 2 == 1) {
            color[static_cast<std::size_t>(count - 1)] = 0;
            color_component(static_cast<Value>(count - 1));
            const Value odd_output_edge =
                inverse[static_cast<std::size_t>(count - 1)];
            if (color[static_cast<std::size_t>(odd_output_edge)] < 0) {
                color[static_cast<std::size_t>(odd_output_edge)] = 0;
                color_component(odd_output_edge);
            } else if (
                color[static_cast<std::size_t>(odd_output_edge)] != 0) {
                throw std::logic_error(
                    "compact odd endpoints have incompatible colors");
            }
        }
        for (Value edge = 0; edge < count; ++edge) {
            if (color[static_cast<std::size_t>(edge)] < 0) {
                color[static_cast<std::size_t>(edge)] = 0;
                color_component(static_cast<Value>(edge));
            }
        }
        for (Value input = 0; input < count; ++input) {
            const bool branch =
                color[static_cast<std::size_t>(input)] != 0;
            if (input != count - 1 || count % 2 == 0) {
                set_encoded_bit(bytes, layout.input_offset,
                    input_address.switch_bit(input / 2),
                    branch != ((input & 1) != 0));
            }
            const Value output =
                permutation[static_cast<std::size_t>(input)];
            if (output != count - 1 || count % 2 == 0) {
                set_encoded_bit(bytes, layout.output_offset,
                    output_address.switch_bit(output / 2),
                    branch != ((output & 1) != 0));
            }
            (branch ? right : left)[static_cast<std::size_t>(input / 2)] =
                output / 2;
        }
    }
    const std::uint64_t right_middle_bit =
        middle_bit_offset +
        sizes.middle_bits(left.size());
    if (threads > 1 && count >= 4096) {
        const std::size_t right_threads = threads / 2;
        const std::size_t left_threads = threads - right_threads;
        FormatSizeCache right_sizes = sizes;
        const SectionAddress right_input_address =
            input_address.child(true, count).with_cache(right_sizes);
        const SectionAddress right_output_address =
            output_address.child(true, count).with_cache(right_sizes);
        auto right_work = std::async(std::launch::async, [&] {
            encode_compact_node(
                std::span<const Value>(right), bytes, layout,
                right_input_address,
                right_output_address, right_middle_bit,
                right_sizes, right_threads, encoder);
        });
        encode_compact_node(
            std::span<const Value>(left), bytes, layout,
            input_address.child(false, count),
            output_address.child(false, count), middle_bit_offset,
            sizes, left_threads, encoder);
        right_work.get();
    } else {
        encode_compact_node(std::span<const Value>(left), bytes, layout,
            input_address.child(false, count),
            output_address.child(false, count), middle_bit_offset,
            sizes, 1, encoder);
        encode_compact_node(std::span<const Value>(right), bytes, layout,
            input_address.child(true, count),
            output_address.child(true, count), right_middle_bit,
            sizes, 1, encoder);
    }
}

} // namespace

template<typename Value>
std::vector<std::byte> compress_compact_with_encoder_impl(
    std::span<const Value> permutation,
    std::size_t threads,
    FactoradicEncoder encoder,
    FormatOptions format) {
    validate_permutation(permutation);
    const std::uint64_t count = permutation.size();
    const Layout layout = make_layout(count, std::move(format));
    if (layout.end_offset > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error(
            "compressed permutation does not fit in memory");
    }
    FormatSizeCache sizes(layout.middle_group_log2);
    std::vector<std::byte> bytes(
        static_cast<std::size_t>(layout.end_offset));
    write_header(bytes, layout);
    if (threads == 0) {
        threads = std::max(1U, std::thread::hardware_concurrency());
    }
    SectionAddress input_address(count, layout.input_clusters,
        layout.middle_placement == MiddlePlacement::embedded_input, sizes);
    SectionAddress output_address(count, layout.output_clusters,
        layout.middle_placement == MiddlePlacement::embedded_output, sizes);
    encode_compact_node(permutation, bytes, layout,
        input_address, output_address, 0, sizes, threads, encoder);
    return bytes;
}

std::vector<std::byte> compress_compact_with_encoder(
    std::span<const std::uint32_t> permutation,
    std::size_t threads,
    FactoradicEncoder encoder,
    FormatOptions format) {
    return compress_compact_with_encoder_impl(
        permutation, threads, encoder, std::move(format));
}

std::vector<std::byte> compress_compact_with_encoder(
    std::span<const std::uint64_t> permutation,
    std::size_t threads,
    FactoradicEncoder encoder,
    FormatOptions format) {
    return compress_compact_with_encoder_impl(
        permutation, threads, encoder, std::move(format));
}

std::vector<std::byte> compress_compact(
    std::span<const std::uint32_t> permutation,
    std::size_t threads,
    FormatOptions format) {
    return compress_compact_with_encoder(
        permutation, threads, encode_factoradic, std::move(format));
}

std::vector<std::byte> compress_compact(
    std::span<const std::uint64_t> permutation,
    std::size_t threads,
    FormatOptions format) {
    return compress_compact_with_encoder_impl(
        permutation, threads, encode_factoradic, std::move(format));
}

} // namespace benes::detail

namespace benes {

std::vector<std::byte> compress(
    std::span<const std::uint32_t> permutation,
    CompressOptions options) {
    Backend backend = options.backend;
    if (backend == Backend::automatic) {
        if (detail::metal_available()) {
            backend = Backend::metal;
        } else if (detail::cuda_available()) {
            backend = Backend::cuda;
        } else if (detail::avx512_available()) {
            backend = Backend::avx512;
        } else {
            backend = Backend::portable_cpu;
        }
    }
    if ((backend == Backend::metal && !detail::metal_available()) ||
        (backend == Backend::cuda && !detail::cuda_available()) ||
        (backend == Backend::avx512 && !detail::avx512_available())) {
        throw std::runtime_error(
            std::string(backend_name(backend)) + " backend is unavailable");
    }
    if (backend == Backend::metal) {
        return detail::compress_compact_metal(
            permutation, options.format);
    }
    if (backend == Backend::cuda) {
        return detail::compress_compact_cuda(
            permutation, options.format);
    }
    if (backend == Backend::avx512) {
        return detail::compress_compact_avx512(
            permutation, options.threads, options.format);
    }
    detail::set_last_device_seconds(0);
    return detail::compress_compact(
        permutation, options.threads, std::move(options.format));
}

std::vector<std::byte> compress(
    std::span<const std::uint64_t> permutation,
    CompressOptions options) {
    Backend backend = options.backend;
    if (backend == Backend::automatic) {
        if (detail::metal_available()) {
            backend = Backend::metal;
        } else if (detail::cuda_available()) {
            backend = Backend::cuda;
        } else if (detail::avx512_available()) {
            backend = Backend::avx512;
        } else {
            backend = Backend::portable_cpu;
        }
    }
    if ((backend == Backend::metal && !detail::metal_available()) ||
        (backend == Backend::cuda && !detail::cuda_available()) ||
        (backend == Backend::avx512 && !detail::avx512_available())) {
        throw std::runtime_error(
            std::string(backend_name(backend)) + " backend is unavailable");
    }
    if (backend == Backend::metal) {
        return detail::compress_compact_metal(
            permutation, options.format);
    }
    if (backend == Backend::cuda) {
        return detail::compress_compact_cuda(
            permutation, options.format);
    }
    if (backend == Backend::avx512) {
        return detail::compress_compact_avx512(
            permutation, options.threads, options.format);
    }
    detail::set_last_device_seconds(0);
    return detail::compress_compact(
        permutation, options.threads, std::move(options.format));
}

void compress_file(
    const std::filesystem::path& input,
    const std::filesystem::path& output,
    InputValueWidth input_width,
    CompressOptions options) {
    std::ifstream stream(input, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open permutation input file");
    }
    const auto length = stream.tellg();
    const std::size_t value_bytes = static_cast<std::size_t>(input_width);
    if ((value_bytes != 4 && value_bytes != 8) ||
        length <= 0 ||
        static_cast<std::uint64_t>(length) % value_bytes != 0 ||
        static_cast<std::uint64_t>(length) >
            std::numeric_limits<std::size_t>::max() ||
        static_cast<std::uint64_t>(length) >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::streamsize>::max())) {
        throw std::invalid_argument(
            "permutation input has an invalid element width or size");
    }
    stream.seekg(0);
    std::vector<std::byte> raw(static_cast<std::size_t>(length));
    stream.read(reinterpret_cast<char*>(raw.data()), length);
    if (!stream) {
        throw std::runtime_error("failed to read permutation input file");
    }
    std::vector<std::byte> encoded;
    if (input_width == InputValueWidth::bits32) {
        std::vector<std::uint32_t> permutation(raw.size() / 4);
        for (std::size_t i = 0; i < permutation.size(); ++i) {
            permutation[i] =
                detail::load_u32(std::span(raw).subspan(i * 4, 4));
        }
        encoded = compress(permutation, options);
    } else {
        std::vector<std::uint64_t> permutation(raw.size() / 8);
        for (std::size_t i = 0; i < permutation.size(); ++i) {
            permutation[i] =
                detail::load_u64(std::span(raw).subspan(i * 8, 8));
        }
        encoded = compress(permutation, options);
    }
    std::ofstream out(output, std::ios::binary | std::ios::trunc);
    if (encoded.size() > static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max())) {
        throw std::length_error(
            "compressed permutation is too large to write");
    }
    out.write(reinterpret_cast<const char*>(encoded.data()),
        static_cast<std::streamsize>(encoded.size()));
    if (!out) {
        throw std::runtime_error("failed to write compressed permutation file");
    }
}

bool backend_available(Backend backend) noexcept {
    switch (backend) {
    case Backend::automatic:
    case Backend::portable_cpu:
        return true;
    case Backend::metal:
        return detail::metal_available();
    case Backend::cuda:
        return detail::cuda_available();
    case Backend::avx512:
        return detail::avx512_available();
    }
    return false;
}

const char* backend_name(Backend backend) noexcept {
    switch (backend) {
    case Backend::automatic: return "automatic";
    case Backend::portable_cpu: return "portable-cpu";
    case Backend::metal: return "metal";
    case Backend::cuda: return "cuda";
    case Backend::avx512: return "avx512";
    }
    return "unknown";
}

CompressedPermutationView::CompressedPermutationView(std::span<const std::byte> bytes)
    : bytes_(bytes) {
    const detail::Layout layout = detail::parse_layout(bytes);
    size_ = layout.size;
    network_size_ = layout.network_size;
    middle_group_log2_ = layout.middle_group_log2;
    middle_placement_ = layout.middle_placement;
    input_clusters_ = layout.input_clusters;
    output_clusters_ = layout.output_clusters;
    input_offset_ = layout.input_offset;
    output_offset_ = layout.output_offset;
    middle_offset_ = layout.middle_offset;
}

std::uint64_t CompressedPermutationView::forward(std::uint64_t input) const {
    const detail::LookupLayout layout{size_, middle_group_log2_,
        middle_placement_, input_clusters_, output_clusters_,
        input_offset_, output_offset_, middle_offset_};
    return detail::forward_lookup(layout, input,
        [this](std::uint64_t offset) { return bytes_[offset]; });
}

std::uint64_t CompressedPermutationView::inverse(std::uint64_t output) const {
    const detail::LookupLayout layout{size_, middle_group_log2_,
        middle_placement_, input_clusters_, output_clusters_,
        input_offset_, output_offset_, middle_offset_};
    return detail::inverse_lookup(layout, output,
        [this](std::uint64_t offset) { return bytes_[offset]; });
}

std::uint64_t CompressedPermutationView::size() const noexcept { return size_; }
std::uint64_t CompressedPermutationView::network_size() const noexcept {
    return network_size_;
}
std::uint64_t CompressedPermutationView::encoded_size() const noexcept {
    return bytes_.size();
}
std::span<const std::byte> CompressedPermutationView::bytes() const noexcept {
    return bytes_;
}

struct CompressedPermutationFile::Impl {
    static constexpr std::size_t page_size = 4096;

    mutable std::ifstream stream;
    std::uint64_t file_size = 0;
    detail::Layout layout{};
    std::size_t page_limit;
    mutable std::mutex mutex;
    mutable std::uint64_t tick = 0;

    struct Page {
        std::array<std::byte, page_size> bytes{};
        std::size_t valid = 0;
        std::uint64_t last_used = 0;
    };
    mutable std::unordered_map<std::uint64_t, Page> pages;

    Impl(const std::filesystem::path& path, std::size_t cache_pages)
        : stream(path, std::ios::binary),
          page_limit(std::max<std::size_t>(1, cache_pages)) {
        if (!stream) {
            throw std::runtime_error("cannot open compressed permutation");
        }
        stream.seekg(0, std::ios::end);
        const auto length = stream.tellg();
        if (length < 0) {
            throw std::runtime_error("cannot determine compressed permutation size");
        }
        file_size = static_cast<std::uint64_t>(length);
        std::array<std::byte, detail::header_size> fixed_header{};
        read_exact(0, fixed_header);
        const std::size_t encoded_header_size = detail::header_size +
            std::to_integer<std::uint8_t>(fixed_header[18]) +
            std::to_integer<std::uint8_t>(fixed_header[19]);
        if (encoded_header_size > detail::header_size + 64 ||
            encoded_header_size > file_size) {
            throw std::invalid_argument(
                "unsupported compressed permutation header size");
        }
        std::vector<std::byte> header(encoded_header_size);
        std::copy(fixed_header.begin(), fixed_header.end(), header.begin());
        if (encoded_header_size > detail::header_size) {
            read_exact(detail::header_size,
                std::span(header).subspan(detail::header_size));
        }
        layout = detail::parse_header_for_file(header, file_size);
        detail::validate_padding(layout, [&](std::uint64_t offset) {
            std::array<std::byte, 1> byte{};
            read_exact(offset, byte);
            return byte[0];
        });
    }

    void read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
        if (offset > static_cast<std::uint64_t>(
                std::numeric_limits<std::streamoff>::max()) ||
            destination.size() > static_cast<std::uint64_t>(
                std::numeric_limits<std::streamsize>::max())) {
            throw std::length_error(
                "compressed permutation file offset is unsupported");
        }
        stream.clear();
        stream.seekg(static_cast<std::streamoff>(offset));
        stream.read(reinterpret_cast<char*>(destination.data()),
            static_cast<std::streamsize>(destination.size()));
        if (!stream) {
            throw std::runtime_error("failed to read compressed permutation");
        }
    }

    const Page& page(std::uint64_t index) const {
        auto found = pages.find(index);
        if (found != pages.end()) {
            found->second.last_used = ++tick;
            return found->second;
        }
        if (pages.size() >= page_limit) {
            auto oldest = std::min_element(pages.begin(), pages.end(),
                [](const auto& left, const auto& right) {
                    return left.second.last_used < right.second.last_used;
                });
            pages.erase(oldest);
        }
        Page loaded;
        const std::uint64_t offset = index * page_size;
        loaded.valid = static_cast<std::size_t>(
            std::min<std::uint64_t>(page_size, file_size - offset));
        read_exact(offset, std::span(loaded.bytes).first(loaded.valid));
        loaded.last_used = ++tick;
        return pages.emplace(index, std::move(loaded)).first->second;
    }

    void read_cached(std::uint64_t offset, std::span<std::byte> destination) const {
        if (offset > file_size || destination.size() > file_size - offset) {
            throw std::runtime_error("compressed permutation read is out of bounds");
        }
        std::lock_guard lock(mutex);
        std::size_t done = 0;
        while (done < destination.size()) {
            const std::uint64_t absolute = offset + done;
            const std::uint64_t page_index = absolute / page_size;
            const std::size_t within = absolute % page_size;
            const Page& source = page(page_index);
            const std::size_t count =
                std::min(destination.size() - done, source.valid - within);
            std::copy_n(source.bytes.begin() +
                    static_cast<std::ptrdiff_t>(within),
                count, destination.begin() + static_cast<std::ptrdiff_t>(done));
            done += count;
        }
    }

    std::uint32_t read32(std::uint64_t offset) const {
        std::array<std::byte, 4> bytes{};
        read_cached(offset, bytes);
        return detail::load_u32(bytes);
    }

    detail::U128 read128(std::uint64_t offset) const {
        std::array<std::byte, 16> bytes{};
        read_cached(offset, bytes);
        return {detail::load_u64(std::span(bytes).first(8)),
            detail::load_u64(std::span(bytes).subspan(8, 8))};
    }

    std::byte read_byte(std::uint64_t offset) const {
        std::array<std::byte, 1> bytes{};
        read_cached(offset, bytes);
        return bytes[0];
    }
};

CompressedPermutationFile::CompressedPermutationFile(
    const std::filesystem::path& path,
    std::size_t cache_pages)
    : impl_(std::make_unique<Impl>(path, cache_pages)) {}

CompressedPermutationFile::~CompressedPermutationFile() = default;
CompressedPermutationFile::CompressedPermutationFile(
    CompressedPermutationFile&&) noexcept = default;
CompressedPermutationFile& CompressedPermutationFile::operator=(
    CompressedPermutationFile&&) noexcept = default;

std::uint64_t CompressedPermutationFile::forward(std::uint64_t input) const {
    return detail::forward_lookup(detail::lookup_layout(impl_->layout), input,
        [this](std::uint64_t offset) { return impl_->read_byte(offset); });
}

std::uint64_t CompressedPermutationFile::inverse(std::uint64_t output) const {
    return detail::inverse_lookup(detail::lookup_layout(impl_->layout), output,
        [this](std::uint64_t offset) { return impl_->read_byte(offset); });
}

std::uint64_t CompressedPermutationFile::size() const noexcept {
    return impl_->layout.size;
}
std::uint64_t CompressedPermutationFile::network_size() const noexcept {
    return impl_->layout.network_size;
}
std::uint64_t CompressedPermutationFile::encoded_size() const noexcept {
    return impl_->file_size;
}

} // namespace benes
