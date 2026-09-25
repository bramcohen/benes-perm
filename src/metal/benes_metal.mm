// SPDX-License-Identifier: Apache-2.0
#include "../internal.hpp"

#import <Metal/Metal.h>

#include "metal_source.hpp"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace benes::detail {
namespace {

void check_permutation(std::span<const std::uint32_t> permutation) {
    if (permutation.empty() ||
        permutation.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid permutation size");
    }
    std::vector<bool> seen(permutation.size());
    for (const std::uint32_t value : permutation) {
        if (value >= permutation.size() || seen[value]) {
            throw std::invalid_argument("input is not a permutation");
        }
        seen[value] = true;
    }
}

void check_permutation(std::span<const std::uint64_t> permutation) {
    if (permutation.empty()) {
        throw std::invalid_argument("invalid permutation size");
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

id<MTLComputePipelineState> make_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        throw std::runtime_error(
            "Metal function not found: " + std::string(name.UTF8String));
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        throw std::runtime_error(
            "Metal pipeline creation failed: " +
            std::string(error.localizedDescription.UTF8String));
    }
    return pipeline;
}

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> initialize;
    id<MTLComputePipelineState> merge;
    id<MTLComputePipelineState> output;
    id<MTLComputePipelineState> route;
    id<MTLComputePipelineState> small_layers;
    id<MTLComputePipelineState> encode;
    id<MTLComputePipelineState> compact_inverse;
    id<MTLComputePipelineState> compact_initialize;
    id<MTLComputePipelineState> compact_join;
    id<MTLComputePipelineState> compact_orient;
    id<MTLComputePipelineState> compact_route;
    id<MTLComputePipelineState> compact_pack;
    id<MTLComputePipelineState> compact_encode;
    id<MTLComputePipelineState> compact64_inverse;
    id<MTLComputePipelineState> compact64_32_initialize;
    id<MTLComputePipelineState> compact64_32_join;
    id<MTLComputePipelineState> compact64_32_orient;
    id<MTLComputePipelineState> compact64_32_route;
    id<MTLComputePipelineState> compact64_initialize;
    id<MTLComputePipelineState> compact64_join;
    id<MTLComputePipelineState> compact64_orient;
    id<MTLComputePipelineState> compact64_route;
    id<MTLComputePipelineState> compact64_pack;
    id<MTLComputePipelineState> compact64_encode;

    MetalContext() {
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            throw std::runtime_error("no Metal device is available");
        }
        NSError* error = nil;
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        NSString* source_text =
            [NSString stringWithUTF8String:benes_metal_source];
        id<MTLLibrary> library =
            [device newLibraryWithSource:source_text options:options error:&error];
        if (library == nil) {
            throw std::runtime_error(
                "Metal shader compilation failed: " +
                std::string(error.localizedDescription.UTF8String));
        }
        initialize = make_pipeline(device, library, @"initialize_parents");
        merge = make_pipeline(device, library, @"merge_cycles");
        output = make_pipeline(device, library, @"output_switches");
        route = make_pipeline(device, library, @"route_layer");
        small_layers = make_pipeline(device, library, @"compress_small_layers");
        encode = make_pipeline(device, library, @"encode_middles");
        compact_inverse = make_pipeline(device, library, @"compact_inverse");
        compact_initialize = make_pipeline(device, library, @"compact_initialize");
        compact_join = make_pipeline(device, library, @"compact_join_pairs");
        compact_orient = make_pipeline(device, library, @"compact_orient_odd");
        compact_route = make_pipeline(device, library, @"compact_route");
        compact_pack = make_pipeline(device, library, @"compact_pack_switches");
        compact_encode = make_pipeline(device, library, @"compact_encode_leaves");
        compact64_inverse = make_pipeline(device, library, @"compact64_inverse");
        compact64_32_initialize =
            make_pipeline(device, library, @"compact64_32_initialize");
        compact64_32_join =
            make_pipeline(device, library, @"compact64_32_join_pairs");
        compact64_32_orient =
            make_pipeline(device, library, @"compact64_32_orient_odd");
        compact64_32_route =
            make_pipeline(device, library, @"compact64_32_route");
        compact64_initialize =
            make_pipeline(device, library, @"compact64_initialize");
        compact64_join =
            make_pipeline(device, library, @"compact64_join_pairs");
        compact64_orient =
            make_pipeline(device, library, @"compact64_orient_odd");
        compact64_route = make_pipeline(device, library, @"compact64_route");
        compact64_pack =
            make_pipeline(device, library, @"compact64_pack_switches");
        compact64_encode =
            make_pipeline(device, library, @"compact64_encode_leaves");
        queue = [device newCommandQueue];
        if (queue == nil) {
            throw std::runtime_error("Metal command queue creation failed");
        }
    }
};

MetalContext& metal_context() {
    static MetalContext context;
    return context;
}

void dispatch(
    id<MTLCommandBuffer> command_buffer,
    id<MTLComputePipelineState> pipeline,
    std::uint32_t count,
    const std::function<void(id<MTLComputeCommandEncoder>)>& bind) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    bind(encoder);
    NSUInteger requested_width = 256;
    if (const char* environment = std::getenv("BENES_METAL_THREADGROUP_SIZE")) {
        requested_width = static_cast<NSUInteger>(std::stoul(environment));
    }
    const NSUInteger width = std::max<NSUInteger>(
        pipeline.threadExecutionWidth,
        std::min(requested_width, pipeline.maxTotalThreadsPerThreadgroup));
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
}

void dispatch_exact(
    id<MTLCommandBuffer> command_buffer,
    id<MTLComputePipelineState> pipeline,
    std::uint32_t count,
    NSUInteger width,
    const std::function<void(id<MTLComputeCommandEncoder>)>& bind) {
    if (width > pipeline.maxTotalThreadsPerThreadgroup ||
        count % width != 0) {
        throw std::runtime_error("invalid fixed Metal threadgroup size");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    bind(encoder);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
}

void dispatch_2d(
    id<MTLCommandBuffer> command_buffer,
    id<MTLComputePipelineState> pipeline,
    std::uint32_t width_count,
    std::uint32_t height_count,
    const std::function<void(id<MTLComputeCommandEncoder>)>& bind) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    bind(encoder);
    NSUInteger requested_width = 256;
    if (const char* environment = std::getenv("BENES_METAL_THREADGROUP_SIZE")) {
        requested_width = static_cast<NSUInteger>(std::stoul(environment));
    }
    const NSUInteger width = std::max<NSUInteger>(
        pipeline.threadExecutionWidth,
        std::min(requested_width, pipeline.maxTotalThreadsPerThreadgroup));
    [encoder dispatchThreads:MTLSizeMake(width_count, height_count, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
}

void dispatch_batched(
    id<MTLCommandBuffer> command_buffer,
    id<MTLComputePipelineState> pipeline,
    std::uint64_t count,
    std::uint32_t alignment,
    const std::function<void(
        id<MTLComputeCommandEncoder>, std::uint64_t, std::uint64_t)>& bind) {
    const std::uint64_t maximum_batch =
        (std::numeric_limits<std::uint32_t>::max() / alignment) * alignment;
    for (std::uint64_t base = 0; base < count;) {
        const std::uint64_t batch = std::min(maximum_batch, count - base);
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        bind(encoder, base, count);
        NSUInteger requested_width = 256;
        if (const char* environment =
                std::getenv("BENES_METAL_THREADGROUP_SIZE")) {
            requested_width = static_cast<NSUInteger>(
                std::stoul(environment));
        }
        const NSUInteger width = alignment == 1
            ? std::max<NSUInteger>(pipeline.threadExecutionWidth,
                  std::min(requested_width,
                      pipeline.maxTotalThreadsPerThreadgroup))
            : alignment;
        if (width > pipeline.maxTotalThreadsPerThreadgroup) {
            throw std::runtime_error("invalid fixed Metal threadgroup size");
        }
        [encoder dispatchThreads:MTLSizeMake(
                static_cast<NSUInteger>(batch), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        base += batch;
    }
}

} // namespace

bool metal_available() noexcept {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

std::vector<std::byte> compress_metal(
    std::span<const std::uint32_t> source) {
    check_permutation(source);
    @autoreleasepool {
        MetalContext& context = metal_context();
        id<MTLDevice> device = context.device;

        const std::uint64_t minimum =
            std::max<std::uint64_t>(32, source.size());
        const std::uint64_t network64 = std::bit_ceil(minimum);
        if (network64 > (std::uint64_t{1} << 31)) {
            throw std::invalid_argument("permutation is too large");
        }
        const auto network_size = static_cast<std::uint32_t>(network64);
        const auto log_size =
            static_cast<std::uint32_t>(std::bit_width(network_size) - 1);
        const std::uint32_t words_per_layer = network_size / 64;
        const std::size_t word_count =
            static_cast<std::size_t>(log_size - 5) * words_per_layer;

        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        id<MTLBuffer> permutation =
            [device newBufferWithLength:network_size * sizeof(std::uint32_t)
                options:shared];
        id<MTLBuffer> temporary =
            [device newBufferWithLength:network_size * sizeof(std::uint32_t)
                options:shared];
        id<MTLBuffer> parents =
            [device newBufferWithLength:network_size * sizeof(std::uint32_t)
                options:shared];
        id<MTLBuffer> input =
            [device newBufferWithLength:std::max<std::size_t>(4, word_count * 4)
                options:shared];
        id<MTLBuffer> output_words =
            [device newBufferWithLength:std::max<std::size_t>(4, word_count * 4)
                options:shared];
        id<MTLBuffer> middles =
            [device newBufferWithLength:(network_size / 32) * sizeof(U128)
                options:shared];
        if (permutation == nil || temporary == nil || parents == nil ||
            input == nil || output_words == nil || middles == nil) {
            throw std::runtime_error("Metal buffer allocation failed");
        }
        auto* permutation_data =
            static_cast<std::uint32_t*>(permutation.contents);
        std::copy(source.begin(), source.end(), permutation_data);
        std::iota(permutation_data + source.size(),
            permutation_data + network_size,
            static_cast<std::uint32_t>(source.size()));
        std::memset(input.contents, 0, input.length);
        std::memset(output_words.contents, 0, output_words.length);

        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        const std::uint32_t last_separate_layer =
            network_size >= 512 ? 9 : 5;
        for (std::uint32_t layer = log_size;
            layer > last_separate_layer; --layer) {
            const std::uint32_t block_size = std::uint32_t{1} << layer;
            const std::uint32_t layer_index = log_size - layer;
            dispatch(command_buffer, context.initialize, network_size,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:parents offset:0 atIndex:0];
                    [encoder setBytes:&block_size length:sizeof(block_size) atIndex:1];
                });
            dispatch(command_buffer, context.merge, network_size,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:permutation offset:0 atIndex:0];
                    [encoder setBuffer:parents offset:0 atIndex:1];
                    [encoder setBytes:&block_size length:sizeof(block_size) atIndex:2];
                });
            dispatch(command_buffer, context.output, network_size,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:parents offset:0 atIndex:0];
                    [encoder setBuffer:input offset:0 atIndex:1];
                    [encoder setBuffer:output_words offset:0 atIndex:2];
                    [encoder setBytes:&block_size length:sizeof(block_size) atIndex:3];
                    [encoder setBytes:&words_per_layer
                        length:sizeof(words_per_layer) atIndex:4];
                    [encoder setBytes:&layer_index
                        length:sizeof(layer_index) atIndex:5];
                });
            dispatch(command_buffer, context.route, network_size,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:permutation offset:0 atIndex:0];
                    [encoder setBuffer:temporary offset:0 atIndex:1];
                    [encoder setBuffer:input offset:0 atIndex:2];
                    [encoder setBuffer:output_words offset:0 atIndex:3];
                    [encoder setBytes:&block_size length:sizeof(block_size) atIndex:4];
                    [encoder setBytes:&words_per_layer
                        length:sizeof(words_per_layer) atIndex:5];
                    [encoder setBytes:&layer_index
                        length:sizeof(layer_index) atIndex:6];
                });
            std::swap(permutation, temporary);
        }
        if (network_size >= 512) {
            dispatch_exact(command_buffer, context.small_layers, network_size, 512,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:permutation offset:0 atIndex:0];
                    [encoder setBuffer:temporary offset:0 atIndex:1];
                    [encoder setBuffer:input offset:0 atIndex:2];
                    [encoder setBuffer:output_words offset:0 atIndex:3];
                    [encoder setBytes:&log_size length:sizeof(log_size) atIndex:4];
                    [encoder setBytes:&words_per_layer
                        length:sizeof(words_per_layer) atIndex:5];
                });
            std::swap(permutation, temporary);
        }
        dispatch(command_buffer, context.encode, network_size,
            [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:permutation offset:0 atIndex:0];
                [encoder setBuffer:middles offset:0 atIndex:1];
            });
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            throw std::runtime_error(
                "Metal compression failed: " +
                std::string(command_buffer.error.localizedDescription.UTF8String));
        }
        set_last_device_seconds(
            command_buffer.GPUEndTime - command_buffer.GPUStartTime);

        const auto input_span = std::span(
            static_cast<const std::uint32_t*>(input.contents), word_count);
        const auto output_span = std::span(
            static_cast<const std::uint32_t*>(output_words.contents), word_count);
        const auto middle_span = std::span(
            static_cast<const U128*>(middles.contents),
            static_cast<std::size_t>(network_size / 32));
        return make_blob(static_cast<std::uint32_t>(source.size()), network_size,
            input_span, output_span, middle_span);
    }
}

std::vector<std::byte> compress_compact_metal(
    std::span<const std::uint32_t> source,
    FormatOptions format) {
    check_permutation(source);
    if (source.size() > 0x7fffffffU) {
        throw std::invalid_argument("permutation is too large for Metal");
    }
    @autoreleasepool {
        MetalContext& context = metal_context();
        id<MTLDevice> device = context.device;
        const auto count = static_cast<std::uint32_t>(source.size());
        const FormatOptions normalized_format =
            normalize_format(count, std::move(format));
        const Layout layout = make_layout(count, normalized_format);
        const CompactTree tree =
            make_compact_tree(count, normalized_format);
        const auto& nodes = tree.nodes;
        const auto& leaves = tree.leaves;
        const auto& levels = tree.levels;
        const std::uint64_t input_buffer_bytes =
            ((tree.input_section_bits + 31) / 32) * 4;
        const std::uint64_t output_buffer_bytes =
            ((tree.output_section_bits + 31) / 32) * 4;
        const std::uint64_t middle_buffer_bytes =
            ((tree.middle_section_bits + 31) / 32) * 4;

        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        const std::size_t word_bytes = source.size() * sizeof(std::uint32_t);
        id<MTLBuffer> permutation = [device newBufferWithBytes:source.data()
            length:word_bytes options:shared];
        id<MTLBuffer> routed =
            [device newBufferWithLength:word_bytes options:shared];
        id<MTLBuffer> inverse =
            [device newBufferWithLength:word_bytes options:shared];
        id<MTLBuffer> parents =
            [device newBufferWithLength:word_bytes options:shared];
        id<MTLBuffer> flips =
            [device newBufferWithLength:word_bytes options:shared];
        id<MTLBuffer> colors =
            [device newBufferWithLength:word_bytes options:shared];
        id<MTLBuffer> node_buffer = [device newBufferWithLength:
            std::max<std::size_t>(1, nodes.size() * sizeof(CompactNode))
            options:shared];
        id<MTLBuffer> leaf_buffer = [device newBufferWithBytes:leaves.data()
            length:std::max<std::size_t>(1, leaves.size() * sizeof(CompactNode))
            options:shared];
        id<MTLBuffer> input_bits = [device newBufferWithLength:
            std::max<std::uint64_t>(4, input_buffer_bytes) options:shared];
        id<MTLBuffer> output_bits = [device newBufferWithLength:
            std::max<std::uint64_t>(4, output_buffer_bytes) options:shared];
        id<MTLBuffer> middle_bits = [device newBufferWithLength:
            std::max<std::uint64_t>(4, middle_buffer_bytes) options:shared];
        if (permutation == nil || routed == nil || inverse == nil ||
            parents == nil || flips == nil || colors == nil ||
            node_buffer == nil || leaf_buffer == nil || input_bits == nil ||
            output_bits == nil || middle_bits == nil) {
            throw std::runtime_error("Metal compact buffer allocation failed");
        }
        if (!nodes.empty()) {
            std::memcpy(node_buffer.contents, nodes.data(),
                nodes.size() * sizeof(CompactNode));
        }
        std::memset(input_bits.contents, 0, input_bits.length);
        std::memset(output_bits.contents, 0, output_bits.length);
        std::memset(middle_bits.contents, 0, middle_bits.length);
        id<MTLBuffer> middle_destination = middle_bits;
        if (normalized_format.middle_placement ==
            MiddlePlacement::embedded_input) {
            middle_destination = input_bits;
        } else if (
            normalized_format.middle_placement ==
            MiddlePlacement::embedded_output) {
            middle_destination = output_bits;
        }

        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        for (const CompactLevel& level : levels) {
            if (level.leaf_count != 0) {
                const std::uint32_t leaf_base = level.leaf_base;
                dispatch_exact(command_buffer, context.compact_encode,
                    level.leaf_count * 32, 32,
                    [&](id<MTLComputeCommandEncoder> encoder) {
                        [encoder setBuffer:permutation offset:0 atIndex:0];
                        [encoder setBuffer:middle_destination offset:0 atIndex:1];
                        [encoder setBuffer:leaf_buffer offset:0 atIndex:2];
                        [encoder setBytes:&leaf_base length:sizeof(leaf_base)
                            atIndex:3];
                    });
            }
            if (level.node_count == 0) {
                continue;
            }
            const std::uint32_t node_base = level.node_base;
            const auto bind_nodes = [&](id<MTLComputeCommandEncoder> encoder,
                                        NSUInteger index) {
                [encoder setBuffer:node_buffer offset:0 atIndex:index];
                [encoder setBytes:&node_base length:sizeof(node_base)
                    atIndex:index + 1];
            };
            dispatch_2d(command_buffer, context.compact_inverse,
                level.maximum_count, level.node_count,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:permutation offset:0 atIndex:0];
                    [encoder setBuffer:inverse offset:0 atIndex:1];
                    bind_nodes(encoder, 2);
                });
            dispatch_2d(command_buffer, context.compact_initialize,
                level.maximum_count, level.node_count,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:parents offset:0 atIndex:0];
                    [encoder setBuffer:flips offset:0 atIndex:1];
                    bind_nodes(encoder, 2);
                });
            dispatch_2d(command_buffer, context.compact_join,
                level.maximum_count / 2, level.node_count,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:inverse offset:0 atIndex:0];
                    [encoder setBuffer:parents offset:0 atIndex:1];
                    bind_nodes(encoder, 2);
                });
            dispatch(command_buffer, context.compact_orient,
                level.node_count,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:inverse offset:0 atIndex:0];
                    [encoder setBuffer:parents offset:0 atIndex:1];
                    [encoder setBuffer:flips offset:0 atIndex:2];
                    [encoder setBuffer:node_buffer offset:0 atIndex:3];
                    [encoder setBytes:&node_base length:sizeof(node_base)
                        atIndex:4];
                });
            dispatch_2d(command_buffer, context.compact_route,
                level.maximum_count, level.node_count,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:permutation offset:0 atIndex:0];
                    [encoder setBuffer:routed offset:0 atIndex:1];
                    [encoder setBuffer:parents offset:0 atIndex:2];
                    [encoder setBuffer:flips offset:0 atIndex:3];
                    [encoder setBuffer:colors offset:0 atIndex:4];
                    bind_nodes(encoder, 5);
                });
            dispatch_2d(command_buffer, context.compact_pack,
                level.maximum_pairs, level.node_count,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:inverse offset:0 atIndex:0];
                    [encoder setBuffer:colors offset:0 atIndex:1];
                    [encoder setBuffer:input_bits offset:0 atIndex:2];
                    [encoder setBuffer:output_bits offset:0 atIndex:3];
                    bind_nodes(encoder, 4);
                    const std::uint32_t leaf_size =
                        std::uint32_t{1} <<
                        normalized_format.middle_group_log2;
                    [encoder setBytes:&leaf_size length:sizeof(leaf_size)
                        atIndex:6];
                });
            std::swap(permutation, routed);
        }
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            throw std::runtime_error(
                "Metal compact compression failed: " +
                std::string(command_buffer.error.localizedDescription.UTF8String));
        }
        set_last_device_seconds(
            command_buffer.GPUEndTime - command_buffer.GPUStartTime);

        const std::uint64_t input_bytes =
            layout.output_offset - layout.input_offset;
        const std::uint64_t output_bytes =
            layout.middle_offset - layout.output_offset;
        const std::uint64_t middle_bytes =
            layout.end_offset - layout.middle_offset;
        std::vector<std::byte> bytes(
            static_cast<std::size_t>(layout.end_offset));
        write_header(bytes, layout);
        std::memcpy(bytes.data() + layout.input_offset, input_bits.contents,
            static_cast<std::size_t>(input_bytes));
        std::memcpy(bytes.data() + layout.output_offset, output_bits.contents,
            static_cast<std::size_t>(output_bytes));
        std::memcpy(bytes.data() + layout.middle_offset, middle_bits.contents,
            static_cast<std::size_t>(middle_bytes));
        return bytes;
    }
}

std::vector<std::byte> compress_compact_metal(
    std::span<const std::uint64_t> source,
    FormatOptions format) {
    check_permutation(source);
    if (source.size() > 0x7fffffffffffffffULL) {
        throw std::invalid_argument("permutation is too large for Metal");
    }
    @autoreleasepool {
        MetalContext& context = metal_context();
        id<MTLDevice> device = context.device;
        const std::uint64_t count = source.size();
        const FormatOptions normalized_format =
            normalize_format(count, std::move(format));
        const Layout layout = make_layout(count, normalized_format);
        const CompactTree64 tree =
            make_compact_tree64(count, normalized_format);
        const auto word_bytes = [](std::uint64_t bits) {
            return bits / 32 * 4 + (bits % 32 != 0 ? 4 : 0);
        };
        const std::uint64_t input_buffer_bytes =
            word_bytes(tree.input_section_bits);
        const std::uint64_t output_buffer_bytes =
            word_bytes(tree.output_section_bits);
        const std::uint64_t middle_buffer_bytes =
            word_bytes(tree.middle_section_bits);
        const auto multiply = [](std::uint64_t left, std::uint64_t right) {
            if (left != 0 &&
                right > std::numeric_limits<std::uint64_t>::max() / left) {
                throw std::overflow_error("Metal work count overflow");
            }
            return left * right;
        };
        const char* force_parent64_environment =
            std::getenv("BENES_METAL_FORCE_PARENT64");
        const bool force_parent64 =
            force_parent64_environment != nullptr &&
            std::string_view(force_parent64_environment) != "0";
        const bool has_parent32 = std::any_of(
            tree.levels.begin(), tree.levels.end(),
            [force_parent64](const CompactLevel64& level) {
                return level.node_count != 0 &&
                    !force_parent64 &&
                    level.maximum_count <= 0x7fffffffULL;
            });
        const bool has_parent64 = std::any_of(
            tree.levels.begin(), tree.levels.end(),
            [force_parent64](const CompactLevel64& level) {
                return level.node_count != 0 &&
                    (force_parent64 ||
                        level.maximum_count > 0x7fffffffULL);
            });

        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        const std::size_t value_bytes =
            checked_size_bytes(source.size(), sizeof(std::uint64_t));
        const std::size_t index32_bytes =
            checked_size_bytes(source.size(), sizeof(std::uint32_t));
        const std::size_t node_bytes =
            checked_size_bytes(tree.nodes.size(), sizeof(CompactNode64));
        const std::size_t leaf_bytes =
            checked_size_bytes(tree.leaves.size(), sizeof(CompactNode64));
        id<MTLBuffer> permutation = [device newBufferWithBytes:source.data()
            length:value_bytes options:shared];
        id<MTLBuffer> routed =
            [device newBufferWithLength:value_bytes options:shared];
        id<MTLBuffer> inverse =
            [device newBufferWithLength:value_bytes options:shared];
        id<MTLBuffer> parents32 = [device newBufferWithLength:
            has_parent32 ? index32_bytes : 4
            options:shared];
        id<MTLBuffer> parents64 = [device newBufferWithLength:
            has_parent64 ? value_bytes : 8 options:shared];
        id<MTLBuffer> locks = [device newBufferWithLength:
            has_parent64 ? index32_bytes : 4
            options:shared];
        id<MTLBuffer> flips = [device newBufferWithLength:
            index32_bytes options:shared];
        id<MTLBuffer> colors = [device newBufferWithLength:
            index32_bytes options:shared];
        id<MTLBuffer> node_buffer = [device newBufferWithLength:
            std::max<std::size_t>(1, node_bytes)
            options:shared];
        id<MTLBuffer> leaf_buffer = [device newBufferWithLength:
            std::max<std::size_t>(1, leaf_bytes)
            options:shared];
        id<MTLBuffer> input_bits = [device newBufferWithLength:
            std::max<std::uint64_t>(4, input_buffer_bytes) options:shared];
        id<MTLBuffer> output_bits = [device newBufferWithLength:
            std::max<std::uint64_t>(4, output_buffer_bytes) options:shared];
        id<MTLBuffer> middle_bits = [device newBufferWithLength:
            std::max<std::uint64_t>(4, middle_buffer_bytes) options:shared];
        if (permutation == nil || routed == nil || inverse == nil ||
            parents32 == nil || parents64 == nil || locks == nil ||
            flips == nil || colors == nil ||
            node_buffer == nil || leaf_buffer == nil || input_bits == nil ||
            output_bits == nil || middle_bits == nil) {
            throw std::runtime_error("Metal compact64 buffer allocation failed");
        }
        if (!tree.nodes.empty()) {
            std::memcpy(node_buffer.contents, tree.nodes.data(),
                node_bytes);
        }
        if (!tree.leaves.empty()) {
            std::memcpy(leaf_buffer.contents, tree.leaves.data(),
                leaf_bytes);
        }
        std::memset(input_bits.contents, 0, input_bits.length);
        std::memset(output_bits.contents, 0, output_bits.length);
        std::memset(middle_bits.contents, 0, middle_bits.length);
        id<MTLBuffer> middle_destination = middle_bits;
        if (normalized_format.middle_placement ==
            MiddlePlacement::embedded_input) {
            middle_destination = input_bits;
        } else if (normalized_format.middle_placement ==
            MiddlePlacement::embedded_output) {
            middle_destination = output_bits;
        }

        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        for (const CompactLevel64& level : tree.levels) {
            if (level.leaf_count != 0) {
                const std::uint64_t work = multiply(level.leaf_count, 32);
                dispatch_batched(command_buffer, context.compact64_encode,
                    work, 32,
                    [&](id<MTLComputeCommandEncoder> encoder,
                        std::uint64_t work_base,
                        std::uint64_t work_count) {
                        [encoder setBuffer:permutation offset:0 atIndex:0];
                        [encoder setBuffer:middle_destination offset:0 atIndex:1];
                        [encoder setBuffer:leaf_buffer offset:0 atIndex:2];
                        [encoder setBytes:&level.leaf_base
                            length:sizeof(level.leaf_base) atIndex:3];
                        [encoder setBytes:&work_base
                            length:sizeof(work_base) atIndex:4];
                        [encoder setBytes:&work_count
                            length:sizeof(work_count) atIndex:5];
                    });
            }
            if (level.node_count == 0) {
                continue;
            }
            const bool parent32 =
                !force_parent64 &&
                level.maximum_count <= 0x7fffffffULL;
            const std::uint64_t element_work =
                multiply(level.maximum_count, level.node_count);
            const auto bind_nodes =
                [&](id<MTLComputeCommandEncoder> encoder,
                    NSUInteger index,
                    std::uint64_t width,
                    std::uint64_t work_base,
                    std::uint64_t work_count) {
                    [encoder setBuffer:node_buffer offset:0 atIndex:index];
                    [encoder setBytes:&level.node_base
                        length:sizeof(level.node_base) atIndex:index + 1];
                    [encoder setBytes:&width
                        length:sizeof(width) atIndex:index + 2];
                    [encoder setBytes:&work_base
                        length:sizeof(work_base) atIndex:index + 3];
                    [encoder setBytes:&work_count
                        length:sizeof(work_count) atIndex:index + 4];
                };
            dispatch_batched(command_buffer, context.compact64_inverse,
                element_work, 1,
                [&](id<MTLComputeCommandEncoder> encoder,
                    std::uint64_t base, std::uint64_t total) {
                    [encoder setBuffer:permutation offset:0 atIndex:0];
                    [encoder setBuffer:inverse offset:0 atIndex:1];
                    bind_nodes(
                        encoder, 2, level.maximum_count, base, total);
                });
            if (parent32) {
                dispatch_batched(
                    command_buffer, context.compact64_32_initialize,
                    element_work, 1,
                    [&](id<MTLComputeCommandEncoder> encoder,
                        std::uint64_t base, std::uint64_t total) {
                        [encoder setBuffer:parents32 offset:0 atIndex:0];
                        [encoder setBuffer:flips offset:0 atIndex:1];
                        bind_nodes(
                            encoder, 2, level.maximum_count, base, total);
                    });
            } else {
                dispatch_batched(command_buffer, context.compact64_initialize,
                    element_work, 1,
                    [&](id<MTLComputeCommandEncoder> encoder,
                        std::uint64_t base, std::uint64_t total) {
                        [encoder setBuffer:parents64 offset:0 atIndex:0];
                        [encoder setBuffer:locks offset:0 atIndex:1];
                        [encoder setBuffer:flips offset:0 atIndex:2];
                        bind_nodes(
                            encoder, 3, level.maximum_count, base, total);
                    });
            }
            const std::uint64_t pair_work =
                multiply(level.maximum_pairs, level.node_count);
            if (parent32) {
                dispatch_batched(command_buffer, context.compact64_32_join,
                    pair_work, 1,
                    [&](id<MTLComputeCommandEncoder> encoder,
                        std::uint64_t base, std::uint64_t total) {
                        [encoder setBuffer:inverse offset:0 atIndex:0];
                        [encoder setBuffer:parents32 offset:0 atIndex:1];
                        bind_nodes(
                            encoder, 2, level.maximum_pairs, base, total);
                    });
                dispatch_batched(command_buffer, context.compact64_32_orient,
                    level.node_count, 1,
                    [&](id<MTLComputeCommandEncoder> encoder,
                        std::uint64_t base, std::uint64_t total) {
                        [encoder setBuffer:inverse offset:0 atIndex:0];
                        [encoder setBuffer:parents32 offset:0 atIndex:1];
                        [encoder setBuffer:flips offset:0 atIndex:2];
                        [encoder setBuffer:node_buffer offset:0 atIndex:3];
                        [encoder setBytes:&level.node_base
                            length:sizeof(level.node_base) atIndex:4];
                        [encoder setBytes:&base length:sizeof(base) atIndex:5];
                        [encoder setBytes:&total
                            length:sizeof(total) atIndex:6];
                    });
                dispatch_batched(command_buffer, context.compact64_32_route,
                    element_work, 1,
                    [&](id<MTLComputeCommandEncoder> encoder,
                        std::uint64_t base, std::uint64_t total) {
                        [encoder setBuffer:permutation offset:0 atIndex:0];
                        [encoder setBuffer:routed offset:0 atIndex:1];
                        [encoder setBuffer:parents32 offset:0 atIndex:2];
                        [encoder setBuffer:flips offset:0 atIndex:3];
                        [encoder setBuffer:colors offset:0 atIndex:4];
                        bind_nodes(
                            encoder, 5, level.maximum_count, base, total);
                    });
            } else {
                dispatch_batched(command_buffer, context.compact64_join,
                    pair_work, 1,
                    [&](id<MTLComputeCommandEncoder> encoder,
                        std::uint64_t base, std::uint64_t total) {
                        [encoder setBuffer:inverse offset:0 atIndex:0];
                        [encoder setBuffer:parents64 offset:0 atIndex:1];
                        [encoder setBuffer:locks offset:0 atIndex:2];
                        bind_nodes(
                            encoder, 3, level.maximum_pairs, base, total);
                    });
                dispatch_batched(command_buffer, context.compact64_orient,
                    level.node_count, 1,
                    [&](id<MTLComputeCommandEncoder> encoder,
                        std::uint64_t base, std::uint64_t total) {
                        [encoder setBuffer:inverse offset:0 atIndex:0];
                        [encoder setBuffer:parents64 offset:0 atIndex:1];
                        [encoder setBuffer:locks offset:0 atIndex:2];
                        [encoder setBuffer:flips offset:0 atIndex:3];
                        [encoder setBuffer:node_buffer offset:0 atIndex:4];
                        [encoder setBytes:&level.node_base
                            length:sizeof(level.node_base) atIndex:5];
                        [encoder setBytes:&base length:sizeof(base) atIndex:6];
                        [encoder setBytes:&total
                            length:sizeof(total) atIndex:7];
                    });
                dispatch_batched(command_buffer, context.compact64_route,
                    element_work, 1,
                    [&](id<MTLComputeCommandEncoder> encoder,
                        std::uint64_t base, std::uint64_t total) {
                        [encoder setBuffer:permutation offset:0 atIndex:0];
                        [encoder setBuffer:routed offset:0 atIndex:1];
                        [encoder setBuffer:parents64 offset:0 atIndex:2];
                        [encoder setBuffer:locks offset:0 atIndex:3];
                        [encoder setBuffer:flips offset:0 atIndex:4];
                        [encoder setBuffer:colors offset:0 atIndex:5];
                        bind_nodes(
                            encoder, 6, level.maximum_count, base, total);
                    });
            }
            dispatch_batched(command_buffer, context.compact64_pack,
                pair_work, 1,
                [&](id<MTLComputeCommandEncoder> encoder,
                    std::uint64_t base, std::uint64_t total) {
                    [encoder setBuffer:inverse offset:0 atIndex:0];
                    [encoder setBuffer:colors offset:0 atIndex:1];
                    [encoder setBuffer:input_bits offset:0 atIndex:2];
                    [encoder setBuffer:output_bits offset:0 atIndex:3];
                    [encoder setBuffer:node_buffer offset:0 atIndex:4];
                    [encoder setBytes:&level.node_base
                        length:sizeof(level.node_base) atIndex:5];
                    const std::uint32_t leaf_size =
                        std::uint32_t{1} <<
                        normalized_format.middle_group_log2;
                    [encoder setBytes:&leaf_size
                        length:sizeof(leaf_size) atIndex:6];
                    [encoder setBytes:&level.maximum_pairs
                        length:sizeof(level.maximum_pairs) atIndex:7];
                    [encoder setBytes:&base length:sizeof(base) atIndex:8];
                    [encoder setBytes:&total length:sizeof(total) atIndex:9];
                });
            std::swap(permutation, routed);
        }
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            throw std::runtime_error(
                "Metal compact64 compression failed: " +
                std::string(command_buffer.error.localizedDescription.UTF8String));
        }
        set_last_device_seconds(
            command_buffer.GPUEndTime - command_buffer.GPUStartTime);

        const std::size_t input_bytes = static_cast<std::size_t>(
            layout.output_offset - layout.input_offset);
        const std::size_t output_bytes = static_cast<std::size_t>(
            layout.middle_offset - layout.output_offset);
        const std::size_t middle_bytes = static_cast<std::size_t>(
            layout.end_offset - layout.middle_offset);
        std::vector<std::byte> bytes(
            static_cast<std::size_t>(layout.end_offset));
        write_header(bytes, layout);
        std::memcpy(bytes.data() + layout.input_offset, input_bits.contents,
            input_bytes);
        std::memcpy(bytes.data() + layout.output_offset, output_bits.contents,
            output_bytes);
        std::memcpy(bytes.data() + layout.middle_offset, middle_bits.contents,
            middle_bytes);
        return bytes;
    }
}

} // namespace benes::detail
