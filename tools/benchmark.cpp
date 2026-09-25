// SPDX-License-Identifier: Apache-2.0
#include <benes/benes.hpp>

#include "internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

std::vector<std::uint32_t> make_permutation(
    std::size_t size,
    std::uint64_t seed) {
    std::vector<std::uint32_t> permutation(size);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::mt19937_64 random(seed);
    std::shuffle(permutation.begin(), permutation.end(), random);
    return permutation;
}

benes::Backend parse_backend(const std::string& name) {
    if (name == "cpu") return benes::Backend::portable_cpu;
    if (name == "metal") return benes::Backend::metal;
    if (name == "cuda") return benes::Backend::cuda;
    if (name == "avx512") return benes::Backend::avx512;
    if (name == "auto") return benes::Backend::automatic;
    throw std::invalid_argument("unknown backend: " + name);
}

void run_case(
    std::size_t size,
    benes::Backend backend,
    std::size_t threads,
    bool use_64_bit_values,
    std::size_t repetitions = 1) {
    const auto permutation = make_permutation(size, 0x5eed0000ULL + size);
    const std::vector<std::uint64_t> wide_permutation(
        permutation.begin(), permutation.end());
    struct Measurement {
        double wall;
        double device;
    };
    std::vector<Measurement> measurements;
    std::vector<std::byte> bytes;
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
        const auto start = std::chrono::steady_clock::now();
        bytes = use_64_bit_values
            ? benes::compress(wide_permutation, {backend, threads})
            : benes::compress(permutation, {backend, threads});
        const auto finish = std::chrono::steady_clock::now();
        measurements.push_back({
            std::chrono::duration<double>(finish - start).count(),
            benes::detail::last_device_seconds()});
    }
    std::sort(measurements.begin(), measurements.end(),
        [](const Measurement& left, const Measurement& right) {
            return left.wall < right.wall;
        });
    const Measurement measurement = measurements[measurements.size() / 2];
    const benes::CompressedPermutationView view(bytes);
    for (std::uint32_t input = 0; input < permutation.size(); ++input) {
        if (view.forward(input) != permutation[input] ||
            view.inverse(permutation[input]) != input) {
            throw std::runtime_error("lookup verification failed");
        }
    }
    const double seconds = measurement.wall;
    const double packed_bytes =
        std::ceil(size * std::ceil(std::log2(std::max<std::size_t>(2, size))) / 8.0);
    const double minimum_bytes = std::ceil(std::lgamma(size + 1) / std::log(2.0) / 8.0);
    const std::size_t encoded_header_size = benes::detail::header_size +
        std::to_integer<std::uint8_t>(bytes[18]) +
        std::to_integer<std::uint8_t>(bytes[19]);
    std::cout << size << ',' << view.network_size() << ',' << threads << ','
              << std::fixed << std::setprecision(6) << seconds << ','
              << measurement.device << ','
              << seconds - measurement.device << ','
              << std::setprecision(2) << size / seconds / 1e6 << ','
              << bytes.size() << ','
              << bytes.size() - encoded_header_size << ','
              << size * (use_64_bit_values
                      ? sizeof(std::uint64_t) : sizeof(std::uint32_t)) << ','
              << static_cast<std::uint64_t>(packed_bytes) << ','
              << static_cast<std::uint64_t>(minimum_bytes) << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        benes::Backend backend = benes::Backend::portable_cpu;
        bool use_64_bit_values = false;
        if (argc > 1) {
            std::string backend_name = argv[1];
            if (backend_name.ends_with("64")) {
                use_64_bit_values = true;
                backend_name.resize(backend_name.size() - 2);
            }
            backend = parse_backend(backend_name);
        }
        if (!benes::backend_available(backend)) {
            throw std::runtime_error(
                std::string("backend unavailable: ") + benes::backend_name(backend));
        }
        const std::size_t maximum_threads =
            std::max(1U, std::thread::hardware_concurrency());
        const auto warmup = make_permutation(32, 1);
        if (use_64_bit_values) {
            const std::vector<std::uint64_t> wide_warmup(
                warmup.begin(), warmup.end());
            (void)benes::compress(wide_warmup, {backend, 1});
        } else {
            (void)benes::compress(warmup, {backend, 1});
        }
        std::cout << "size,network,threads,seconds,device_seconds,host_seconds,"
                     "melements_per_second,"
                     "blob_bytes,payload_bytes,input_bytes,packed_bytes,"
                     "minimum_bytes\n";
        if (argc > 2) {
            const std::size_t size = std::stoull(argv[2]);
            const std::size_t threads =
                argc > 3 ? std::stoull(argv[3]) : maximum_threads;
            const std::size_t repetitions =
                argc > 4 ? std::stoull(argv[4]) : 3;
            run_case(size, backend, threads, use_64_bit_values, repetitions);
            return 0;
        }
        for (const std::size_t size :
            {31U, 32U, 33U, 63U, 64U, 65U, 127U, 128U, 129U,
                1024U, 4096U, 16384U, 65536U}) {
            run_case(size, backend, 1, use_64_bit_values);
            if (size >= 4096) {
                for (std::size_t threads = 2; threads <= maximum_threads;
                    threads *= 2) {
                    run_case(size, backend,
                        std::min(threads, maximum_threads), use_64_bit_values);
                }
                if ((maximum_threads & (maximum_threads - 1)) != 0) {
                    run_case(size, backend, maximum_threads,
                        use_64_bit_values);
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark failure: " << error.what() << '\n';
        return 1;
    }
}
