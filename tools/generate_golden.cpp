// SPDX-License-Identifier: Apache-2.0
#include <benes/benes.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

void write(
    const std::filesystem::path& directory,
    const std::string& name,
    std::span<const std::uint32_t> permutation) {
    const auto bytes = benes::compress(permutation,
        {.backend = benes::Backend::portable_cpu, .threads = 1});
    std::ofstream output(directory / name, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return 2;
    }
    const std::filesystem::path directory = argv[1];
    std::filesystem::create_directories(directory);

    std::vector<std::uint32_t> identity(32);
    std::iota(identity.begin(), identity.end(), 0);
    write(directory, "identity-32.benes", identity);

    std::vector<std::uint32_t> reverse(33);
    std::iota(reverse.rbegin(), reverse.rend(), 0);
    write(directory, "reverse-33.benes", reverse);

    std::vector<std::uint32_t> random(47);
    std::iota(random.begin(), random.end(), 0);
    std::mt19937_64 generator(0x474f4c44454eULL);
    std::shuffle(random.begin(), random.end(), generator);
    write(directory, "random-47.benes", random);
    return 0;
}
