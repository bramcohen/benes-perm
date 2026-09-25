# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.25)
project(benes-perm VERSION 0.1.0 LANGUAGES CXX)

include(CheckLanguage)
include(CTest)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BENES_ENABLE_METAL "Build the Metal compression backend" ${APPLE})
option(BENES_ENABLE_CUDA "Build the CUDA compression backend" OFF)
option(BENES_ENABLE_AVX512 "Build the AVX-512 compression backend" ON)
option(BENES_BUILD_TOOLS "Build diagnostic and benchmark tools" ON)

add_library(benes
    src/backends_stub.cpp
    src/benes.cpp
    src/cpu/benes_cpu.cpp)
target_include_directories(benes
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_features(benes PUBLIC cxx_std_20)
add_library(benes::benes ALIAS benes)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(benes PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

if(BENES_ENABLE_METAL AND APPLE)
    enable_language(OBJCXX)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        src/metal/benes.metal)
    file(READ src/metal/benes.metal BENES_METAL_SOURCE)
    configure_file(src/metal/metal_source.hpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/generated/metal_source.hpp @ONLY)
    target_sources(benes PRIVATE src/metal/benes_metal.mm)
    target_compile_definitions(benes PRIVATE BENES_HAS_METAL=1)
    target_include_directories(benes PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
    set_source_files_properties(src/metal/benes_metal.mm
        PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(benes PRIVATE "-framework Foundation" "-framework Metal")
endif()

if(BENES_ENABLE_CUDA)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "BENES_ENABLE_CUDA is ON but no CUDA compiler was found")
    endif()
    enable_language(CUDA)
    target_sources(benes PRIVATE src/cuda/benes_cuda.cu)
    target_compile_definitions(benes PRIVATE BENES_HAS_CUDA=1)
    set_target_properties(benes PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
endif()

if(BENES_ENABLE_AVX512 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    target_sources(benes PRIVATE src/avx512/benes_avx512.cpp)
    target_compile_definitions(benes PRIVATE BENES_HAS_AVX512=1)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        set_source_files_properties(src/avx512/benes_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl")
    endif()
endif()

if(BUILD_TESTING)
    add_executable(benes-tests tests/test_benes.cpp)
    target_link_libraries(benes-tests PRIVATE benes)
    target_compile_definitions(benes-tests PRIVATE
        BENES_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests")
    add_test(NAME benes-tests COMMAND benes-tests)
endif()

if(BENES_BUILD_TOOLS)
    add_executable(benes-benchmark tools/benchmark.cpp)
    target_link_libraries(benes-benchmark PRIVATE benes)
    target_include_directories(benes-benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_executable(benes-generate-golden tools/generate_golden.cpp)
    target_link_libraries(benes-generate-golden PRIVATE benes)
    add_executable(benes-lookup-benchmark tools/lookup_benchmark.cpp)
    target_link_libraries(benes-lookup-benchmark PRIVATE benes)
endif()

install(TARGETS benes
    EXPORT benes-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT benes-targets
    FILE benes-targets.cmake
    NAMESPACE benes::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/benes)
configure_package_config_file(
    cmake/benes-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/benes-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/benes)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/benes-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/benes-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/benes-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/benes)
