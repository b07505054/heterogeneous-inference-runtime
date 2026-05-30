cmake_minimum_required(VERSION 3.15)

project(edge_onnx_cpp_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(NOT DEFINED ONNXRUNTIME_DIR)
    message(FATAL_ERROR "Please set ONNXRUNTIME_DIR to your ONNX Runtime installation path")
endif()

include_directories(${ONNXRUNTIME_DIR}/include)

add_executable(edge_onnx_cpp main.cpp)

if(WIN32)
    target_link_libraries(edge_onnx_cpp ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)
elseif(APPLE)
    target_link_libraries(edge_onnx_cpp ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib)
    set_target_properties(edge_onnx_cpp PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
    )
else()
    target_link_libraries(edge_onnx_cpp ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
endif()

option(ENABLE_GOOGLE_BENCHMARK "Build Google Benchmark targets" OFF)

if(ENABLE_GOOGLE_BENCHMARK)
    find_package(benchmark REQUIRED)

    add_executable(benchmark_onnx_cpp benchmark_onnx_cpp.cpp)

    target_include_directories(benchmark_onnx_cpp PRIVATE ${ONNXRUNTIME_DIR}/include)

    if(WIN32)
        target_link_libraries(
            benchmark_onnx_cpp
            ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
            benchmark::benchmark
        )
    elseif(APPLE)
        target_link_libraries(
            benchmark_onnx_cpp
            ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib
            benchmark::benchmark
        )
        set_target_properties(benchmark_onnx_cpp PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
        )
    else()
        target_link_libraries(
            benchmark_onnx_cpp
            ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
            benchmark::benchmark
        )
    endif()
endif()