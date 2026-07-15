#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr const char* kKernelId = "portable_fused_matmul_bias_relu_int8_symmetric_packed_b";
constexpr const char* kPackedLayout = "packed_b_transposed_nxk";
constexpr const char* kPackingScheme = "b_transposed_nxk_contiguous";

struct Args {
    long m = 0, n = 0, k = 0;
    std::string a_path, b_path, bias_path, out_path;
    std::string kernel_id, packed_layout, packing_scheme;
    double activation_scale = 0.0;
    double weight_scale = 0.0;
    int activation_zero_point = 0;
    int weight_zero_point = 0;
    int repeats = 1;
};

[[noreturn]] void fail(const std::string& msg) {
    std::cerr << "error: " << msg << "\n";
    std::exit(1);
}

template <typename T>
std::vector<T> read_exact(const std::string& path, size_t count, const char* dtype_name) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) fail("cannot open input file: " + path);
    in.seekg(0, std::ios::end);
    std::streampos size = in.tellg();
    in.seekg(0, std::ios::beg);
    const size_t expected = count * sizeof(T);
    if (size < 0 || static_cast<size_t>(size) != expected) {
        fail("input file '" + path + "' has " + std::to_string(static_cast<long long>(size)) +
             " bytes, expected exactly " + std::to_string(expected) + " bytes (" +
             std::to_string(count) + " " + dtype_name + " elements)");
    }
    std::vector<T> data(count);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(expected));
    if (!in.good() && !in.eof()) fail("short read on input file: " + path);
    return data;
}

void write_floats(const std::string& path, const std::vector<float>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out.good()) fail("cannot open output file for writing: " + path);
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!out.good()) fail("failed writing output file: " + path);
}

void run_int8_matmul_bias_relu_packed_b(
    const int8_t* a, const int8_t* packed_b, const float* bias, float* out,
    long M, long N, long K, float activation_scale, float weight_scale,
    int activation_zero_point, int weight_zero_point) {
    if (activation_zero_point == 0 && weight_zero_point == 0) {
        for (long i = 0; i < M; ++i) {
            const int8_t* a_row = a + i * K;
            for (long j = 0; j < N; ++j) {
                const int8_t* b_col_packed = packed_b + j * K;
                int32_t acc = 0;
                for (long kk = 0; kk < K; ++kk) {
                    acc += static_cast<int32_t>(a_row[kk]) * static_cast<int32_t>(b_col_packed[kk]);
                }
                const float dequant = static_cast<float>(acc) * activation_scale * weight_scale;
                out[i * N + j] = std::max(0.0f, dequant + bias[j]);
            }
        }
        return;
    }
    for (long i = 0; i < M; ++i) {
        const int8_t* a_row = a + i * K;
        for (long j = 0; j < N; ++j) {
            const int8_t* b_col_packed = packed_b + j * K;
            int32_t acc = 0;
            for (long kk = 0; kk < K; ++kk) {
                const int32_t av = static_cast<int32_t>(a_row[kk]) - activation_zero_point;
                const int32_t bv = static_cast<int32_t>(b_col_packed[kk]) - weight_zero_point;
                acc += av * bv; // real int8 x int8 -> int32 arithmetic over compiler-packed B.
            }
            const float dequant = static_cast<float>(acc) * activation_scale * weight_scale;
            out[i * N + j] = std::max(0.0f, dequant + bias[j]);
        }
    }
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) fail(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (flag == "--m") args.m = std::stol(next("--m"));
        else if (flag == "--n") args.n = std::stol(next("--n"));
        else if (flag == "--k") args.k = std::stol(next("--k"));
        else if (flag == "--a-int8") args.a_path = next("--a-int8");
        else if (flag == "--b-packed-int8") args.b_path = next("--b-packed-int8");
        else if (flag == "--bias") args.bias_path = next("--bias");
        else if (flag == "--out") args.out_path = next("--out");
        else if (flag == "--kernel-id") args.kernel_id = next("--kernel-id");
        else if (flag == "--activation-scale") args.activation_scale = std::stod(next("--activation-scale"));
        else if (flag == "--weight-scale") args.weight_scale = std::stod(next("--weight-scale"));
        else if (flag == "--activation-zero-point") args.activation_zero_point = std::stoi(next("--activation-zero-point"));
        else if (flag == "--weight-zero-point") args.weight_zero_point = std::stoi(next("--weight-zero-point"));
        else if (flag == "--packed-layout") args.packed_layout = next("--packed-layout");
        else if (flag == "--packing-scheme") args.packing_scheme = next("--packing-scheme");
        else if (flag == "--repeats") args.repeats = std::stoi(next("--repeats"));
        else fail("unknown argument: " + flag);
    }
    if (args.kernel_id != kKernelId) fail("unknown kernel_id requested: '" + args.kernel_id + "' (no silent substitution)");
    if (args.packed_layout != kPackedLayout) fail("packed_layout mismatch: '" + args.packed_layout + "'");
    if (args.packing_scheme != kPackingScheme) fail("packing_scheme mismatch: '" + args.packing_scheme + "'");
    if (args.m <= 0 || args.n <= 0 || args.k <= 0) fail("--m, --n, --k must all be positive integers");
    if (args.a_path.empty() || args.b_path.empty() || args.bias_path.empty() || args.out_path.empty()) fail("--a-int8, --b-packed-int8, --bias, and --out are required");
    if (args.activation_scale <= 0.0 || args.weight_scale <= 0.0) fail("scales must be positive");
    if (args.activation_zero_point != 0 || args.weight_zero_point != 0) fail("Slice 3B supports zero_point=0 only");
    if (args.repeats <= 0) fail("--repeats must be positive");
    return args;
}
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    const std::vector<int8_t> a = read_exact<int8_t>(args.a_path, static_cast<size_t>(args.m) * args.k, "i8");
    const std::vector<int8_t> packed_b = read_exact<int8_t>(args.b_path, static_cast<size_t>(args.n) * args.k, "packed_i8");
    const std::vector<float> bias = read_exact<float>(args.bias_path, static_cast<size_t>(args.n), "f32");
    std::vector<float> out(static_cast<size_t>(args.m) * args.n);
    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(args.repeats));
    for (int r = 0; r < args.repeats; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_int8_matmul_bias_relu_packed_b(a.data(), packed_b.data(), bias.data(), out.data(), args.m, args.n, args.k,
                                           static_cast<float>(args.activation_scale), static_cast<float>(args.weight_scale),
                                           args.activation_zero_point, args.weight_zero_point);
        auto t1 = std::chrono::high_resolution_clock::now();
        samples_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    write_floats(args.out_path, out);
    std::ostringstream json;
    json << "{\n";
    json << "  \"kernel_id\": \"" << kKernelId << "\",\n";
    json << "  \"backend\": \"cpu\",\n";
    json << "  \"dtype\": \"int8_static_symmetric\",\n";
    json << "  \"accumulator_dtype\": \"int32\",\n";
    json << "  \"output_dtype\": \"f32\",\n";
    json << "  \"arithmetic\": \"int8_times_int8_accumulate_int32_dequantize_fp32_packed_b\",\n";
    json << "  \"packed_layout\": \"" << kPackedLayout << "\",\n";
    json << "  \"packing_scheme\": \"" << kPackingScheme << "\",\n";
    json << "  \"runtime_packed_weight_transform\": false,\n";
    json << "  \"m\": " << args.m << ", \"n\": " << args.n << ", \"k\": " << args.k << ",\n";
    json << "  \"activation_scale\": " << args.activation_scale << ",\n";
    json << "  \"weight_scale\": " << args.weight_scale << ",\n";
    json << "  \"activation_zero_point\": " << args.activation_zero_point << ",\n";
    json << "  \"weight_zero_point\": " << args.weight_zero_point << ",\n";
    json << "  \"thread_count\": 1,\n";
    json << "  \"partition_axis\": \"none\",\n";
    json << "  \"partition_strategy\": \"serial\",\n";
    json << "  \"samples_ms\": [";
    for (size_t i = 0; i < samples_ms.size(); ++i) { if (i) json << ", "; json << samples_ms[i]; }
    json << "],\n  \"exit_status\": 0\n}\n";
    std::cout << json.str();
    return 0;
}
