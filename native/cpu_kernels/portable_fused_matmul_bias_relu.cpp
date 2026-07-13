// Phase P1C: portable_fused_matmul_bias_relu — a portable scalar C++
// executable implementing a small, FROZEN candidate family of runtime
// kernel contracts, all sharing one implementation framework:
//
//   backend:    cpu
//   op:         fused_matmul_bias_relu (hir.fused_matmul_bias_relu)
//   dtype:      f32
//   threads:    1 (fixed, always)
//
// Candidates differ ONLY in (block_m, block_n, block_k) -- the one intended
// implementation dimension for this phase. Every candidate uses the exact
// same algorithm (one-pass fused tile-local accumulator, bias-add + ReLU
// fused into the tile-store loop, remainder-tile handling via std::min),
// the exact same dtype, the exact same thread count (1), the exact same
// compiler flags, and the exact same correctness tolerance. No NEON, no
// AVX, no OpenMP, no pthreads, no auto-tuning -- compiler auto-vectorization
// only, exactly as in Phase P1B.
//
// This reuses the exact tiled-fused algorithm validated by
// ml-graph-compiler-runtime's Phase 1 / R1 CPU fused-schedule-discovery work
// (apps/run_cpu_fused_schedule_discovery.cpp: run_fused_tiled_matmul_bias_relu)
// as new, independent source in this repo, extended here (Phase P1C) from a
// single fixed tile shape to a static, immutable candidate table designed for
// the Raspberry Pi 5 Cortex-A76 cache hierarchy (L1D 64 KiB/core, L2 512 KiB
// private/core, L3 2 MiB shared, cache line 64 B) -- see
// DOC/result/... P1C report for the full per-candidate working-set rationale.
//
// Candidate identity is explicit and immutable: --kernel-id selects exactly
// one row of the static table below by exact string match; an unrecognized
// ID is a hard failure, never a silent substitution or fallback.
//
// Semantics: out[M,N] = max(0, A[M,K] . B[K,N] + bias[N]), row-major.
//
// I/O contract: A, B, bias, and the output are flat, native-endian,
// row-major float32 binary files with no header (the caller already knows
// M, N, K and validates them before invoking this executable). This
// executable performs its OWN shape/argument validation and refuses to
// silently guess or truncate on a bad input file size.
//
// Exit codes: 0 on success. Non-zero with a message on stderr for any
// argument, file, shape, or unknown-candidate error -- never a silent
// fallback or substitution.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kThreadCount = 1;

struct Candidate {
    const char* kernel_id;
    int block_m;
    int block_n;
    int block_k;
};

// Frozen P1C candidate table (see the P1C report for the working-set
// rationale behind every row). This table is the single source of truth
// for candidate identity in this executable -- adding a candidate means
// adding one row here, nowhere else in the algorithm.
constexpr Candidate kCandidates[] = {
    {"portable_fused_matmul_bias_relu_bm32_bn32_bk32",   32,  32,  32 },  // baseline (P1B)
    {"portable_fused_matmul_bias_relu_bm48_bn48_bk48",   48,  48,  48 },  // larger square
    {"portable_fused_matmul_bias_relu_bm64_bn64_bk64",   64,  64,  64 },  // near L1D boundary
    {"portable_fused_matmul_bias_relu_bm128_bn128_bk32", 128, 128, 32 },  // exceeds L1D, fits L2 (wide)
    {"portable_fused_matmul_bias_relu_bm128_bn32_bk32",  128, 32,  32 },  // M-dominant rectangular
    {"portable_fused_matmul_bias_relu_bm32_bn128_bk32",  32,  128, 32 },  // N-dominant rectangular
    {"portable_fused_matmul_bias_relu_bm32_bn32_bk128",  32,  32,  128},  // BLOCK_K variation
    {"portable_fused_matmul_bias_relu_bm64_bn64_bk128",  64,  64,  128},  // exceeds L1D, fits L2 (deep K)
};
constexpr int kNumCandidates = sizeof(kCandidates) / sizeof(kCandidates[0]);

const Candidate* find_candidate(const std::string& kernel_id) {
    for (const Candidate& c : kCandidates) {
        if (kernel_id == c.kernel_id) return &c;
    }
    return nullptr;
}

struct Args {
    long m = 0, n = 0, k = 0;
    std::string a_path, b_path, bias_path, out_path;
    std::string kernel_id;
    int repeats = 1;
};

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "error: " << message << "\n";
    std::exit(1);
}

std::vector<float> read_floats(const std::string& path, size_t expected_count) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        fail("cannot open input file: " + path);
    }
    in.seekg(0, std::ios::end);
    std::streampos size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size < 0 || static_cast<size_t>(size) != expected_count * sizeof(float)) {
        fail("input file '" + path + "' has " + std::to_string(static_cast<long long>(size)) +
             " bytes, expected exactly " + std::to_string(expected_count * sizeof(float)) +
             " bytes (" + std::to_string(expected_count) + " f32 elements) -- refusing to guess");
    }
    std::vector<float> data(expected_count);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    if (!in.good() && !in.eof()) {
        fail("short read on input file: " + path);
    }
    return data;
}

void write_floats(const std::string& path, const std::vector<float>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out.good()) {
        fail("cannot open output file for writing: " + path);
    }
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!out.good()) {
        fail("failed writing output file: " + path);
    }
}

// The one shared fused-tile algorithm for every candidate in the table.
// Only (block_m, block_n, block_k) vary across candidates; the algorithm,
// dtype, thread count (1), and semantics are identical for all of them.
// Handles M/N/K not evenly divisible by the tile size (remainder tiles) --
// same std::min-based handling as every candidate, no special-casing.
void run_fused_tiled_matmul_bias_relu(
    const float* a, const float* b, const float* bias, float* out,
    long M, long N, long K,
    int block_m, int block_n, int block_k
) {
    std::vector<float> tile_scratch(static_cast<size_t>(block_m) * static_cast<size_t>(block_n));

    for (long ii = 0; ii < M; ii += block_m) {
        const long i_end = std::min(ii + block_m, M);
        for (long jj = 0; jj < N; jj += block_n) {
            const long j_end = std::min(jj + block_n, N);
            const long tile_rows = i_end - ii;
            const long tile_cols = j_end - jj;
            const size_t scratch_size = static_cast<size_t>(tile_rows) * static_cast<size_t>(tile_cols);
            std::fill(tile_scratch.begin(), tile_scratch.begin() + static_cast<long>(scratch_size), 0.0f);

            for (long kk = 0; kk < K; kk += block_k) {
                const long k_end = std::min(kk + block_k, K);
                for (long i = ii; i < i_end; ++i) {
                    const float* a_row = a + static_cast<size_t>(i) * K;
                    float* scratch_row = tile_scratch.data() + static_cast<size_t>(i - ii) * tile_cols;
                    for (long k = kk; k < k_end; ++k) {
                        const float a_value = a_row[k];
                        const float* b_row = b + static_cast<size_t>(k) * N + jj;
                        for (long j = 0; j < tile_cols; ++j) {
                            scratch_row[j] += a_value * b_row[j];
                        }
                    }
                }
            }

            for (long i = ii; i < i_end; ++i) {
                const float* scratch_row = tile_scratch.data() + static_cast<size_t>(i - ii) * tile_cols;
                float* out_row = out + static_cast<size_t>(i) * N + jj;
                for (long j = 0; j < tile_cols; ++j) {
                    const float with_bias = scratch_row[j] + bias[jj + j];
                    out_row[j] = std::max(0.0f, with_bias);
                }
            }
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
        else if (flag == "--a") args.a_path = next("--a");
        else if (flag == "--b") args.b_path = next("--b");
        else if (flag == "--bias") args.bias_path = next("--bias");
        else if (flag == "--out") args.out_path = next("--out");
        else if (flag == "--repeats") args.repeats = std::stoi(next("--repeats"));
        else if (flag == "--kernel-id") args.kernel_id = next("--kernel-id");
        else if (flag == "--list-candidates") {
            for (const Candidate& c : kCandidates) {
                std::cout << c.kernel_id << " bm" << c.block_m << "_bn" << c.block_n
                          << "_bk" << c.block_k << "\n";
            }
            std::exit(0);
        } else {
            fail("unknown argument: " + flag);
        }
    }
    if (args.kernel_id.empty()) {
        fail("--kernel-id is required (no default candidate -- exact dispatch only)");
    }
    if (args.m <= 0 || args.n <= 0 || args.k <= 0) {
        fail("--m, --n, --k must all be positive integers");
    }
    if (args.a_path.empty() || args.b_path.empty() || args.bias_path.empty() || args.out_path.empty()) {
        fail("--a, --b, --bias, and --out are all required");
    }
    if (args.repeats <= 0) {
        fail("--repeats must be a positive integer");
    }
    return args;
}

} // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    const Candidate* candidate = find_candidate(args.kernel_id);
    if (!candidate) {
        std::ostringstream known;
        for (int i = 0; i < kNumCandidates; ++i) {
            if (i) known << ", ";
            known << kCandidates[i].kernel_id;
        }
        fail("unknown kernel_id requested: '" + args.kernel_id + "' (no silent substitution). "
             "Known candidates: " + known.str());
    }

    const std::vector<float> a = read_floats(args.a_path, static_cast<size_t>(args.m) * args.k);
    const std::vector<float> b = read_floats(args.b_path, static_cast<size_t>(args.k) * args.n);
    const std::vector<float> bias = read_floats(args.bias_path, static_cast<size_t>(args.n));
    std::vector<float> out(static_cast<size_t>(args.m) * args.n);

    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(args.repeats));
    for (int r = 0; r < args.repeats; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_fused_tiled_matmul_bias_relu(a.data(), b.data(), bias.data(), out.data(),
                                          args.m, args.n, args.k,
                                          candidate->block_m, candidate->block_n, candidate->block_k);
        auto t1 = std::chrono::high_resolution_clock::now();
        samples_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    write_floats(args.out_path, out);

    // Emit a small, self-contained JSON report on stdout (not the
    // ExecutionPlan schema -- a minimal per-invocation dispatch/timing
    // record for the calling adapter to parse).
    std::ostringstream json;
    json << "{\n";
    json << "  \"kernel_id\": \"" << candidate->kernel_id << "\",\n";
    json << "  \"backend\": \"cpu\",\n";
    json << "  \"dtype\": \"f32\",\n";
    json << "  \"m\": " << args.m << ", \"n\": " << args.n << ", \"k\": " << args.k << ",\n";
    json << "  \"block_m\": " << candidate->block_m << ", \"block_n\": " << candidate->block_n
         << ", \"block_k\": " << candidate->block_k << ", \"thread_count\": " << kThreadCount << ",\n";
    json << "  \"samples_ms\": [";
    for (size_t i = 0; i < samples_ms.size(); ++i) {
        if (i) json << ", ";
        json << samples_ms[i];
    }
    json << "],\n";
    json << "  \"exit_status\": 0\n";
    json << "}\n";
    std::cout << json.str();

    return 0;
}
