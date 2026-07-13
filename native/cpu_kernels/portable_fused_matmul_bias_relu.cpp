// Phase P1D: portable_fused_matmul_bias_relu — extends the Phase P1C
// portable scalar tile-candidate family with a SEPARATE, orthogonal
// decomposition dimension: thread count and output partitioning.
//
//   backend:    cpu
//   op:         fused_matmul_bias_relu (hir.fused_matmul_bias_relu)
//   dtype:      f32
//   tile:       fixed per --kernel-id (see kCandidates), unchanged from P1C
//   threads:    1, 2, or 4 (this invocation), --partition-axis/--partition-strategy
//
// Kernel/tile identity (--kernel-id) and thread schedule
// (--thread-count/--partition-axis/--partition-strategy) are two SEPARATE,
// independently-validated decisions, matching the compiler-side schema
// split (kernel_selection vs. thread_schedule, Phase P1D). Every thread
// count/partition combination uses the EXACT SAME tiled algorithm as P1B/P1C
// (one-pass fused tile-local accumulator, bias-add + ReLU fused into the
// tile-store loop, remainder-tile handling via std::min), the same dtype,
// the same compiler flags, the same correctness tolerance -- only the
// output row/column range each thread is responsible for differs. No NEON,
// no AVX, no OpenMP, no hidden thread pool, no work stealing, no nested
// parallelism, no atomics in the hot numerical path (every thread owns a
// strictly disjoint output region; synchronization is only the final
// std::thread::join(), never touched during compute).
//
// Threading contract (thread_schedule_contract_v1, mirrors the compiler
// schema exactly):
//   thread_count=1            requires partition_axis=none,   partition_strategy=serial
//   thread_count=2|4          requires partition_axis=m|n,    partition_strategy=contiguous_chunks
// Invalid combinations are a hard failure, never silently reinterpreted.
//
// Partitioning: split M divides output ROW ranges across threads (ceiling
// division: thread t owns rows [t*chunk, min((t+1)*chunk, M)), chunk =
// ceil(M/thread_count) -- some trailing threads legitimately own an empty
// range when thread_count exceeds M, handled as a no-op, never a crash).
// Split N divides output COLUMN ranges the same way. Every output element
// is computed by exactly one thread, exactly once.
//
// I/O contract, exit codes, and the kKernelId candidate table are otherwise
// unchanged from Phase P1C.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Candidate {
    const char* kernel_id;
    int block_m;
    int block_n;
    int block_k;
};

// Frozen P1C candidate table (see the P1C report for the working-set
// rationale behind every row). Unchanged in P1D -- tile identity stays
// fixed at bm32_bn128_bk32 for the thread experiment; this table exists
// only so the executable can still serve every P1B/P1C tile candidate too.
constexpr Candidate kCandidates[] = {
    {"portable_fused_matmul_bias_relu_bm32_bn32_bk32",   32,  32,  32 },
    {"portable_fused_matmul_bias_relu_bm48_bn48_bk48",   48,  48,  48 },
    {"portable_fused_matmul_bias_relu_bm64_bn64_bk64",   64,  64,  64 },
    {"portable_fused_matmul_bias_relu_bm128_bn128_bk32", 128, 128, 32 },
    {"portable_fused_matmul_bias_relu_bm128_bn32_bk32",  128, 32,  32 },
    {"portable_fused_matmul_bias_relu_bm32_bn128_bk32",  32,  128, 32 },
    {"portable_fused_matmul_bias_relu_bm32_bn32_bk128",  32,  32,  128},
    {"portable_fused_matmul_bias_relu_bm64_bn64_bk128",  64,  64,  128},
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
    int thread_count = 1;
    std::string partition_axis = "none";
    std::string partition_strategy = "serial";
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

// The one shared fused-tile algorithm, generalized (Phase P1D) to operate
// on an arbitrary disjoint [m_start,m_end) x [n_start,n_end) output
// sub-region instead of always the full [0,M) x [0,N) range. Passing
// m_start=0, m_end=M, n_start=0, n_end=N reproduces the exact P1B/P1C
// single-thread behavior byte-for-byte -- this is the SAME function, not a
// parallel reimplementation. K is never partitioned in this phase.
void run_fused_tiled_matmul_bias_relu_range(
    const float* a, const float* b, const float* bias, float* out,
    long N, long K,
    long m_start, long m_end, long n_start, long n_end,
    int block_m, int block_n, int block_k
) {
    if (m_start >= m_end || n_start >= n_end) return;  // empty range: no-op
    std::vector<float> tile_scratch(static_cast<size_t>(block_m) * static_cast<size_t>(block_n));

    for (long ii = m_start; ii < m_end; ii += block_m) {
        const long i_end = std::min(ii + block_m, m_end);
        for (long jj = n_start; jj < n_end; jj += block_n) {
            const long j_end = std::min(jj + block_n, n_end);
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

// Ceiling-division contiguous-chunk boundary for worker `t` of `count`
// splitting range [0, total). Some trailing workers legitimately get an
// empty [start,start) range when count > total -- callers must treat that
// as a valid no-op, never an error.
void chunk_bounds(long total, int count, int t, long& start, long& end) {
    const long chunk = (total + count - 1) / count;
    start = std::min(static_cast<long>(t) * chunk, total);
    end   = std::min(static_cast<long>(t + 1) * chunk, total);
}

// Dispatches thread_count worker threads (or runs serially in-process for
// thread_count==1, with zero std::thread overhead), each computing a
// disjoint output row range (partition_axis=="m") or column range
// (partition_axis=="n"). No shared output ownership: every worker writes
// only its own [m_start,m_end) or [n_start,n_end) sub-region, never
// touched by any other worker. No atomics anywhere in this function or in
// run_fused_tiled_matmul_bias_relu_range's hot loops -- std::thread::join()
// is the only synchronization point, after all compute is done.
void dispatch(
    const float* a, const float* b, const float* bias, float* out,
    long M, long N, long K,
    int block_m, int block_n, int block_k,
    int thread_count, const std::string& partition_axis
) {
    if (thread_count == 1) {
        run_fused_tiled_matmul_bias_relu_range(a, b, bias, out, N, K, 0, M, 0, N,
                                                block_m, block_n, block_k);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(thread_count));
    for (int t = 0; t < thread_count; ++t) {
        if (partition_axis == "m") {
            long m_start, m_end;
            chunk_bounds(M, thread_count, t, m_start, m_end);
            workers.emplace_back([=]() {
                run_fused_tiled_matmul_bias_relu_range(a, b, bias, out, N, K,
                                                        m_start, m_end, 0, N,
                                                        block_m, block_n, block_k);
            });
        } else {  // partition_axis == "n"
            long n_start, n_end;
            chunk_bounds(N, thread_count, t, n_start, n_end);
            workers.emplace_back([=]() {
                run_fused_tiled_matmul_bias_relu_range(a, b, bias, out, N, K,
                                                        0, M, n_start, n_end,
                                                        block_m, block_n, block_k);
            });
        }
    }
    for (std::thread& w : workers) w.join();
}

Args parse_args(int argc, char** argv) {
    Args args;
    bool thread_count_given = false, axis_given = false, strategy_given = false;
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
        else if (flag == "--thread-count") {
            args.thread_count = std::stoi(next("--thread-count"));
            thread_count_given = true;
        } else if (flag == "--partition-axis") {
            args.partition_axis = next("--partition-axis");
            axis_given = true;
        } else if (flag == "--partition-strategy") {
            args.partition_strategy = next("--partition-strategy");
            strategy_given = true;
        } else if (flag == "--list-candidates") {
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

    // Thread-schedule contract validation (thread_schedule_contract_v1).
    // Defaults (thread_count=1, axis=none, strategy=serial) reproduce exact
    // P1B/P1C behavior when no thread flags are given at all -- the
    // documented backward-compatible default.
    if (!thread_count_given && (axis_given || strategy_given)) {
        fail("--partition-axis/--partition-strategy given without --thread-count -- "
             "the schedule must be fully specified together, never partially");
    }
    if (args.thread_count != 1 && args.thread_count != 2 && args.thread_count != 4) {
        fail("unsupported --thread-count " + std::to_string(args.thread_count) +
             " (supported: 1, 2, 4) -- refusing to silently clamp or round");
    }
    if (args.thread_count == 1) {
        if (args.partition_axis != "none" || args.partition_strategy != "serial") {
            fail("thread_count=1 requires partition_axis=none and partition_strategy=serial, "
                 "got partition_axis='" + args.partition_axis + "' partition_strategy='" +
                 args.partition_strategy + "' -- refusing to silently reinterpret");
        }
    } else {
        if (args.partition_axis != "m" && args.partition_axis != "n") {
            fail("thread_count>1 requires an explicit partition_axis of 'm' or 'n', got '" +
                 args.partition_axis + "' -- refusing to silently downgrade to serial");
        }
        if (args.partition_strategy != "contiguous_chunks") {
            fail("thread_count>1 requires partition_strategy=contiguous_chunks, got '" +
                 args.partition_strategy + "'");
        }
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
        dispatch(a.data(), b.data(), bias.data(), out.data(),
                  args.m, args.n, args.k,
                  candidate->block_m, candidate->block_n, candidate->block_k,
                  args.thread_count, args.partition_axis);
        auto t1 = std::chrono::high_resolution_clock::now();
        samples_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    write_floats(args.out_path, out);

    // Emit a small, self-contained JSON report on stdout (not the
    // ExecutionPlan schema -- a minimal per-invocation dispatch/timing
    // record for the calling adapter to parse). Self-reports the ACTUAL
    // dispatched kernel_id/thread_count/partition_axis/partition_strategy
    // so the adapter can verify the contract was honored exactly, never
    // silently substituted.
    std::ostringstream json;
    json << "{\n";
    json << "  \"kernel_id\": \"" << candidate->kernel_id << "\",\n";
    json << "  \"backend\": \"cpu\",\n";
    json << "  \"dtype\": \"f32\",\n";
    json << "  \"m\": " << args.m << ", \"n\": " << args.n << ", \"k\": " << args.k << ",\n";
    json << "  \"block_m\": " << candidate->block_m << ", \"block_n\": " << candidate->block_n
         << ", \"block_k\": " << candidate->block_k << ",\n";
    json << "  \"thread_count\": " << args.thread_count << ",\n";
    json << "  \"partition_axis\": \"" << args.partition_axis << "\",\n";
    json << "  \"partition_strategy\": \"" << args.partition_strategy << "\",\n";
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
