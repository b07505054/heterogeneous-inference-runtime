#include <benchmark/benchmark.h>
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <random>
#include <string>
#include <vector>

struct OrtBenchmarkState {
    Ort::Env env;
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info;
    std::vector<float> input_values;
    std::vector<int64_t> input_shape;
    std::string input_name_storage;
    std::string output_name_storage;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    explicit OrtBenchmarkState(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "onnx_cpp_benchmark"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          input_shape{1, 3, 224, 224},
          input_values(1 * 3 * 224 * 224) {
        options.SetIntraOpNumThreads(4);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        std::wstring w_model_path(model_path.begin(), model_path.end());
        session = std::make_unique<Ort::Session>(env, w_model_path.c_str(), options);
#else
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), options);
#endif

        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name_alloc = session->GetInputNameAllocated(0, allocator);
        auto output_name_alloc = session->GetOutputNameAllocated(0, allocator);

        input_name_storage = input_name_alloc.get();
        output_name_storage = output_name_alloc.get();

        input_names = {input_name_storage.c_str()};
        output_names = {output_name_storage.c_str()};

        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (float& v : input_values) {
            v = dist(gen);
        }
    }

    Ort::Value make_input() {
        return Ort::Value::CreateTensor<float>(
            memory_info,
            input_values.data(),
            input_values.size(),
            input_shape.data(),
            input_shape.size()
        );
    }
};

static void BM_ONNXRuntimeCpp(benchmark::State& state) {
    const std::string model_path =
        state.range(0) == 0
            ? "models/mobilenet_v2_optimized.onnx"
            : "models/mobilenet_v2_fp32.onnx";

    OrtBenchmarkState ort_state(model_path);
    auto input_tensor = ort_state.make_input();

    for (int i = 0; i < 20; ++i) {
        ort_state.session->Run(
            Ort::RunOptions{nullptr},
            ort_state.input_names.data(),
            &input_tensor,
            1,
            ort_state.output_names.data(),
            1
        );
    }

    for (auto _ : state) {
        auto output_tensors = ort_state.session->Run(
            Ort::RunOptions{nullptr},
            ort_state.input_names.data(),
            &input_tensor,
            1,
            ort_state.output_names.data(),
            1
        );
        benchmark::DoNotOptimize(output_tensors);
    }

    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_ONNXRuntimeCpp)
    ->Arg(0)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();