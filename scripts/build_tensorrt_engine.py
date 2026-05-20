import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

onnx_path = "models/mobilenet_v2_optimized.onnx"
engine_path = "models/mobilenet_v2_fp16.engine"

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX")

config = builder.create_builder_config()
config.set_memory_pool_limit(
    trt.MemoryPoolType.WORKSPACE,
    1 << 30,
)

config.set_flag(trt.BuilderFlag.FP16)

input_tensor = network.get_input(0)
input_name = input_tensor.name

profile = builder.create_optimization_profile()
profile.set_shape(
    input_name,
    min=(1, 3, 224, 224),
    opt=(1, 3, 224, 224),
    max=(1, 3, 224, 224),
)
config.add_optimization_profile(profile)

serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    raise RuntimeError("Failed to build TensorRT engine")

with open(engine_path, "wb") as f:
    f.write(serialized_engine)

print(f"Input name: {input_name}")
print(f"Saved TensorRT engine to {engine_path}")