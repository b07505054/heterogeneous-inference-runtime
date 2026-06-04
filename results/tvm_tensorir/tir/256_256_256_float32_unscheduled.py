# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((256, 256), "float32"), B: T.Buffer((256, 256), "float32"), bias: T.Buffer((256,), "float32"), matmul_bias_relu: T.Buffer((256, 256), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        matmul = T.sblock_alloc_buffer((256, 256))
        for i, j, k in T.grid(256, 256, 256):
            with T.sblock("matmul"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                T.reads(A[v_i, v_k], B[v_k, v_j])
                T.writes(matmul[v_i, v_j])
                with T.init():
                    matmul[v_i, v_j] = T.float32(0.0)
                matmul[v_i, v_j] = matmul[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]
        for i, j in T.grid(256, 256):
            with T.sblock("matmul_bias_relu"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(matmul[v_i, v_j], bias[v_j])
                T.writes(matmul_bias_relu[v_i, v_j])
                matmul_bias_relu[v_i, v_j] = T.max(matmul[v_i, v_j] + bias[v_j], T.float32(0.0))
