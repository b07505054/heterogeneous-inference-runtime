# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((64, 64), "float32"), B: T.Buffer((64, 64), "float32"), bias: T.Buffer((64,), "float32"), matmul_bias_relu: T.Buffer((64, 64), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        matmul = T.sblock_alloc_buffer((64, 64))
        for i_0 in T.parallel(4):
            for j_0 in range(4):
                for k_0, i_1, k_1 in T.grid(8, 16, 8):
                    for j_1 in T.vectorized(16):
                        with T.sblock("matmul"):
                            v_i = T.axis.spatial(64, i_0 * 16 + i_1)
                            v_j = T.axis.spatial(64, j_0 * 16 + j_1)
                            v_k = T.axis.reduce(64, k_0 * 8 + k_1)
                            T.reads(A[v_i, v_k], B[v_k, v_j])
                            T.writes(matmul[v_i, v_j])
                            with T.init():
                                matmul[v_i, v_j] = T.float32(0.0)
                            matmul[v_i, v_j] = matmul[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]
                for ax0, ax1 in T.grid(16, 16):
                    with T.sblock("matmul_bias_relu"):
                        v_i = T.axis.spatial(64, i_0 * 16 + ax0)
                        v_j = T.axis.spatial(64, j_0 * 16 + ax1)
                        T.reads(matmul[v_i, v_j], bias[v_j])
                        T.writes(matmul_bias_relu[v_i, v_j])
                        matmul_bias_relu[v_i, v_j] = T.max(matmul[v_i, v_j] + bias[v_j], T.float32(0.0))
