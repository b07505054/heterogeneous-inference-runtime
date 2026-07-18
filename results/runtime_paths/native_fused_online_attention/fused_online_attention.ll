; ModuleID = 'native/fused_online_attention.cpp'
source_filename = "native/fused_online_attention.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@.str = private unnamed_addr constant [30 x i8] c"hir.fused_online_attention.v1\00", align 1
@__cpu_model = external dso_local local_unnamed_addr global { i32, i32, i32, [1 x i32] }
@.str.1 = private unnamed_addr constant [21 x i8] c"avx2_fma_unavailable\00", align 1
@.str.2 = private unnamed_addr constant [28 x i8] c"invalid_softmax_denominator\00", align 1
@.str.3 = private unnamed_addr constant [3 x i8] c"ok\00", align 1
@.str.4 = private unnamed_addr constant [13 x i8] c"null_pointer\00", align 1
@.str.5 = private unnamed_addr constant [18 x i8] c"invalid_dimension\00", align 1
@.str.6 = private unnamed_addr constant [20 x i8] c"invalid_gqa_mapping\00", align 1
@.str.7 = private unnamed_addr constant [25 x i8] c"invalid_query_head_range\00", align 1
@.str.8 = private unnamed_addr constant [19 x i8] c"unsupported_stride\00", align 1
@.str.9 = private unnamed_addr constant [14 x i8] c"invalid_scale\00", align 1
@.str.10 = private unnamed_addr constant [16 x i8] c"causal_required\00", align 1
@.str.11 = private unnamed_addr constant [29 x i8] c"invalid_tile_or_worker_count\00", align 1
@.str.12 = private unnamed_addr constant [30 x i8] c"invalid_causal_position_range\00", align 1
@.str.13 = private unnamed_addr constant [49 x i8] c"cannot create std::vector larger than max_size()\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef nonnull ptr @hir_fused_attention_artifact_version() local_unnamed_addr #0 {
  ret ptr @.str
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @hir_fused_attention_has_avx2() local_unnamed_addr #1 {
  %1 = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12), align 4
  %2 = and i32 %1, 1024
  %3 = icmp eq i32 %2, 0
  %4 = lshr i32 %1, 14
  %5 = and i32 %4, 1
  %6 = select i1 %3, i32 0, i32 %5
  ret i32 %6
}

; Function Attrs: mustprogress uwtable
define dso_local { i32, ptr } @hir_fused_online_attention_scalar(ptr noundef readonly captures(address_is_null) %0, ptr noundef writeonly captures(address_is_null) %1) local_unnamed_addr #2 personality ptr @__gxx_personality_v0 {
  %3 = tail call fastcc { i32, ptr } @_ZN12_GLOBAL__N_18validateEPK23HirFusedAttentionParams(ptr noundef readonly %0)
  %4 = extractvalue { i32, ptr } %3, 0
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %430

6:                                                ; preds = %2
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 232
  %8 = load i64, ptr %7, align 8, !tbaa !5
  %9 = shl i64 %8, 2
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %11 = load i64, ptr %10, align 8, !tbaa !14
  %12 = shl i64 %11, 2
  %13 = add nsw i64 %11, %8
  %14 = icmp ugt i64 %13, 2305843009213693951
  br i1 %14, label %15, label %16

15:                                               ; preds = %6
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.13) #13
  unreachable

16:                                               ; preds = %6
  %17 = icmp eq i64 %13, 0
  br i1 %17, label %26, label %18

18:                                               ; preds = %16
  %19 = shl nuw nsw i64 %13, 2
  %20 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %19) #14
  %21 = getelementptr inbounds nuw float, ptr %20, i64 %13
  store float 0.000000e+00, ptr %20, align 4, !tbaa !15
  %22 = icmp eq i64 %13, 1
  br i1 %22, label %26, label %23

23:                                               ; preds = %18
  %24 = getelementptr i8, ptr %20, i64 4
  %25 = add nsw i64 %19, -4
  tail call void @llvm.memset.p0.i64(ptr align 4 %24, i8 0, i64 %25, i1 false), !tbaa !15
  br label %26

26:                                               ; preds = %23, %18, %16
  %27 = phi ptr [ %21, %18 ], [ %21, %23 ], [ null, %16 ]
  %28 = phi ptr [ %20, %18 ], [ %20, %23 ], [ null, %16 ]
  %29 = getelementptr float, ptr %28, i64 %8
  %30 = icmp eq ptr %1, null
  br i1 %30, label %40, label %31

31:                                               ; preds = %26
  %32 = add i64 %12, %9
  %33 = tail call i64 @llvm.umax.i64(i64 %9, i64 %12)
  store i64 %32, ptr %1, align 8, !tbaa !16
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i64 %33, ptr %34, align 8, !tbaa !16
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i64 1, ptr %35, align 8, !tbaa !16
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i64 0, ptr %36, align 8, !tbaa !16
  %37 = getelementptr inbounds nuw i8, ptr %1, i64 32
  store i64 %9, ptr %37, align 8, !tbaa !16
  %38 = getelementptr inbounds nuw i8, ptr %1, i64 40
  store i64 %12, ptr %38, align 8, !tbaa !16
  %39 = getelementptr inbounds nuw i8, ptr %1, i64 48
  store i64 %32, ptr %39, align 8, !tbaa !16
  br label %40

40:                                               ; preds = %31, %26
  %41 = getelementptr inbounds nuw i8, ptr %0, i64 256
  %42 = load i64, ptr %41, align 8, !tbaa !17
  %43 = getelementptr inbounds nuw i8, ptr %0, i64 64
  %44 = load i64, ptr %43, align 8, !tbaa !18
  %45 = sdiv i64 %42, %44
  %46 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %47 = load i64, ptr %46, align 8, !tbaa !19
  %48 = icmp sgt i64 %47, 0
  br i1 %48, label %49, label %419

49:                                               ; preds = %40
  %50 = getelementptr inbounds nuw i8, ptr %0, i64 56
  %51 = load i64, ptr %50, align 8, !tbaa !20
  %52 = icmp sgt i64 %51, 0
  %53 = getelementptr inbounds nuw i8, ptr %0, i64 80
  %54 = getelementptr inbounds nuw i8, ptr %0, i64 88
  %55 = getelementptr inbounds nuw i8, ptr %0, i64 96
  %56 = getelementptr inbounds nuw i8, ptr %0, i64 216
  %57 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %59 = getelementptr inbounds nuw i8, ptr %0, i64 112
  %60 = getelementptr inbounds nuw i8, ptr %0, i64 120
  %61 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %62 = getelementptr inbounds nuw i8, ptr %0, i64 104
  %63 = getelementptr inbounds nuw i8, ptr %0, i64 136
  %64 = getelementptr inbounds nuw i8, ptr %0, i64 208
  %65 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %66 = getelementptr inbounds nuw i8, ptr %0, i64 144
  %67 = getelementptr inbounds nuw i8, ptr %0, i64 152
  %68 = getelementptr inbounds nuw i8, ptr %0, i64 160
  %69 = getelementptr inbounds nuw i8, ptr %0, i64 168
  %70 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %71 = getelementptr inbounds nuw i8, ptr %0, i64 176
  %72 = getelementptr inbounds nuw i8, ptr %0, i64 184
  %73 = getelementptr inbounds nuw i8, ptr %0, i64 192
  %74 = getelementptr inbounds nuw i8, ptr %0, i64 200
  br i1 %52, label %75, label %419

75:                                               ; preds = %49
  %76 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %77 = getelementptr inbounds nuw i8, ptr %0, i64 248
  %78 = load i64, ptr %77, align 8, !tbaa !21
  %79 = load i64, ptr %76, align 8, !tbaa !22
  %80 = icmp sgt i64 %79, 0
  br i1 %80, label %81, label %419

81:                                               ; preds = %75
  %82 = load i64, ptr %10, align 8, !tbaa !14
  %83 = freeze i64 %82
  %84 = shl i64 %83, 2
  %85 = icmp eq i64 %83, 0
  %86 = load ptr, ptr %0, align 8, !tbaa !23
  %87 = load i64, ptr %53, align 8, !tbaa !16
  %88 = load i64, ptr %54, align 8, !tbaa !16
  %89 = load i64, ptr %55, align 8, !tbaa !16
  %90 = load i64, ptr %56, align 8, !tbaa !24
  %91 = load i64, ptr %57, align 8, !tbaa !25
  %92 = icmp sgt i64 %91, 0
  %93 = icmp sgt i64 %83, 0
  br i1 %92, label %94, label %417

94:                                               ; preds = %81
  %95 = load i64, ptr %7, align 8, !tbaa !5
  %96 = and i64 %83, 3
  %97 = icmp ult i64 %83, 4
  %98 = and i64 %83, 9223372036854775804
  %99 = icmp eq i64 %96, 0
  %100 = and i64 %83, 3
  %101 = icmp ult i64 %83, 4
  %102 = and i64 %83, 9223372036854775804
  %103 = getelementptr inbounds i8, ptr %29, i64 4
  %104 = getelementptr inbounds i8, ptr %29, i64 8
  %105 = getelementptr inbounds i8, ptr %29, i64 12
  %106 = icmp eq i64 %100, 0
  %107 = and i64 %83, 1
  %108 = icmp eq i64 %83, 1
  %109 = and i64 %83, 9223372036854775806
  %110 = icmp eq i64 %107, 0
  %111 = and i64 %83, 3
  %112 = icmp ult i64 %83, 4
  %113 = and i64 %83, 9223372036854775804
  %114 = icmp eq i64 %111, 0
  br label %115

115:                                              ; preds = %414, %94
  %116 = phi i64 [ 0, %94 ], [ %415, %414 ]
  %117 = mul nsw i64 %116, %87
  %118 = getelementptr float, ptr %86, i64 %117
  br label %119

119:                                              ; preds = %411, %115
  %120 = phi i64 [ 0, %115 ], [ %412, %411 ]
  %121 = add nsw i64 %120, %78
  %122 = sdiv i64 %121, %45
  %123 = mul nsw i64 %120, %88
  %124 = getelementptr float, ptr %118, i64 %123
  br label %125

125:                                              ; preds = %156, %119
  %126 = phi i64 [ 0, %119 ], [ %157, %156 ]
  br i1 %85, label %128, label %127

127:                                              ; preds = %125
  tail call void @llvm.memset.p0.i64(ptr align 4 %29, i8 0, i64 %84, i1 false), !tbaa !15
  br label %128

128:                                              ; preds = %127, %125
  %129 = mul nsw i64 %126, %89
  %130 = getelementptr float, ptr %124, i64 %129
  %131 = add nsw i64 %126, %90
  br label %188

132:                                              ; preds = %407
  %133 = load ptr, ptr %70, align 8, !tbaa !26
  %134 = load i64, ptr %71, align 8, !tbaa !16
  %135 = mul nsw i64 %134, %116
  %136 = load i64, ptr %72, align 8, !tbaa !16
  %137 = mul nsw i64 %136, %120
  %138 = load i64, ptr %73, align 8, !tbaa !16
  %139 = mul nsw i64 %138, %126
  %140 = getelementptr float, ptr %133, i64 %137
  %141 = getelementptr float, ptr %140, i64 %135
  %142 = getelementptr float, ptr %141, i64 %139
  br i1 %93, label %409, label %156

143:                                              ; preds = %159, %409
  %144 = phi i64 [ 0, %409 ], [ %185, %159 ]
  br i1 %114, label %156, label %145

145:                                              ; preds = %143, %145
  %146 = phi i64 [ %153, %145 ], [ %144, %143 ]
  %147 = phi i64 [ %154, %145 ], [ 0, %143 ]
  %148 = getelementptr inbounds nuw float, ptr %29, i64 %146
  %149 = load float, ptr %148, align 4, !tbaa !15
  %150 = fdiv float %149, %246
  %151 = mul nsw i64 %146, %410
  %152 = getelementptr inbounds float, ptr %142, i64 %151
  store float %150, ptr %152, align 4, !tbaa !15
  %153 = add nuw nsw i64 %146, 1
  %154 = add i64 %147, 1
  %155 = icmp eq i64 %154, %111
  br i1 %155, label %156, label %145, !llvm.loop !27

156:                                              ; preds = %143, %145, %132
  %157 = add nuw nsw i64 %126, 1
  %158 = icmp eq i64 %157, %79
  br i1 %158, label %411, label %125, !llvm.loop !29

159:                                              ; preds = %409, %159
  %160 = phi i64 [ %185, %159 ], [ 0, %409 ]
  %161 = phi i64 [ %186, %159 ], [ 0, %409 ]
  %162 = getelementptr inbounds nuw float, ptr %29, i64 %160
  %163 = load float, ptr %162, align 4, !tbaa !15
  %164 = fdiv float %163, %246
  %165 = mul nsw i64 %160, %410
  %166 = getelementptr inbounds float, ptr %142, i64 %165
  store float %164, ptr %166, align 4, !tbaa !15
  %167 = or disjoint i64 %160, 1
  %168 = getelementptr inbounds nuw float, ptr %29, i64 %167
  %169 = load float, ptr %168, align 4, !tbaa !15
  %170 = fdiv float %169, %246
  %171 = mul nsw i64 %167, %410
  %172 = getelementptr inbounds float, ptr %142, i64 %171
  store float %170, ptr %172, align 4, !tbaa !15
  %173 = or disjoint i64 %160, 2
  %174 = getelementptr inbounds nuw float, ptr %29, i64 %173
  %175 = load float, ptr %174, align 4, !tbaa !15
  %176 = fdiv float %175, %246
  %177 = mul nsw i64 %173, %410
  %178 = getelementptr inbounds float, ptr %142, i64 %177
  store float %176, ptr %178, align 4, !tbaa !15
  %179 = or disjoint i64 %160, 3
  %180 = getelementptr inbounds nuw float, ptr %29, i64 %179
  %181 = load float, ptr %180, align 4, !tbaa !15
  %182 = fdiv float %181, %246
  %183 = mul nsw i64 %179, %410
  %184 = getelementptr inbounds float, ptr %142, i64 %183
  store float %182, ptr %184, align 4, !tbaa !15
  %185 = add nuw nsw i64 %160, 4
  %186 = add i64 %161, 4
  %187 = icmp eq i64 %186, %113
  br i1 %187, label %143, label %159, !llvm.loop !31

188:                                              ; preds = %244, %128
  %189 = phi i64 [ %248, %244 ], [ %95, %128 ]
  %190 = phi i64 [ %194, %244 ], [ 0, %128 ]
  %191 = phi float [ %246, %244 ], [ 0.000000e+00, %128 ]
  %192 = phi float [ %245, %244 ], [ 0xFFF0000000000000, %128 ]
  %193 = tail call i64 @llvm.smin.i64(i64 %91, i64 %189)
  %194 = add nsw i64 %190, %95
  %195 = tail call i64 @llvm.smin.i64(i64 %91, i64 %194)
  %196 = icmp slt i64 %190, %195
  br i1 %196, label %249, label %197

197:                                              ; preds = %334, %188
  %198 = phi float [ 0xFFF0000000000000, %188 ], [ %335, %334 ]
  %199 = tail call float @llvm.fabs.f32(float %198)
  %200 = fcmp ueq float %199, 0x7FF0000000000000
  br i1 %200, label %244, label %201

201:                                              ; preds = %197
  %202 = fcmp olt float %192, %198
  %203 = select i1 %202, float %198, float %192
  %204 = tail call float @llvm.fabs.f32(float %192)
  %205 = fcmp ueq float %204, 0x7FF0000000000000
  br i1 %205, label %209, label %206

206:                                              ; preds = %201
  %207 = fsub float %192, %203
  %208 = tail call noundef float @expf(float noundef %207) #15, !tbaa !32
  br label %209

209:                                              ; preds = %206, %201
  %210 = phi float [ %208, %206 ], [ 0.000000e+00, %201 ]
  %211 = fmul float %191, %210
  br i1 %93, label %212, label %349

212:                                              ; preds = %209
  br i1 %101, label %338, label %226

213:                                              ; preds = %352, %213
  %214 = phi i64 [ %222, %213 ], [ %190, %352 ]
  %215 = phi float [ %221, %213 ], [ %211, %352 ]
  %216 = sub nsw i64 %214, %190
  %217 = getelementptr inbounds float, ptr %28, i64 %216
  %218 = load float, ptr %217, align 4, !tbaa !15
  %219 = fsub float %218, %203
  %220 = tail call noundef float @expf(float noundef %219) #15, !tbaa !32
  %221 = fadd float %215, %220
  %222 = add nsw i64 %214, 1
  %223 = icmp slt i64 %222, %195
  %224 = icmp slt i64 %214, %131
  %225 = select i1 %223, i1 %224, i1 false
  br i1 %225, label %213, label %244, !llvm.loop !33

226:                                              ; preds = %212, %226
  %227 = phi i64 [ %241, %226 ], [ 0, %212 ]
  %228 = phi i64 [ %242, %226 ], [ 0, %212 ]
  %229 = getelementptr inbounds nuw float, ptr %29, i64 %227
  %230 = load float, ptr %229, align 4, !tbaa !15
  %231 = fmul float %210, %230
  store float %231, ptr %229, align 4, !tbaa !15
  %232 = getelementptr inbounds float, ptr %103, i64 %227
  %233 = load float, ptr %232, align 4, !tbaa !15
  %234 = fmul float %210, %233
  store float %234, ptr %232, align 4, !tbaa !15
  %235 = getelementptr inbounds float, ptr %104, i64 %227
  %236 = load float, ptr %235, align 4, !tbaa !15
  %237 = fmul float %210, %236
  store float %237, ptr %235, align 4, !tbaa !15
  %238 = getelementptr inbounds float, ptr %105, i64 %227
  %239 = load float, ptr %238, align 4, !tbaa !15
  %240 = fmul float %210, %239
  store float %240, ptr %238, align 4, !tbaa !15
  %241 = add nuw nsw i64 %227, 4
  %242 = add i64 %228, 4
  %243 = icmp eq i64 %242, %102
  br i1 %243, label %338, label %226, !llvm.loop !34

244:                                              ; preds = %213, %401, %349, %197
  %245 = phi float [ %192, %197 ], [ %203, %349 ], [ %203, %401 ], [ %203, %213 ]
  %246 = phi float [ %191, %197 ], [ %211, %349 ], [ %402, %401 ], [ %221, %213 ]
  %247 = icmp slt i64 %194, %91
  %248 = add i64 %189, %95
  br i1 %247, label %188, label %407, !llvm.loop !35

249:                                              ; preds = %188, %334
  %250 = phi i64 [ %336, %334 ], [ %190, %188 ]
  %251 = phi float [ %335, %334 ], [ 0xFFF0000000000000, %188 ]
  %252 = icmp sgt i64 %250, %131
  br i1 %252, label %331, label %253

253:                                              ; preds = %249
  %254 = load ptr, ptr %58, align 8, !tbaa !36
  %255 = load i64, ptr %59, align 8, !tbaa !16
  %256 = mul nsw i64 %255, %116
  %257 = load i64, ptr %60, align 8, !tbaa !16
  %258 = mul nsw i64 %257, %122
  %259 = load i64, ptr %61, align 8, !tbaa !16
  %260 = mul nsw i64 %259, %250
  %261 = getelementptr float, ptr %254, i64 %258
  %262 = getelementptr float, ptr %261, i64 %256
  %263 = getelementptr float, ptr %262, i64 %260
  %264 = load i64, ptr %62, align 8, !tbaa !16
  %265 = load i64, ptr %63, align 8, !tbaa !16
  br i1 %93, label %266, label %323

266:                                              ; preds = %253
  br i1 %97, label %305, label %267

267:                                              ; preds = %266, %267
  %268 = phi i64 [ %302, %267 ], [ 0, %266 ]
  %269 = phi float [ %301, %267 ], [ 0.000000e+00, %266 ]
  %270 = phi i64 [ %303, %267 ], [ 0, %266 ]
  %271 = mul nsw i64 %268, %264
  %272 = getelementptr inbounds float, ptr %130, i64 %271
  %273 = load float, ptr %272, align 4, !tbaa !15
  %274 = mul nsw i64 %268, %265
  %275 = getelementptr inbounds float, ptr %263, i64 %274
  %276 = load float, ptr %275, align 4, !tbaa !15
  %277 = tail call float @llvm.fmuladd.f32(float %273, float %276, float %269)
  %278 = or disjoint i64 %268, 1
  %279 = mul nsw i64 %278, %264
  %280 = getelementptr inbounds float, ptr %130, i64 %279
  %281 = load float, ptr %280, align 4, !tbaa !15
  %282 = mul nsw i64 %278, %265
  %283 = getelementptr inbounds float, ptr %263, i64 %282
  %284 = load float, ptr %283, align 4, !tbaa !15
  %285 = tail call float @llvm.fmuladd.f32(float %281, float %284, float %277)
  %286 = or disjoint i64 %268, 2
  %287 = mul nsw i64 %286, %264
  %288 = getelementptr inbounds float, ptr %130, i64 %287
  %289 = load float, ptr %288, align 4, !tbaa !15
  %290 = mul nsw i64 %286, %265
  %291 = getelementptr inbounds float, ptr %263, i64 %290
  %292 = load float, ptr %291, align 4, !tbaa !15
  %293 = tail call float @llvm.fmuladd.f32(float %289, float %292, float %285)
  %294 = or disjoint i64 %268, 3
  %295 = mul nsw i64 %294, %264
  %296 = getelementptr inbounds float, ptr %130, i64 %295
  %297 = load float, ptr %296, align 4, !tbaa !15
  %298 = mul nsw i64 %294, %265
  %299 = getelementptr inbounds float, ptr %263, i64 %298
  %300 = load float, ptr %299, align 4, !tbaa !15
  %301 = tail call float @llvm.fmuladd.f32(float %297, float %300, float %293)
  %302 = add nuw nsw i64 %268, 4
  %303 = add i64 %270, 4
  %304 = icmp eq i64 %303, %98
  br i1 %304, label %305, label %267, !llvm.loop !37

305:                                              ; preds = %267, %266
  %306 = phi float [ poison, %266 ], [ %301, %267 ]
  %307 = phi i64 [ 0, %266 ], [ %302, %267 ]
  %308 = phi float [ 0.000000e+00, %266 ], [ %301, %267 ]
  br i1 %99, label %323, label %309

309:                                              ; preds = %305, %309
  %310 = phi i64 [ %320, %309 ], [ %307, %305 ]
  %311 = phi float [ %319, %309 ], [ %308, %305 ]
  %312 = phi i64 [ %321, %309 ], [ 0, %305 ]
  %313 = mul nsw i64 %310, %264
  %314 = getelementptr inbounds float, ptr %130, i64 %313
  %315 = load float, ptr %314, align 4, !tbaa !15
  %316 = mul nsw i64 %310, %265
  %317 = getelementptr inbounds float, ptr %263, i64 %316
  %318 = load float, ptr %317, align 4, !tbaa !15
  %319 = tail call float @llvm.fmuladd.f32(float %315, float %318, float %311)
  %320 = add nuw nsw i64 %310, 1
  %321 = add i64 %312, 1
  %322 = icmp eq i64 %321, %96
  br i1 %322, label %323, label %309, !llvm.loop !38

323:                                              ; preds = %305, %309, %253
  %324 = phi float [ 0.000000e+00, %253 ], [ %306, %305 ], [ %319, %309 ]
  %325 = load float, ptr %64, align 8, !tbaa !39
  %326 = fmul float %324, %325
  %327 = sub nsw i64 %250, %190
  %328 = getelementptr inbounds float, ptr %28, i64 %327
  store float %326, ptr %328, align 4, !tbaa !15
  %329 = fcmp olt float %251, %326
  %330 = select i1 %329, float %326, float %251
  br label %334

331:                                              ; preds = %249
  %332 = sub nsw i64 %250, %190
  %333 = getelementptr inbounds float, ptr %28, i64 %332
  store float 0xFFF0000000000000, ptr %333, align 4, !tbaa !15
  br label %334

334:                                              ; preds = %331, %323
  %335 = phi float [ %251, %331 ], [ %330, %323 ]
  %336 = add nsw i64 %250, 1
  %337 = icmp eq i64 %336, %193
  br i1 %337, label %197, label %249, !llvm.loop !40

338:                                              ; preds = %226, %212
  %339 = phi i64 [ 0, %212 ], [ %241, %226 ]
  br i1 %106, label %349, label %340

340:                                              ; preds = %338, %340
  %341 = phi i64 [ %346, %340 ], [ %339, %338 ]
  %342 = phi i64 [ %347, %340 ], [ 0, %338 ]
  %343 = getelementptr inbounds nuw float, ptr %29, i64 %341
  %344 = load float, ptr %343, align 4, !tbaa !15
  %345 = fmul float %210, %344
  store float %345, ptr %343, align 4, !tbaa !15
  %346 = add nuw nsw i64 %341, 1
  %347 = add i64 %342, 1
  %348 = icmp eq i64 %347, %100
  br i1 %348, label %349, label %340, !llvm.loop !41

349:                                              ; preds = %338, %340, %209
  %350 = icmp sle i64 %190, %131
  %351 = select i1 %196, i1 %350, i1 false
  br i1 %351, label %352, label %244

352:                                              ; preds = %349
  %353 = load ptr, ptr %65, align 8, !tbaa !42
  %354 = load i64, ptr %66, align 8, !tbaa !16
  %355 = mul nsw i64 %354, %116
  %356 = load i64, ptr %67, align 8, !tbaa !16
  %357 = mul nsw i64 %356, %122
  %358 = load i64, ptr %68, align 8, !tbaa !16
  %359 = getelementptr float, ptr %353, i64 %357
  %360 = getelementptr float, ptr %359, i64 %355
  br i1 %93, label %361, label %213

361:                                              ; preds = %352
  %362 = load i64, ptr %69, align 8, !tbaa !16
  br label %363

363:                                              ; preds = %401, %361
  %364 = phi i64 [ %190, %361 ], [ %403, %401 ]
  %365 = phi float [ %211, %361 ], [ %402, %401 ]
  %366 = sub nsw i64 %364, %190
  %367 = getelementptr inbounds float, ptr %28, i64 %366
  %368 = load float, ptr %367, align 4, !tbaa !15
  %369 = fsub float %368, %203
  %370 = tail call noundef float @expf(float noundef %369) #15, !tbaa !32
  %371 = mul nsw i64 %364, %358
  %372 = getelementptr float, ptr %360, i64 %371
  br i1 %108, label %392, label %373

373:                                              ; preds = %363, %373
  %374 = phi i64 [ %389, %373 ], [ 0, %363 ]
  %375 = phi i64 [ %390, %373 ], [ 0, %363 ]
  %376 = mul nsw i64 %374, %362
  %377 = getelementptr inbounds float, ptr %372, i64 %376
  %378 = load float, ptr %377, align 4, !tbaa !15
  %379 = getelementptr inbounds nuw float, ptr %29, i64 %374
  %380 = load float, ptr %379, align 4, !tbaa !15
  %381 = tail call float @llvm.fmuladd.f32(float %370, float %378, float %380)
  store float %381, ptr %379, align 4, !tbaa !15
  %382 = or disjoint i64 %374, 1
  %383 = mul nsw i64 %382, %362
  %384 = getelementptr inbounds float, ptr %372, i64 %383
  %385 = load float, ptr %384, align 4, !tbaa !15
  %386 = getelementptr inbounds nuw float, ptr %29, i64 %382
  %387 = load float, ptr %386, align 4, !tbaa !15
  %388 = tail call float @llvm.fmuladd.f32(float %370, float %385, float %387)
  store float %388, ptr %386, align 4, !tbaa !15
  %389 = add nuw nsw i64 %374, 2
  %390 = add i64 %375, 2
  %391 = icmp eq i64 %390, %109
  br i1 %391, label %392, label %373, !llvm.loop !43

392:                                              ; preds = %373, %363
  %393 = phi i64 [ 0, %363 ], [ %389, %373 ]
  br i1 %110, label %401, label %394

394:                                              ; preds = %392
  %395 = mul nsw i64 %393, %362
  %396 = getelementptr inbounds float, ptr %372, i64 %395
  %397 = load float, ptr %396, align 4, !tbaa !15
  %398 = getelementptr inbounds nuw float, ptr %29, i64 %393
  %399 = load float, ptr %398, align 4, !tbaa !15
  %400 = tail call float @llvm.fmuladd.f32(float %370, float %397, float %399)
  store float %400, ptr %398, align 4, !tbaa !15
  br label %401

401:                                              ; preds = %392, %394
  %402 = fadd float %365, %370
  %403 = add nsw i64 %364, 1
  %404 = icmp slt i64 %403, %195
  %405 = icmp slt i64 %364, %131
  %406 = select i1 %404, i1 %405, i1 false
  br i1 %406, label %363, label %244, !llvm.loop !33

407:                                              ; preds = %244
  %408 = tail call i1 @llvm.is.fpclass.f32(float %246, i32 384)
  br i1 %408, label %132, label %419

409:                                              ; preds = %132
  %410 = load i64, ptr %74, align 8, !tbaa !16
  br i1 %112, label %143, label %159

411:                                              ; preds = %156
  %412 = add nuw nsw i64 %120, 1
  %413 = icmp eq i64 %412, %51
  br i1 %413, label %414, label %119, !llvm.loop !44

414:                                              ; preds = %411
  %415 = add nuw nsw i64 %116, 1
  %416 = icmp eq i64 %415, %47
  br i1 %416, label %419, label %115, !llvm.loop !45

417:                                              ; preds = %81
  br i1 %85, label %419, label %418

418:                                              ; preds = %417
  tail call void @llvm.memset.p0.i64(ptr align 4 %29, i8 0, i64 %84, i1 false), !tbaa !15
  br label %419

419:                                              ; preds = %414, %407, %418, %417, %75, %49, %40
  %420 = phi ptr [ @.str.3, %40 ], [ @.str.3, %49 ], [ @.str.3, %75 ], [ @.str.2, %418 ], [ @.str.2, %417 ], [ @.str.2, %407 ], [ @.str.3, %414 ]
  %421 = phi i32 [ 0, %40 ], [ 0, %49 ], [ 0, %75 ], [ 10, %418 ], [ 10, %417 ], [ 10, %407 ], [ 0, %414 ]
  %422 = icmp eq ptr %28, null
  br i1 %422, label %427, label %423

423:                                              ; preds = %419
  %424 = ptrtoint ptr %27 to i64
  %425 = ptrtoint ptr %28 to i64
  %426 = sub i64 %424, %425
  tail call void @_ZdlPvm(ptr noundef nonnull %28, i64 noundef %426) #16
  br label %427

427:                                              ; preds = %423, %419
  %428 = insertvalue { i32, ptr } poison, i32 %421, 0
  %429 = insertvalue { i32, ptr } %428, ptr %420, 1
  br label %430

430:                                              ; preds = %2, %427
  %431 = phi { i32, ptr } [ %429, %427 ], [ %3, %2 ]
  ret { i32, ptr } %431
}

; Function Attrs: mustprogress uwtable
define dso_local { i32, ptr } @hir_fused_online_attention_avx2(ptr noundef readonly captures(address_is_null) %0, ptr noundef writeonly captures(address_is_null) %1) local_unnamed_addr #3 personality ptr @__gxx_personality_v0 {
  %3 = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12), align 4
  %4 = and i32 %3, 17408
  %5 = icmp eq i32 %4, 17408
  br i1 %5, label %6, label %675

6:                                                ; preds = %2
  %7 = tail call fastcc { i32, ptr } @_ZN12_GLOBAL__N_18validateEPK23HirFusedAttentionParams(ptr noundef readonly %0)
  %8 = extractvalue { i32, ptr } %7, 0
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %10, label %675

10:                                               ; preds = %6
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 232
  %12 = load i64, ptr %11, align 8, !tbaa !5
  %13 = shl i64 %12, 2
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %15 = load i64, ptr %14, align 8, !tbaa !14
  %16 = shl i64 %15, 2
  %17 = add nsw i64 %15, %12
  %18 = icmp ugt i64 %17, 2305843009213693951
  br i1 %18, label %19, label %20

19:                                               ; preds = %10
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.13) #13
  unreachable

20:                                               ; preds = %10
  %21 = icmp eq i64 %17, 0
  br i1 %21, label %30, label %22

22:                                               ; preds = %20
  %23 = shl nuw nsw i64 %17, 2
  %24 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %23) #14
  %25 = getelementptr inbounds nuw float, ptr %24, i64 %17
  store float 0.000000e+00, ptr %24, align 4, !tbaa !15
  %26 = icmp eq i64 %17, 1
  br i1 %26, label %30, label %27

27:                                               ; preds = %22
  %28 = getelementptr i8, ptr %24, i64 4
  %29 = add nsw i64 %23, -4
  tail call void @llvm.memset.p0.i64(ptr align 4 %28, i8 0, i64 %29, i1 false), !tbaa !15
  br label %30

30:                                               ; preds = %27, %22, %20
  %31 = phi ptr [ %25, %22 ], [ %25, %27 ], [ null, %20 ]
  %32 = phi ptr [ %24, %22 ], [ %24, %27 ], [ null, %20 ]
  %33 = getelementptr float, ptr %32, i64 %12
  %34 = icmp eq ptr %1, null
  br i1 %34, label %44, label %35

35:                                               ; preds = %30
  %36 = add i64 %16, %13
  %37 = tail call i64 @llvm.umax.i64(i64 %13, i64 %16)
  store i64 %36, ptr %1, align 8, !tbaa !16
  %38 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i64 %37, ptr %38, align 8, !tbaa !16
  %39 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i64 1, ptr %39, align 8, !tbaa !16
  %40 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i64 0, ptr %40, align 8, !tbaa !16
  %41 = getelementptr inbounds nuw i8, ptr %1, i64 32
  store i64 %13, ptr %41, align 8, !tbaa !16
  %42 = getelementptr inbounds nuw i8, ptr %1, i64 40
  store i64 %16, ptr %42, align 8, !tbaa !16
  %43 = getelementptr inbounds nuw i8, ptr %1, i64 48
  store i64 %36, ptr %43, align 8, !tbaa !16
  br label %44

44:                                               ; preds = %35, %30
  %45 = getelementptr inbounds nuw i8, ptr %0, i64 256
  %46 = load i64, ptr %45, align 8, !tbaa !17
  %47 = getelementptr inbounds nuw i8, ptr %0, i64 64
  %48 = load i64, ptr %47, align 8, !tbaa !18
  %49 = sdiv i64 %46, %48
  %50 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %51 = load i64, ptr %50, align 8, !tbaa !19
  %52 = icmp sgt i64 %51, 0
  br i1 %52, label %53, label %664

53:                                               ; preds = %44
  %54 = getelementptr inbounds nuw i8, ptr %0, i64 56
  %55 = getelementptr inbounds nuw i8, ptr %0, i64 248
  %56 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %57 = getelementptr inbounds nuw i8, ptr %0, i64 80
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 88
  %59 = getelementptr inbounds nuw i8, ptr %0, i64 96
  %60 = getelementptr inbounds nuw i8, ptr %0, i64 216
  %61 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %62 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %63 = getelementptr inbounds nuw i8, ptr %0, i64 112
  %64 = getelementptr inbounds nuw i8, ptr %0, i64 120
  %65 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %66 = getelementptr inbounds nuw i8, ptr %0, i64 104
  %67 = getelementptr inbounds nuw i8, ptr %0, i64 136
  %68 = getelementptr inbounds nuw i8, ptr %0, i64 208
  %69 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %70 = getelementptr inbounds nuw i8, ptr %0, i64 144
  %71 = getelementptr inbounds nuw i8, ptr %0, i64 152
  %72 = getelementptr inbounds nuw i8, ptr %0, i64 160
  %73 = getelementptr inbounds nuw i8, ptr %0, i64 168
  %74 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %75 = getelementptr inbounds nuw i8, ptr %0, i64 176
  %76 = getelementptr inbounds nuw i8, ptr %0, i64 184
  %77 = getelementptr inbounds nuw i8, ptr %0, i64 192
  %78 = load i64, ptr %54, align 8, !tbaa !20
  %79 = icmp sgt i64 %78, 0
  br i1 %79, label %80, label %664

80:                                               ; preds = %53
  %81 = getelementptr inbounds i8, ptr %33, i64 32
  %82 = getelementptr inbounds i8, ptr %33, i64 64
  %83 = getelementptr i8, ptr %33, i64 4
  %84 = getelementptr i8, ptr %33, i64 8
  %85 = getelementptr i8, ptr %33, i64 12
  br label %86

86:                                               ; preds = %80, %658
  %87 = phi i64 [ %659, %658 ], [ %51, %80 ]
  %88 = phi i64 [ %660, %658 ], [ %78, %80 ]
  %89 = phi i64 [ %661, %658 ], [ %78, %80 ]
  %90 = phi i64 [ %662, %658 ], [ 0, %80 ]
  %91 = icmp sgt i64 %89, 0
  br i1 %91, label %92, label %658

92:                                               ; preds = %86
  %93 = load i64, ptr %56, align 8, !tbaa !22
  %94 = icmp sgt i64 %93, 0
  br i1 %94, label %95, label %658

95:                                               ; preds = %92, %651
  %96 = phi i64 [ %652, %651 ], [ %88, %92 ]
  %97 = phi i64 [ %653, %651 ], [ %93, %92 ]
  %98 = phi i64 [ %654, %651 ], [ 0, %92 ]
  %99 = load i64, ptr %55, align 8, !tbaa !21
  %100 = add nsw i64 %99, %98
  %101 = sdiv i64 %100, %49
  %102 = icmp sgt i64 %97, 0
  br i1 %102, label %103, label %651

103:                                              ; preds = %95
  %104 = load i64, ptr %14, align 8, !tbaa !14
  br label %105

105:                                              ; preds = %645, %103
  %106 = phi i64 [ %592, %645 ], [ %104, %103 ]
  %107 = phi i64 [ %646, %645 ], [ 0, %103 ]
  %108 = icmp eq i64 %106, 0
  br i1 %108, label %111, label %109

109:                                              ; preds = %105
  %110 = shl i64 %106, 2
  tail call void @llvm.memset.p0.i64(ptr align 4 %33, i8 0, i64 %110, i1 false), !tbaa !15
  br label %111

111:                                              ; preds = %109, %105
  %112 = load ptr, ptr %0, align 8, !tbaa !23
  %113 = load i64, ptr %57, align 8, !tbaa !16
  %114 = mul nsw i64 %113, %90
  %115 = load i64, ptr %58, align 8, !tbaa !16
  %116 = mul nsw i64 %115, %98
  %117 = load i64, ptr %59, align 8, !tbaa !16
  %118 = mul nsw i64 %117, %107
  %119 = getelementptr float, ptr %112, i64 %116
  %120 = getelementptr float, ptr %119, i64 %114
  %121 = getelementptr float, ptr %120, i64 %118
  %122 = load i64, ptr %60, align 8, !tbaa !24
  %123 = add nsw i64 %122, %107
  %124 = load i64, ptr %61, align 8, !tbaa !25
  %125 = icmp sgt i64 %124, 0
  br i1 %125, label %126, label %150

126:                                              ; preds = %111
  %127 = load i64, ptr %11, align 8, !tbaa !5
  %128 = icmp slt i64 %106, 8
  %129 = icmp sgt i64 %106, 0
  %130 = add i64 %106, -8
  %131 = lshr i64 %130, 3
  %132 = add nuw nsw i64 %131, 1
  %133 = add nuw nsw i64 %131, 1
  %134 = and i64 %132, 3
  %135 = icmp ult i64 %130, 24
  %136 = and i64 %132, 4611686018427387900
  %137 = icmp eq i64 %134, 0
  %138 = and i64 %106, 3
  %139 = icmp ult i64 %106, 4
  %140 = and i64 %106, 9223372036854775804
  %141 = icmp eq i64 %138, 0
  %142 = and i64 %133, 3
  %143 = icmp ult i64 %130, 24
  %144 = and i64 %133, 4611686018427387900
  %145 = icmp eq i64 %142, 0
  %146 = and i64 %133, 3
  %147 = icmp ult i64 %130, 24
  %148 = and i64 %133, 4611686018427387900
  %149 = icmp eq i64 %146, 0
  br label %153

150:                                              ; preds = %571, %111
  %151 = phi float [ 0.000000e+00, %111 ], [ %573, %571 ]
  %152 = tail call i1 @llvm.is.fpclass.f32(float %151, i32 384)
  br i1 %152, label %576, label %664

153:                                              ; preds = %571, %126
  %154 = phi i64 [ %127, %126 ], [ %575, %571 ]
  %155 = phi float [ 0.000000e+00, %126 ], [ %573, %571 ]
  %156 = phi i64 [ 0, %126 ], [ %159, %571 ]
  %157 = phi float [ 0xFFF0000000000000, %126 ], [ %572, %571 ]
  %158 = tail call i64 @llvm.smin.i64(i64 %124, i64 %154)
  %159 = add nsw i64 %156, %127
  %160 = tail call i64 @llvm.smin.i64(i64 %124, i64 %159)
  %161 = icmp slt i64 %156, %160
  br i1 %161, label %166, label %162

162:                                              ; preds = %367, %153
  %163 = phi float [ 0xFFF0000000000000, %153 ], [ %368, %367 ]
  %164 = tail call float @llvm.fabs.f32(float %163)
  %165 = fcmp ueq float %164, 0x7FF0000000000000
  br i1 %165, label %571, label %371

166:                                              ; preds = %153, %367
  %167 = phi i64 [ %369, %367 ], [ %156, %153 ]
  %168 = phi float [ %368, %367 ], [ 0xFFF0000000000000, %153 ]
  %169 = icmp sgt i64 %167, %123
  br i1 %169, label %170, label %173

170:                                              ; preds = %166
  %171 = sub nsw i64 %167, %156
  %172 = getelementptr inbounds float, ptr %32, i64 %171
  store float 0xFFF0000000000000, ptr %172, align 4, !tbaa !15
  br label %367

173:                                              ; preds = %166
  %174 = load ptr, ptr %62, align 8, !tbaa !36
  %175 = load i64, ptr %63, align 8, !tbaa !16
  %176 = mul nsw i64 %175, %90
  %177 = load i64, ptr %64, align 8, !tbaa !16
  %178 = mul nsw i64 %177, %101
  %179 = load i64, ptr %65, align 8, !tbaa !16
  %180 = mul nsw i64 %179, %167
  %181 = getelementptr float, ptr %174, i64 %178
  %182 = getelementptr float, ptr %181, i64 %176
  %183 = getelementptr float, ptr %182, i64 %180
  %184 = load i64, ptr %66, align 8, !tbaa !16
  %185 = load i64, ptr %67, align 8, !tbaa !16
  %186 = icmp ne i64 %184, 1
  %187 = icmp ne i64 %185, 1
  %188 = or i1 %186, %187
  br i1 %188, label %191, label %189

189:                                              ; preds = %173
  br i1 %128, label %281, label %190

190:                                              ; preds = %189
  br i1 %135, label %262, label %231

191:                                              ; preds = %173
  br i1 %129, label %192, label %359

192:                                              ; preds = %191
  br i1 %139, label %341, label %193

193:                                              ; preds = %192, %193
  %194 = phi i64 [ %228, %193 ], [ 0, %192 ]
  %195 = phi float [ %227, %193 ], [ 0.000000e+00, %192 ]
  %196 = phi i64 [ %229, %193 ], [ 0, %192 ]
  %197 = mul nsw i64 %194, %184
  %198 = getelementptr inbounds float, ptr %121, i64 %197
  %199 = load float, ptr %198, align 4, !tbaa !15
  %200 = mul nsw i64 %194, %185
  %201 = getelementptr inbounds float, ptr %183, i64 %200
  %202 = load float, ptr %201, align 4, !tbaa !15
  %203 = tail call float @llvm.fmuladd.f32(float %199, float %202, float %195)
  %204 = or disjoint i64 %194, 1
  %205 = mul nsw i64 %204, %184
  %206 = getelementptr inbounds float, ptr %121, i64 %205
  %207 = load float, ptr %206, align 4, !tbaa !15
  %208 = mul nsw i64 %204, %185
  %209 = getelementptr inbounds float, ptr %183, i64 %208
  %210 = load float, ptr %209, align 4, !tbaa !15
  %211 = tail call float @llvm.fmuladd.f32(float %207, float %210, float %203)
  %212 = or disjoint i64 %194, 2
  %213 = mul nsw i64 %212, %184
  %214 = getelementptr inbounds float, ptr %121, i64 %213
  %215 = load float, ptr %214, align 4, !tbaa !15
  %216 = mul nsw i64 %212, %185
  %217 = getelementptr inbounds float, ptr %183, i64 %216
  %218 = load float, ptr %217, align 4, !tbaa !15
  %219 = tail call float @llvm.fmuladd.f32(float %215, float %218, float %211)
  %220 = or disjoint i64 %194, 3
  %221 = mul nsw i64 %220, %184
  %222 = getelementptr inbounds float, ptr %121, i64 %221
  %223 = load float, ptr %222, align 4, !tbaa !15
  %224 = mul nsw i64 %220, %185
  %225 = getelementptr inbounds float, ptr %183, i64 %224
  %226 = load float, ptr %225, align 4, !tbaa !15
  %227 = tail call float @llvm.fmuladd.f32(float %223, float %226, float %219)
  %228 = add nuw nsw i64 %194, 4
  %229 = add i64 %196, 4
  %230 = icmp eq i64 %229, %140
  br i1 %230, label %341, label %193, !llvm.loop !37

231:                                              ; preds = %190, %231
  %232 = phi i64 [ %259, %231 ], [ 8, %190 ]
  %233 = phi <8 x float> [ %258, %231 ], [ zeroinitializer, %190 ]
  %234 = phi i64 [ %253, %231 ], [ 0, %190 ]
  %235 = phi i64 [ %260, %231 ], [ 0, %190 ]
  %236 = getelementptr inbounds nuw float, ptr %121, i64 %234
  %237 = load <8 x float>, ptr %236, align 1, !tbaa !46
  %238 = getelementptr inbounds nuw float, ptr %183, i64 %234
  %239 = load <8 x float>, ptr %238, align 1, !tbaa !46
  %240 = tail call noundef <8 x float> @llvm.fma.v8f32(<8 x float> %237, <8 x float> %239, <8 x float> %233)
  %241 = add nuw nsw i64 %232, 8
  %242 = getelementptr inbounds nuw float, ptr %121, i64 %232
  %243 = load <8 x float>, ptr %242, align 1, !tbaa !46
  %244 = getelementptr inbounds nuw float, ptr %183, i64 %232
  %245 = load <8 x float>, ptr %244, align 1, !tbaa !46
  %246 = tail call noundef <8 x float> @llvm.fma.v8f32(<8 x float> %243, <8 x float> %245, <8 x float> %240)
  %247 = add nuw nsw i64 %232, 16
  %248 = getelementptr inbounds nuw float, ptr %121, i64 %241
  %249 = load <8 x float>, ptr %248, align 1, !tbaa !46
  %250 = getelementptr inbounds nuw float, ptr %183, i64 %241
  %251 = load <8 x float>, ptr %250, align 1, !tbaa !46
  %252 = tail call noundef <8 x float> @llvm.fma.v8f32(<8 x float> %249, <8 x float> %251, <8 x float> %246)
  %253 = add nuw nsw i64 %232, 24
  %254 = getelementptr inbounds nuw float, ptr %121, i64 %247
  %255 = load <8 x float>, ptr %254, align 1, !tbaa !46
  %256 = getelementptr inbounds nuw float, ptr %183, i64 %247
  %257 = load <8 x float>, ptr %256, align 1, !tbaa !46
  %258 = tail call noundef <8 x float> @llvm.fma.v8f32(<8 x float> %255, <8 x float> %257, <8 x float> %252)
  %259 = add nuw nsw i64 %232, 32
  %260 = add i64 %235, 4
  %261 = icmp eq i64 %260, %136
  br i1 %261, label %262, label %231, !llvm.loop !47

262:                                              ; preds = %231, %190
  %263 = phi i64 [ poison, %190 ], [ %253, %231 ]
  %264 = phi <8 x float> [ poison, %190 ], [ %258, %231 ]
  %265 = phi i64 [ 8, %190 ], [ %259, %231 ]
  %266 = phi <8 x float> [ zeroinitializer, %190 ], [ %258, %231 ]
  %267 = phi i64 [ 0, %190 ], [ %253, %231 ]
  br i1 %137, label %281, label %268

268:                                              ; preds = %262, %268
  %269 = phi i64 [ %278, %268 ], [ %265, %262 ]
  %270 = phi <8 x float> [ %277, %268 ], [ %266, %262 ]
  %271 = phi i64 [ %269, %268 ], [ %267, %262 ]
  %272 = phi i64 [ %279, %268 ], [ 0, %262 ]
  %273 = getelementptr inbounds nuw float, ptr %121, i64 %271
  %274 = load <8 x float>, ptr %273, align 1, !tbaa !46
  %275 = getelementptr inbounds nuw float, ptr %183, i64 %271
  %276 = load <8 x float>, ptr %275, align 1, !tbaa !46
  %277 = tail call noundef <8 x float> @llvm.fma.v8f32(<8 x float> %274, <8 x float> %276, <8 x float> %270)
  %278 = add nuw nsw i64 %269, 8
  %279 = add i64 %272, 1
  %280 = icmp eq i64 %279, %134
  br i1 %280, label %281, label %268, !llvm.loop !48

281:                                              ; preds = %262, %268, %189
  %282 = phi i64 [ 0, %189 ], [ %263, %262 ], [ %269, %268 ]
  %283 = phi <8 x float> [ zeroinitializer, %189 ], [ %264, %262 ], [ %277, %268 ]
  %284 = shufflevector <8 x float> %283, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %285 = shufflevector <8 x float> %283, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %286 = fadd <4 x float> %284, %285
  %287 = tail call noundef <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %286, <4 x float> %286)
  %288 = tail call noundef <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %287, <4 x float> %287)
  %289 = extractelement <4 x float> %288, i64 0
  %290 = icmp slt i64 %282, %106
  br i1 %290, label %291, label %359

291:                                              ; preds = %281
  %292 = sub i64 %106, %282
  %293 = and i64 %292, 3
  %294 = icmp eq i64 %293, 0
  br i1 %294, label %307, label %295

295:                                              ; preds = %291, %295
  %296 = phi float [ %303, %295 ], [ %289, %291 ]
  %297 = phi i64 [ %304, %295 ], [ %282, %291 ]
  %298 = phi i64 [ %305, %295 ], [ 0, %291 ]
  %299 = getelementptr inbounds nuw float, ptr %121, i64 %297
  %300 = load float, ptr %299, align 4, !tbaa !15
  %301 = getelementptr inbounds nuw float, ptr %183, i64 %297
  %302 = load float, ptr %301, align 4, !tbaa !15
  %303 = tail call float @llvm.fmuladd.f32(float %300, float %302, float %296)
  %304 = add nuw nsw i64 %297, 1
  %305 = add i64 %298, 1
  %306 = icmp eq i64 %305, %293
  br i1 %306, label %307, label %295, !llvm.loop !49

307:                                              ; preds = %295, %291
  %308 = phi float [ poison, %291 ], [ %303, %295 ]
  %309 = phi float [ %289, %291 ], [ %303, %295 ]
  %310 = phi i64 [ %282, %291 ], [ %304, %295 ]
  %311 = sub i64 %282, %106
  %312 = icmp ugt i64 %311, -4
  br i1 %312, label %359, label %313

313:                                              ; preds = %307, %313
  %314 = phi float [ %338, %313 ], [ %309, %307 ]
  %315 = phi i64 [ %339, %313 ], [ %310, %307 ]
  %316 = getelementptr inbounds nuw float, ptr %121, i64 %315
  %317 = load float, ptr %316, align 4, !tbaa !15
  %318 = getelementptr inbounds nuw float, ptr %183, i64 %315
  %319 = load float, ptr %318, align 4, !tbaa !15
  %320 = tail call float @llvm.fmuladd.f32(float %317, float %319, float %314)
  %321 = add nuw nsw i64 %315, 1
  %322 = getelementptr inbounds nuw float, ptr %121, i64 %321
  %323 = load float, ptr %322, align 4, !tbaa !15
  %324 = getelementptr inbounds nuw float, ptr %183, i64 %321
  %325 = load float, ptr %324, align 4, !tbaa !15
  %326 = tail call float @llvm.fmuladd.f32(float %323, float %325, float %320)
  %327 = add nuw nsw i64 %315, 2
  %328 = getelementptr inbounds nuw float, ptr %121, i64 %327
  %329 = load float, ptr %328, align 4, !tbaa !15
  %330 = getelementptr inbounds nuw float, ptr %183, i64 %327
  %331 = load float, ptr %330, align 4, !tbaa !15
  %332 = tail call float @llvm.fmuladd.f32(float %329, float %331, float %326)
  %333 = add nuw nsw i64 %315, 3
  %334 = getelementptr inbounds nuw float, ptr %121, i64 %333
  %335 = load float, ptr %334, align 4, !tbaa !15
  %336 = getelementptr inbounds nuw float, ptr %183, i64 %333
  %337 = load float, ptr %336, align 4, !tbaa !15
  %338 = tail call float @llvm.fmuladd.f32(float %335, float %337, float %332)
  %339 = add nuw nsw i64 %315, 4
  %340 = icmp eq i64 %339, %106
  br i1 %340, label %359, label %313, !llvm.loop !50

341:                                              ; preds = %193, %192
  %342 = phi float [ poison, %192 ], [ %227, %193 ]
  %343 = phi i64 [ 0, %192 ], [ %228, %193 ]
  %344 = phi float [ 0.000000e+00, %192 ], [ %227, %193 ]
  br i1 %141, label %359, label %345

345:                                              ; preds = %341, %345
  %346 = phi i64 [ %356, %345 ], [ %343, %341 ]
  %347 = phi float [ %355, %345 ], [ %344, %341 ]
  %348 = phi i64 [ %357, %345 ], [ 0, %341 ]
  %349 = mul nsw i64 %346, %184
  %350 = getelementptr inbounds float, ptr %121, i64 %349
  %351 = load float, ptr %350, align 4, !tbaa !15
  %352 = mul nsw i64 %346, %185
  %353 = getelementptr inbounds float, ptr %183, i64 %352
  %354 = load float, ptr %353, align 4, !tbaa !15
  %355 = tail call float @llvm.fmuladd.f32(float %351, float %354, float %347)
  %356 = add nuw nsw i64 %346, 1
  %357 = add i64 %348, 1
  %358 = icmp eq i64 %357, %138
  br i1 %358, label %359, label %345, !llvm.loop !51

359:                                              ; preds = %307, %313, %341, %345, %281, %191
  %360 = phi float [ 0.000000e+00, %191 ], [ %289, %281 ], [ %342, %341 ], [ %355, %345 ], [ %308, %307 ], [ %338, %313 ]
  %361 = load float, ptr %68, align 8, !tbaa !39
  %362 = fmul float %360, %361
  %363 = sub nsw i64 %167, %156
  %364 = getelementptr inbounds float, ptr %32, i64 %363
  store float %362, ptr %364, align 4, !tbaa !15
  %365 = fcmp olt float %168, %362
  %366 = select i1 %365, float %362, float %168
  br label %367

367:                                              ; preds = %359, %170
  %368 = phi float [ %168, %170 ], [ %366, %359 ]
  %369 = add nsw i64 %167, 1
  %370 = icmp eq i64 %369, %158
  br i1 %370, label %162, label %166, !llvm.loop !52

371:                                              ; preds = %162
  %372 = fcmp olt float %157, %163
  %373 = select i1 %372, float %163, float %157
  %374 = tail call float @llvm.fabs.f32(float %157)
  %375 = fcmp ueq float %374, 0x7FF0000000000000
  br i1 %375, label %379, label %376

376:                                              ; preds = %371
  %377 = fsub float %157, %373
  %378 = tail call noundef float @expf(float noundef %377) #15, !tbaa !32
  br label %379

379:                                              ; preds = %376, %371
  %380 = phi float [ %378, %376 ], [ 0.000000e+00, %371 ]
  %381 = fmul float %155, %380
  %382 = insertelement <8 x float> poison, float %380, i64 0
  %383 = shufflevector <8 x float> %382, <8 x float> poison, <8 x i32> zeroinitializer
  br i1 %128, label %399, label %384

384:                                              ; preds = %379
  br i1 %143, label %385, label %419

385:                                              ; preds = %419, %384
  %386 = phi i64 [ poison, %384 ], [ %432, %419 ]
  %387 = phi i64 [ 8, %384 ], [ %436, %419 ]
  %388 = phi i64 [ 0, %384 ], [ %432, %419 ]
  br i1 %145, label %399, label %389

389:                                              ; preds = %385, %389
  %390 = phi i64 [ %396, %389 ], [ %387, %385 ]
  %391 = phi i64 [ %390, %389 ], [ %388, %385 ]
  %392 = phi i64 [ %397, %389 ], [ 0, %385 ]
  %393 = getelementptr inbounds nuw float, ptr %33, i64 %391
  %394 = load <8 x float>, ptr %393, align 1, !tbaa !46
  %395 = fmul <8 x float> %383, %394
  store <8 x float> %395, ptr %393, align 1, !tbaa !46
  %396 = add nuw nsw i64 %390, 8
  %397 = add i64 %392, 1
  %398 = icmp eq i64 %397, %142
  br i1 %398, label %399, label %389, !llvm.loop !53

399:                                              ; preds = %385, %389, %379
  %400 = phi i64 [ 0, %379 ], [ %386, %385 ], [ %390, %389 ]
  %401 = icmp slt i64 %400, %106
  br i1 %401, label %402, label %439

402:                                              ; preds = %399
  %403 = sub i64 %106, %400
  %404 = and i64 %403, 3
  %405 = icmp eq i64 %404, 0
  br i1 %405, label %415, label %406

406:                                              ; preds = %402, %406
  %407 = phi i64 [ %412, %406 ], [ %400, %402 ]
  %408 = phi i64 [ %413, %406 ], [ 0, %402 ]
  %409 = getelementptr inbounds nuw float, ptr %33, i64 %407
  %410 = load float, ptr %409, align 4, !tbaa !15
  %411 = fmul float %380, %410
  store float %411, ptr %409, align 4, !tbaa !15
  %412 = add nuw nsw i64 %407, 1
  %413 = add i64 %408, 1
  %414 = icmp eq i64 %413, %404
  br i1 %414, label %415, label %406, !llvm.loop !54

415:                                              ; preds = %406, %402
  %416 = phi i64 [ %400, %402 ], [ %412, %406 ]
  %417 = sub i64 %400, %106
  %418 = icmp ugt i64 %417, -4
  br i1 %418, label %439, label %454

419:                                              ; preds = %384, %419
  %420 = phi i64 [ %436, %419 ], [ 8, %384 ]
  %421 = phi i64 [ %432, %419 ], [ 0, %384 ]
  %422 = phi i64 [ %437, %419 ], [ 0, %384 ]
  %423 = getelementptr inbounds nuw float, ptr %33, i64 %421
  %424 = load <8 x float>, ptr %423, align 1, !tbaa !46
  %425 = fmul <8 x float> %383, %424
  store <8 x float> %425, ptr %423, align 1, !tbaa !46
  %426 = getelementptr inbounds nuw float, ptr %33, i64 %420
  %427 = load <8 x float>, ptr %426, align 1, !tbaa !46
  %428 = fmul <8 x float> %383, %427
  store <8 x float> %428, ptr %426, align 1, !tbaa !46
  %429 = getelementptr inbounds float, ptr %81, i64 %420
  %430 = load <8 x float>, ptr %429, align 1, !tbaa !46
  %431 = fmul <8 x float> %383, %430
  store <8 x float> %431, ptr %429, align 1, !tbaa !46
  %432 = add nuw nsw i64 %420, 24
  %433 = getelementptr inbounds float, ptr %82, i64 %420
  %434 = load <8 x float>, ptr %433, align 1, !tbaa !46
  %435 = fmul <8 x float> %383, %434
  store <8 x float> %435, ptr %433, align 1, !tbaa !46
  %436 = add nuw nsw i64 %420, 32
  %437 = add i64 %422, 4
  %438 = icmp eq i64 %437, %144
  br i1 %438, label %385, label %419, !llvm.loop !55

439:                                              ; preds = %415, %454, %399
  %440 = icmp sle i64 %156, %123
  %441 = select i1 %161, i1 %440, i1 false
  br i1 %441, label %442, label %571

442:                                              ; preds = %439
  %443 = load ptr, ptr %69, align 8, !tbaa !42
  %444 = load i64, ptr %70, align 8, !tbaa !16
  %445 = mul nsw i64 %444, %90
  %446 = load i64, ptr %71, align 8, !tbaa !16
  %447 = mul nsw i64 %446, %101
  %448 = load i64, ptr %72, align 8, !tbaa !16
  %449 = getelementptr float, ptr %443, i64 %447
  %450 = getelementptr float, ptr %449, i64 %445
  %451 = load i64, ptr %73, align 8, !tbaa !16
  %452 = icmp ne i64 %451, 1
  %453 = or i1 %128, %452
  br label %470

454:                                              ; preds = %415, %454
  %455 = phi i64 [ %468, %454 ], [ %416, %415 ]
  %456 = getelementptr inbounds nuw float, ptr %33, i64 %455
  %457 = load float, ptr %456, align 4, !tbaa !15
  %458 = fmul float %380, %457
  store float %458, ptr %456, align 4, !tbaa !15
  %459 = getelementptr float, ptr %83, i64 %455
  %460 = load float, ptr %459, align 4, !tbaa !15
  %461 = fmul float %380, %460
  store float %461, ptr %459, align 4, !tbaa !15
  %462 = getelementptr float, ptr %84, i64 %455
  %463 = load float, ptr %462, align 4, !tbaa !15
  %464 = fmul float %380, %463
  store float %464, ptr %462, align 4, !tbaa !15
  %465 = getelementptr float, ptr %85, i64 %455
  %466 = load float, ptr %465, align 4, !tbaa !15
  %467 = fmul float %380, %466
  store float %467, ptr %465, align 4, !tbaa !15
  %468 = add nuw nsw i64 %455, 4
  %469 = icmp eq i64 %468, %106
  br i1 %469, label %439, label %454, !llvm.loop !56

470:                                              ; preds = %566, %442
  %471 = phi i64 [ %156, %442 ], [ %567, %566 ]
  %472 = phi float [ %381, %442 ], [ %478, %566 ]
  %473 = sub nsw i64 %471, %156
  %474 = getelementptr inbounds float, ptr %32, i64 %473
  %475 = load float, ptr %474, align 4, !tbaa !15
  %476 = fsub float %475, %373
  %477 = tail call noundef float @expf(float noundef %476) #15, !tbaa !32
  %478 = fadd float %472, %477
  %479 = mul nsw i64 %471, %448
  %480 = getelementptr float, ptr %450, i64 %479
  %481 = insertelement <8 x float> poison, float %477, i64 0
  %482 = shufflevector <8 x float> %481, <8 x float> poison, <8 x i32> zeroinitializer
  br i1 %453, label %530, label %483

483:                                              ; preds = %470
  br i1 %147, label %514, label %484

484:                                              ; preds = %483, %484
  %485 = phi i64 [ %511, %484 ], [ 8, %483 ]
  %486 = phi i64 [ %505, %484 ], [ 0, %483 ]
  %487 = phi i64 [ %512, %484 ], [ 0, %483 ]
  %488 = getelementptr inbounds nuw float, ptr %33, i64 %486
  %489 = getelementptr inbounds nuw float, ptr %480, i64 %486
  %490 = load <8 x float>, ptr %489, align 1, !tbaa !46
  %491 = load <8 x float>, ptr %488, align 1, !tbaa !46
  %492 = tail call noundef <8 x float> @llvm.fma.v8f32(<8 x float> %490, <8 x float> %482, <8 x float> %491)
  store <8 x float> %492, ptr %488, align 1, !tbaa !46
  %493 = add nuw nsw i64 %485, 8
  %494 = getelementptr inbounds nuw float, ptr %33, i64 %485
  %495 = getelementptr inbounds nuw float, ptr %480, i64 %485
  %496 = load <8 x float>, ptr %495, align 1, !tbaa !46
  %497 = load <8 x float>, ptr %494, align 1, !tbaa !46
  %498 = tail call noundef <8 x float> @llvm.fma.v8f32(<8 x float> %496, <8 x float> %482, <8 x float> %497)
  store <8 x float> %498, ptr %494, align 1, !tbaa !46
  %499 = add nuw nsw i64 %485, 16
  %500 = getelementptr inbounds nuw float, ptr %33, i64 %493
  %501 = getelementptr inbounds nuw float, ptr %480, i64 %493
  %502 = load <8 x float>, ptr %501, align 1, !tbaa !46
  %503 = load <8 x float>, ptr %500, align 1, !tbaa !46
  %504 = tail call noundef <8 x float> @llvm.fma.v8f32(<8 x float> %502, <8 x float> %482, <8 x float> %503)
  store <8 x float> %504, ptr %500, align 1, !tbaa !46
  %505 = add nuw nsw i64 %485, 24
  %506 = getelementptr inbounds nuw float, ptr %33, i64 %499
  %507 = getelementptr inbounds nuw float, ptr %480, i64 %499
  %508 = load <8 x float>, ptr %507, align 1, !tbaa !46
  %509 = load <8 x float>, ptr %506, align 1, !tbaa !46
  %510 = tail call noundef <8 x float> @llvm.fma.v8f32(<8 x float> %508, <8 x float> %482, <8 x float> %509)
  store <8 x float> %510, ptr %506, align 1, !tbaa !46
  %511 = add nuw nsw i64 %485, 32
  %512 = add i64 %487, 4
  %513 = icmp eq i64 %512, %148
  br i1 %513, label %514, label %484, !llvm.loop !57

514:                                              ; preds = %484, %483
  %515 = phi i64 [ poison, %483 ], [ %505, %484 ]
  %516 = phi i64 [ 8, %483 ], [ %511, %484 ]
  %517 = phi i64 [ 0, %483 ], [ %505, %484 ]
  br i1 %149, label %530, label %518

518:                                              ; preds = %514, %518
  %519 = phi i64 [ %527, %518 ], [ %516, %514 ]
  %520 = phi i64 [ %519, %518 ], [ %517, %514 ]
  %521 = phi i64 [ %528, %518 ], [ 0, %514 ]
  %522 = getelementptr inbounds nuw float, ptr %33, i64 %520
  %523 = getelementptr inbounds nuw float, ptr %480, i64 %520
  %524 = load <8 x float>, ptr %523, align 1, !tbaa !46
  %525 = load <8 x float>, ptr %522, align 1, !tbaa !46
  %526 = tail call noundef <8 x float> @llvm.fma.v8f32(<8 x float> %524, <8 x float> %482, <8 x float> %525)
  store <8 x float> %526, ptr %522, align 1, !tbaa !46
  %527 = add nuw nsw i64 %519, 8
  %528 = add i64 %521, 1
  %529 = icmp eq i64 %528, %146
  br i1 %529, label %530, label %518, !llvm.loop !58

530:                                              ; preds = %514, %518, %470
  %531 = phi i64 [ 0, %470 ], [ %515, %514 ], [ %519, %518 ]
  %532 = icmp slt i64 %531, %106
  br i1 %532, label %533, label %566

533:                                              ; preds = %530
  %534 = sub i64 %106, %531
  %535 = add i64 %531, 1
  %536 = and i64 %534, 1
  %537 = icmp eq i64 %536, 0
  br i1 %537, label %546, label %538

538:                                              ; preds = %533
  %539 = mul nsw i64 %531, %451
  %540 = getelementptr inbounds float, ptr %480, i64 %539
  %541 = load float, ptr %540, align 4, !tbaa !15
  %542 = getelementptr inbounds nuw float, ptr %33, i64 %531
  %543 = load float, ptr %542, align 4, !tbaa !15
  %544 = tail call float @llvm.fmuladd.f32(float %477, float %541, float %543)
  store float %544, ptr %542, align 4, !tbaa !15
  %545 = add nuw nsw i64 %531, 1
  br label %546

546:                                              ; preds = %538, %533
  %547 = phi i64 [ %531, %533 ], [ %545, %538 ]
  %548 = icmp eq i64 %106, %535
  br i1 %548, label %566, label %549

549:                                              ; preds = %546, %549
  %550 = phi i64 [ %564, %549 ], [ %547, %546 ]
  %551 = mul nsw i64 %550, %451
  %552 = getelementptr inbounds float, ptr %480, i64 %551
  %553 = load float, ptr %552, align 4, !tbaa !15
  %554 = getelementptr inbounds nuw float, ptr %33, i64 %550
  %555 = load float, ptr %554, align 4, !tbaa !15
  %556 = tail call float @llvm.fmuladd.f32(float %477, float %553, float %555)
  store float %556, ptr %554, align 4, !tbaa !15
  %557 = add nuw nsw i64 %550, 1
  %558 = mul nsw i64 %557, %451
  %559 = getelementptr inbounds float, ptr %480, i64 %558
  %560 = load float, ptr %559, align 4, !tbaa !15
  %561 = getelementptr inbounds nuw float, ptr %33, i64 %557
  %562 = load float, ptr %561, align 4, !tbaa !15
  %563 = tail call float @llvm.fmuladd.f32(float %477, float %560, float %562)
  store float %563, ptr %561, align 4, !tbaa !15
  %564 = add nuw nsw i64 %550, 2
  %565 = icmp eq i64 %564, %106
  br i1 %565, label %566, label %549, !llvm.loop !59

566:                                              ; preds = %546, %549, %530
  %567 = add nsw i64 %471, 1
  %568 = icmp slt i64 %567, %160
  %569 = icmp slt i64 %471, %123
  %570 = select i1 %568, i1 %569, i1 false
  br i1 %570, label %470, label %571, !llvm.loop !60

571:                                              ; preds = %566, %439, %162
  %572 = phi float [ %157, %162 ], [ %373, %439 ], [ %373, %566 ]
  %573 = phi float [ %155, %162 ], [ %381, %439 ], [ %478, %566 ]
  %574 = icmp slt i64 %159, %124
  %575 = add i64 %154, %127
  br i1 %574, label %153, label %150, !llvm.loop !61

576:                                              ; preds = %150
  %577 = load ptr, ptr %74, align 8, !tbaa !26
  %578 = load i64, ptr %75, align 8, !tbaa !16
  %579 = mul nsw i64 %578, %90
  %580 = load i64, ptr %76, align 8, !tbaa !16
  %581 = mul nsw i64 %580, %98
  %582 = load i64, ptr %77, align 8, !tbaa !16
  %583 = mul nsw i64 %582, %107
  %584 = getelementptr float, ptr %577, i64 %581
  %585 = getelementptr float, ptr %584, i64 %579
  %586 = getelementptr float, ptr %585, i64 %583
  %587 = fdiv float 1.000000e+00, %151
  %588 = insertelement <8 x float> poison, float %587, i64 0
  %589 = shufflevector <8 x float> %588, <8 x float> poison, <8 x i32> zeroinitializer
  %590 = icmp slt i64 %106, 8
  br i1 %590, label %591, label %612

591:                                              ; preds = %612, %576
  %592 = phi i64 [ %106, %576 ], [ %620, %612 ]
  %593 = phi i64 [ 0, %576 ], [ %613, %612 ]
  %594 = icmp slt i64 %593, %592
  br i1 %594, label %595, label %645

595:                                              ; preds = %591
  %596 = and i64 %592, 3
  %597 = icmp eq i64 %596, 0
  br i1 %597, label %608, label %598

598:                                              ; preds = %595, %598
  %599 = phi i64 [ %605, %598 ], [ %593, %595 ]
  %600 = phi i64 [ %606, %598 ], [ 0, %595 ]
  %601 = getelementptr inbounds nuw float, ptr %33, i64 %599
  %602 = load float, ptr %601, align 4, !tbaa !15
  %603 = fdiv float %602, %151
  %604 = getelementptr inbounds nuw float, ptr %586, i64 %599
  store float %603, ptr %604, align 4, !tbaa !15
  %605 = add nuw nsw i64 %599, 1
  %606 = add i64 %600, 1
  %607 = icmp eq i64 %606, %596
  br i1 %607, label %608, label %598, !llvm.loop !62

608:                                              ; preds = %598, %595
  %609 = phi i64 [ %593, %595 ], [ %605, %598 ]
  %610 = sub i64 %593, %592
  %611 = icmp ugt i64 %610, -4
  br i1 %611, label %645, label %622

612:                                              ; preds = %576, %612
  %613 = phi i64 [ %619, %612 ], [ 8, %576 ]
  %614 = phi i64 [ %613, %612 ], [ 0, %576 ]
  %615 = getelementptr inbounds nuw float, ptr %586, i64 %614
  %616 = getelementptr inbounds nuw float, ptr %33, i64 %614
  %617 = load <8 x float>, ptr %616, align 1, !tbaa !46
  %618 = fmul <8 x float> %589, %617
  store <8 x float> %618, ptr %615, align 1, !tbaa !46
  %619 = add nuw nsw i64 %613, 8
  %620 = load i64, ptr %14, align 8, !tbaa !14
  %621 = icmp sgt i64 %619, %620
  br i1 %621, label %591, label %612, !llvm.loop !63

622:                                              ; preds = %608, %622
  %623 = phi i64 [ %643, %622 ], [ %609, %608 ]
  %624 = getelementptr inbounds nuw float, ptr %33, i64 %623
  %625 = load float, ptr %624, align 4, !tbaa !15
  %626 = fdiv float %625, %151
  %627 = getelementptr inbounds nuw float, ptr %586, i64 %623
  store float %626, ptr %627, align 4, !tbaa !15
  %628 = add nuw nsw i64 %623, 1
  %629 = getelementptr inbounds nuw float, ptr %33, i64 %628
  %630 = load float, ptr %629, align 4, !tbaa !15
  %631 = fdiv float %630, %151
  %632 = getelementptr inbounds nuw float, ptr %586, i64 %628
  store float %631, ptr %632, align 4, !tbaa !15
  %633 = add nuw nsw i64 %623, 2
  %634 = getelementptr inbounds nuw float, ptr %33, i64 %633
  %635 = load float, ptr %634, align 4, !tbaa !15
  %636 = fdiv float %635, %151
  %637 = getelementptr inbounds nuw float, ptr %586, i64 %633
  store float %636, ptr %637, align 4, !tbaa !15
  %638 = add nuw nsw i64 %623, 3
  %639 = getelementptr inbounds nuw float, ptr %33, i64 %638
  %640 = load float, ptr %639, align 4, !tbaa !15
  %641 = fdiv float %640, %151
  %642 = getelementptr inbounds nuw float, ptr %586, i64 %638
  store float %641, ptr %642, align 4, !tbaa !15
  %643 = add nuw nsw i64 %623, 4
  %644 = icmp eq i64 %643, %592
  br i1 %644, label %645, label %622, !llvm.loop !64

645:                                              ; preds = %608, %622, %591
  %646 = add nuw nsw i64 %107, 1
  %647 = load i64, ptr %56, align 8, !tbaa !22
  %648 = icmp slt i64 %646, %647
  br i1 %648, label %105, label %649, !llvm.loop !65

649:                                              ; preds = %645
  %650 = load i64, ptr %54, align 8, !tbaa !20
  br label %651

651:                                              ; preds = %649, %95
  %652 = phi i64 [ %650, %649 ], [ %96, %95 ]
  %653 = phi i64 [ %647, %649 ], [ %97, %95 ]
  %654 = add nuw nsw i64 %98, 1
  %655 = icmp slt i64 %654, %652
  br i1 %655, label %95, label %656, !llvm.loop !66

656:                                              ; preds = %651
  %657 = load i64, ptr %50, align 8, !tbaa !19
  br label %658

658:                                              ; preds = %656, %92, %86
  %659 = phi i64 [ %657, %656 ], [ %87, %86 ], [ %87, %92 ]
  %660 = phi i64 [ %652, %656 ], [ %88, %86 ], [ %88, %92 ]
  %661 = phi i64 [ %652, %656 ], [ %89, %86 ], [ %89, %92 ]
  %662 = add nuw nsw i64 %90, 1
  %663 = icmp slt i64 %662, %659
  br i1 %663, label %86, label %664, !llvm.loop !68

664:                                              ; preds = %658, %150, %53, %44
  %665 = phi ptr [ @.str.3, %44 ], [ @.str.3, %53 ], [ @.str.2, %150 ], [ @.str.3, %658 ]
  %666 = phi i32 [ 0, %44 ], [ 0, %53 ], [ 10, %150 ], [ 0, %658 ]
  %667 = icmp eq ptr %32, null
  br i1 %667, label %672, label %668

668:                                              ; preds = %664
  %669 = ptrtoint ptr %31 to i64
  %670 = ptrtoint ptr %32 to i64
  %671 = sub i64 %669, %670
  tail call void @_ZdlPvm(ptr noundef nonnull %32, i64 noundef %671) #16
  br label %672

672:                                              ; preds = %668, %664
  %673 = insertvalue { i32, ptr } poison, i32 %666, 0
  %674 = insertvalue { i32, ptr } %673, ptr %665, 1
  br label %675

675:                                              ; preds = %672, %6, %2
  %676 = phi { i32, ptr } [ { i32 11, ptr @.str.1 }, %2 ], [ %674, %672 ], [ %7, %6 ]
  ret { i32, ptr } %676
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define internal fastcc { i32, ptr } @_ZN12_GLOBAL__N_18validateEPK23HirFusedAttentionParams(ptr noundef readonly captures(address_is_null) %0) unnamed_addr #4 {
  %2 = icmp eq ptr %0, null
  br i1 %2, label %150, label %3

3:                                                ; preds = %1
  %4 = load ptr, ptr %0, align 8, !tbaa !23
  %5 = icmp eq ptr %4, null
  br i1 %5, label %150, label %6

6:                                                ; preds = %3
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %8 = load ptr, ptr %7, align 8, !tbaa !36
  %9 = icmp eq ptr %8, null
  br i1 %9, label %150, label %10

10:                                               ; preds = %6
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %12 = load ptr, ptr %11, align 8, !tbaa !42
  %13 = icmp eq ptr %12, null
  br i1 %13, label %150, label %14

14:                                               ; preds = %10
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %16 = load ptr, ptr %15, align 8, !tbaa !26
  %17 = icmp eq ptr %16, null
  br i1 %17, label %150, label %18

18:                                               ; preds = %14
  %19 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %20 = load i64, ptr %19, align 8, !tbaa !19
  %21 = icmp slt i64 %20, 1
  br i1 %21, label %150, label %22

22:                                               ; preds = %18
  %23 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %24 = load i64, ptr %23, align 8, !tbaa !22
  %25 = icmp slt i64 %24, 1
  br i1 %25, label %150, label %26

26:                                               ; preds = %22
  %27 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %28 = load i64, ptr %27, align 8, !tbaa !25
  %29 = icmp slt i64 %28, 1
  br i1 %29, label %150, label %30

30:                                               ; preds = %26
  %31 = getelementptr inbounds nuw i8, ptr %0, i64 56
  %32 = load i64, ptr %31, align 8, !tbaa !20
  %33 = icmp slt i64 %32, 1
  br i1 %33, label %150, label %34

34:                                               ; preds = %30
  %35 = getelementptr inbounds nuw i8, ptr %0, i64 64
  %36 = load i64, ptr %35, align 8, !tbaa !18
  %37 = icmp slt i64 %36, 1
  br i1 %37, label %150, label %38

38:                                               ; preds = %34
  %39 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %40 = load i64, ptr %39, align 8, !tbaa !14
  %41 = icmp slt i64 %40, 1
  br i1 %41, label %150, label %42

42:                                               ; preds = %38
  %43 = urem i64 %32, %36
  %44 = icmp eq i64 %43, 0
  br i1 %44, label %45, label %150

45:                                               ; preds = %42
  %46 = getelementptr inbounds nuw i8, ptr %0, i64 256
  %47 = load i64, ptr %46, align 8, !tbaa !17
  %48 = srem i64 %47, %36
  %49 = icmp eq i64 %48, 0
  br i1 %49, label %50, label %150

50:                                               ; preds = %45
  %51 = getelementptr inbounds nuw i8, ptr %0, i64 248
  %52 = load i64, ptr %51, align 8, !tbaa !21
  %53 = icmp slt i64 %52, 0
  %54 = add nuw nsw i64 %52, %32
  %55 = icmp sgt i64 %54, %47
  %56 = select i1 %53, i1 true, i1 %55
  br i1 %56, label %150, label %57

57:                                               ; preds = %50
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 80
  %59 = load i64, ptr %58, align 8, !tbaa !16
  %60 = icmp sgt i64 %59, 0
  br i1 %60, label %61, label %150

61:                                               ; preds = %57
  %62 = getelementptr inbounds nuw i8, ptr %0, i64 88
  %63 = load i64, ptr %62, align 8, !tbaa !16
  %64 = icmp sgt i64 %63, 0
  br i1 %64, label %65, label %150

65:                                               ; preds = %61
  %66 = getelementptr inbounds nuw i8, ptr %0, i64 96
  %67 = load i64, ptr %66, align 8, !tbaa !16
  %68 = icmp sgt i64 %67, 0
  br i1 %68, label %69, label %150

69:                                               ; preds = %65
  %70 = getelementptr inbounds nuw i8, ptr %0, i64 104
  %71 = load i64, ptr %70, align 8, !tbaa !16
  %72 = icmp sgt i64 %71, 0
  br i1 %72, label %73, label %150

73:                                               ; preds = %69
  %74 = getelementptr inbounds nuw i8, ptr %0, i64 112
  %75 = load i64, ptr %74, align 8, !tbaa !16
  %76 = icmp sgt i64 %75, 0
  br i1 %76, label %77, label %150

77:                                               ; preds = %73
  %78 = getelementptr inbounds nuw i8, ptr %0, i64 120
  %79 = load i64, ptr %78, align 8, !tbaa !16
  %80 = icmp sgt i64 %79, 0
  br i1 %80, label %81, label %150

81:                                               ; preds = %77
  %82 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %83 = load i64, ptr %82, align 8, !tbaa !16
  %84 = icmp sgt i64 %83, 0
  br i1 %84, label %85, label %150

85:                                               ; preds = %81
  %86 = getelementptr inbounds nuw i8, ptr %0, i64 136
  %87 = load i64, ptr %86, align 8, !tbaa !16
  %88 = icmp sgt i64 %87, 0
  br i1 %88, label %89, label %150

89:                                               ; preds = %85
  %90 = getelementptr inbounds nuw i8, ptr %0, i64 144
  %91 = load i64, ptr %90, align 8, !tbaa !16
  %92 = icmp sgt i64 %91, 0
  br i1 %92, label %93, label %150

93:                                               ; preds = %89
  %94 = getelementptr inbounds nuw i8, ptr %0, i64 152
  %95 = load i64, ptr %94, align 8, !tbaa !16
  %96 = icmp sgt i64 %95, 0
  br i1 %96, label %97, label %150

97:                                               ; preds = %93
  %98 = getelementptr inbounds nuw i8, ptr %0, i64 160
  %99 = load i64, ptr %98, align 8, !tbaa !16
  %100 = icmp sgt i64 %99, 0
  br i1 %100, label %101, label %150

101:                                              ; preds = %97
  %102 = getelementptr inbounds nuw i8, ptr %0, i64 168
  %103 = load i64, ptr %102, align 8, !tbaa !16
  %104 = icmp sgt i64 %103, 0
  br i1 %104, label %105, label %150

105:                                              ; preds = %101
  %106 = getelementptr inbounds nuw i8, ptr %0, i64 176
  %107 = load i64, ptr %106, align 8, !tbaa !16
  %108 = icmp sgt i64 %107, 0
  br i1 %108, label %109, label %150

109:                                              ; preds = %105
  %110 = getelementptr inbounds nuw i8, ptr %0, i64 184
  %111 = load i64, ptr %110, align 8, !tbaa !16
  %112 = icmp sgt i64 %111, 0
  br i1 %112, label %113, label %150

113:                                              ; preds = %109
  %114 = getelementptr inbounds nuw i8, ptr %0, i64 192
  %115 = load i64, ptr %114, align 8, !tbaa !16
  %116 = icmp sgt i64 %115, 0
  br i1 %116, label %117, label %150

117:                                              ; preds = %113
  %118 = getelementptr inbounds nuw i8, ptr %0, i64 200
  %119 = load i64, ptr %118, align 8, !tbaa !16
  %120 = icmp sgt i64 %119, 0
  br i1 %120, label %121, label %150

121:                                              ; preds = %117
  %122 = getelementptr inbounds nuw i8, ptr %0, i64 208
  %123 = load float, ptr %122, align 8, !tbaa !39
  %124 = tail call i1 @llvm.is.fpclass.f32(float %123, i32 384)
  br i1 %124, label %125, label %150

125:                                              ; preds = %121
  %126 = getelementptr inbounds nuw i8, ptr %0, i64 212
  %127 = load i32, ptr %126, align 4, !tbaa !69
  %128 = icmp eq i32 %127, 1
  br i1 %128, label %129, label %150

129:                                              ; preds = %125
  %130 = getelementptr inbounds nuw i8, ptr %0, i64 224
  %131 = load i64, ptr %130, align 8, !tbaa !70
  %132 = icmp slt i64 %131, 1
  br i1 %132, label %150, label %133

133:                                              ; preds = %129
  %134 = getelementptr inbounds nuw i8, ptr %0, i64 232
  %135 = load i64, ptr %134, align 8, !tbaa !5
  %136 = icmp slt i64 %135, 1
  br i1 %136, label %150, label %137

137:                                              ; preds = %133
  %138 = getelementptr inbounds nuw i8, ptr %0, i64 240
  %139 = load i64, ptr %138, align 8, !tbaa !71
  %140 = icmp slt i64 %139, 1
  br i1 %140, label %150, label %141

141:                                              ; preds = %137
  %142 = getelementptr inbounds nuw i8, ptr %0, i64 216
  %143 = load i64, ptr %142, align 8, !tbaa !24
  %144 = icmp slt i64 %143, 0
  %145 = add nuw nsw i64 %143, %24
  %146 = icmp samesign ugt i64 %145, %28
  %147 = select i1 %144, i1 true, i1 %146
  %148 = select i1 %147, i32 9, i32 0
  %149 = select i1 %147, ptr @.str.12, ptr @.str.3
  br label %150

150:                                              ; preds = %141, %105, %109, %113, %89, %93, %97, %73, %77, %81, %57, %61, %65, %129, %133, %137, %125, %121, %69, %85, %101, %117, %50, %42, %45, %18, %22, %26, %30, %34, %38, %1, %3, %6, %10, %14
  %151 = phi i32 [ 1, %14 ], [ 1, %10 ], [ 1, %6 ], [ 1, %3 ], [ 1, %1 ], [ 2, %38 ], [ 2, %34 ], [ 2, %30 ], [ 2, %26 ], [ 2, %22 ], [ 2, %18 ], [ 3, %45 ], [ 3, %42 ], [ 4, %50 ], [ 5, %117 ], [ 5, %101 ], [ 5, %85 ], [ 5, %69 ], [ 6, %121 ], [ 7, %125 ], [ 8, %137 ], [ 8, %133 ], [ 8, %129 ], [ 5, %65 ], [ 5, %61 ], [ 5, %57 ], [ 5, %81 ], [ 5, %77 ], [ 5, %73 ], [ 5, %97 ], [ 5, %93 ], [ 5, %89 ], [ 5, %113 ], [ 5, %109 ], [ 5, %105 ], [ %148, %141 ]
  %152 = phi ptr [ @.str.4, %14 ], [ @.str.4, %10 ], [ @.str.4, %6 ], [ @.str.4, %3 ], [ @.str.4, %1 ], [ @.str.5, %38 ], [ @.str.5, %34 ], [ @.str.5, %30 ], [ @.str.5, %26 ], [ @.str.5, %22 ], [ @.str.5, %18 ], [ @.str.6, %45 ], [ @.str.6, %42 ], [ @.str.7, %50 ], [ @.str.8, %117 ], [ @.str.8, %101 ], [ @.str.8, %85 ], [ @.str.8, %69 ], [ @.str.9, %121 ], [ @.str.10, %125 ], [ @.str.11, %137 ], [ @.str.11, %133 ], [ @.str.11, %129 ], [ @.str.8, %65 ], [ @.str.8, %61 ], [ @.str.8, %57 ], [ @.str.8, %81 ], [ @.str.8, %77 ], [ @.str.8, %73 ], [ @.str.8, %97 ], [ @.str.8, %93 ], [ @.str.8, %89 ], [ @.str.8, %113 ], [ @.str.8, %109 ], [ @.str.8, %105 ], [ %149, %141 ]
  %153 = insertvalue { i32, ptr } poison, i32 %151, 0
  %154 = insertvalue { i32, ptr } %153, ptr %152, 1
  ret { i32, ptr } %154
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #5

; Function Attrs: cold noreturn
declare void @_ZSt20__throw_length_errorPKc(ptr noundef) local_unnamed_addr #6

; Function Attrs: nobuiltin allocsize(0)
declare noalias noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #7

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #8

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.is.fpclass.f32(float, i32 immarg) #5

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare float @expf(float noundef) local_unnamed_addr #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float>, <4 x float>) #10

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>) #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #11

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #11

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #11

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #2 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #3 = { mustprogress uwtable "min-legal-vector-width"="256" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #4 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #5 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { cold noreturn "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #7 = { nobuiltin allocsize(0) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #8 = { nobuiltin nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #9 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #10 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #11 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #12 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #13 = { cold noreturn }
attributes #14 = { builtin allocsize(0) }
attributes #15 = { nounwind }
attributes #16 = { builtin nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 21.1.8 (6ubuntu1)"}
!5 = !{!6, !11, i64 232}
!6 = !{!"_ZTS23HirFusedAttentionParams", !7, i64 0, !7, i64 8, !7, i64 16, !7, i64 24, !11, i64 32, !11, i64 40, !11, i64 48, !11, i64 56, !11, i64 64, !11, i64 72, !9, i64 80, !9, i64 112, !9, i64 144, !9, i64 176, !12, i64 208, !13, i64 212, !11, i64 216, !11, i64 224, !11, i64 232, !11, i64 240, !11, i64 248, !11, i64 256}
!7 = !{!"p1 float", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!"long", !9, i64 0}
!12 = !{!"float", !9, i64 0}
!13 = !{!"int", !9, i64 0}
!14 = !{!6, !11, i64 72}
!15 = !{!12, !12, i64 0}
!16 = !{!11, !11, i64 0}
!17 = !{!6, !11, i64 256}
!18 = !{!6, !11, i64 64}
!19 = !{!6, !11, i64 32}
!20 = !{!6, !11, i64 56}
!21 = !{!6, !11, i64 248}
!22 = !{!6, !11, i64 40}
!23 = !{!6, !7, i64 0}
!24 = !{!6, !11, i64 216}
!25 = !{!6, !11, i64 48}
!26 = !{!6, !7, i64 24}
!27 = distinct !{!27, !28}
!28 = !{!"llvm.loop.unroll.disable"}
!29 = distinct !{!29, !30}
!30 = !{!"llvm.loop.mustprogress"}
!31 = distinct !{!31, !30}
!32 = !{!13, !13, i64 0}
!33 = distinct !{!33, !30}
!34 = distinct !{!34, !30}
!35 = distinct !{!35, !30}
!36 = !{!6, !7, i64 8}
!37 = distinct !{!37, !30}
!38 = distinct !{!38, !28}
!39 = !{!6, !12, i64 208}
!40 = distinct !{!40, !30}
!41 = distinct !{!41, !28}
!42 = !{!6, !7, i64 16}
!43 = distinct !{!43, !30}
!44 = distinct !{!44, !30}
!45 = distinct !{!45, !30}
!46 = !{!9, !9, i64 0}
!47 = distinct !{!47, !30}
!48 = distinct !{!48, !28}
!49 = distinct !{!49, !28}
!50 = distinct !{!50, !30}
!51 = distinct !{!51, !28}
!52 = distinct !{!52, !30}
!53 = distinct !{!53, !28}
!54 = distinct !{!54, !28}
!55 = distinct !{!55, !30}
!56 = distinct !{!56, !30}
!57 = distinct !{!57, !30}
!58 = distinct !{!58, !28}
!59 = distinct !{!59, !30}
!60 = distinct !{!60, !30}
!61 = distinct !{!61, !30}
!62 = distinct !{!62, !28}
!63 = distinct !{!63, !30}
!64 = distinct !{!64, !30}
!65 = distinct !{!65, !30}
!66 = distinct !{!66, !30, !67}
!67 = !{!"llvm.loop.unswitch.partial.disable"}
!68 = distinct !{!68, !30, !67}
!69 = !{!6, !13, i64 212}
!70 = !{!6, !11, i64 224}
!71 = !{!6, !11, i64 240}
