	.file	"fused_online_attention.cpp"
	.text
	.globl	hir_fused_attention_artifact_version # -- Begin function hir_fused_attention_artifact_version
	.p2align	4
	.type	hir_fused_attention_artifact_version,@function
hir_fused_attention_artifact_version:   # @hir_fused_attention_artifact_version
	.cfi_startproc
# %bb.0:
	leaq	.L.str(%rip), %rax
	retq
.Lfunc_end0:
	.size	hir_fused_attention_artifact_version, .Lfunc_end0-hir_fused_attention_artifact_version
	.cfi_endproc
                                        # -- End function
	.globl	hir_fused_attention_has_avx2    # -- Begin function hir_fused_attention_has_avx2
	.p2align	4
	.type	hir_fused_attention_has_avx2,@function
hir_fused_attention_has_avx2:           # @hir_fused_attention_has_avx2
	.cfi_startproc
# %bb.0:
	movl	__cpu_model+12(%rip), %eax
	movl	%eax, %ecx
	shrl	$10, %ecx
	shrl	$14, %eax
	andl	%ecx, %eax
	andl	$1, %eax
	retq
.Lfunc_end1:
	.size	hir_fused_attention_has_avx2, .Lfunc_end1-hir_fused_attention_has_avx2
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function hir_fused_online_attention_scalar
.LCPI2_0:
	.long	0xff800000                      # float -Inf
	.text
	.globl	hir_fused_online_attention_scalar
	.p2align	4
	.type	hir_fused_online_attention_scalar,@function
hir_fused_online_attention_scalar:      # @hir_fused_online_attention_scalar
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$360, %rsp                      # imm = 0x168
	.cfi_def_cfa_offset 416
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbx
	movq	%rdi, %r12
	callq	_ZN12_GLOBAL__N_18validateEPK23HirFusedAttentionParams
	testl	%eax, %eax
	jne	.LBB2_91
# %bb.1:
	movq	72(%r12), %r15
	movq	232(%r12), %r14
	leaq	(%r15,%r14), %r13
	movq	%r13, %rax
	shrq	$61, %rax
	jne	.LBB2_92
# %bb.2:
	testq	%r13, %r13
	movq	%r12, 48(%rsp)                  # 8-byte Spill
	je	.LBB2_9
# %bb.3:
	leaq	(,%r13,4), %r12
	movq	%r12, %rdi
	callq	_Znwm@PLT
	movq	%rax, %rbp
	leaq	(%rax,%r13,4), %rax
	movq	%rax, 184(%rsp)                 # 8-byte Spill
	movl	$0, (%rbp)
	cmpq	$1, %r13
	je	.LBB2_5
# %bb.4:
	movq	%rbp, %rdi
	addq	$4, %rdi
	addq	$-4, %r12
	xorl	%esi, %esi
	movq	%r12, %rdx
	callq	memset@PLT
.LBB2_5:
	movq	%rbp, %r13
	movq	48(%rsp), %r12                  # 8-byte Reload
	testq	%rbx, %rbx
	je	.LBB2_7
.LBB2_6:
	leaq	(,%r14,4), %rax
	shlq	$2, %r15
	cmpq	%r15, %rax
	movq	%r15, %rcx
	cmovaq	%rax, %rcx
	leaq	(%r15,%rax), %rdx
	movq	%rdx, (%rbx)
	movq	%rcx, 8(%rbx)
	movq	$1, 16(%rbx)
	movq	$0, 24(%rbx)
	movq	%rax, 32(%rbx)
	movq	%r15, 40(%rbx)
	movq	%rdx, 48(%rbx)
.LBB2_7:
	movq	64(%r12), %rcx
	movq	256(%r12), %rax
	movq	%rax, %rdx
	orq	%rcx, %rdx
	shrq	$32, %rdx
	je	.LBB2_10
# %bb.8:
	cqto
	idivq	%rcx
	movq	%rax, %rdx
	jmp	.LBB2_11
.LBB2_9:
	movq	$0, 184(%rsp)                   # 8-byte Folded Spill
	xorl	%r13d, %r13d
	testq	%rbx, %rbx
	jne	.LBB2_6
	jmp	.LBB2_7
.LBB2_10:
                                        # kill: def $eax killed $eax killed $rax
	xorl	%edx, %edx
	divl	%ecx
	movl	%eax, %edx
.LBB2_11:
	movq	32(%r12), %rsi
	xorl	%eax, %eax
	leaq	.L.str.3(%rip), %rcx
	movq	%rsi, 264(%rsp)                 # 8-byte Spill
	testq	%rsi, %rsi
	jle	.LBB2_83
# %bb.12:
	movq	56(%r12), %rsi
	movq	%rsi, 256(%rsp)                 # 8-byte Spill
	testq	%rsi, %rsi
	jle	.LBB2_83
# %bb.13:
	movq	40(%r12), %rsi
	movq	%rsi, 296(%rsp)                 # 8-byte Spill
	testq	%rsi, %rsi
	jle	.LBB2_83
# %bb.14:
	leaq	(,%r14,4), %rdi
	addq	%r13, %rdi
	movq	72(%r12), %r10
	leaq	(,%r10,4), %rax
	movq	%rax, 192(%rsp)                 # 8-byte Spill
	movq	48(%r12), %rax
	testq	%rax, %rax
	jle	.LBB2_87
# %bb.15:
	movq	%rax, 208(%rsp)                 # 8-byte Spill
	movq	248(%r12), %rcx
	movq	(%r12), %r15
	movq	80(%r12), %rax
	movq	88(%r12), %rsi
	movq	96(%r12), %r8
	movq	216(%r12), %r9
	movq	%r9, 288(%rsp)                  # 8-byte Spill
	movq	232(%r12), %r9
	movl	%r10d, %ebp
	andl	$3, %ebp
	movabsq	$9223372036854775804, %r11      # imm = 0x7FFFFFFFFFFFFFFC
	movq	%r10, %rbx
	andq	%r11, %rbx
	movq	%rbx, 144(%rsp)                 # 8-byte Spill
	leaq	12(%rdi), %rbx
	movq	%rbx, 200(%rsp)                 # 8-byte Spill
	movq	%r15, %rbx
	orq	$2, %r11
	andq	%r10, %r11
	movq	%r11, 328(%rsp)                 # 8-byte Spill
	shlq	$2, %rax
	movq	%rax, 216(%rsp)                 # 8-byte Spill
	shlq	$2, %rsi
	movq	%rsi, 224(%rsp)                 # 8-byte Spill
	shlq	$2, %r8
	movq	%r8, 272(%rsp)                  # 8-byte Spill
	movq	%r9, 64(%rsp)                   # 8-byte Spill
	leaq	(,%r9,4), %rax
	movq	%rax, 312(%rsp)                 # 8-byte Spill
	leaq	4(%r13,%r14,4), %r14
	movq	$0, 32(%rsp)                    # 8-byte Folded Spill
	movq	$0, 136(%rsp)                   # 8-byte Folded Spill
	movq	%rdx, 240(%rsp)                 # 8-byte Spill
	movq	%r10, 16(%rsp)                  # 8-byte Spill
	movq	%rbp, 152(%rsp)                 # 8-byte Spill
	movq	%r13, 352(%rsp)                 # 8-byte Spill
	movq	%rdi, 40(%rsp)                  # 8-byte Spill
	movq	%rcx, 232(%rsp)                 # 8-byte Spill
.LBB2_16:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_17 Depth 2
                                        #       Child Loop BB2_22 Depth 3
                                        #         Child Loop BB2_27 Depth 4
                                        #           Child Loop BB2_31 Depth 5
                                        #             Child Loop BB2_37 Depth 6
                                        #             Child Loop BB2_40 Depth 6
                                        #           Child Loop BB2_48 Depth 5
                                        #           Child Loop BB2_51 Depth 5
                                        #           Child Loop BB2_65 Depth 5
                                        #           Child Loop BB2_55 Depth 5
                                        #             Child Loop BB2_58 Depth 6
                                        #         Child Loop BB2_76 Depth 4
                                        #         Child Loop BB2_79 Depth 4
	movq	$0, 112(%rsp)                   # 8-byte Folded Spill
	movq	%rbx, 104(%rsp)                 # 8-byte Spill
	xorl	%eax, %eax
	movq	%rbx, 280(%rsp)                 # 8-byte Spill
.LBB2_17:                               #   Parent Loop BB2_16 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB2_22 Depth 3
                                        #         Child Loop BB2_27 Depth 4
                                        #           Child Loop BB2_31 Depth 5
                                        #             Child Loop BB2_37 Depth 6
                                        #             Child Loop BB2_40 Depth 6
                                        #           Child Loop BB2_48 Depth 5
                                        #           Child Loop BB2_51 Depth 5
                                        #           Child Loop BB2_65 Depth 5
                                        #           Child Loop BB2_55 Depth 5
                                        #             Child Loop BB2_58 Depth 6
                                        #         Child Loop BB2_76 Depth 4
                                        #         Child Loop BB2_79 Depth 4
	movq	%rax, 248(%rsp)                 # 8-byte Spill
	addq	%rcx, %rax
	movq	%rax, %rcx
	orq	%rdx, %rcx
	shrq	$32, %rcx
	movq	%rdx, %rcx
	je	.LBB2_19
# %bb.18:                               #   in Loop: Header=BB2_17 Depth=2
	cqto
	idivq	%rcx
	jmp	.LBB2_20
.LBB2_19:                               #   in Loop: Header=BB2_17 Depth=2
                                        # kill: def $eax killed $eax killed $rax
	xorl	%edx, %edx
	divl	%ecx
                                        # kill: def $eax killed $eax def $rax
.LBB2_20:                               #   in Loop: Header=BB2_17 Depth=2
	movq	%rax, 304(%rsp)                 # 8-byte Spill
	leaq	(,%rax,4), %rax
	movq	%rax, 160(%rsp)                 # 8-byte Spill
	movq	$0, 120(%rsp)                   # 8-byte Folded Spill
	movq	104(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	movq	$0, 128(%rsp)                   # 8-byte Folded Spill
	movq	40(%rsp), %rdi                  # 8-byte Reload
	movq	64(%rsp), %rcx                  # 8-byte Reload
	jmp	.LBB2_22
	.p2align	4
.LBB2_21:                               #   in Loop: Header=BB2_22 Depth=3
	movq	128(%rsp), %rcx                 # 8-byte Reload
	incq	%rcx
	movq	80(%rsp), %rax                  # 8-byte Reload
	addq	272(%rsp), %rax                 # 8-byte Folded Reload
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	addq	$4, 120(%rsp)                   # 8-byte Folded Spill
	movq	%rcx, 128(%rsp)                 # 8-byte Spill
	cmpq	296(%rsp), %rcx                 # 8-byte Folded Reload
	movq	40(%rsp), %rdi                  # 8-byte Reload
	movq	64(%rsp), %rcx                  # 8-byte Reload
	movq	280(%rsp), %rbx                 # 8-byte Reload
	je	.LBB2_80
.LBB2_22:                               #   Parent Loop BB2_16 Depth=1
                                        #     Parent Loop BB2_17 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB2_27 Depth 4
                                        #           Child Loop BB2_31 Depth 5
                                        #             Child Loop BB2_37 Depth 6
                                        #             Child Loop BB2_40 Depth 6
                                        #           Child Loop BB2_48 Depth 5
                                        #           Child Loop BB2_51 Depth 5
                                        #           Child Loop BB2_65 Depth 5
                                        #           Child Loop BB2_55 Depth 5
                                        #             Child Loop BB2_58 Depth 6
                                        #         Child Loop BB2_76 Depth 4
                                        #         Child Loop BB2_79 Depth 4
	testq	%r10, %r10
	je	.LBB2_24
# %bb.23:                               #   in Loop: Header=BB2_22 Depth=3
	xorl	%esi, %esi
	movq	192(%rsp), %rdx                 # 8-byte Reload
	callq	memset@PLT
	movq	64(%rsp), %rcx                  # 8-byte Reload
	movq	16(%rsp), %r10                  # 8-byte Reload
.LBB2_24:                               #   in Loop: Header=BB2_22 Depth=3
	movq	288(%rsp), %rax                 # 8-byte Reload
	movq	128(%rsp), %rdx                 # 8-byte Reload
	addq	%rdx, %rax
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	vxorps	%xmm4, %xmm4, %xmm4
	movq	$0, 72(%rsp)                    # 8-byte Folded Spill
	movq	%rcx, %rdx
	xorl	%ebx, %ebx
	vmovd	.LCPI2_0(%rip), %xmm1           # xmm1 = [-Inf,0.0E+0,0.0E+0,0.0E+0]
	movq	208(%rsp), %rax                 # 8-byte Reload
	jmp	.LBB2_27
	.p2align	4
.LBB2_25:                               #   in Loop: Header=BB2_27 Depth=4
	vmovd	8(%rsp), %xmm1                  # 4-byte Folded Reload
                                        # xmm1 = mem[0],zero,zero,zero
.LBB2_26:                               #   in Loop: Header=BB2_27 Depth=4
	movq	64(%rsp), %rcx                  # 8-byte Reload
	movq	320(%rsp), %rdx                 # 8-byte Reload
	addq	%rcx, %rdx
	movq	72(%rsp), %rax                  # 8-byte Reload
	addq	312(%rsp), %rax                 # 8-byte Folded Reload
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	movq	208(%rsp), %rax                 # 8-byte Reload
	cmpq	%rax, %rbx
	jge	.LBB2_71
.LBB2_27:                               #   Parent Loop BB2_16 Depth=1
                                        #     Parent Loop BB2_17 Depth=2
                                        #       Parent Loop BB2_22 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB2_31 Depth 5
                                        #             Child Loop BB2_37 Depth 6
                                        #             Child Loop BB2_40 Depth 6
                                        #           Child Loop BB2_48 Depth 5
                                        #           Child Loop BB2_51 Depth 5
                                        #           Child Loop BB2_65 Depth 5
                                        #           Child Loop BB2_55 Depth 5
                                        #             Child Loop BB2_58 Depth 6
	cmpq	%rdx, %rax
	movq	%rdx, 320(%rsp)                 # 8-byte Spill
	movq	%rdx, %rsi
	cmovlq	%rax, %rsi
	movq	%rbx, %r11
	addq	%rcx, %rbx
	cmpq	%rbx, %rax
	movq	%rbx, %rcx
	cmovlq	%rax, %rcx
	vmovss	.LCPI2_0(%rip), %xmm0           # xmm0 = [-Inf,0.0E+0,0.0E+0,0.0E+0]
	movq	%rcx, 176(%rsp)                 # 8-byte Spill
	cmpq	%rcx, %r11
	movq	%rbx, 56(%rsp)                  # 8-byte Spill
	movq	%r11, 88(%rsp)                  # 8-byte Spill
	jge	.LBB2_42
# %bb.28:                               #   in Loop: Header=BB2_27 Depth=4
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movq	%r11, %rdx
	vmovss	.LCPI2_0(%rip), %xmm0           # xmm0 = [-Inf,0.0E+0,0.0E+0,0.0E+0]
	movq	%rsi, 24(%rsp)                  # 8-byte Spill
	jmp	.LBB2_31
	.p2align	4
.LBB2_29:                               #   in Loop: Header=BB2_31 Depth=5
	movq	%rdx, %rax
	subq	%r11, %rax
	movl	$-8388608, (%r13,%rax,4)        # imm = 0xFF800000
	incq	%rdx
	addq	$4, %rcx
	cmpq	%rsi, %rdx
	je	.LBB2_42
.LBB2_31:                               #   Parent Loop BB2_16 Depth=1
                                        #     Parent Loop BB2_17 Depth=2
                                        #       Parent Loop BB2_22 Depth=3
                                        #         Parent Loop BB2_27 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB2_37 Depth 6
                                        #             Child Loop BB2_40 Depth 6
	cmpq	96(%rsp), %rdx                  # 8-byte Folded Reload
	jg	.LBB2_29
# %bb.32:                               #   in Loop: Header=BB2_31 Depth=5
	testq	%r10, %r10
	jle	.LBB2_35
# %bb.33:                               #   in Loop: Header=BB2_31 Depth=5
	movq	8(%r12), %rsi
	movq	104(%r12), %rdi
	movq	112(%r12), %r11
	movq	120(%r12), %rbx
	movq	128(%r12), %rax
	movq	136(%r12), %r8
	cmpq	$4, %r10
	movq	%r11, 8(%rsp)                   # 8-byte Spill
	movq	%rbx, 168(%rsp)                 # 8-byte Spill
	jae	.LBB2_36
# %bb.34:                               #   in Loop: Header=BB2_31 Depth=5
	vxorps	%xmm2, %xmm2, %xmm2
	xorl	%r9d, %r9d
	jmp	.LBB2_38
.LBB2_35:                               #   in Loop: Header=BB2_31 Depth=5
	vxorps	%xmm2, %xmm2, %xmm2
	jmp	.LBB2_41
.LBB2_36:                               #   in Loop: Header=BB2_31 Depth=5
	leaq	(,%rdi,4), %r9
	leaq	(%r9,%r9,2), %r15
	movq	%rdi, %r12
	shlq	$4, %r12
	leaq	(,%r8,4), %r9
	leaq	(%r9,%r9,2), %rbp
	movq	160(%rsp), %r9                  # 8-byte Reload
	imulq	%rbx, %r9
	movq	32(%rsp), %rbx                  # 8-byte Reload
	imulq	%r11, %rbx
	addq	%r9, %rbx
	movq	%rax, %r10
	imulq	%rcx, %r10
	addq	%rsi, %r10
	addq	%rbx, %r10
	movq	%r8, %r13
	shlq	$4, %r13
	vxorps	%xmm2, %xmm2, %xmm2
	movq	80(%rsp), %rbx                  # 8-byte Reload
	xorl	%r9d, %r9d
	movq	144(%rsp), %r11                 # 8-byte Reload
	.p2align	4
.LBB2_37:                               #   Parent Loop BB2_16 Depth=1
                                        #     Parent Loop BB2_17 Depth=2
                                        #       Parent Loop BB2_22 Depth=3
                                        #         Parent Loop BB2_27 Depth=4
                                        #           Parent Loop BB2_31 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	vmovss	(%rbx), %xmm3                   # xmm3 = mem[0],zero,zero,zero
	vfmadd132ss	(%r10), %xmm2, %xmm3    # xmm3 = (xmm3 * mem) + xmm2
	vmovss	(%rbx,%rdi,4), %xmm2            # xmm2 = mem[0],zero,zero,zero
	vfmadd132ss	(%r10,%r8,4), %xmm3, %xmm2 # xmm2 = (xmm2 * mem) + xmm3
	vmovss	(%rbx,%rdi,8), %xmm3            # xmm3 = mem[0],zero,zero,zero
	vfmadd132ss	(%r10,%r8,8), %xmm2, %xmm3 # xmm3 = (xmm3 * mem) + xmm2
	vmovss	(%rbx,%r15), %xmm2              # xmm2 = mem[0],zero,zero,zero
	vfmadd132ss	(%r10,%rbp), %xmm3, %xmm2 # xmm2 = (xmm2 * mem) + xmm3
	addq	$4, %r9
	addq	%r12, %rbx
	addq	%r13, %r10
	cmpq	%r9, %r11
	jne	.LBB2_37
.LBB2_38:                               #   in Loop: Header=BB2_31 Depth=5
	movq	152(%rsp), %rbp                 # 8-byte Reload
	testq	%rbp, %rbp
	movq	352(%rsp), %r13                 # 8-byte Reload
	movq	48(%rsp), %r12                  # 8-byte Reload
	movq	56(%rsp), %rbx                  # 8-byte Reload
	movq	88(%rsp), %r11                  # 8-byte Reload
	je	.LBB2_41
# %bb.39:                               #   in Loop: Header=BB2_31 Depth=5
	shlq	$2, %r9
	movq	%r9, %r10
	imulq	%r8, %r10
	movq	%r12, %r11
	movq	%rbx, %r12
	movq	%r13, %rbx
	movq	%rbp, %r13
	movq	168(%rsp), %rbp                 # 8-byte Reload
	imulq	160(%rsp), %rbp                 # 8-byte Folded Reload
	movq	8(%rsp), %r15                   # 8-byte Reload
	imulq	32(%rsp), %r15                  # 8-byte Folded Reload
	addq	%rbp, %r15
	movq	%r13, %rbp
	movq	%rbx, %r13
	movq	%r12, %rbx
	movq	%r11, %r12
	movq	88(%rsp), %r11                  # 8-byte Reload
	imulq	%rcx, %rax
	addq	%r15, %rax
	addq	%r10, %rax
	addq	%rax, %rsi
	shlq	$2, %r8
	imulq	%rdi, %r9
	addq	80(%rsp), %r9                   # 8-byte Folded Reload
	shlq	$2, %rdi
	movq	%rbp, %rax
	.p2align	4
.LBB2_40:                               #   Parent Loop BB2_16 Depth=1
                                        #     Parent Loop BB2_17 Depth=2
                                        #       Parent Loop BB2_22 Depth=3
                                        #         Parent Loop BB2_27 Depth=4
                                        #           Parent Loop BB2_31 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	vmovss	(%r9), %xmm3                    # xmm3 = mem[0],zero,zero,zero
	vfmadd231ss	(%rsi), %xmm3, %xmm2    # xmm2 = (xmm3 * mem) + xmm2
	addq	%r8, %rsi
	addq	%rdi, %r9
	decq	%rax
	jne	.LBB2_40
.LBB2_41:                               #   in Loop: Header=BB2_31 Depth=5
	vmulss	208(%r12), %xmm2, %xmm2
	movq	%rdx, %rax
	subq	%r11, %rax
	vmovss	%xmm2, (%r13,%rax,4)
	vmaxss	%xmm0, %xmm2, %xmm0
	movq	16(%rsp), %r10                  # 8-byte Reload
	movq	24(%rsp), %rsi                  # 8-byte Reload
	incq	%rdx
	addq	$4, %rcx
	cmpq	%rsi, %rdx
	jne	.LBB2_31
.LBB2_42:                               #   in Loop: Header=BB2_27 Depth=4
	vmovd	%xmm0, %eax
	andl	$2147483647, %eax               # imm = 0x7FFFFFFF
	cmpl	$2139095039, %eax               # imm = 0x7F7FFFFF
	jg	.LBB2_26
# %bb.43:                               #   in Loop: Header=BB2_27 Depth=4
	vmovd	%xmm1, %eax
	vmaxss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, 8(%rsp)                  # 4-byte Spill
	andl	$2147483647, %eax               # imm = 0x7FFFFFFF
	vxorps	%xmm0, %xmm0, %xmm0
	cmpl	$2139095039, %eax               # imm = 0x7F7FFFFF
	jg	.LBB2_45
# %bb.44:                               #   in Loop: Header=BB2_27 Depth=4
	vsubss	8(%rsp), %xmm1, %xmm0           # 4-byte Folded Reload
	vmovss	%xmm4, 24(%rsp)                 # 4-byte Spill
	callq	expf@PLT
	movq	88(%rsp), %r11                  # 8-byte Reload
	vmovss	24(%rsp), %xmm4                 # 4-byte Reload
                                        # xmm4 = mem[0],zero,zero,zero
	movq	16(%rsp), %r10                  # 8-byte Reload
.LBB2_45:                               #   in Loop: Header=BB2_27 Depth=4
	testq	%r10, %r10
	movq	200(%rsp), %rdx                 # 8-byte Reload
	jle	.LBB2_52
# %bb.46:                               #   in Loop: Header=BB2_27 Depth=4
	xorl	%eax, %eax
	cmpq	$4, %r10
	jb	.LBB2_49
# %bb.47:                               #   in Loop: Header=BB2_27 Depth=4
	movq	144(%rsp), %rcx                 # 8-byte Reload
	.p2align	4
.LBB2_48:                               #   Parent Loop BB2_16 Depth=1
                                        #     Parent Loop BB2_17 Depth=2
                                        #       Parent Loop BB2_22 Depth=3
                                        #         Parent Loop BB2_27 Depth=4
                                        # =>        This Inner Loop Header: Depth=5
	vmulss	-12(%rdx,%rax,4), %xmm0, %xmm1
	vmovss	%xmm1, -12(%rdx,%rax,4)
	vmulss	-8(%rdx,%rax,4), %xmm0, %xmm1
	vmovss	%xmm1, -8(%rdx,%rax,4)
	vmulss	-4(%rdx,%rax,4), %xmm0, %xmm1
	vmovss	%xmm1, -4(%rdx,%rax,4)
	vmulss	(%rdx,%rax,4), %xmm0, %xmm1
	vmovss	%xmm1, (%rdx,%rax,4)
	addq	$4, %rax
	cmpq	%rax, %rcx
	jne	.LBB2_48
.LBB2_49:                               #   in Loop: Header=BB2_27 Depth=4
	testb	$3, %r10b
	je	.LBB2_52
# %bb.50:                               #   in Loop: Header=BB2_27 Depth=4
	movq	40(%rsp), %rcx                  # 8-byte Reload
	leaq	(%rcx,%rax,4), %rax
	xorl	%ecx, %ecx
	.p2align	4
.LBB2_51:                               #   Parent Loop BB2_16 Depth=1
                                        #     Parent Loop BB2_17 Depth=2
                                        #       Parent Loop BB2_22 Depth=3
                                        #         Parent Loop BB2_27 Depth=4
                                        # =>        This Inner Loop Header: Depth=5
	vmulss	(%rax,%rcx,4), %xmm0, %xmm1
	vmovss	%xmm1, (%rax,%rcx,4)
	incq	%rcx
	cmpq	%rcx, %rbp
	jne	.LBB2_51
.LBB2_52:                               #   in Loop: Header=BB2_27 Depth=4
	vmulss	%xmm0, %xmm4, %xmm4
	cmpq	176(%rsp), %r11                 # 8-byte Folded Reload
	setl	%al
	cmpq	96(%rsp), %r11                  # 8-byte Folded Reload
	setle	%cl
	andb	%al, %cl
	cmpb	$1, %cl
	jne	.LBB2_25
# %bb.53:                               #   in Loop: Header=BB2_27 Depth=4
	testq	%r10, %r10
	jle	.LBB2_64
# %bb.54:                               #   in Loop: Header=BB2_27 Depth=4
	movq	16(%r12), %rax
	movq	144(%r12), %rcx
	movq	%rcx, %rdx
	imulq	136(%rsp), %rdx                 # 8-byte Folded Reload
	movq	152(%r12), %rsi
	movq	%rsi, %rdi
	imulq	304(%rsp), %rdi                 # 8-byte Folded Reload
	movq	160(%r12), %r8
	leaq	(%rax,%rdi,4), %rdi
	leaq	(%rdi,%rdx,4), %rdx
	movq	%rdx, 336(%rsp)                 # 8-byte Spill
	movq	168(%r12), %rbx
	imulq	160(%rsp), %rsi                 # 8-byte Folded Reload
	imulq	32(%rsp), %rcx                  # 8-byte Folded Reload
	addq	%rsi, %rcx
	movq	%r8, %rbp
	imulq	72(%rsp), %rbp                  # 8-byte Folded Reload
	addq	%rax, %rbp
	addq	%rcx, %rbp
	movq	%r8, 344(%rsp)                  # 8-byte Spill
	leaq	(,%r8,4), %rax
	movq	%rax, 168(%rsp)                 # 8-byte Spill
	leaq	(,%rbx,8), %r15
	movq	%r11, %r12
	.p2align	4
.LBB2_55:                               #   Parent Loop BB2_16 Depth=1
                                        #     Parent Loop BB2_17 Depth=2
                                        #       Parent Loop BB2_22 Depth=3
                                        #         Parent Loop BB2_27 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB2_58 Depth 6
	vmovss	%xmm4, 24(%rsp)                 # 4-byte Spill
	movq	%r12, %rax
	subq	%r11, %rax
	vmovss	(%r13,%rax,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	vsubss	8(%rsp), %xmm0, %xmm0           # 4-byte Folded Reload
	callq	expf@PLT
	movq	16(%rsp), %r10                  # 8-byte Reload
	cmpq	$1, %r10
	jne	.LBB2_57
# %bb.56:                               #   in Loop: Header=BB2_55 Depth=5
	xorl	%eax, %eax
	vmovss	24(%rsp), %xmm4                 # 4-byte Reload
                                        # xmm4 = mem[0],zero,zero,zero
	jmp	.LBB2_59
	.p2align	4
.LBB2_57:                               #   in Loop: Header=BB2_55 Depth=5
	movq	%rbp, %rcx
	xorl	%eax, %eax
	movq	328(%rsp), %rdx                 # 8-byte Reload
	vmovss	24(%rsp), %xmm4                 # 4-byte Reload
                                        # xmm4 = mem[0],zero,zero,zero
	.p2align	4
.LBB2_58:                               #   Parent Loop BB2_16 Depth=1
                                        #     Parent Loop BB2_17 Depth=2
                                        #       Parent Loop BB2_22 Depth=3
                                        #         Parent Loop BB2_27 Depth=4
                                        #           Parent Loop BB2_55 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	vmovss	(%rcx), %xmm1                   # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	-4(%r14,%rax,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, -4(%r14,%rax,4)
	vmovss	(%rcx,%rbx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	(%r14,%rax,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, (%r14,%rax,4)
	addq	$2, %rax
	addq	%r15, %rcx
	cmpq	%rax, %rdx
	jne	.LBB2_58
.LBB2_59:                               #   in Loop: Header=BB2_55 Depth=5
	testb	$1, %r10b
	movq	88(%rsp), %r11                  # 8-byte Reload
	je	.LBB2_61
# %bb.60:                               #   in Loop: Header=BB2_55 Depth=5
	movq	%r12, %rcx
	imulq	344(%rsp), %rcx                 # 8-byte Folded Reload
	movq	336(%rsp), %rdx                 # 8-byte Reload
	leaq	(%rdx,%rcx,4), %rcx
	movq	%rax, %rdx
	imulq	%rbx, %rdx
	vmovss	(%rcx,%rdx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	movq	40(%rsp), %rcx                  # 8-byte Reload
	vfmadd213ss	(%rcx,%rax,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, (%rcx,%rax,4)
.LBB2_61:                               #   in Loop: Header=BB2_55 Depth=5
	vaddss	%xmm0, %xmm4, %xmm4
	leaq	1(%r12), %rax
	cmpq	176(%rsp), %rax                 # 8-byte Folded Reload
	jge	.LBB2_68
# %bb.62:                               #   in Loop: Header=BB2_55 Depth=5
	addq	168(%rsp), %rbp                 # 8-byte Folded Reload
	cmpq	96(%rsp), %r12                  # 8-byte Folded Reload
	movq	%rax, %r12
	jl	.LBB2_55
# %bb.63:                               #   in Loop: Header=BB2_27 Depth=4
	vmovd	8(%rsp), %xmm1                  # 4-byte Folded Reload
                                        # xmm1 = mem[0],zero,zero,zero
	movq	48(%rsp), %r12                  # 8-byte Reload
	movq	16(%rsp), %r10                  # 8-byte Reload
	jmp	.LBB2_69
.LBB2_64:                               #   in Loop: Header=BB2_27 Depth=4
	movq	%r13, %rbx
	.p2align	4
.LBB2_65:                               #   Parent Loop BB2_16 Depth=1
                                        #     Parent Loop BB2_17 Depth=2
                                        #       Parent Loop BB2_22 Depth=3
                                        #         Parent Loop BB2_27 Depth=4
                                        # =>        This Inner Loop Header: Depth=5
	vmovss	%xmm4, 24(%rsp)                 # 4-byte Spill
	vmovss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	vsubss	8(%rsp), %xmm0, %xmm0           # 4-byte Folded Reload
	movq	%r11, %r15
	callq	expf@PLT
	vmovss	24(%rsp), %xmm4                 # 4-byte Reload
                                        # xmm4 = mem[0],zero,zero,zero
	vaddss	%xmm0, %xmm4, %xmm4
	leaq	1(%r15), %rax
	cmpq	176(%rsp), %rax                 # 8-byte Folded Reload
	jge	.LBB2_67
# %bb.66:                               #   in Loop: Header=BB2_65 Depth=5
	addq	$4, %rbx
	cmpq	96(%rsp), %r15                  # 8-byte Folded Reload
	movq	%rax, %r11
	jl	.LBB2_65
.LBB2_67:                               #   in Loop: Header=BB2_27 Depth=4
	vmovd	8(%rsp), %xmm1                  # 4-byte Folded Reload
                                        # xmm1 = mem[0],zero,zero,zero
	movq	16(%rsp), %r10                  # 8-byte Reload
	jmp	.LBB2_70
.LBB2_68:                               #   in Loop: Header=BB2_27 Depth=4
	vmovd	8(%rsp), %xmm1                  # 4-byte Folded Reload
                                        # xmm1 = mem[0],zero,zero,zero
	movq	48(%rsp), %r12                  # 8-byte Reload
.LBB2_69:                               #   in Loop: Header=BB2_27 Depth=4
	movq	152(%rsp), %rbp                 # 8-byte Reload
.LBB2_70:                               #   in Loop: Header=BB2_27 Depth=4
	movq	56(%rsp), %rbx                  # 8-byte Reload
	jmp	.LBB2_26
	.p2align	4
.LBB2_71:                               #   in Loop: Header=BB2_22 Depth=3
	vmovd	%xmm4, %eax
	testl	%eax, %eax
	setns	%cl
	movl	%eax, %edx
	andl	$2147483647, %edx               # imm = 0x7FFFFFFF
	addl	$-8388608, %edx                 # imm = 0xFF800000
	cmpl	$2130706432, %edx               # imm = 0x7F000000
	setb	%dl
	andb	%cl, %dl
	decl	%eax
	cmpl	$8388607, %eax                  # imm = 0x7FFFFF
	setb	%al
	orb	%dl, %al
	je	.LBB2_85
# %bb.72:                               #   in Loop: Header=BB2_22 Depth=3
	testq	%r10, %r10
	jle	.LBB2_21
# %bb.73:                               #   in Loop: Header=BB2_22 Depth=3
	movq	24(%r12), %rax
	movq	176(%r12), %r15
	movq	184(%r12), %rsi
	movq	192(%r12), %rdx
	movq	200(%r12), %rcx
	cmpq	$4, %r10
	jae	.LBB2_75
# %bb.74:                               #   in Loop: Header=BB2_22 Depth=3
	xorl	%r8d, %r8d
	jmp	.LBB2_77
	.p2align	4
.LBB2_75:                               #   in Loop: Header=BB2_22 Depth=3
	leaq	(,%rcx,4), %r8
	leaq	(%r8,%r8,2), %r9
	movq	32(%rsp), %r8                   # 8-byte Reload
	imulq	%r15, %r8
	movq	112(%rsp), %r11                 # 8-byte Reload
	imulq	%rsi, %r11
	addq	%r8, %r11
	movq	%rdx, %r10
	imulq	120(%rsp), %r10                 # 8-byte Folded Reload
	addq	%rax, %r10
	addq	%r11, %r10
	movq	%rcx, %r11
	shlq	$4, %r11
	xorl	%r8d, %r8d
	movq	144(%rsp), %rbx                 # 8-byte Reload
	movq	200(%rsp), %rdi                 # 8-byte Reload
	.p2align	4
.LBB2_76:                               #   Parent Loop BB2_16 Depth=1
                                        #     Parent Loop BB2_17 Depth=2
                                        #       Parent Loop BB2_22 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vmovss	-12(%rdi,%r8,4), %xmm0          # xmm0 = mem[0],zero,zero,zero
	vdivss	%xmm4, %xmm0, %xmm0
	vmovss	%xmm0, (%r10)
	vmovss	-8(%rdi,%r8,4), %xmm0           # xmm0 = mem[0],zero,zero,zero
	vdivss	%xmm4, %xmm0, %xmm0
	vmovss	%xmm0, (%r10,%rcx,4)
	vmovss	-4(%rdi,%r8,4), %xmm0           # xmm0 = mem[0],zero,zero,zero
	vdivss	%xmm4, %xmm0, %xmm0
	vmovss	%xmm0, (%r10,%rcx,8)
	vmovss	(%rdi,%r8,4), %xmm0             # xmm0 = mem[0],zero,zero,zero
	vdivss	%xmm4, %xmm0, %xmm0
	vmovss	%xmm0, (%r10,%r9)
	addq	$4, %r8
	addq	%r11, %r10
	cmpq	%r8, %rbx
	jne	.LBB2_76
.LBB2_77:                               #   in Loop: Header=BB2_22 Depth=3
	movq	16(%rsp), %r10                  # 8-byte Reload
	testb	$3, %r10b
	je	.LBB2_21
# %bb.78:                               #   in Loop: Header=BB2_22 Depth=3
	leaq	(,%r8,4), %r9
	imulq	%rcx, %r9
	imulq	32(%rsp), %r15                  # 8-byte Folded Reload
	imulq	112(%rsp), %rsi                 # 8-byte Folded Reload
	addq	%r15, %rsi
	imulq	120(%rsp), %rdx                 # 8-byte Folded Reload
	addq	%rsi, %rdx
	addq	%r9, %rdx
	addq	%rdx, %rax
	shlq	$2, %rcx
	movq	40(%rsp), %rdx                  # 8-byte Reload
	leaq	(%rdx,%r8,4), %rdx
	xorl	%esi, %esi
	.p2align	4
.LBB2_79:                               #   Parent Loop BB2_16 Depth=1
                                        #     Parent Loop BB2_17 Depth=2
                                        #       Parent Loop BB2_22 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vmovss	(%rdx,%rsi,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	vdivss	%xmm4, %xmm0, %xmm0
	vmovss	%xmm0, (%rax)
	incq	%rsi
	addq	%rcx, %rax
	cmpq	%rsi, %rbp
	jne	.LBB2_79
	jmp	.LBB2_21
.LBB2_80:                               #   in Loop: Header=BB2_17 Depth=2
	movq	248(%rsp), %rcx                 # 8-byte Reload
	incq	%rcx
	movq	104(%rsp), %rax                 # 8-byte Reload
	addq	224(%rsp), %rax                 # 8-byte Folded Reload
	movq	%rax, 104(%rsp)                 # 8-byte Spill
	movq	%rcx, %rax
	addq	$4, 112(%rsp)                   # 8-byte Folded Spill
	cmpq	256(%rsp), %rcx                 # 8-byte Folded Reload
	movq	240(%rsp), %rdx                 # 8-byte Reload
	movq	232(%rsp), %rcx                 # 8-byte Reload
	jne	.LBB2_17
# %bb.81:                               #   in Loop: Header=BB2_16 Depth=1
	movq	136(%rsp), %r8                  # 8-byte Reload
	incq	%r8
	addq	216(%rsp), %rbx                 # 8-byte Folded Reload
	addq	$4, 32(%rsp)                    # 8-byte Folded Spill
	xorl	%eax, %eax
	movq	%r8, 136(%rsp)                  # 8-byte Spill
	cmpq	264(%rsp), %r8                  # 8-byte Folded Reload
	jne	.LBB2_16
# %bb.82:
	leaq	.L.str.3(%rip), %rdx
	jmp	.LBB2_89
.LBB2_83:
	movq	%rcx, %rdx
.LBB2_89:
	testq	%r13, %r13
	je	.LBB2_91
# %bb.90:
	movq	184(%rsp), %rsi                 # 8-byte Reload
	subq	%r13, %rsi
	movq	%r13, %rdi
	movl	%eax, %ebx
	movq	%rdx, %r14
	callq	_ZdlPvm@PLT
	movq	%r14, %rdx
	movl	%ebx, %eax
.LBB2_91:
	addq	$360, %rsp                      # imm = 0x168
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB2_85:
	.cfi_def_cfa_offset 416
	movl	$10, %eax
	leaq	.L.str.2(%rip), %rdx
	jmp	.LBB2_89
.LBB2_87:
	movl	$10, %eax
	leaq	.L.str.2(%rip), %rdx
	testq	%r10, %r10
	je	.LBB2_89
# %bb.88:
	xorl	%esi, %esi
	movq	%rdx, %rbx
	movq	192(%rsp), %rdx                 # 8-byte Reload
	callq	memset@PLT
	movq	%rbx, %rdx
	movl	$10, %eax
	jmp	.LBB2_89
.LBB2_92:
	leaq	.L.str.13(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Lfunc_end2:
	.size	hir_fused_online_attention_scalar, .Lfunc_end2-hir_fused_online_attention_scalar
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function hir_fused_online_attention_avx2
.LCPI3_0:
	.long	0xff800000                      # float -Inf
.LCPI3_1:
	.long	0x3f800000                      # float 1
	.text
	.globl	hir_fused_online_attention_avx2
	.p2align	4
	.type	hir_fused_online_attention_avx2,@function
hir_fused_online_attention_avx2:        # @hir_fused_online_attention_avx2
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$312, %rsp                      # imm = 0x138
	.cfi_def_cfa_offset 368
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	__cpu_model+12(%rip), %eax
	notl	%eax
	testl	$17408, %eax                    # imm = 0x4400
	jne	.LBB3_1
# %bb.3:
	movq	%rsi, %r12
	movq	%rdi, %rbx
	callq	_ZN12_GLOBAL__N_18validateEPK23HirFusedAttentionParams
	testl	%eax, %eax
	jne	.LBB3_2
# %bb.4:
	movq	72(%rbx), %r15
	movq	232(%rbx), %rsi
	leaq	(%r15,%rsi), %r13
	movq	%r13, %rax
	shrq	$61, %rax
	jne	.LBB3_136
# %bb.5:
	movq	%rbx, %r11
	testq	%r13, %r13
	movq	%rbx, 80(%rsp)                  # 8-byte Spill
	je	.LBB3_6
# %bb.7:
	movq	%rsi, 48(%rsp)                  # 8-byte Spill
	leaq	(,%r13,4), %r14
	movq	%r14, %rdi
	callq	_Znwm@PLT
	movq	%rax, %rbx
	leaq	(%rax,%r13,4), %rbp
	movl	$0, (%rax)
	cmpq	$1, %r13
	je	.LBB3_9
# %bb.8:
	movq	%rbx, %rdi
	addq	$4, %rdi
	addq	$-4, %r14
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset@PLT
.LBB3_9:
	movq	80(%rsp), %r11                  # 8-byte Reload
	movq	%rbx, %r13
	movq	48(%rsp), %rsi                  # 8-byte Reload
	testq	%r12, %r12
	je	.LBB3_12
.LBB3_11:
	leaq	(,%rsi,4), %rax
	shlq	$2, %r15
	cmpq	%r15, %rax
	movq	%r15, %rcx
	cmovaq	%rax, %rcx
	leaq	(%r15,%rax), %rdx
	movq	%rdx, (%r12)
	movq	%rcx, 8(%r12)
	movq	$1, 16(%r12)
	movq	$0, 24(%r12)
	movq	%rax, 32(%r12)
	movq	%r15, 40(%r12)
	movq	%rdx, 48(%r12)
.LBB3_12:
	movq	64(%r11), %rcx
	movq	256(%r11), %rax
	movq	%rax, %rdx
	orq	%rcx, %rdx
	shrq	$32, %rdx
	je	.LBB3_13
# %bb.14:
	cqto
	idivq	%rcx
	movq	%rax, %rdi
	jmp	.LBB3_15
.LBB3_1:
	leaq	.L.str.1(%rip), %rdx
	movl	$11, %eax
.LBB3_2:
	addq	$312, %rsp                      # imm = 0x138
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.LBB3_6:
	.cfi_def_cfa_offset 368
	xorl	%ebp, %ebp
	xorl	%r13d, %r13d
	testq	%r12, %r12
	jne	.LBB3_11
	jmp	.LBB3_12
.LBB3_13:
                                        # kill: def $eax killed $eax killed $rax
	xorl	%edx, %edx
	divl	%ecx
	movl	%eax, %edi
.LBB3_15:
	movq	32(%r11), %rdx
	xorl	%eax, %eax
	leaq	.L.str.3(%rip), %rcx
	testq	%rdx, %rdx
	jle	.LBB3_16
# %bb.17:
	movq	56(%r11), %rcx
	testq	%rcx, %rcx
	jle	.LBB3_18
# %bb.19:
	movq	%rbp, 184(%rsp)                 # 8-byte Spill
	leaq	(,%rsi,4), %rbp
	addq	%r13, %rbp
	leaq	12(%rbp), %rax
	movq	%rax, 248(%rsp)                 # 8-byte Spill
	movq	$0, 8(%rsp)                     # 8-byte Folded Spill
	movq	$0, 72(%rsp)                    # 8-byte Folded Spill
	movq	%r13, 120(%rsp)                 # 8-byte Spill
	movq	%rdi, 240(%rsp)                 # 8-byte Spill
	jmp	.LBB3_20
.LBB3_16:
	movq	%rcx, %rdx
	testq	%r13, %r13
	jne	.LBB3_35
	jmp	.LBB3_2
.LBB3_18:
	leaq	.L.str.3(%rip), %rdx
	testq	%r13, %r13
	jne	.LBB3_35
	jmp	.LBB3_2
.LBB3_131:                              #   in Loop: Header=BB3_20 Depth=1
	movq	32(%r11), %rdx
.LBB3_132:                              #   in Loop: Header=BB3_20 Depth=1
	movq	72(%rsp), %rsi                  # 8-byte Reload
	incq	%rsi
	addq	$4, 8(%rsp)                     # 8-byte Folded Spill
	movq	%rsi, 72(%rsp)                  # 8-byte Spill
	cmpq	%rdx, %rsi
	jge	.LBB3_133
.LBB3_20:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_23 Depth 2
                                        #       Child Loop BB3_28 Depth 3
                                        #         Child Loop BB3_37 Depth 4
                                        #           Child Loop BB3_39 Depth 5
                                        #             Child Loop BB3_58 Depth 6
                                        #             Child Loop BB3_62 Depth 6
                                        #             Child Loop BB3_68 Depth 6
                                        #             Child Loop BB3_72 Depth 6
                                        #             Child Loop BB3_53 Depth 6
                                        #             Child Loop BB3_56 Depth 6
                                        #           Child Loop BB3_94 Depth 5
                                        #           Child Loop BB3_84 Depth 5
                                        #           Child Loop BB3_89 Depth 5
                                        #           Child Loop BB3_95 Depth 5
                                        #           Child Loop BB3_97 Depth 5
                                        #             Child Loop BB3_102 Depth 6
                                        #             Child Loop BB3_106 Depth 6
                                        #             Child Loop BB3_113 Depth 6
                                        #         Child Loop BB3_135 Depth 4
                                        #         Child Loop BB3_123 Depth 4
                                        #         Child Loop BB3_127 Depth 4
	testq	%rcx, %rcx
	jle	.LBB3_132
# %bb.21:                               #   in Loop: Header=BB3_20 Depth=1
	movq	40(%r11), %rsi
	testq	%rsi, %rsi
	jle	.LBB3_132
# %bb.22:                               #   in Loop: Header=BB3_20 Depth=1
	movq	$0, 64(%rsp)                    # 8-byte Folded Spill
	xorl	%edx, %edx
	jmp	.LBB3_23
.LBB3_129:                              #   in Loop: Header=BB3_23 Depth=2
	movq	56(%r11), %rcx
	movq	240(%rsp), %rdi                 # 8-byte Reload
.LBB3_130:                              #   in Loop: Header=BB3_23 Depth=2
	movq	192(%rsp), %rdx                 # 8-byte Reload
	incq	%rdx
	addq	$4, 64(%rsp)                    # 8-byte Folded Spill
	cmpq	%rcx, %rdx
	jge	.LBB3_131
.LBB3_23:                               #   Parent Loop BB3_20 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB3_28 Depth 3
                                        #         Child Loop BB3_37 Depth 4
                                        #           Child Loop BB3_39 Depth 5
                                        #             Child Loop BB3_58 Depth 6
                                        #             Child Loop BB3_62 Depth 6
                                        #             Child Loop BB3_68 Depth 6
                                        #             Child Loop BB3_72 Depth 6
                                        #             Child Loop BB3_53 Depth 6
                                        #             Child Loop BB3_56 Depth 6
                                        #           Child Loop BB3_94 Depth 5
                                        #           Child Loop BB3_84 Depth 5
                                        #           Child Loop BB3_89 Depth 5
                                        #           Child Loop BB3_95 Depth 5
                                        #           Child Loop BB3_97 Depth 5
                                        #             Child Loop BB3_102 Depth 6
                                        #             Child Loop BB3_106 Depth 6
                                        #             Child Loop BB3_113 Depth 6
                                        #         Child Loop BB3_135 Depth 4
                                        #         Child Loop BB3_123 Depth 4
                                        #         Child Loop BB3_127 Depth 4
	movq	248(%r11), %rax
	movq	%rdx, 192(%rsp)                 # 8-byte Spill
	addq	%rdx, %rax
	movq	%rax, %rdx
	orq	%rdi, %rdx
	shrq	$32, %rdx
	je	.LBB3_24
# %bb.25:                               #   in Loop: Header=BB3_23 Depth=2
	cqto
	idivq	%rdi
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	testq	%rsi, %rsi
	jg	.LBB3_27
	jmp	.LBB3_130
.LBB3_24:                               #   in Loop: Header=BB3_23 Depth=2
                                        # kill: def $eax killed $eax killed $rax
	xorl	%edx, %edx
	divl	%edi
                                        # kill: def $eax killed $eax def $rax
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	testq	%rsi, %rsi
	jle	.LBB3_130
.LBB3_27:                               #   in Loop: Header=BB3_23 Depth=2
	movq	72(%r11), %r9
	movq	88(%rsp), %rax                  # 8-byte Reload
	leaq	(,%rax,4), %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	xorl	%r15d, %r15d
	movq	$0, 128(%rsp)                   # 8-byte Folded Spill
	jmp	.LBB3_28
	.p2align	4
.LBB3_128:                              #   in Loop: Header=BB3_28 Depth=3
	movq	128(%rsp), %rcx                 # 8-byte Reload
	incq	%rcx
	movq	40(%r11), %rsi
	addq	$4, %r15
	movq	%rcx, 128(%rsp)                 # 8-byte Spill
	cmpq	%rsi, %rcx
	jge	.LBB3_129
.LBB3_28:                               #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB3_37 Depth 4
                                        #           Child Loop BB3_39 Depth 5
                                        #             Child Loop BB3_58 Depth 6
                                        #             Child Loop BB3_62 Depth 6
                                        #             Child Loop BB3_68 Depth 6
                                        #             Child Loop BB3_72 Depth 6
                                        #             Child Loop BB3_53 Depth 6
                                        #             Child Loop BB3_56 Depth 6
                                        #           Child Loop BB3_94 Depth 5
                                        #           Child Loop BB3_84 Depth 5
                                        #           Child Loop BB3_89 Depth 5
                                        #           Child Loop BB3_95 Depth 5
                                        #           Child Loop BB3_97 Depth 5
                                        #             Child Loop BB3_102 Depth 6
                                        #             Child Loop BB3_106 Depth 6
                                        #             Child Loop BB3_113 Depth 6
                                        #         Child Loop BB3_135 Depth 4
                                        #         Child Loop BB3_123 Depth 4
                                        #         Child Loop BB3_127 Depth 4
	testq	%r9, %r9
	movq	%r9, 32(%rsp)                   # 8-byte Spill
	je	.LBB3_30
# %bb.29:                               #   in Loop: Header=BB3_28 Depth=3
	leaq	(,%r9,4), %rdx
	movq	%rbp, %rdi
	xorl	%esi, %esi
	movq	%r11, %rbx
	vzeroupper
	callq	memset@PLT
	movq	%rbx, %r11
	movq	32(%rsp), %r9                   # 8-byte Reload
.LBB3_30:                               #   in Loop: Header=BB3_28 Depth=3
	movq	48(%r11), %rax
	testq	%rax, %rax
	movq	%r15, 200(%rsp)                 # 8-byte Spill
	jle	.LBB3_31
# %bb.36:                               #   in Loop: Header=BB3_28 Depth=3
	movq	%rax, 208(%rsp)                 # 8-byte Spill
	movq	(%r11), %rax
	movq	80(%r11), %rdx
	movq	%rdx, %rsi
	imulq	72(%rsp), %rsi                  # 8-byte Folded Reload
	movq	88(%r11), %rcx
	movq	%rcx, %rdi
	imulq	192(%rsp), %rdi                 # 8-byte Folded Reload
	movq	96(%r11), %r10
	movq	%r10, %r8
	movq	128(%rsp), %r9                  # 8-byte Reload
	imulq	%r9, %r8
	leaq	(%rax,%rdi,4), %rdi
	leaq	(%rdi,%rsi,4), %rsi
	leaq	(%rsi,%r8,4), %rsi
	movq	%rsi, 280(%rsp)                 # 8-byte Spill
	movq	216(%r11), %r12
	addq	%r9, %r12
	movq	32(%rsp), %r9                   # 8-byte Reload
	movq	232(%r11), %rdi
	leaq	-8(%r9), %r8
	movq	%r8, 152(%rsp)                  # 8-byte Spill
	shrq	$3, %r8
	incq	%r8
	movl	%r8d, %esi
	andl	$3, %esi
	movq	%rsi, 104(%rsp)                 # 8-byte Spill
	movq	%r8, 224(%rsp)                  # 8-byte Spill
	andq	$-4, %r8
	movq	%r8, 136(%rsp)                  # 8-byte Spill
	movl	%r9d, %esi
	andl	$3, %esi
	movq	%rsi, 216(%rsp)                 # 8-byte Spill
	movq	%r9, %r8
	movabsq	$9223372036854775804, %rsi      # imm = 0x7FFFFFFFFFFFFFFC
	andq	%rsi, %r8
	movq	%r8, 272(%rsp)                  # 8-byte Spill
	leaq	(,%rdi,4), %rsi
	movq	%rsi, 256(%rsp)                 # 8-byte Spill
	imulq	8(%rsp), %rdx                   # 8-byte Folded Reload
	imulq	64(%rsp), %rcx                  # 8-byte Folded Reload
	addq	%rdx, %rcx
	imulq	%r15, %r10
	addq	%rcx, %r10
	movq	%rdi, %rcx
	leaq	(%rax,%r10), %r15
	addq	$96, %r15
	addq	%rax, %r10
	movq	%r10, 144(%rsp)                 # 8-byte Spill
	movq	208(%rsp), %rax                 # 8-byte Reload
	vxorps	%xmm6, %xmm6, %xmm6
	movq	$0, 96(%rsp)                    # 8-byte Folded Spill
	movq	%rdi, %rdx
	xorl	%ebx, %ebx
	vmovd	.LCPI3_0(%rip), %xmm1           # xmm1 = [-Inf,0.0E+0,0.0E+0,0.0E+0]
	movq	%r12, 112(%rsp)                 # 8-byte Spill
	movq	%rdi, 264(%rsp)                 # 8-byte Spill
	jmp	.LBB3_37
	.p2align	4
.LBB3_42:                               #   in Loop: Header=BB3_37 Depth=4
	movq	24(%rsp), %rdx                  # 8-byte Reload
.LBB3_117:                              #   in Loop: Header=BB3_37 Depth=4
	movq	264(%rsp), %rcx                 # 8-byte Reload
	addq	%rcx, %rdx
	movq	96(%rsp), %rax                  # 8-byte Reload
	addq	256(%rsp), %rax                 # 8-byte Folded Reload
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	movq	208(%rsp), %rax                 # 8-byte Reload
	cmpq	%rax, %rbx
	jge	.LBB3_32
.LBB3_37:                               #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB3_39 Depth 5
                                        #             Child Loop BB3_58 Depth 6
                                        #             Child Loop BB3_62 Depth 6
                                        #             Child Loop BB3_68 Depth 6
                                        #             Child Loop BB3_72 Depth 6
                                        #             Child Loop BB3_53 Depth 6
                                        #             Child Loop BB3_56 Depth 6
                                        #           Child Loop BB3_94 Depth 5
                                        #           Child Loop BB3_84 Depth 5
                                        #           Child Loop BB3_89 Depth 5
                                        #           Child Loop BB3_95 Depth 5
                                        #           Child Loop BB3_97 Depth 5
                                        #             Child Loop BB3_102 Depth 6
                                        #             Child Loop BB3_106 Depth 6
                                        #             Child Loop BB3_113 Depth 6
	cmpq	%rdx, %rax
	movq	%rdx, 24(%rsp)                  # 8-byte Spill
	movq	%rdx, %rsi
	cmovlq	%rax, %rsi
	movq	%rbx, %r10
	addq	%rcx, %rbx
	cmpq	%rbx, %rax
	movq	%rbx, %rcx
	cmovlq	%rax, %rcx
	vmovss	.LCPI3_0(%rip), %xmm0           # xmm0 = [-Inf,0.0E+0,0.0E+0,0.0E+0]
	movq	%rcx, 232(%rsp)                 # 8-byte Spill
	cmpq	%rcx, %r10
	movq	%rbx, 176(%rsp)                 # 8-byte Spill
	movq	%r10, 48(%rsp)                  # 8-byte Spill
	jge	.LBB3_41
# %bb.38:                               #   in Loop: Header=BB3_37 Depth=4
	movq	96(%rsp), %rcx                  # 8-byte Reload
	movq	%r10, %rdx
	vmovss	.LCPI3_0(%rip), %xmm0           # xmm0 = [-Inf,0.0E+0,0.0E+0,0.0E+0]
	movq	%rsi, 40(%rsp)                  # 8-byte Spill
	jmp	.LBB3_39
.LBB3_49:                               #   in Loop: Header=BB3_39 Depth=5
	movq	%r12, %r14
	movq	%r13, %r12
	vxorps	%xmm2, %xmm2, %xmm2
	.p2align	4
.LBB3_73:                               #   in Loop: Header=BB3_39 Depth=5
	movq	80(%rsp), %r11                  # 8-byte Reload
	vmulss	208(%r11), %xmm2, %xmm2
	movq	%rdx, %rax
	movq	48(%rsp), %r10                  # 8-byte Reload
	subq	%r10, %rax
	movq	%r12, %r13
	vmovss	%xmm2, (%r12,%rax,4)
	vmaxss	%xmm0, %xmm2, %xmm0
	movq	176(%rsp), %rbx                 # 8-byte Reload
	movq	40(%rsp), %rsi                  # 8-byte Reload
	movq	%r14, %r12
.LBB3_74:                               #   in Loop: Header=BB3_39 Depth=5
	incq	%rdx
	addq	$4, %rcx
	cmpq	%rsi, %rdx
	je	.LBB3_41
.LBB3_39:                               #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB3_58 Depth 6
                                        #             Child Loop BB3_62 Depth 6
                                        #             Child Loop BB3_68 Depth 6
                                        #             Child Loop BB3_72 Depth 6
                                        #             Child Loop BB3_53 Depth 6
                                        #             Child Loop BB3_56 Depth 6
	cmpq	%r12, %rdx
	jle	.LBB3_43
# %bb.40:                               #   in Loop: Header=BB3_39 Depth=5
	movq	%rdx, %rax
	subq	%r10, %rax
	movl	$-8388608, (%r13,%rax,4)        # imm = 0xFF800000
	jmp	.LBB3_74
	.p2align	4
.LBB3_43:                               #   in Loop: Header=BB3_39 Depth=5
	movq	8(%r11), %rsi
	movq	104(%r11), %r10
	movq	112(%r11), %rbx
	movq	120(%r11), %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movq	128(%r11), %rax
	movq	136(%r11), %r11
	movq	%r10, %rdi
	xorq	$1, %rdi
	movq	%r11, %r8
	xorq	$1, %r8
	orq	%rdi, %r8
	je	.LBB3_44
# %bb.48:                               #   in Loop: Header=BB3_39 Depth=5
	testq	%r9, %r9
	jle	.LBB3_49
# %bb.50:                               #   in Loop: Header=BB3_39 Depth=5
	cmpq	$4, %r9
	movq	%rax, 168(%rsp)                 # 8-byte Spill
	movq	%rbx, 160(%rsp)                 # 8-byte Spill
	jae	.LBB3_52
# %bb.51:                               #   in Loop: Header=BB3_39 Depth=5
	vxorps	%xmm2, %xmm2, %xmm2
	xorl	%ebx, %ebx
	jmp	.LBB3_54
	.p2align	4
.LBB3_44:                               #   in Loop: Header=BB3_39 Depth=5
	cmpq	$8, %r9
	jge	.LBB3_46
# %bb.45:                               #   in Loop: Header=BB3_39 Depth=5
	movq	%r12, %r14
	movq	%r13, %r12
	vxorps	%xmm2, %xmm2, %xmm2
	xorl	%r10d, %r10d
	jmp	.LBB3_64
.LBB3_46:                               #   in Loop: Header=BB3_39 Depth=5
	cmpq	$24, 152(%rsp)                  # 8-byte Folded Reload
	jae	.LBB3_57
# %bb.47:                               #   in Loop: Header=BB3_39 Depth=5
	movq	%r12, %r14
	vxorps	%xmm2, %xmm2, %xmm2
	movl	$8, %r11d
	xorl	%r10d, %r10d
	movq	%r13, %r12
	cmpq	$0, 104(%rsp)                   # 8-byte Folded Reload
	jne	.LBB3_61
	jmp	.LBB3_64
.LBB3_52:                               #   in Loop: Header=BB3_39 Depth=5
	leaq	(,%r10,4), %rdi
	leaq	(%rdi,%rdi,2), %r14
	movq	%r10, %r12
	shlq	$4, %r12
	leaq	(,%r11,4), %rdi
	leaq	(%rdi,%rdi,2), %r13
	movq	56(%rsp), %rdi                  # 8-byte Reload
	imulq	16(%rsp), %rdi                  # 8-byte Folded Reload
	movq	8(%rsp), %r8                    # 8-byte Reload
	imulq	%rbx, %r8
	addq	%rdi, %r8
	movq	%rax, %rdi
	imulq	%rcx, %rdi
	addq	%rsi, %rdi
	addq	%r8, %rdi
	movq	%r11, %r9
	shlq	$4, %r9
	vxorps	%xmm2, %xmm2, %xmm2
	movq	144(%rsp), %r8                  # 8-byte Reload
	xorl	%ebx, %ebx
	movq	272(%rsp), %rax                 # 8-byte Reload
	.p2align	4
.LBB3_53:                               #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        #           Parent Loop BB3_39 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	vmovss	(%r8), %xmm3                    # xmm3 = mem[0],zero,zero,zero
	vfmadd132ss	(%rdi), %xmm2, %xmm3    # xmm3 = (xmm3 * mem) + xmm2
	vmovss	(%r8,%r10,4), %xmm2             # xmm2 = mem[0],zero,zero,zero
	vfmadd132ss	(%rdi,%r11,4), %xmm3, %xmm2 # xmm2 = (xmm2 * mem) + xmm3
	vmovss	(%r8,%r10,8), %xmm3             # xmm3 = mem[0],zero,zero,zero
	vfmadd132ss	(%rdi,%r11,8), %xmm2, %xmm3 # xmm3 = (xmm3 * mem) + xmm2
	vmovss	(%r8,%r14), %xmm2               # xmm2 = mem[0],zero,zero,zero
	vfmadd132ss	(%rdi,%r13), %xmm3, %xmm2 # xmm2 = (xmm2 * mem) + xmm3
	addq	$4, %rbx
	addq	%r12, %r8
	addq	%r9, %rdi
	cmpq	%rbx, %rax
	jne	.LBB3_53
.LBB3_54:                               #   in Loop: Header=BB3_39 Depth=5
	cmpq	$0, 216(%rsp)                   # 8-byte Folded Reload
	movq	120(%rsp), %r12                 # 8-byte Reload
	movq	32(%rsp), %r9                   # 8-byte Reload
	movq	112(%rsp), %r14                 # 8-byte Reload
	movq	168(%rsp), %rax                 # 8-byte Reload
	je	.LBB3_73
# %bb.55:                               #   in Loop: Header=BB3_39 Depth=5
	shlq	$2, %rbx
	movq	%rbx, %rdi
	imulq	%r11, %rdi
	movq	16(%rsp), %r13                  # 8-byte Reload
	imulq	56(%rsp), %r13                  # 8-byte Folded Reload
	movq	160(%rsp), %r8                  # 8-byte Reload
	imulq	8(%rsp), %r8                    # 8-byte Folded Reload
	addq	%r13, %r8
	imulq	%rcx, %rax
	addq	%r8, %rax
	addq	%rdi, %rax
	addq	%rax, %rsi
	shlq	$2, %r11
	imulq	%r10, %rbx
	addq	144(%rsp), %rbx                 # 8-byte Folded Reload
	shlq	$2, %r10
	movq	216(%rsp), %rax                 # 8-byte Reload
	.p2align	4
.LBB3_56:                               #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        #           Parent Loop BB3_39 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	vmovss	(%rbx), %xmm3                   # xmm3 = mem[0],zero,zero,zero
	vfmadd231ss	(%rsi), %xmm3, %xmm2    # xmm2 = (xmm3 * mem) + xmm2
	addq	%r11, %rsi
	addq	%r10, %rbx
	decq	%rax
	jne	.LBB3_56
	jmp	.LBB3_73
.LBB3_57:                               #   in Loop: Header=BB3_39 Depth=5
	movq	56(%rsp), %rdi                  # 8-byte Reload
	imulq	16(%rsp), %rdi                  # 8-byte Folded Reload
	movq	8(%rsp), %r8                    # 8-byte Reload
	imulq	%rbx, %r8
	addq	%rdi, %r8
	movq	%rax, %rdi
	imulq	%rcx, %rdi
	addq	%r8, %rdi
	addq	%rsi, %rdi
	addq	$96, %rdi
	vxorps	%xmm2, %xmm2, %xmm2
	movq	136(%rsp), %r8                  # 8-byte Reload
	xorl	%r10d, %r10d
	.p2align	4
.LBB3_58:                               #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        #           Parent Loop BB3_39 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	vmovups	-96(%r15,%r10,4), %ymm3
	vmovups	-64(%r15,%r10,4), %ymm4
	vmovups	-32(%r15,%r10,4), %ymm5
	vfmadd132ps	-96(%rdi,%r10,4), %ymm2, %ymm3 # ymm3 = (ymm3 * mem) + ymm2
	vfmadd231ps	-64(%rdi,%r10,4), %ymm4, %ymm3 # ymm3 = (ymm4 * mem) + ymm3
	vmovups	(%r15,%r10,4), %ymm4
	vfmadd231ps	-32(%rdi,%r10,4), %ymm5, %ymm3 # ymm3 = (ymm5 * mem) + ymm3
	vmovaps	%ymm3, %ymm2
	vfmadd231ps	(%rdi,%r10,4), %ymm4, %ymm2 # ymm2 = (ymm4 * mem) + ymm2
	addq	$32, %r10
	addq	$-4, %r8
	jne	.LBB3_58
# %bb.59:                               #   in Loop: Header=BB3_39 Depth=5
	movq	%r12, %r14
	leaq	8(%r10), %r11
	movq	%r13, %r12
	cmpq	$0, 104(%rsp)                   # 8-byte Folded Reload
	je	.LBB3_64
.LBB3_61:                               #   in Loop: Header=BB3_39 Depth=5
	movq	%rbx, %rdi
	imulq	72(%rsp), %rdi                  # 8-byte Folded Reload
	movq	16(%rsp), %r8                   # 8-byte Reload
	imulq	88(%rsp), %r8                   # 8-byte Folded Reload
	movq	%rax, %r9
	imulq	%rdx, %r9
	leaq	(%rsi,%r8,4), %r8
	leaq	(%r8,%rdi,4), %rdi
	leaq	(%rdi,%r9,4), %rdi
	movq	104(%rsp), %r8                  # 8-byte Reload
	movq	280(%rsp), %r9                  # 8-byte Reload
	.p2align	4
.LBB3_62:                               #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        #           Parent Loop BB3_39 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	vmovups	(%r9,%r10,4), %ymm3
	vfmadd231ps	(%rdi,%r10,4), %ymm3, %ymm2 # ymm2 = (ymm3 * mem) + ymm2
	movq	%r11, %r10
	addq	$8, %r11
	decq	%r8
	jne	.LBB3_62
# %bb.63:                               #   in Loop: Header=BB3_39 Depth=5
	addq	$-8, %r11
	movq	%r11, %r10
	movq	32(%rsp), %r9                   # 8-byte Reload
.LBB3_64:                               #   in Loop: Header=BB3_39 Depth=5
	vextractf128	$1, %ymm2, %xmm3
	vaddps	%xmm3, %xmm2, %xmm2
	vhaddps	%xmm2, %xmm2, %xmm2
	vhaddps	%xmm2, %xmm2, %xmm2
	movq	%r10, %r11
	subq	%r9, %r11
	jge	.LBB3_73
# %bb.65:                               #   in Loop: Header=BB3_39 Depth=5
	movl	%r9d, %edi
	subl	%r10d, %edi
	andl	$3, %edi
	je	.LBB3_66
# %bb.67:                               #   in Loop: Header=BB3_39 Depth=5
	movq	56(%rsp), %r8                   # 8-byte Reload
	imulq	16(%rsp), %r8                   # 8-byte Folded Reload
	movq	8(%rsp), %r9                    # 8-byte Reload
	movq	%rbx, %r13
	imulq	%rbx, %r9
	addq	%r8, %r9
	movq	%rax, %r8
	imulq	%rcx, %r8
	addq	%r9, %r8
	leaq	(%r8,%r10,4), %r8
	addq	%rsi, %r8
	movq	144(%rsp), %r9                  # 8-byte Reload
	leaq	(%r9,%r10,4), %r9
	xorl	%ebx, %ebx
	.p2align	4
.LBB3_68:                               #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        #           Parent Loop BB3_39 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	vmovss	(%r9,%rbx,4), %xmm3             # xmm3 = mem[0],zero,zero,zero
	vfmadd231ss	(%r8,%rbx,4), %xmm3, %xmm2 # xmm2 = (xmm3 * mem) + xmm2
	incq	%rbx
	cmpq	%rbx, %rdi
	jne	.LBB3_68
# %bb.69:                               #   in Loop: Header=BB3_39 Depth=5
	addq	%rbx, %r10
	movq	32(%rsp), %r9                   # 8-byte Reload
	cmpq	$-4, %r11
	ja	.LBB3_73
	jmp	.LBB3_71
.LBB3_66:                               #   in Loop: Header=BB3_39 Depth=5
	movq	%rbx, %r13
	cmpq	$-4, %r11
	ja	.LBB3_73
.LBB3_71:                               #   in Loop: Header=BB3_39 Depth=5
	movq	%r9, %rdi
	subq	%r10, %rdi
	movq	16(%rsp), %r8                   # 8-byte Reload
	imulq	56(%rsp), %r8                   # 8-byte Folded Reload
	imulq	8(%rsp), %r13                   # 8-byte Folded Reload
	addq	%r8, %r13
	imulq	%rcx, %rax
	addq	%r13, %rax
	leaq	(%rax,%r10,4), %rax
	addq	%rsi, %rax
	addq	$12, %rax
	leaq	(%r15,%r10,4), %rsi
	addq	$-84, %rsi
	xorl	%r8d, %r8d
	.p2align	4
.LBB3_72:                               #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        #           Parent Loop BB3_39 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	vmovss	-12(%rsi,%r8,4), %xmm3          # xmm3 = mem[0],zero,zero,zero
	vmovss	-8(%rsi,%r8,4), %xmm4           # xmm4 = mem[0],zero,zero,zero
	vfmadd132ss	-12(%rax,%r8,4), %xmm2, %xmm3 # xmm3 = (xmm3 * mem) + xmm2
	vfmadd231ss	-8(%rax,%r8,4), %xmm4, %xmm3 # xmm3 = (xmm4 * mem) + xmm3
	vmovss	-4(%rsi,%r8,4), %xmm4           # xmm4 = mem[0],zero,zero,zero
	vfmadd132ss	-4(%rax,%r8,4), %xmm3, %xmm4 # xmm4 = (xmm4 * mem) + xmm3
	vmovss	(%rsi,%r8,4), %xmm2             # xmm2 = mem[0],zero,zero,zero
	vfmadd132ss	(%rax,%r8,4), %xmm4, %xmm2 # xmm2 = (xmm2 * mem) + xmm4
	addq	$4, %r8
	cmpq	%r8, %rdi
	jne	.LBB3_72
	jmp	.LBB3_73
	.p2align	4
.LBB3_41:                               #   in Loop: Header=BB3_37 Depth=4
	vmovd	%xmm0, %eax
	andl	$2147483647, %eax               # imm = 0x7FFFFFFF
	cmpl	$2139095039, %eax               # imm = 0x7F7FFFFF
	jg	.LBB3_42
# %bb.75:                               #   in Loop: Header=BB3_37 Depth=4
	vmovd	%xmm1, %eax
	vmaxss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, 40(%rsp)                 # 4-byte Spill
	andl	$2147483647, %eax               # imm = 0x7FFFFFFF
	vxorps	%xmm0, %xmm0, %xmm0
	cmpl	$2139095039, %eax               # imm = 0x7F7FFFFF
	jg	.LBB3_77
# %bb.76:                               #   in Loop: Header=BB3_37 Depth=4
	vsubss	40(%rsp), %xmm1, %xmm0          # 4-byte Folded Reload
	vmovss	%xmm6, 16(%rsp)                 # 4-byte Spill
	movq	%r11, %r14
	vzeroupper
	callq	expf@PLT
	movq	%r14, %r11
	movq	48(%rsp), %r10                  # 8-byte Reload
	vmovss	16(%rsp), %xmm6                 # 4-byte Reload
                                        # xmm6 = mem[0],zero,zero,zero
	movq	32(%rsp), %r9                   # 8-byte Reload
.LBB3_77:                               #   in Loop: Header=BB3_37 Depth=4
	cmpq	$8, %r9
	jge	.LBB3_79
# %bb.78:                               #   in Loop: Header=BB3_37 Depth=4
	xorl	%eax, %eax
	jmp	.LBB3_86
	.p2align	4
.LBB3_79:                               #   in Loop: Header=BB3_37 Depth=4
	vbroadcastss	%xmm0, %ymm1
	cmpq	$24, 152(%rsp)                  # 8-byte Folded Reload
	jae	.LBB3_93
# %bb.80:                               #   in Loop: Header=BB3_37 Depth=4
	movl	$8, %ecx
	xorl	%eax, %eax
	movq	24(%rsp), %rdx                  # 8-byte Reload
	testb	$3, 224(%rsp)                   # 1-byte Folded Reload
	jne	.LBB3_83
	jmp	.LBB3_87
.LBB3_93:                               #   in Loop: Header=BB3_37 Depth=4
	movq	136(%rsp), %rcx                 # 8-byte Reload
	xorl	%eax, %eax
	.p2align	4
.LBB3_94:                               #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        # =>        This Inner Loop Header: Depth=5
	vmulps	(%rbp,%rax,4), %ymm1, %ymm2
	vmovups	%ymm2, (%rbp,%rax,4)
	vmulps	32(%rbp,%rax,4), %ymm1, %ymm2
	vmovups	%ymm2, 32(%rbp,%rax,4)
	vmulps	64(%rbp,%rax,4), %ymm1, %ymm2
	vmovups	%ymm2, 64(%rbp,%rax,4)
	vmulps	96(%rbp,%rax,4), %ymm1, %ymm2
	vmovups	%ymm2, 96(%rbp,%rax,4)
	addq	$32, %rax
	addq	$-4, %rcx
	jne	.LBB3_94
# %bb.81:                               #   in Loop: Header=BB3_37 Depth=4
	leaq	8(%rax), %rcx
	movq	24(%rsp), %rdx                  # 8-byte Reload
	testb	$3, 224(%rsp)                   # 1-byte Folded Reload
	je	.LBB3_87
.LBB3_83:                               #   in Loop: Header=BB3_37 Depth=4
	movq	104(%rsp), %rdx                 # 8-byte Reload
	.p2align	4
.LBB3_84:                               #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        # =>        This Inner Loop Header: Depth=5
	vmulps	(%rbp,%rax,4), %ymm1, %ymm2
	vmovups	%ymm2, (%rbp,%rax,4)
	movq	%rcx, %rax
	addq	$8, %rcx
	decq	%rdx
	jne	.LBB3_84
# %bb.85:                               #   in Loop: Header=BB3_37 Depth=4
	addq	$-8, %rcx
	movq	%rcx, %rax
.LBB3_86:                               #   in Loop: Header=BB3_37 Depth=4
	movq	24(%rsp), %rdx                  # 8-byte Reload
.LBB3_87:                               #   in Loop: Header=BB3_37 Depth=4
	movq	%rax, %rcx
	subq	%r9, %rcx
	jge	.LBB3_91
# %bb.88:                               #   in Loop: Header=BB3_37 Depth=4
	movl	%r9d, %edx
	subl	%eax, %edx
	andl	$3, %edx
	je	.LBB3_90
	.p2align	4
.LBB3_89:                               #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        # =>        This Inner Loop Header: Depth=5
	vmulss	(%rbp,%rax,4), %xmm0, %xmm1
	vmovss	%xmm1, (%rbp,%rax,4)
	incq	%rax
	decq	%rdx
	jne	.LBB3_89
.LBB3_90:                               #   in Loop: Header=BB3_37 Depth=4
	cmpq	$-4, %rcx
	movq	248(%rsp), %rcx                 # 8-byte Reload
	movq	24(%rsp), %rdx                  # 8-byte Reload
	ja	.LBB3_91
	.p2align	4
.LBB3_95:                               #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        # =>        This Inner Loop Header: Depth=5
	vmulss	-12(%rcx,%rax,4), %xmm0, %xmm1
	vmovss	%xmm1, -12(%rcx,%rax,4)
	vmulss	-8(%rcx,%rax,4), %xmm0, %xmm1
	vmovss	%xmm1, -8(%rcx,%rax,4)
	vmulss	-4(%rcx,%rax,4), %xmm0, %xmm1
	vmovss	%xmm1, -4(%rcx,%rax,4)
	vmulss	(%rcx,%rax,4), %xmm0, %xmm1
	vmovss	%xmm1, (%rcx,%rax,4)
	addq	$4, %rax
	cmpq	%rax, %r9
	jne	.LBB3_95
.LBB3_91:                               #   in Loop: Header=BB3_37 Depth=4
	vmulss	%xmm0, %xmm6, %xmm6
	cmpq	232(%rsp), %r10                 # 8-byte Folded Reload
	setl	%al
	cmpq	%r12, %r10
	setle	%cl
	andb	%al, %cl
	cmpb	$1, %cl
	jne	.LBB3_92
# %bb.96:                               #   in Loop: Header=BB3_37 Depth=4
	cmpq	$8, %r9
	setl	%al
	movq	16(%r11), %rcx
	movq	144(%r11), %rdx
	movq	%rdx, %rsi
	imulq	72(%rsp), %rsi                  # 8-byte Folded Reload
	movq	152(%r11), %rdi
	movq	%rdi, %r8
	imulq	88(%rsp), %r8                   # 8-byte Folded Reload
	movq	160(%r11), %r9
	leaq	(%rcx,%r8,4), %r8
	leaq	(%r8,%rsi,4), %rsi
	movq	%rsi, 160(%rsp)                 # 8-byte Spill
	movq	168(%r11), %rsi
	cmpq	$1, %rsi
	setne	%r8b
	orb	%al, %r8b
	movb	%r8b, 7(%rsp)                   # 1-byte Spill
	imulq	56(%rsp), %rdi                  # 8-byte Folded Reload
	imulq	8(%rsp), %rdx                   # 8-byte Folded Reload
	addq	%rdi, %rdx
	movq	%r9, %rbx
	imulq	96(%rsp), %rbx                  # 8-byte Folded Reload
	addq	%rdx, %rbx
	leaq	(%rcx,%rbx), %r14
	addq	$96, %r14
	movq	%r9, 168(%rsp)                  # 8-byte Spill
	leaq	(,%r9,4), %rax
	movq	%rax, 304(%rsp)                 # 8-byte Spill
	leaq	(,%rsi,4), %rax
	movq	%rax, 288(%rsp)                 # 8-byte Spill
	addq	%rcx, %rbx
	movq	%rsi, 296(%rsp)                 # 8-byte Spill
	leaq	(,%rsi,8), %r13
	movq	%r10, %r12
	.p2align	4
.LBB3_97:                               #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB3_102 Depth 6
                                        #             Child Loop BB3_106 Depth 6
                                        #             Child Loop BB3_113 Depth 6
	vmovss	%xmm6, 16(%rsp)                 # 4-byte Spill
	movq	%r12, %rax
	subq	%r10, %rax
	movq	120(%rsp), %rcx                 # 8-byte Reload
	vmovss	(%rcx,%rax,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	vsubss	40(%rsp), %xmm0, %xmm0          # 4-byte Folded Reload
	vzeroupper
	callq	expf@PLT
	movq	%r12, %rax
	imulq	168(%rsp), %rax                 # 8-byte Folded Reload
	movq	160(%rsp), %rcx                 # 8-byte Reload
	leaq	(%rcx,%rax,4), %rcx
	cmpb	$0, 7(%rsp)                     # 1-byte Folded Reload
	je	.LBB3_99
# %bb.98:                               #   in Loop: Header=BB3_97 Depth=5
	xorl	%eax, %eax
	movq	32(%rsp), %r9                   # 8-byte Reload
	vmovss	16(%rsp), %xmm6                 # 4-byte Reload
                                        # xmm6 = mem[0],zero,zero,zero
	movq	48(%rsp), %r10                  # 8-byte Reload
	jmp	.LBB3_108
	.p2align	4
.LBB3_99:                               #   in Loop: Header=BB3_97 Depth=5
	vbroadcastss	%xmm0, %ymm1
	cmpq	$24, 152(%rsp)                  # 8-byte Folded Reload
	movq	32(%rsp), %r9                   # 8-byte Reload
	vmovss	16(%rsp), %xmm6                 # 4-byte Reload
                                        # xmm6 = mem[0],zero,zero,zero
	movq	48(%rsp), %r10                  # 8-byte Reload
	jae	.LBB3_101
# %bb.100:                              #   in Loop: Header=BB3_97 Depth=5
	movl	$8, %edx
	xorl	%eax, %eax
	testb	$3, 224(%rsp)                   # 1-byte Folded Reload
	jne	.LBB3_105
	jmp	.LBB3_108
	.p2align	4
.LBB3_101:                              #   in Loop: Header=BB3_97 Depth=5
	movq	136(%rsp), %rdx                 # 8-byte Reload
	xorl	%eax, %eax
	.p2align	4
.LBB3_102:                              #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        #           Parent Loop BB3_97 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	vmovups	-96(%r14,%rax,4), %ymm2
	vfmadd213ps	(%rbp,%rax,4), %ymm1, %ymm2 # ymm2 = (ymm1 * ymm2) + mem
	vmovups	%ymm2, (%rbp,%rax,4)
	vmovups	-64(%r14,%rax,4), %ymm2
	vfmadd213ps	32(%rbp,%rax,4), %ymm1, %ymm2 # ymm2 = (ymm1 * ymm2) + mem
	vmovups	%ymm2, 32(%rbp,%rax,4)
	vmovups	-32(%r14,%rax,4), %ymm2
	vfmadd213ps	64(%rbp,%rax,4), %ymm1, %ymm2 # ymm2 = (ymm1 * ymm2) + mem
	vmovups	%ymm2, 64(%rbp,%rax,4)
	vmovups	(%r14,%rax,4), %ymm2
	vfmadd213ps	96(%rbp,%rax,4), %ymm1, %ymm2 # ymm2 = (ymm1 * ymm2) + mem
	vmovups	%ymm2, 96(%rbp,%rax,4)
	addq	$32, %rax
	addq	$-4, %rdx
	jne	.LBB3_102
# %bb.103:                              #   in Loop: Header=BB3_97 Depth=5
	leaq	8(%rax), %rdx
	testb	$3, 224(%rsp)                   # 1-byte Folded Reload
	je	.LBB3_108
.LBB3_105:                              #   in Loop: Header=BB3_97 Depth=5
	movq	104(%rsp), %rsi                 # 8-byte Reload
	.p2align	4
.LBB3_106:                              #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        #           Parent Loop BB3_97 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	vmovups	(%rcx,%rax,4), %ymm2
	vfmadd213ps	(%rbp,%rax,4), %ymm1, %ymm2 # ymm2 = (ymm1 * ymm2) + mem
	vmovups	%ymm2, (%rbp,%rax,4)
	movq	%rdx, %rax
	addq	$8, %rdx
	decq	%rsi
	jne	.LBB3_106
# %bb.107:                              #   in Loop: Header=BB3_97 Depth=5
	addq	$-8, %rdx
	movq	%rdx, %rax
.LBB3_108:                              #   in Loop: Header=BB3_97 Depth=5
	cmpq	%rax, %r9
	jle	.LBB3_114
# %bb.109:                              #   in Loop: Header=BB3_97 Depth=5
	movl	%r9d, %esi
	subl	%eax, %esi
	leaq	1(%rax), %rdx
	testb	$1, %sil
	je	.LBB3_111
# %bb.110:                              #   in Loop: Header=BB3_97 Depth=5
	movq	%rax, %rsi
	imulq	296(%rsp), %rsi                 # 8-byte Folded Reload
	vmovss	(%rcx,%rsi,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	(%rbp,%rax,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, (%rbp,%rax,4)
	movq	%rdx, %rax
.LBB3_111:                              #   in Loop: Header=BB3_97 Depth=5
	cmpq	%rdx, %r9
	je	.LBB3_114
# %bb.112:                              #   in Loop: Header=BB3_97 Depth=5
	leaq	1(%rax), %rcx
	movq	288(%rsp), %rdx                 # 8-byte Reload
	imulq	%rdx, %rcx
	imulq	%rax, %rdx
	movq	%rbx, %rsi
	.p2align	4
.LBB3_113:                              #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        #         Parent Loop BB3_37 Depth=4
                                        #           Parent Loop BB3_97 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	vmovss	(%rsi,%rdx), %xmm1              # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	(%rbp,%rax,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, (%rbp,%rax,4)
	vmovss	(%rsi,%rcx), %xmm1              # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	4(%rbp,%rax,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 4(%rbp,%rax,4)
	addq	$2, %rax
	addq	%r13, %rsi
	cmpq	%rax, %r9
	jne	.LBB3_113
.LBB3_114:                              #   in Loop: Header=BB3_97 Depth=5
	vaddss	%xmm0, %xmm6, %xmm6
	leaq	1(%r12), %rax
	cmpq	232(%rsp), %rax                 # 8-byte Folded Reload
	jge	.LBB3_116
# %bb.115:                              #   in Loop: Header=BB3_97 Depth=5
	movq	304(%rsp), %rcx                 # 8-byte Reload
	addq	%rcx, %r14
	addq	%rcx, %rbx
	cmpq	112(%rsp), %r12                 # 8-byte Folded Reload
	movq	%rax, %r12
	jl	.LBB3_97
.LBB3_116:                              #   in Loop: Header=BB3_37 Depth=4
	vmovd	40(%rsp), %xmm1                 # 4-byte Folded Reload
                                        # xmm1 = mem[0],zero,zero,zero
	movq	80(%rsp), %r11                  # 8-byte Reload
	movq	120(%rsp), %r13                 # 8-byte Reload
	movq	112(%rsp), %r12                 # 8-byte Reload
	movq	24(%rsp), %rdx                  # 8-byte Reload
	movq	176(%rsp), %rbx                 # 8-byte Reload
	jmp	.LBB3_117
	.p2align	4
.LBB3_92:                               #   in Loop: Header=BB3_37 Depth=4
	vmovd	40(%rsp), %xmm1                 # 4-byte Folded Reload
                                        # xmm1 = mem[0],zero,zero,zero
	jmp	.LBB3_117
	.p2align	4
.LBB3_31:                               #   in Loop: Header=BB3_28 Depth=3
	vxorps	%xmm6, %xmm6, %xmm6
.LBB3_32:                               #   in Loop: Header=BB3_28 Depth=3
	vmovd	%xmm6, %eax
	testl	%eax, %eax
	setns	%cl
	movl	%eax, %edx
	andl	$2147483647, %edx               # imm = 0x7FFFFFFF
	addl	$-8388608, %edx                 # imm = 0xFF800000
	cmpl	$2130706432, %edx               # imm = 0x7F000000
	setb	%dl
	andb	%cl, %dl
	decl	%eax
	cmpl	$8388607, %eax                  # imm = 0x7FFFFF
	setb	%al
	orb	%dl, %al
	je	.LBB3_33
# %bb.118:                              #   in Loop: Header=BB3_28 Depth=3
	movq	24(%r11), %rax
	movq	176(%r11), %rsi
	movq	184(%r11), %rcx
	movq	192(%r11), %rdx
	cmpq	$8, %r9
	jge	.LBB3_134
# %bb.119:                              #   in Loop: Header=BB3_28 Depth=3
	xorl	%edi, %edi
	movq	200(%rsp), %r15                 # 8-byte Reload
	movq	%r9, %r10
	jmp	.LBB3_120
	.p2align	4
.LBB3_134:                              #   in Loop: Header=BB3_28 Depth=3
	vmovss	.LCPI3_1(%rip), %xmm0           # xmm0 = [1.0E+0,0.0E+0,0.0E+0,0.0E+0]
	vdivss	%xmm6, %xmm0, %xmm0
	vbroadcastss	%xmm0, %ymm0
	movq	8(%rsp), %rdi                   # 8-byte Reload
	imulq	%rsi, %rdi
	movq	64(%rsp), %r9                   # 8-byte Reload
	imulq	%rcx, %r9
	addq	%rdi, %r9
	movq	%rdx, %r8
	movq	200(%rsp), %r15                 # 8-byte Reload
	imulq	%r15, %r8
	addq	%rax, %r8
	addq	%r9, %r8
	xorl	%r9d, %r9d
	.p2align	4
.LBB3_135:                              #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vmulps	(%rbp,%r9,4), %ymm0, %ymm1
	vmovups	%ymm1, (%r8,%r9,4)
	movq	72(%r11), %r10
	leaq	8(%r9), %rdi
	addq	$16, %r9
	cmpq	%r10, %r9
	movq	%rdi, %r9
	jle	.LBB3_135
.LBB3_120:                              #   in Loop: Header=BB3_28 Depth=3
	movq	%rdi, %r8
	subq	%r10, %r8
	movq	%r10, %r9
	jge	.LBB3_128
# %bb.121:                              #   in Loop: Header=BB3_28 Depth=3
	movq	%r9, %r12
	andq	$3, %r9
	je	.LBB3_125
# %bb.122:                              #   in Loop: Header=BB3_28 Depth=3
	movq	%r11, %r14
	movq	8(%rsp), %r10                   # 8-byte Reload
	imulq	%rsi, %r10
	movq	64(%rsp), %r11                  # 8-byte Reload
	imulq	%rcx, %r11
	addq	%r10, %r11
	movq	%rdx, %r10
	imulq	%r15, %r10
	addq	%r11, %r10
	leaq	(%r10,%rdi,4), %r10
	addq	%rax, %r10
	leaq	(,%rdi,4), %rbx
	addq	%rbp, %rbx
	xorl	%r11d, %r11d
	.p2align	4
.LBB3_123:                              #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vmovss	(%rbx,%r11,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	vdivss	%xmm6, %xmm0, %xmm0
	vmovss	%xmm0, (%r10,%r11,4)
	incq	%r11
	cmpq	%r11, %r9
	jne	.LBB3_123
# %bb.124:                              #   in Loop: Header=BB3_28 Depth=3
	addq	%r11, %rdi
	movq	%r14, %r11
.LBB3_125:                              #   in Loop: Header=BB3_28 Depth=3
	cmpq	$-4, %r8
	movq	%r12, %r9
	ja	.LBB3_128
# %bb.126:                              #   in Loop: Header=BB3_28 Depth=3
	movq	%r9, %r8
	subq	%rdi, %r8
	imulq	8(%rsp), %rsi                   # 8-byte Folded Reload
	imulq	64(%rsp), %rcx                  # 8-byte Folded Reload
	addq	%rsi, %rcx
	imulq	%r15, %rdx
	addq	%rcx, %rdx
	leaq	(%rdx,%rdi,4), %rcx
	addq	%rcx, %rax
	addq	$12, %rax
	leaq	12(,%rdi,4), %rcx
	addq	%rbp, %rcx
	xorl	%edx, %edx
	.p2align	4
.LBB3_127:                              #   Parent Loop BB3_20 Depth=1
                                        #     Parent Loop BB3_23 Depth=2
                                        #       Parent Loop BB3_28 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vmovss	-12(%rcx,%rdx,4), %xmm0         # xmm0 = mem[0],zero,zero,zero
	vdivss	%xmm6, %xmm0, %xmm0
	vmovss	%xmm0, -12(%rax,%rdx,4)
	vmovss	-8(%rcx,%rdx,4), %xmm0          # xmm0 = mem[0],zero,zero,zero
	vdivss	%xmm6, %xmm0, %xmm0
	vmovss	%xmm0, -8(%rax,%rdx,4)
	vmovss	-4(%rcx,%rdx,4), %xmm0          # xmm0 = mem[0],zero,zero,zero
	vdivss	%xmm6, %xmm0, %xmm0
	vmovss	%xmm0, -4(%rax,%rdx,4)
	vmovss	(%rcx,%rdx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	vdivss	%xmm6, %xmm0, %xmm0
	vmovss	%xmm0, (%rax,%rdx,4)
	addq	$4, %rdx
	cmpq	%rdx, %r8
	jne	.LBB3_127
	jmp	.LBB3_128
.LBB3_33:
	movl	$10, %eax
	movq	184(%rsp), %rbp                 # 8-byte Reload
	leaq	.L.str.2(%rip), %rdx
	testq	%r13, %r13
	je	.LBB3_2
.LBB3_35:
	subq	%r13, %rbp
	movq	%r13, %rdi
	movq	%rbp, %rsi
	movl	%eax, %ebx
	movq	%rdx, %r14
	vzeroupper
	callq	_ZdlPvm@PLT
	movq	%r14, %rdx
	movl	%ebx, %eax
	jmp	.LBB3_2
.LBB3_133:
	xorl	%eax, %eax
	leaq	.L.str.3(%rip), %rdx
	movq	184(%rsp), %rbp                 # 8-byte Reload
	testq	%r13, %r13
	jne	.LBB3_35
	jmp	.LBB3_2
.LBB3_136:
	leaq	.L.str.13(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Lfunc_end3:
	.size	hir_fused_online_attention_avx2, .Lfunc_end3-hir_fused_online_attention_avx2
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function _ZN12_GLOBAL__N_18validateEPK23HirFusedAttentionParams
	.type	_ZN12_GLOBAL__N_18validateEPK23HirFusedAttentionParams,@function
_ZN12_GLOBAL__N_18validateEPK23HirFusedAttentionParams: # @_ZN12_GLOBAL__N_18validateEPK23HirFusedAttentionParams
	.cfi_startproc
# %bb.0:
	leaq	.L.str.4(%rip), %rdx
	movl	$1, %eax
	testq	%rdi, %rdi
	je	.LBB4_45
# %bb.1:
	cmpq	$0, (%rdi)
	je	.LBB4_45
# %bb.2:
	cmpq	$0, 8(%rdi)
	je	.LBB4_45
# %bb.3:
	cmpq	$0, 16(%rdi)
	je	.LBB4_45
# %bb.4:
	cmpq	$0, 24(%rdi)
	je	.LBB4_45
# %bb.5:
	leaq	.L.str.5(%rip), %rdx
	movl	$2, %eax
	cmpq	$0, 32(%rdi)
	jle	.LBB4_45
# %bb.6:
	movq	40(%rdi), %r8
	testq	%r8, %r8
	jle	.LBB4_45
# %bb.7:
	movq	48(%rdi), %r9
	testq	%r9, %r9
	jle	.LBB4_45
# %bb.8:
	movq	56(%rdi), %rcx
	testq	%rcx, %rcx
	jle	.LBB4_45
# %bb.9:
	movq	64(%rdi), %r10
	testq	%r10, %r10
	jle	.LBB4_45
# %bb.10:
	cmpq	$0, 72(%rdi)
	jle	.LBB4_45
# %bb.11:
	movq	%rcx, %rax
	orq	%r10, %rax
	shrq	$32, %rax
	je	.LBB4_13
# %bb.12:
	movq	%rcx, %rax
	xorl	%edx, %edx
	divq	%r10
	movq	%rdx, %rsi
	jmp	.LBB4_14
.LBB4_13:
	movl	%ecx, %eax
	xorl	%edx, %edx
	divl	%r10d
	movl	%edx, %esi
.LBB4_14:
	leaq	.L.str.6(%rip), %rdx
	movl	$3, %eax
	testq	%rsi, %rsi
	jne	.LBB4_45
# %bb.15:
	movq	%rdx, %r11
	movq	256(%rdi), %rsi
	movq	%rsi, %rax
	orq	%r10, %rax
	shrq	$32, %rax
	je	.LBB4_17
# %bb.16:
	movq	%rsi, %rax
	cqto
	idivq	%r10
	jmp	.LBB4_18
.LBB4_17:
	movl	%esi, %eax
	xorl	%edx, %edx
	divl	%r10d
                                        # kill: def $edx killed $edx def $rdx
.LBB4_18:
	testq	%rdx, %rdx
	movq	%r11, %rdx
	movl	$3, %eax
	je	.LBB4_19
.LBB4_45:
	retq
.LBB4_19:
	movq	248(%rdi), %rax
	testq	%rax, %rax
	sets	%dl
	addq	%rax, %rcx
	cmpq	%rsi, %rcx
	setg	%al
	orb	%dl, %al
	je	.LBB4_20
# %bb.46:
	leaq	.L.str.7(%rip), %rdx
	movl	$4, %eax
	retq
.LBB4_20:
	leaq	.L.str.8(%rip), %rdx
	movl	$5, %eax
	cmpq	$0, 80(%rdi)
	jle	.LBB4_45
# %bb.21:
	cmpq	$0, 88(%rdi)
	jle	.LBB4_45
# %bb.22:
	cmpq	$0, 96(%rdi)
	jle	.LBB4_45
# %bb.23:
	cmpq	$0, 104(%rdi)
	jle	.LBB4_45
# %bb.24:
	cmpq	$0, 112(%rdi)
	jle	.LBB4_45
# %bb.25:
	cmpq	$0, 120(%rdi)
	jle	.LBB4_45
# %bb.26:
	cmpq	$0, 128(%rdi)
	jle	.LBB4_45
# %bb.27:
	cmpq	$0, 136(%rdi)
	jle	.LBB4_45
# %bb.28:
	cmpq	$0, 144(%rdi)
	jle	.LBB4_45
# %bb.29:
	cmpq	$0, 152(%rdi)
	jle	.LBB4_45
# %bb.30:
	cmpq	$0, 160(%rdi)
	jle	.LBB4_45
# %bb.31:
	cmpq	$0, 168(%rdi)
	jle	.LBB4_45
# %bb.32:
	cmpq	$0, 176(%rdi)
	jle	.LBB4_45
# %bb.33:
	cmpq	$0, 184(%rdi)
	jle	.LBB4_45
# %bb.34:
	cmpq	$0, 192(%rdi)
	jle	.LBB4_45
# %bb.35:
	cmpq	$0, 200(%rdi)
	jle	.LBB4_45
# %bb.36:
	movl	208(%rdi), %eax
	testl	%eax, %eax
	sets	%cl
	movl	%eax, %edx
	andl	$2147483647, %edx               # imm = 0x7FFFFFFF
	addl	$-8388608, %edx                 # imm = 0xFF800000
	cmpl	$2130706432, %edx               # imm = 0x7F000000
	setae	%dl
	orb	%cl, %dl
	decl	%eax
	cmpl	$8388607, %eax                  # imm = 0x7FFFFF
	setae	%al
	testb	%dl, %al
	jne	.LBB4_42
# %bb.37:
	cmpl	$1, 212(%rdi)
	jne	.LBB4_43
# %bb.38:
	leaq	.L.str.11(%rip), %rdx
	cmpq	$0, 224(%rdi)
	jle	.LBB4_44
# %bb.39:
	cmpq	$0, 232(%rdi)
	jle	.LBB4_44
# %bb.40:
	cmpq	$0, 240(%rdi)
	movl	$8, %eax
	jle	.LBB4_45
# %bb.41:
	movq	216(%rdi), %rax
	testq	%rax, %rax
	sets	%cl
	addq	%rax, %r8
	cmpq	%r9, %r8
	seta	%al
	orb	%cl, %al
	movzbl	%al, %ecx
	leal	(%rcx,%rcx,8), %eax
	leaq	.L.str.12(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	testb	%cl, %cl
	cmovneq	%rsi, %rdx
	retq
.LBB4_42:
	leaq	.L.str.9(%rip), %rdx
	movl	$6, %eax
	retq
.LBB4_43:
	leaq	.L.str.10(%rip), %rdx
	movl	$7, %eax
	retq
.LBB4_44:
	movl	$8, %eax
	retq
.Lfunc_end4:
	.size	_ZN12_GLOBAL__N_18validateEPK23HirFusedAttentionParams, .Lfunc_end4-_ZN12_GLOBAL__N_18validateEPK23HirFusedAttentionParams
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"hir.fused_online_attention.v1"
	.size	.L.str, 30

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"avx2_fma_unavailable"
	.size	.L.str.1, 21

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"invalid_softmax_denominator"
	.size	.L.str.2, 28

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"ok"
	.size	.L.str.3, 3

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"null_pointer"
	.size	.L.str.4, 13

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"invalid_dimension"
	.size	.L.str.5, 18

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"invalid_gqa_mapping"
	.size	.L.str.6, 20

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"invalid_query_head_range"
	.size	.L.str.7, 25

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"unsupported_stride"
	.size	.L.str.8, 19

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"invalid_scale"
	.size	.L.str.9, 14

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"causal_required"
	.size	.L.str.10, 16

	.type	.L.str.11,@object               # @.str.11
.L.str.11:
	.asciz	"invalid_tile_or_worker_count"
	.size	.L.str.11, 29

	.type	.L.str.12,@object               # @.str.12
.L.str.12:
	.asciz	"invalid_causal_position_range"
	.size	.L.str.12, 30

	.type	.L.str.13,@object               # @.str.13
.L.str.13:
	.asciz	"cannot create std::vector larger than max_size()"
	.size	.L.str.13, 49

	.ident	"Ubuntu clang version 21.1.8 (6ubuntu1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
