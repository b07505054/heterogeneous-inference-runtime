# Contiguous vs Page-Major Differential Stage Breakdown

- Repetitions: 80
- Inner iterations: 500
- Timer overhead: 0.000035000 ms

Summary: direct exported-kernel timing shows no native page-major latency gap at 64 tokens. The measured direct gap is -0.000175810 ms, while the runtime-integrated benchmark gap is +0.014318081 ms. Paged-only metadata/cache stages are measurable but too small to explain the integrated gap.

## 1 Tokens
- Direct exported-kernel gap: -0.000017518 ms
- Modeled stage gap: 0.000006181 ms
- Unclassified differential remainder: -0.000005370 ms

| Stage | Contiguous ms | Page-major ms | Delta ms | Ratio | % direct gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation_setup | 0.000003408 | 0.000004260 | 0.000000852 | 1.25 | -4.86 |
| metadata_preparation | 0.000003110 | 0.000003111 | 0.000000001 | 1.00 | -0.01 |
| qk_score_generation_plus_max | 0.000014666 | 0.000020148 | 0.000005482 | 1.37 | -31.29 |
| softmax_exp_sum | 0.000036408 | 0.000036852 | 0.000000444 | 1.01 | -2.53 |
| standalone_reciprocal | 0.000000000 | 0.000000000 | 0.000000000 |  | -0.00 |
| v_accumulation | 0.000037594 | 0.000016962 | -0.000020632 | 0.45 | 117.78 |
| output_store_read | 0.000060296 | 0.000060296 | 0.000000000 | 1.00 | -0.00 |
| logical_page_count_setup | 0.000000000 | 0.000003481 | 0.000003481 |  | -19.87 |
| block_table_validation_cache | 0.000000000 | 0.000001334 | 0.000001334 |  | -7.62 |
| k_page_base_generation (diagnostic proxy, not exclusive) | 0.000000000 | 0.000007630 | 0.000007630 |  | -43.56 |
| v_page_base_generation (diagnostic proxy, not exclusive) | 0.000000000 | 0.000007628 | 0.000007628 |  | -43.54 |
| page_loop_tail_handling (diagnostic proxy, not exclusive) | 0.000000000 | 0.000003112 | 0.000003112 |  | -17.76 |
| contiguous_capacity_base_setup | 0.000003110 | 0.000000000 | -0.000003110 | 0.00 | 17.75 |
| contiguous_stride_setup | 0.000000000 | 0.000000000 | 0.000000000 |  | -0.00 |

## 7 Tokens
- Direct exported-kernel gap: -0.000040667 ms
- Modeled stage gap: -0.000005001 ms
- Unclassified differential remainder: -0.000017279 ms

| Stage | Contiguous ms | Page-major ms | Delta ms | Ratio | % direct gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation_setup | 0.000003408 | 0.000004258 | 0.000000850 | 1.25 | -2.09 |
| metadata_preparation | 0.000003110 | 0.000003110 | 0.000000000 | 1.00 | -0.00 |
| qk_score_generation_plus_max | 0.000065574 | 0.000070370 | 0.000004796 | 1.07 | -11.79 |
| softmax_exp_sum | 0.000115962 | 0.000122074 | 0.000006112 | 1.05 | -15.03 |
| standalone_reciprocal | 0.000000000 | 0.000000000 | 0.000000000 |  | -0.00 |
| v_accumulation | 0.000122592 | 0.000085740 | -0.000036852 | 0.70 | 90.62 |
| output_store_read | 0.000060296 | 0.000060296 | 0.000000000 | 1.00 | -0.00 |
| logical_page_count_setup | 0.000000000 | 0.000003482 | 0.000003482 |  | -8.56 |
| block_table_validation_cache | 0.000000000 | 0.000001334 | 0.000001334 |  | -3.28 |
| k_page_base_generation (diagnostic proxy, not exclusive) | 0.000000000 | 0.000007630 | 0.000007630 |  | -18.76 |
| v_page_base_generation (diagnostic proxy, not exclusive) | 0.000000000 | 0.000007628 | 0.000007628 |  | -18.76 |
| page_loop_tail_handling (diagnostic proxy, not exclusive) | 0.000000000 | 0.000003111 | 0.000003111 |  | -7.65 |
| contiguous_capacity_base_setup | 0.000003110 | 0.000000000 | -0.000003110 | 0.00 | 7.65 |
| contiguous_stride_setup | 0.000000000 | 0.000000000 | 0.000000000 |  | -0.00 |

## 8 Tokens
- Direct exported-kernel gap: -0.000045944 ms
- Modeled stage gap: -0.000013887 ms
- Unclassified differential remainder: -0.000013722 ms

| Stage | Contiguous ms | Page-major ms | Delta ms | Ratio | % direct gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation_setup | 0.000003408 | 0.000004258 | 0.000000850 | 1.25 | -1.85 |
| metadata_preparation | 0.000003111 | 0.000003110 | -0.000000001 | 1.00 | 0.00 |
| qk_score_generation_plus_max | 0.000073889 | 0.000079222 | 0.000005333 | 1.07 | -11.61 |
| softmax_exp_sum | 0.000135702 | 0.000134444 | -0.000001258 | 0.99 | 2.74 |
| standalone_reciprocal | 0.000000000 | 0.000000000 | 0.000000000 |  | -0.00 |
| v_accumulation | 0.000136776 | 0.000097924 | -0.000038852 | 0.72 | 84.56 |
| output_store_read | 0.000060296 | 0.000060296 | 0.000000000 | 1.00 | -0.00 |
| logical_page_count_setup | 0.000000000 | 0.000003482 | 0.000003482 |  | -7.58 |
| block_table_validation_cache | 0.000000000 | 0.000001334 | 0.000001334 |  | -2.90 |
| k_page_base_generation (diagnostic proxy, not exclusive) | 0.000000000 | 0.000007630 | 0.000007630 |  | -16.61 |
| v_page_base_generation (diagnostic proxy, not exclusive) | 0.000000000 | 0.000007630 | 0.000007630 |  | -16.61 |
| page_loop_tail_handling (diagnostic proxy, not exclusive) | 0.000000000 | 0.000003112 | 0.000003112 |  | -6.77 |
| contiguous_capacity_base_setup | 0.000003110 | 0.000000000 | -0.000003110 | 0.00 | 6.77 |
| contiguous_stride_setup | 0.000000000 | 0.000000000 | 0.000000000 |  | -0.00 |

## 9 Tokens
- Direct exported-kernel gap: -0.000040556 ms
- Modeled stage gap: 0.000009461 ms
- Unclassified differential remainder: -0.000016297 ms

| Stage | Contiguous ms | Page-major ms | Delta ms | Ratio | % direct gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation_setup | 0.000003408 | 0.000004260 | 0.000000852 | 1.25 | -2.10 |
| metadata_preparation | 0.000003110 | 0.000003110 | 0.000000000 | 1.00 | -0.00 |
| qk_score_generation_plus_max | 0.000082222 | 0.000091406 | 0.000009184 | 1.11 | -22.65 |
| softmax_exp_sum | 0.000153444 | 0.000151555 | -0.000001889 | 0.99 | 4.66 |
| standalone_reciprocal | 0.000000000 | 0.000000000 | 0.000000000 |  | -0.00 |
| v_accumulation | 0.000150926 | 0.000113926 | -0.000037000 | 0.75 | 91.23 |
| output_store_read | 0.000060296 | 0.000060296 | 0.000000000 | 1.00 | -0.00 |
| logical_page_count_setup | 0.000000000 | 0.000003482 | 0.000003482 |  | -8.59 |
| block_table_validation_cache | 0.000000000 | 0.000004222 | 0.000004222 |  | -10.41 |
| k_page_base_generation (diagnostic proxy, not exclusive) | 0.000000000 | 0.000015296 | 0.000015296 |  | -37.72 |
| v_page_base_generation (diagnostic proxy, not exclusive) | 0.000000000 | 0.000015296 | 0.000015296 |  | -37.72 |
| page_loop_tail_handling (diagnostic proxy, not exclusive) | 0.000000000 | 0.000003112 | 0.000003112 |  | -7.67 |
| contiguous_capacity_base_setup | 0.000003110 | 0.000000000 | -0.000003110 | 0.00 | 7.67 |
| contiguous_stride_setup | 0.000000000 | 0.000000000 | 0.000000000 |  | -0.00 |

## 16 Tokens
- Direct exported-kernel gap: -0.000055463 ms
- Modeled stage gap: -0.000014758 ms
- Unclassified differential remainder: -0.000006872 ms

| Stage | Contiguous ms | Page-major ms | Delta ms | Ratio | % direct gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation_setup | 0.000003408 | 0.000004258 | 0.000000850 | 1.25 | -1.53 |
| metadata_preparation | 0.000003111 | 0.000003110 | -0.000000001 | 1.00 | 0.00 |
| qk_score_generation_plus_max | 0.000140482 | 0.000152000 | 0.000011518 | 1.08 | -20.77 |
| softmax_exp_sum | 0.000293332 | 0.000292962 | -0.000000370 | 1.00 | 0.67 |
| standalone_reciprocal | 0.000000000 | 0.000000000 | 0.000000000 |  | -0.00 |
| v_accumulation | 0.000262962 | 0.000197778 | -0.000065184 | 0.75 | 117.53 |
| output_store_read | 0.000060296 | 0.000060296 | 0.000000000 | 1.00 | -0.00 |
| logical_page_count_setup | 0.000000000 | 0.000003482 | 0.000003482 |  | -6.28 |
| block_table_validation_cache | 0.000000000 | 0.000004224 | 0.000004224 |  | -7.62 |
| k_page_base_generation (diagnostic proxy, not exclusive) | 0.000000000 | 0.000015260 | 0.000015260 |  | -27.51 |
| v_page_base_generation (diagnostic proxy, not exclusive) | 0.000000000 | 0.000015296 | 0.000015296 |  | -27.58 |
| page_loop_tail_handling (diagnostic proxy, not exclusive) | 0.000000000 | 0.000003110 | 0.000003110 |  | -5.61 |
| contiguous_capacity_base_setup | 0.000003110 | 0.000000000 | -0.000003110 | 0.00 | 5.61 |
| contiguous_stride_setup | 0.000000000 | 0.000000000 | 0.000000000 |  | -0.00 |

## 32 Tokens
- Direct exported-kernel gap: -0.000077220 ms
- Modeled stage gap: -0.000016095 ms
- Unclassified differential remainder: 0.000002873 ms

| Stage | Contiguous ms | Page-major ms | Delta ms | Ratio | % direct gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation_setup | 0.000003408 | 0.000004258 | 0.000000850 | 1.25 | -1.10 |
| metadata_preparation | 0.000003112 | 0.000003112 | 0.000000000 | 1.00 | -0.00 |
| qk_score_generation_plus_max | 0.000274851 | 0.000296998 | 0.000022147 | 1.08 | -28.68 |
| softmax_exp_sum | 0.000573886 | 0.000568849 | -0.000005037 | 0.99 | 6.52 |
| standalone_reciprocal | 0.000000000 | 0.000000000 | 0.000000000 |  | -0.00 |
| v_accumulation | 0.000502275 | 0.000397110 | -0.000105165 | 0.79 | 136.19 |
| output_store_read | 0.000060296 | 0.000060296 | 0.000000000 | 1.00 | -0.00 |
| logical_page_count_setup | 0.000000000 | 0.000003482 | 0.000003482 |  | -4.51 |
| block_table_validation_cache | 0.000000000 | 0.000006740 | 0.000006740 |  | -8.73 |
| k_page_base_generation (diagnostic proxy, not exclusive) | 0.000000000 | 0.000030556 | 0.000030556 |  | -39.57 |
| v_page_base_generation (diagnostic proxy, not exclusive) | 0.000000000 | 0.000030444 | 0.000030444 |  | -39.43 |
| page_loop_tail_handling (diagnostic proxy, not exclusive) | 0.000000000 | 0.000003112 | 0.000003112 |  | -4.03 |
| contiguous_capacity_base_setup | 0.000003110 | 0.000000000 | -0.000003110 | 0.00 | 4.03 |
| contiguous_stride_setup | 0.000000000 | 0.000000000 | 0.000000000 |  | -0.00 |

## 64 Tokens
- Direct exported-kernel gap: -0.000175810 ms
- Modeled stage gap: -0.000034870 ms
- Unclassified differential remainder: -0.000015940 ms

| Stage | Contiguous ms | Page-major ms | Delta ms | Ratio | % direct gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation_setup | 0.000003408 | 0.000004258 | 0.000000850 | 1.25 | -0.48 |
| metadata_preparation | 0.000003110 | 0.000003110 | 0.000000000 | 1.00 | -0.00 |
| qk_score_generation_plus_max | 0.000542720 | 0.000585960 | 0.000043240 | 1.08 | -24.59 |
| softmax_exp_sum | 0.001127660 | 0.001115880 | -0.000011780 | 0.99 | 6.70 |
| standalone_reciprocal | 0.000000000 | 0.000000000 | 0.000000000 |  | -0.00 |
| v_accumulation | 0.001000250 | 0.000795922 | -0.000204328 | 0.80 | 116.22 |
| output_store_read | 0.000060260 | 0.000060296 | 0.000000036 | 1.00 | -0.02 |
| logical_page_count_setup | 0.000000000 | 0.000003482 | 0.000003482 |  | -1.98 |
| block_table_validation_cache | 0.000000000 | 0.000011740 | 0.000011740 |  | -6.68 |
| k_page_base_generation (diagnostic proxy, not exclusive) | 0.000000000 | 0.000061110 | 0.000061110 |  | -34.76 |
| v_page_base_generation (diagnostic proxy, not exclusive) | 0.000000000 | 0.000061112 | 0.000061112 |  | -34.76 |
| page_loop_tail_handling (diagnostic proxy, not exclusive) | 0.000000000 | 0.000003110 | 0.000003110 |  | -1.77 |
| contiguous_capacity_base_setup | 0.000003110 | 0.000000000 | -0.000003110 | 0.00 | 1.77 |
| contiguous_stride_setup | 0.000000000 | 0.000000000 | 0.000000000 |  | -0.00 |
