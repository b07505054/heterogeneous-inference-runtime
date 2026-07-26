# Integrated Attention Overhead Report

Timer median overhead: 0.0920 us

## 1 Tokens
Contiguous steady median: 11.8878 us
Paged steady median: 19.1044 us
Gap: 7.2167 us
Measured delta: 11.9391 us
Unclassified gap: -4.7225 us
Gap accounted: 165.44%

| Stage | Contiguous us | Paged us | Delta us | Ratio | % gap |
|---|---:|---:|---:|---:|---:|
| block_table_copy_current | 0.0000 | 1.1919 | 1.1919 |  | 16.52 |
| block_table_copy_old_path_simulated | 0.0000 | 1.0492 | 1.0492 |  | 14.54 |
| block_table_lookup | 0.0000 | 0.3025 | 0.3025 |  | 4.19 |
| block_table_python_to_native_materialization_current | 0.0000 | 2.4391 | 2.4391 |  | 33.80 |
| contiguous_kv_buffer_lookup | 0.2042 | 0.0000 | -0.2042 | 0.00 | -2.83 |
| ctypes_ffi_argument_preparation | 5.1821 | 9.1830 | 4.0010 | 1.77 | 55.44 |
| native_function_call | 0.0000 | 0.0000 | 0.0000 |  | 0.00 |
| output_handling_conversion | 0.3817 | 0.3814 | -0.0004 | 1.00 | -0.01 |
| page_session_invariant_validation | 0.0000 | 1.3479 | 1.3479 |  | 18.68 |
| physical_kv_pool_lookup | 0.0000 | 0.3182 | 0.3182 |  | 4.41 |
| physical_page_metadata_preparation | 0.0000 | 1.1449 | 1.1449 |  | 15.86 |
| request_session_lookup | 0.0000 | 0.2060 | 0.2060 |  | 2.85 |
| shape_and_valid_token_calculation | 0.1971 | 0.0000 | -0.1971 | 0.00 | -2.73 |
| telemetry_accounting | 0.2716 | 0.2681 | -0.0035 | 0.99 | -0.05 |
| valid_token_calculation | 0.0000 | 0.3437 | 0.3437 |  | 4.76 |

## 7 Tokens
Contiguous steady median: 12.2413 us
Paged steady median: 19.3846 us
Gap: 7.1433 us
Measured delta: 12.0201 us
Unclassified gap: -4.8768 us
Gap accounted: 168.27%

| Stage | Contiguous us | Paged us | Delta us | Ratio | % gap |
|---|---:|---:|---:|---:|---:|
| block_table_copy_current | 0.0000 | 1.1931 | 1.1931 |  | 16.70 |
| block_table_copy_old_path_simulated | 0.0000 | 1.0313 | 1.0313 |  | 14.44 |
| block_table_lookup | 0.0000 | 0.3208 | 0.3208 |  | 4.49 |
| block_table_python_to_native_materialization_current | 0.0000 | 2.4211 | 2.4211 |  | 33.89 |
| contiguous_kv_buffer_lookup | 0.2038 | 0.0000 | -0.2038 | 0.00 | -2.85 |
| ctypes_ffi_argument_preparation | 5.2266 | 9.2815 | 4.0550 | 1.78 | 56.77 |
| native_function_call | 0.0000 | 0.0000 | 0.0000 |  | 0.00 |
| output_handling_conversion | 0.3851 | 0.3819 | -0.0032 | 0.99 | -0.04 |
| page_session_invariant_validation | 0.0000 | 1.3760 | 1.3760 |  | 19.26 |
| physical_kv_pool_lookup | 0.0000 | 0.3257 | 0.3257 |  | 4.56 |
| physical_page_metadata_preparation | 0.0000 | 1.1559 | 1.1559 |  | 16.18 |
| request_session_lookup | 0.0000 | 0.2057 | 0.2057 |  | 2.88 |
| shape_and_valid_token_calculation | 0.2039 | 0.0000 | -0.2039 | 0.00 | -2.85 |
| telemetry_accounting | 0.2711 | 0.2664 | -0.0047 | 0.98 | -0.07 |
| valid_token_calculation | 0.0000 | 0.3512 | 0.3512 |  | 4.92 |

## 8 Tokens
Contiguous steady median: 12.2757 us
Paged steady median: 19.5405 us
Gap: 7.2648 us
Measured delta: 12.1441 us
Unclassified gap: -4.8793 us
Gap accounted: 167.16%

| Stage | Contiguous us | Paged us | Delta us | Ratio | % gap |
|---|---:|---:|---:|---:|---:|
| block_table_copy_current | 0.0000 | 1.1826 | 1.1826 |  | 16.28 |
| block_table_copy_old_path_simulated | 0.0000 | 1.0354 | 1.0354 |  | 14.25 |
| block_table_lookup | 0.0000 | 0.3024 | 0.3024 |  | 4.16 |
| block_table_python_to_native_materialization_current | 0.0000 | 2.4563 | 2.4563 |  | 33.81 |
| contiguous_kv_buffer_lookup | 0.2040 | 0.0000 | -0.2040 | 0.00 | -2.81 |
| ctypes_ffi_argument_preparation | 5.1735 | 9.3470 | 4.1735 | 1.81 | 57.45 |
| native_function_call | 0.0000 | 0.0000 | 0.0000 |  | 0.00 |
| output_handling_conversion | 0.3842 | 0.3809 | -0.0033 | 0.99 | -0.05 |
| page_session_invariant_validation | 0.0000 | 1.3631 | 1.3631 |  | 18.76 |
| physical_kv_pool_lookup | 0.0000 | 0.3166 | 0.3166 |  | 4.36 |
| physical_page_metadata_preparation | 0.0000 | 1.1694 | 1.1694 |  | 16.10 |
| request_session_lookup | 0.0000 | 0.2057 | 0.2057 |  | 2.83 |
| shape_and_valid_token_calculation | 0.1973 | 0.0000 | -0.1973 | 0.00 | -2.72 |
| telemetry_accounting | 0.2673 | 0.2628 | -0.0045 | 0.98 | -0.06 |
| valid_token_calculation | 0.0000 | 0.3482 | 0.3482 |  | 4.79 |

## 9 Tokens
Contiguous steady median: 12.3591 us
Paged steady median: 19.6896 us
Gap: 7.3306 us
Measured delta: 12.4077 us
Unclassified gap: -5.0772 us
Gap accounted: 169.26%

| Stage | Contiguous us | Paged us | Delta us | Ratio | % gap |
|---|---:|---:|---:|---:|---:|
| block_table_copy_current | 0.0000 | 1.2711 | 1.2711 |  | 17.34 |
| block_table_copy_old_path_simulated | 0.0000 | 1.1030 | 1.1030 |  | 15.05 |
| block_table_lookup | 0.0000 | 0.3059 | 0.3059 |  | 4.17 |
| block_table_python_to_native_materialization_current | 0.0000 | 2.5261 | 2.5261 |  | 34.46 |
| contiguous_kv_buffer_lookup | 0.2033 | 0.0000 | -0.2033 | 0.00 | -2.77 |
| ctypes_ffi_argument_preparation | 5.2136 | 9.3326 | 4.1189 | 1.79 | 56.19 |
| native_function_call | 0.0000 | 0.0000 | 0.0000 |  | 0.00 |
| output_handling_conversion | 0.3805 | 0.3808 | 0.0003 | 1.00 | 0.00 |
| page_session_invariant_validation | 0.0000 | 1.4758 | 1.4758 |  | 20.13 |
| physical_kv_pool_lookup | 0.0000 | 0.3162 | 0.3162 |  | 4.31 |
| physical_page_metadata_preparation | 0.0000 | 1.1632 | 1.1632 |  | 15.87 |
| request_session_lookup | 0.0000 | 0.2044 | 0.2044 |  | 2.79 |
| shape_and_valid_token_calculation | 0.2039 | 0.0000 | -0.2039 | 0.00 | -2.78 |
| telemetry_accounting | 0.2703 | 0.2574 | -0.0129 | 0.95 | -0.18 |
| valid_token_calculation | 0.0000 | 0.3428 | 0.3428 |  | 4.68 |

## 16 Tokens
Contiguous steady median: 12.6417 us
Paged steady median: 19.9137 us
Gap: 7.2720 us
Measured delta: 12.4257 us
Unclassified gap: -5.1537 us
Gap accounted: 170.87%

| Stage | Contiguous us | Paged us | Delta us | Ratio | % gap |
|---|---:|---:|---:|---:|---:|
| block_table_copy_current | 0.0000 | 1.2730 | 1.2730 |  | 17.51 |
| block_table_copy_old_path_simulated | 0.0000 | 1.1116 | 1.1116 |  | 15.29 |
| block_table_lookup | 0.0000 | 0.3039 | 0.3039 |  | 4.18 |
| block_table_python_to_native_materialization_current | 0.0000 | 2.5286 | 2.5286 |  | 34.77 |
| contiguous_kv_buffer_lookup | 0.2032 | 0.0000 | -0.2032 | 0.00 | -2.79 |
| ctypes_ffi_argument_preparation | 5.2272 | 9.3185 | 4.0913 | 1.78 | 56.26 |
| native_function_call | 0.0000 | 0.0000 | 0.0000 |  | 0.00 |
| output_handling_conversion | 0.3736 | 0.3806 | 0.0070 | 1.02 | 0.10 |
| page_session_invariant_validation | 0.0000 | 1.4892 | 1.4892 |  | 20.48 |
| physical_kv_pool_lookup | 0.0000 | 0.3194 | 0.3194 |  | 4.39 |
| physical_page_metadata_preparation | 0.0000 | 1.1565 | 1.1565 |  | 15.90 |
| request_session_lookup | 0.0000 | 0.2043 | 0.2043 |  | 2.81 |
| shape_and_valid_token_calculation | 0.1963 | 0.0000 | -0.1963 | 0.00 | -2.70 |
| telemetry_accounting | 0.2657 | 0.2612 | -0.0046 | 0.98 | -0.06 |
| valid_token_calculation | 0.0000 | 0.3450 | 0.3450 |  | 4.74 |

## 32 Tokens
Contiguous steady median: 13.2505 us
Paged steady median: 21.0948 us
Gap: 7.8443 us
Measured delta: 13.4063 us
Unclassified gap: -5.5620 us
Gap accounted: 170.91%

| Stage | Contiguous us | Paged us | Delta us | Ratio | % gap |
|---|---:|---:|---:|---:|---:|
| block_table_copy_current | 0.0000 | 1.4236 | 1.4236 |  | 18.15 |
| block_table_copy_old_path_simulated | 0.0000 | 1.2749 | 1.2749 |  | 16.25 |
| block_table_lookup | 0.0000 | 0.3088 | 0.3088 |  | 3.94 |
| block_table_python_to_native_materialization_current | 0.0000 | 2.7168 | 2.7168 |  | 34.63 |
| contiguous_kv_buffer_lookup | 0.2041 | 0.0000 | -0.2041 | 0.00 | -2.60 |
| ctypes_ffi_argument_preparation | 5.2636 | 9.5883 | 4.3247 | 1.82 | 55.13 |
| native_function_call | 0.0000 | 0.0000 | 0.0000 |  | 0.00 |
| output_handling_conversion | 0.3787 | 0.3864 | 0.0077 | 1.02 | 0.10 |
| page_session_invariant_validation | 0.0000 | 1.7078 | 1.7078 |  | 21.77 |
| physical_kv_pool_lookup | 0.0000 | 0.3169 | 0.3169 |  | 4.04 |
| physical_page_metadata_preparation | 0.0000 | 1.1779 | 1.1779 |  | 15.02 |
| request_session_lookup | 0.0000 | 0.2057 | 0.2057 |  | 2.62 |
| shape_and_valid_token_calculation | 0.1974 | 0.0000 | -0.1974 | 0.00 | -2.52 |
| telemetry_accounting | 0.2665 | 0.2623 | -0.0042 | 0.98 | -0.05 |
| valid_token_calculation | 0.0000 | 0.3471 | 0.3471 |  | 4.43 |

## 64 Tokens
Contiguous steady median: 14.4578 us
Paged steady median: 23.0187 us
Gap: 8.5609 us
Measured delta: 15.0719 us
Unclassified gap: -6.5110 us
Gap accounted: 176.06%

| Stage | Contiguous us | Paged us | Delta us | Ratio | % gap |
|---|---:|---:|---:|---:|---:|
| block_table_copy_current | 0.0000 | 1.7023 | 1.7023 |  | 19.88 |
| block_table_copy_old_path_simulated | 0.0000 | 1.5341 | 1.5341 |  | 17.92 |
| block_table_lookup | 0.0000 | 0.3326 | 0.3326 |  | 3.88 |
| block_table_python_to_native_materialization_current | 0.0000 | 2.9616 | 2.9616 |  | 34.59 |
| contiguous_kv_buffer_lookup | 0.2083 | 0.0000 | -0.2083 | 0.00 | -2.43 |
| ctypes_ffi_argument_preparation | 5.2160 | 9.8971 | 4.6811 | 1.90 | 54.68 |
| native_function_call | 0.0000 | 0.0000 | 0.0000 |  | 0.00 |
| output_handling_conversion | 0.3874 | 0.3869 | -0.0005 | 1.00 | -0.01 |
| page_session_invariant_validation | 0.0000 | 2.2280 | 2.2280 |  | 26.03 |
| physical_kv_pool_lookup | 0.0000 | 0.3279 | 0.3279 |  | 3.83 |
| physical_page_metadata_preparation | 0.0000 | 1.1623 | 1.1623 |  | 13.58 |
| request_session_lookup | 0.0000 | 0.2108 | 0.2108 |  | 2.46 |
| shape_and_valid_token_calculation | 0.2085 | 0.0000 | -0.2085 | 0.00 | -2.43 |
| telemetry_accounting | 0.2671 | 0.2625 | -0.0046 | 0.98 | -0.05 |
| valid_token_calculation | 0.0000 | 0.3533 | 0.3533 |  | 4.13 |
