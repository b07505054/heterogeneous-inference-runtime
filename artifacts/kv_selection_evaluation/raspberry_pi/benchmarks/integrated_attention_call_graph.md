# Integrated Attention Call Graph

## Contiguous Decode

benchmark/session caller
-> deployment.execution_plan.contiguous_kv_cache.ContiguousKVAttentionSession.decode(q: array('f')) -> array('f')
-> state and valid-token validation
-> _product([num_kv_heads, head_dim])
-> _ptr(q), _ptr(k_cache), _ptr(v_cache), _ptr(output), _ptr(workspace)
-> configured ctypes function hir_cpu_attention_decode_contiguous_kv_fp32
-> output copy: array('f', self._output)
-> counter/accounting updates

## Page-Major Paged Decode

benchmark/session caller
-> deployment.execution_plan.paged_kv_cache.PagedKVAttentionSession.decode(q: array('f')) -> array('f')
-> state validation
-> _validate_live()
-> KVPageManager.validate_invariants()
-> valid_token_count(request_id)
-> block_table(request_id)
-> cached native-compatible block-table validation or refresh
-> _fp(q), _fp(k_pages), _fp(v_pages), _ip(block_table), _ip(physical_page_cache), _fp(output), _fp(workspace)
-> configured ctypes function hir_cpu_attention_decode_paged_kv_page_major_fp32
-> output copy: array('f', self.out)
-> counter/accounting updates
