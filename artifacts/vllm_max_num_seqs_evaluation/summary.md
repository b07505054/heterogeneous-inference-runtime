# Real vLLM `max_num_seqs` measured-policy evaluation

| workload | candidate | effective max_num_seqs | sessions | requests | success | failure | TTFT p50/p95/p99 ms | TPOT p50/p95/p99 ms | E2E p50/p95/p99 ms | output tok/s | req/s | queue p95 | peak MiB | OOM |
|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|
| S1 | vllm_max_num_seqs_1 | 1 | 3 | 150 | 150 | 0 | 150.668678/152.837134/153.95077 | 11.365326/11.448763/11.481389 | 321.198829/324.123517/324.969131 | 49.811313 | 3.113207 | not_available | 2902 | 0 |
| S1 | vllm_max_num_seqs_2 | 2 | 3 | 150 | 150 | 0 | 151.017493/153.764522/154.120749 | 11.387512/11.490227/11.509563 | 321.618244/325.895779/326.103196 | 49.672522 | 3.104533 | not_available | 2862 | 0 |
| S1 | vllm_max_num_seqs_4 | 4 | 3 | 150 | 150 | 0 | 150.606924/153.373527/154.433098 | 11.377766/11.454005/11.475393 | 321.222583/324.945109/325.757779 | 50.006748 | 3.125422 | not_available | 2892 | 0 |
| S1 | vllm_max_num_seqs_8 | 8 | 3 | 150 | 150 | 0 | 150.552014/152.813209/154.133434 | 11.37701/11.455255/11.468214 | 321.158843/324.417606/325.641917 | 49.767009 | 3.110438 | not_available | 2906 | 0 |
| S1 | vllm_max_num_seqs_default | not_available | 3 | 150 | 150 | 0 | 153.714851/155.248445/155.616765 | 11.448299/11.531296/11.553306 | 325.612901/327.769401/328.259692 | 49.297917 | 3.08112 | not_available | 3126 | 0 |
| S2 | vllm_max_num_seqs_1 | 1 | 3 | 150 | 150 | 0 | 1397.823881/1408.412295/1410.482443 | 11.485537/11.54892/11.575473 | 1662.152898/1674.381771/1675.991089 | 57.739338 | 2.405806 | not_available | 2902 | 0 |
| S2 | vllm_max_num_seqs_2 | 2 | 3 | 150 | 150 | 0 | 4221.637028/4379.505801/4383.441526 | 164.578887/170.50476/171.106336 | 8136.446297/8145.849364/8177.979177 | 11.807839 | 0.491993 | not_available | 2862 | 0 |
| S2 | vllm_max_num_seqs_4 | 4 | 3 | 150 | 150 | 0 | 630.674526/641.600991/643.703799 | 170.576269/184.803106/184.836378 | 4554.396831/4556.055992/4556.851138 | 20.476386 | 0.853183 | not_available | 2894 | 0 |
| S2 | vllm_max_num_seqs_8 | 8 | 3 | 150 | 150 | 0 | 630.386798/633.333418/643.538251 | 170.586404/184.815843/184.830219 | 4554.34813/4556.105294/4559.566462 | 20.457355 | 0.85239 | not_available | 2906 | 0 |
| S2 | vllm_max_num_seqs_default | not_available | 3 | 150 | 150 | 0 | 632.054553/649.8632/652.384548 | 170.618215/184.888853/184.90489 | 4556.453268/4573.862748/4576.287636 | 20.418469 | 0.85077 | not_available | 3106 | 0 |
| S3 | vllm_max_num_seqs_1 | 1 | 3 | 150 | 150 | 0 | 3715.525796/3741.789208/3745.597789 | 11.53857/11.595693/11.623378 | 4073.22946/4102.205995/4104.93824 | 62.785642 | 1.962051 | not_available | 2902 | 0 |
| S3 | vllm_max_num_seqs_2 | 2 | 3 | 150 | 150 | 0 | 16471.537876/16650.255297/16738.820906 | 167.057147/171.805377/171.843258 | 21721.542669/21895.335888/21906.870895 | 11.777128 | 0.368035 | not_available | 2862 | 0 |
| S3 | vllm_max_num_seqs_4 | 4 | 3 | 150 | 150 | 0 | 6542.276452/6550.508303/6594.435212 | 170.626559/181.193916/182.524175 | 11833.893867/11878.532966/11927.470042 | 20.933956 | 0.654186 | not_available | 2894 | 0 |
| S3 | vllm_max_num_seqs_8 | 8 | 3 | 150 | 150 | 0 | 925.068657/938.321685/939.076759 | 171.454795/191.476745/191.506083 | 6240.075054/6242.527962/6243.337724 | 37.34995 | 1.167186 | not_available | 2928 | 0 |
| S3 | vllm_max_num_seqs_default | not_available | 3 | 150 | 150 | 0 | 862.584952/1187.696671/1188.980401 | 172.841682/194.403571/194.461953 | 6333.205737/6531.194172/6533.182237 | 36.492536 | 1.140392 | not_available | 3106 | 0 |

All 45 baseline sessions and nine independent compiler-plan proof sessions used real vLLM 0.24.0 execution on the NVIDIA GeForce GTX 1650 with Max-Q Design. The default flag was omitted; its resolved effective value was not exposed by vLLM and is reported as `not_available`. Queue wait, separate prefill/decode time, and exact KV-cache usage were not exposed and are not reported as zero.

Compiler selections: S1 latency=8, throughput=4, balanced=4; S2 latency=8, throughput=1, balanced=1; S3 latency=8, throughput=1, balanced=1. Every independent proof session executed the exact selected value with `runtime_policy_reselection_count=0`.

Truth boundary: target/model/workload-specific measured policy for one vLLM setting. This is not a predictive cost model, universal optimum, scheduler-internal control, multi-GPU evidence, or production SLO guarantee.

Recommended next slice: `max_num_batched_tokens`.
