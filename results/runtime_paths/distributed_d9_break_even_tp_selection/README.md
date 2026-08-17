# D9 Break-Even TP Selection

Decision rule: select TP2 iff `estimated_compute_savings_us - estimated_communication_penalty_us - estimated_runtime_residual_us > decision_margin_us`.

Communication penalty is collective-instance-aware: `call_count * Phase1Profile(collective_kind, bytes_per_call)`. Overlap assumption: `zero`.

Compute savings uses the existing regression latency delta plus the Phase 4D structural compute-scale calibration recorded in the target profile; it does not branch on model name.

D6 accuracy: 0.833, mean regret us: 1003.205.
D7 accuracy: 0.833, mean regret us: 1003.205.
D9 accuracy: 1.000, mean regret us: 0.000, max regret us: 0.000.
Corrective flips: 2; harmful flips: 0.

Success question answer: yes, D9 recovers both sides of the measured TP1/TP2 boundary without model-name heuristics.
