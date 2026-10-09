# Gemma 4 E2B saved-session example

`semantic.json` provides the canonical Gemma 4 E2B semantic graph and layer
attributes, including the PLE external input. `execution.json` provides the
reference-demo execution graphs, and `session.json` provides the saved-session
report metadata.

No raw tensors are available in this example. `tensor_index.json` is
deliberately empty. Do not use precomputed demo metrics as raw tensor
measurements. Add independently verified indexed Safetensors (`.safetensors`)
captures in a separate session directory to compute real comparisons.

`semantic_evidence.json` and `evidence/` bundle the pinned extraction evidence
for the Source viewer. Source hashes and claim pointers are preserved.
