# Implementation plan: finish one step before starting the next

> Moved from the user's repository `~/Projects/pytorch-collector` at commit
> `fe82ff9` with its uncommitted module-only change (`collect(model,
> modules=[...])`, step 6) on 2026-10-20; this copy is the one maintained from
> now on. Unchanged but for this note and its links to the acceptance records,
> `docs/verification/`, which stayed in that repository with `examples/`, the
> tests of the examples and the `pytorch_collector.equivalence` package.

Status: **all four steps are complete and the first release `0.1.0` passed the
standalone wheel acceptance.** Modules, operators and explicitly typed caches
are captured from one execution; real cached consecutive calls and generate
pass. The runtime flow is in the [design](DESIGN.md); delivery evidence and
reproduction steps are in stage 4 (pytorch-collector's
`docs/verification/stage4.md`), with stages 1 (pytorch-collector's
`docs/verification/stage1.md`), 2 (pytorch-collector's
`docs/verification/stage2.md`) and 3 (pytorch-collector's
`docs/verification/stage3.md`) kept as historical checkpoints.

First-release goal: `collect(model)` automatically observes every registered
module and every visible operator, covers every target-model forward inside the
scope, distinguished by `forward_id`; a real Gemma `model.generate()` must pass
acceptance. Snapshots uniformly exclude registered parameters and recognized
weight representations and keep buffers, dynamic caches and unrecognized derived
results. A uniform in-memory snapshot is read after the whole scope succeeds; if
any capture fails, nothing is readable and neither the model nor the KV cache is
rolled back. The default snapshot budget is 1 GiB, overridable through
`memory_limit_bytes`, failing immediately when exceeded; the measurement is in
step 3. The acceptance scope is CPU eager and excludes the inside of fused
kernels.

What follows is the development order, not a capture level for the user to
choose; no `level` or per-model configuration is added. The first release had no
`modules` option either; step 6 adds it as an opt-in that leaves the default
capture unchanged. Parameters are discovered automatically, special-object
compatibility is reused by type and version, and the uniform snapshot rules are
in the [design](DESIGN.md).

## Test arrangement: small-model regression + Gemma 4 E2B acceptance

-   tiny CPU models (Exact checks of records, aliases, exceptions, resource
    cleanup and budget edges, no checkpoint needed): The full suite before every
    implementation commit
-   Gemma 4 E2B real weights (Module/operator compatibility, consecutive
    inference and snapshot overhead on a real model): From step 1 on, the
    relevant scenario before each stage completes

Real-model acceptance uses the PyTorch checkpoint of `google/gemma-4-E2B-it`,
with the model and tokenizer revision and compatible PyTorch/Transformers
versions pinned during preparation. Reduced configurations or random
initialization may add quick tests but do not count as passing real-weight
acceptance. When weights or resources are unavailable, that is recorded as
incomplete; a stage or the first release is never declared passed through a
skip.

The first round uses CPU eager, text only, batch 1 and a fixed short input; the
model is in eval, with the same dtype and attention backend, random sampling and
automatic compilation off. A run without capture is the baseline against which
the captured run is compared, with identical inputs and initial model state and
separately re-initialized caches. Logits are compared in shape, dtype and value
with a predetermined error threshold and the maximum error reported; generated
tokens are only a supplementary check.

**Compatibility work comes first:** step 1 supports ordinary Mappings and checks
the `UserDict`, `ModelOutput` and `Cache` objects the pinned version actually
passes to modules. The first release adapts only the explicit cache/inner-state
types the Gemma acceptance really uses, listing every tensor field to capture,
the stable paths, and the handling of cycles and repeated references, finished
before the cached acceptance. Tensors in all fields share the parameter filter
and budget rules; unknown cache types or incompatible versions fail explicitly,
extended on demand later without scanning arbitrary object attributes. Tests may
not be made to pass by ignoring extra runtime tensors or removing hooks, and
this work introduces no KV/token semantic analysis. The cache is part of Gemma's
official inputs and outputs, see the [official interface][hf-gemma4-fwd].

[hf-gemma4-fwd]: https://huggingface.co/docs/transformers/v5.7.0/model_doc/gemma4#transformers.Gemma4ForCausalLM.forward

## Step 1: get module capture working first

**Outcome:** the standalone environment pins CPython 3.11.15, PyTorch 2.9.0 and
pytest 8.4.2; the Gemma integration environment uses Transformers 5.17.0. 86
small tests pass; a single uncached forward of the full Gemma 4 E2B-it real
weights passes with 700 actually executed modules, 701 module calls and 1,926
snapshots, logits element-wise identical to the baseline. This checkpoint used a
1 GiB development budget while the real acceptance used an explicit 2 GiB; the
official default and its measurement are in step 3.

**Deliverable:** a standalone Python package and virtual environment
establishing the `collect(model)` lifecycle, automatically observing
root/parent/child modules, producing input/output snapshots, `forward_id`s and
call numbers; module records have `op=None`. This is a development checkpoint
without operator observation.

**Completed with this step:** target claiming, partial-install rollback, no-grad
restoration, multiple sequential root calls, immediate clone, cumulative
snapshot budget, hook order and call pairing checks. Module inputs and outputs
automatically exclude registered parameters and verifiable parameter aliases
through `named_parameters()`; buffers are not an active exclusion source.
Tracking of weight representations created during computation is completed in
step 2 with operator observation. The failure flag takes effect immediately and
clears the staged results of the whole capture; `records` is unreadable before a
successful exit. The budget check and optional override ship with the snapshot
logic; the default is used for development measurement first, with step 3 fixing
the official number.

**Acceptance:** tiny CPU models check the module record set, kwargs / multiple
outputs / `None`, original dtype, repeated calls and shared modules; zero root
calls return an empty result, numbering and pairing of different forwards do not
mix, and in-place modification does not pollute snapshots. Budget edges and
overrun, a later forward's failure making earlier results unreadable, no
recovery from an internally caught exception, and no leftover hooks/claims on
normal and abnormal exit are all verified. Module regressions for the known
F6/F7 are added so result trustworthiness is not judged only at the end.

Registered parameters with biases, norms, frozen and shared parameters are
verified without relying on attribute names or `requires_grad`; ordinary buffers
are kept, and a tensor with parameter identity follows the parameter exclusion
rule. Modifying a successfully returned snapshot is allowed and verified not to
affect the model or other observation points. After an exception only collector
resource cleanup is checked; undoing model/cache updates that already happened
is not required.

**Gemma acceptance:** with the container compatibility needed for this scenario
done, run a short-text single forward on the real weights with `use_cache=False`
first. Check the records of every actually executed registered module that meet
the snapshot rules, the original snapshot dtype, logits identical to the
uncaptured baseline, and hook cleanup after exit. Passing this only proves a
single uncached call of module capture; full weight-representation filtering is
completed in step 2, and cached and real generate are accepted separately in
step 3.

**Commit:** `feat: collect module snapshots safely`; container compatibility and
its regressions are committed as separate logical changes.

## Step 2: add operator capture

**Outcome:** 140 small tests pass. On the pinned version, a real Gemma
short-text single uncached forward captures modules and operators together:
12,134 snapshots, 261,715,877 bytes; logits identical to the independent
baseline. The baseline observed 39 kinds and 5,223 operator calls; the 4,717
operator calls with qualifying tensor records match the complete order and
identity of the module records item by item. Explicit copy/cast rules,
derived-storage write invalidation, internal copy isolation and ownership-based
dispatch mode cleanup are implemented. This checkpoint used the 1 GiB
development default; the official value is in step 3.

**Deliverable:** operator observation added automatically to the same
`collect(model)`; reusing forward/module identity, the snapshot budget and the
result logic, visible operations are captured at the same time through the
pinned-version `TorchDispatchMode`.

**Completed with this step:** inputs to capture are copied before execution, the
original operator runs once, the collector's own operations are excluded, and
the dispatch mode is restored after exceptions. Only operations during root
calls are captured, keeping real operator names and the owning module call.
Explicit weight-representation tracking rules cover recognized transposes,
views, copies and casts; identity is determined before output snapshots, and all
module/operator IN/OUT share the filtering result. Unknown derivations are
captured as ordinary tensors.

**Acceptance:** confirm the visible forms of add, mm, softmax, view/reshape and
factory operations on the pinned version first, then check tensor values; verify
in-place, `out=`, return-value aliasing, operator exceptions and clone failure.
Both kinds of records come from one execution and the outputs match the
uncaptured run; no self-capture recursion, and operations outside the root call
or between two forwards do not leak into the result. Operator and module
snapshots share the budget, and a failure on either observation path fails the
whole capture.

Parameter-filter regressions: embedding indices and outputs are kept and the
weight is not copied; Linear activations and outputs are kept and weight/bias
are not copied; shared weights across modules, parameter views and supported
copy/cast produce no parameter snapshots. Unrecognized results such as `W *
scale` are still captured, and the rule "anything depending on a weight is
excluded" is not propagated. Filtering is checked to happen before clone and
budget deduction; observation points containing only filtered parameters still
keep call identity and pairing checks. The supported representation conversions
are listed without claiming to recognize arbitrary custom derivations.

**Gemma acceptance:** observe modules and operators together in the step 1 real
short-text scenario, checking the call attribution of both record kinds, the
actual visible operator names, and logits identical to the uncaptured baseline.
Verify that parameter filtering covers the embedding, the output head and the
observed parameter views/conversions, so a weight is not copied again at another
observation point. Confirm there is no repeated execution or self-capture, and
record the filtered snapshot count, bytes and runtime; fused operators are still
accepted only at their visible boundary.

**Commit:** `feat: capture dispatched tensor operations`.

## Step 3: verify consecutive capture and fix the default budget

**Status: complete.** 177 small tests pass with no skips. The pinned
Transformers 5.17.0 `DynamicCache` and its two inner layer types are adapted;
the real Gemma hand-written 1 prefill + 3 decodes and the real generate both
pass, saving 513,295,160 and 490,164,856 bytes respectively. Step-by-step
logits, generated tokens and the complete ordered record check pass. A
267,222,053-byte budget is exceeded on the second forward; after rebuilding the
cache and inputs, a new capture with the budget argument omitted recovers fully.
The official default budget is 1 GiB, about 2.1 times the largest measured
scenario. Details in the stage 3 record (pytorch-collector's
`docs/verification/stage3.md`).

**Deliverable:** combined verification of consecutive forwards and real
generate, plus the snapshot usage record of Gemma 4 E2B after uniform parameter
filtering, from which the default `memory_limit_bytes` is fixed and documented.
The observation range and filtering rules are fixed; the caller may adjust only
the budget.

**Acceptance:** consecutive calls of tiny CPU models cover changing input
shapes, conditional branches and shared modules, checking module/op forward
attribution, call pairing and cumulative bytes. When an earlier forward succeeds
and a later one fails at a different observation point, nothing is readable; a
single forward within budget but cumulatively over it fails immediately. After
the caller prepares the needed state again and raises the budget, a new capture
captures fully. The measured model, inputs, forward count and actual snapshot
usage are recorded to justify the default, without describing snapshot bytes as
total process memory.

**Gemma acceptance:** with the cache types and fields the pinned version needs
adapted, enable `use_cache=True` inside one capture, run 1 prefill + 3 decode
forwards by hand, and check the 4 `forward_id`s, each forward's module/op
pairing and the cumulative budget. With the same fixed token input, run the
baseline and the captured path with independent caches and compare logits step
by step. Verify full success with an explicitly sufficient budget, then lower it
so a later forward overruns, verifying that nothing is readable and resources
are cleaned up; after the caller rebuilds the cache, inputs and needed state,
verify recovery in a new capture.

Separately accept the direct entry point with a real `with collect(model):
model.generate(...)`, fixed to CPU eager, text only, `do_sample=False`,
`use_cache=True` and a short generation limit. Compare the uncaptured baseline
in an independent state with the captured run's step-by-step logits, generated
tokens and actual number of root forwards; check numbering by actual calls
rather than taking the token limit as the forward count. Check that operations
outside root calls during generation do not leak into snapshots. Both the
hand-written loop and the real generate must pass; this does not extend to
arbitrary generation configurations or multimodal inputs.

Use the real scenario's filtered record count, snapshot bytes, peak process
memory and runtime to choose the default budget, reporting snapshot budget and
process memory separately and not reusing the old "copy the whole weight table
each time" estimate. If the complete scenario in the agreed scope cannot finish
within available resources, record the blocker; do not shrink the observation
range, filter extra runtime data or downsample and then declare a pass.

**Commit:** `test: verify multi-forward capture and memory limits`; the official
default and its measurement use `feat: set measured snapshot memory budget`.
Logic fixes found along the way are committed individually.

## Step 4: complete first-release acceptance and standalone delivery

**Status: complete.** Exiting a capture from another thread, which left the
original thread's resources uncleaned, is fixed with 9 new regressions; the
other exception combinations were already covered. 186 tests pass with no skips
in both the source environment and the standalone wheel environment. The
external environment does not copy `src`, confirming that the non-editable
package in site-packages is imported; the basic example passes with PyTorch as
the only dependency, and after installing the integration dependencies the real
single forward, consecutive prefill + decode, real generate, budget failure and
default-budget recovery all pass again. The short generation example passes on
both the default and the explicit budget path. Delivered: the `0.1.0` wheel, the
source distribution and the complete evidence (pytorch-collector's
`docs/verification/stage4.md`).

**Deliverable:** runnable examples showing real generate, multiple forwards and
the optional budget, Gemma acceptance run instructions, the wheel, and complete
acceptance records for the small models and the real weights. The records
include dependency versions, model revision, inputs, dtype/backend, error
thresholds, comparison results, forward/record counts, the parameter filtering
and cache support range, budget and resource usage.

**Acceptance:** cover the remaining exception combinations: a root pre-hook
calling a submodule, late-registered hooks, an interruption swallowed
internally, cross-thread/recursion, mode restoration, target replacement and
unsupported data structures. Fix what is found without rewriting tests that
already exist. Finally, in a clean standalone environment, install the wheel
from outside the source directory and run the examples and the full suite.

**Gemma acceptance:** with the installed wheel, rerun the single forward,
hand-written prefill + decode, real generate, budget failure and recovery after
the caller's reset. The first release is declared complete only after both the
small-model regression and the real-model acceptance pass; a missing, skipped or
failed real acceptance does not count.

**Commit:** each logic fix found is committed on its own; the delivery record
uses `docs: verify standalone collector usage`. The first release is declared
complete only after all four steps pass.

## Step 5: record data flow and export the execution graph

**Status: complete.** Version `0.2.0`; the existing `collect`, `Capture.records`
and `Record` are unchanged. 232 tests pass and the full Gemma acceptance passes,
with evidence in the stage 5 acceptance record (pytorch-collector's
`docs/verification/stage5.md`). Data flow adds about 8–11% to the capture time
of Gemma's four forwards and stays on by default under the 10% criterion.

**Deliverable:** `capture.ops`, `capture.edges`, `to_model_explorer`, defined in
[design step 6](DESIGN.md). The runtime still depends on PyTorch only, and the
export function only returns a dict.

**Acceptance:** small models cover sequential connections, residual fan-out,
repeated calls of shared modules, parameters and buffers, in-place / `out=` /
`_foreach` writes, writes through views, multi-output operators, operations
outside the root and state across forwards; assert exactly one source per tensor
input, and that after failure `ops` / `edges` are unreadable together with
`records`. The export survives a JSON round trip, the node count equals the
operator calls of the selected forwards, and both ends of every link are in the
same graph.

**Real-model acceptance:** the integration group runs prefill + decode with
tiny-random-Llama and a `DynamicCache`, asserting that every `constant` consumer
is one of the listed entry operators and that decode's KV sources point at
prefill. Gemma 4 E2B runs once in the same fixed configuration, recording the
count of each `kind`, the `constant` entry list with reasons, whether decoder
layers of the same kind are isomorphic subgraphs, and the capture time with data
flow on and off. If the increase exceeds 10%, data flow becomes opt-in.

**Commit:** `refactor: separate container traversal from snapshot copying`,
`docs: design data-flow edges for one captured execution`, `feat: record
operator data-flow edges from tensor identity`, `feat: attribute parameter and
buffer inputs by registered name`, `feat: reassign tensor producers after
in-place and out= writes`, `test: verify cache-carried edges across prefill and
decode`, `feat: export one capture as a Model Explorer graph`, `perf: share one
traversal between data flow and snapshots`, `test: verify real-model edge
attribution and layer isomorphism`; the delivery record uses `docs: record
stage-five graph verification` and `build: prepare the 0.2.0 distribution`.

## Step 6: capture only selected modules

**Status: implemented, not released.** `collect(model, modules=[...])`, defined
in [design step 7](DESIGN.md): hooks only on the selected modules (and the root,
to number forwards), no dispatch mode, module records only, `ops` / `edges` /
`outputs` and the graph export refused, any device. The default capture is
unchanged: the 251 existing tests pass unmodified, with 11 new ones.

**Acceptance:** small models check that the records are exactly the selected
modules' records of a full capture, bit for bit, including a tiny Gemma's
`DynamicCache` and per-layer input over three cached forwards; that only the
selected modules and the root carry hooks, no dispatch mode is active and
everything is removed after success, a model exception and a budget failure;
root selection, calls outside the root, shared-module aliases, parameter and
view exclusion, unknown paths and objects, a selection changed before entry, the
meta device (always) and MPS (when present).

**Real-model measurement:** Gemma 4 E2B-it at the pinned revision, BF16, eager
attention, prompt "The capital of France is Paris, and the capital of Germany
is" (13 tokens), one prefill and 24 greedy decodes with one `DynamicCache`, 8
intra-op threads, Apple M2 Max, one process per run after a warm-up generation.
Anchors: `model.language_model.layers.{i}`, `.pre_feedforward_layernorm` and
`.per_layer_input_gate` for the 35 layers (105 modules).

-   CPU, no capture (—): —: 46 / 51 MiB: 2.30 / 2.28 s
-   CPU, full capture (4,807,495,333): 384,410 (127,380 module, 257,030
    operator): 4.82 / 4.83 GiB: 13.25 / 13.84 s
-   CPU, anchors only (459,514,406): 48,185: 366 / 397 MiB: 3.11 / 3.16 s
-   MPS, no capture (—): —: 39 / 30 MiB footprint: 2.63 / 2.69 s
-   MPS, full capture (refused: `Snapshots support CPU tensors only`): : :
-   MPS, anchors kept on MPS (459,514,406): 48,185: 420 / 495 MiB footprint:
    3.69 / 3.83 s
-   MPS, anchors copied to CPU in the hook (459,514,406): 48,185: 390 / 389 MiB
    footprint: 12.63 / 14.72 s

Memory is the increase over the warmed-up process, sampled every 20 ms: RSS on
CPU; on MPS the snapshots live in Metal memory, so the physical footprint is the
comparable number. On CPU the anchor records equal the full capture's records of
the same modules bit for bit (48,185 records, 0 different), and on MPS the kept
and the copied records are bit-identical; logits and tokens of every captured
run equal the uncaptured run on the same device bit for bit. Of the anchor
bytes, 397.6 MB are `past_key_values` and 39.0 MB `shared_kv_states` received by
each layer, 22.9 MB everything else.

## Rules for every step

- Advance one step at a time and report its result; every logic change is its
  own commit, with tests committed alongside the implementation. Run the full
  small-model regression and `git diff --check` before every implementation
  commit; run the corresponding Gemma acceptance before a stage completes, and
  rerun the affected real scenarios when later changes touch the relevant paths.
- Correctness protection ships with the corresponding feature; steps 3 and 4 are
  for the combined verification of consecutive capture, default measurement and
  delivery, and failure handling or basic budget checks may not be deferred to
  them. Passing the earlier steps is not passing the first release.
- Use only this repository's own environment, never an old package or venv.
  Small-model regression downloads no checkpoints; Gemma acceptance prefers the
  local weights with the verified revision and, when missing, obtains them as an
  acceptance preparation item with the source recorded.
- Step 1 pinned and measured the Python/PyTorch/Transformers combination;
  dependencies are in pyproject.toml and uv.lock. The collector runtime depends
  on PyTorch only; environments using Transformers models supply the verified
  version, and integration tests install their dependencies separately.
  Parameter recognition is generic, cache rules are reused by type/version, and
  no per-model configuration is added.
- `collector.py` manages the scope, call identity and uniform records; the
  private `_snapshots.py` handles value traversal, parameter-representation
  filtering and the budget; the private `_dispatch.py` handles the dispatch
  entry and mode ownership cleanup. The private `_provenance.py` handles the
  operator call table and tensor identity sources, and `graph.py` handles
  export. Modules and operators share results, budget and failure state.

HF semantic selection, GPU, file saving, KV/token interpretation, module-level
projected graphs and external debugger integration are not part of step 5 and
are decided separately. Test results of older projects are not borrowed as
acceptance for this repository.
