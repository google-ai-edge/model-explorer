# Collector design: one scope, every forward captured

> Moved from the user's repository `~/Projects/pytorch-collector` at commit
> `fe82ff9` with its uncommitted module-only change (`collect(model,
> modules=[...])`, step 7) on 2026-10-20; this copy is the one maintained from
> now on. Unchanged but for this note and its links to the acceptance records,
> `docs/verification/`, which stayed in that repository with `examples/`, the
> tests of the examples and the `pytorch_collector.equivalence` package.

Status: the first release `0.1.0` passed the stage 4 standalone wheel acceptance
(pytorch-collector's `docs/verification/stage4.md`), covering modules,
operators, parameter-representation filtering, cache adaptation, real
consecutive generation and failure recovery. An op does not stand for an actual
device kernel, and reading tensors inside fused kernels is not promised.

The whole flow has five steps: **enter the scope → automatic preparation →
capture forward by forward → verify and clean up → read the result**.

## Step 1: a `with` block marks the capture range

```python
with collect(model) as capture:
    output = model.generate(
        **inputs, do_sample=False, use_cache=True, max_new_tokens=4
    )

records = capture.records
```

Every registered module of the target model is observed, and so is every
operator visible at the dispatch layer during each forward of the target model.
Snapshots exclude model parameters and recognized weight representations and
keep every other supported input and output tensor; the rules are in step 3.
There is no `level` option, no sampling or downsampling, and no per-model
configuration; the opt-in `modules` option of step 7 narrows a capture to the
boundaries of chosen modules, and leaving it out keeps everything described
here.

One `with` may contain several sequential `model(...)` calls or one real
`model.generate(...)`; every root forward gets its own `forward_id`. The first
release had to pass acceptance on a real generate of Gemma 4 E2B under the fixed
CPU eager, text-only, cached, deterministic configuration. Other models or
generation configurations need their own verification.

The caller organizes the loop or the generation; the collector only
distinguishes forwards. It does not infer turns, prefill or decode, and it does
not create, update or reset a KV cache. Cache tensors are read only to make
snapshots.

Besides `modules`, the only option is `memory_limit_bytes`: **1 GiB** by
default, overridable by the caller. The default comes from the stage 3
measurements of the full Gemma on a short prompt: about 489.5 MiB for four
hand-written cached forwards and about 467.5 MiB for a real generate; 1 GiB is
about 2.1 times the larger observation, and full recovery with the budget
argument omitted has been verified. The value is not guaranteed to suffice for
arbitrary input lengths or call counts; changing the budget does not change what
is captured.

## Step 2: entering the `with` prepares everything

The snapshot budget is validated and the target object is claimed first, then
`torch.no_grad()` is entered, module hooks are installed and a pinned-version
`TorchDispatchMode` is enabled. A failure halfway through preparation rolls back
the resources already taken.

Root, parent and child modules are found through `model.named_modules()`. A
shared object gets one set of hooks, named by the path of its first appearance;
the root path is `""`. Capture and lifecycle checks never install twice.

The parameter identity set is built through `model.named_parameters()` and
includes weights, biases, norm parameters, frozen and shared parameters. It does
not rely on the name containing `weight` and keeps no per-model lists. The rules
recognizing parameters and their representations are shared by all models.

Operators are attributed to the innermost tracked registered module call.
Visible operators inside temporarily created or unregistered modules go to the
enclosing tracked call; such modules add no module-boundary records.

**Nothing is readable yet.** The model's train/eval state is unchanged, and
computation that prepares inputs before entering the model is not recorded.

## Step 3: every model execution captures both kinds of tensors

Every root entry checks the thread and the call state, allocates a new
`forward_id` and opens the call. Calling the root model again after a previous
call finished is allowed; recursive entry of the root model is not. Module and
operator `call_id`s are unique across the whole capture; pairing state and the
failure flag are maintained from the first observation on.

-   Module entry: Establish the call identity; after existing pre-hooks ran,
    filter and copy the actual inputs by the snapshot rules
-   Operator call: Filter and copy the inputs → run the original operator once →
    identify the outputs, filter and copy them, associate with the current
    module call
-   Module exit: After existing post-hooks ran, filter and copy the actual
    outputs by the snapshot rules, then close the call

Module and operator observations interleave within one call; the model is never
rerun. Recording pauses after each root call ends and resumes when the next
begins; computation before entering the root model and between two root calls is
out of scope.

Both observations share one path: **expand the supported data → recognize and
exclude parameters → check the budget → `detach().clone()` on the spot**,
keeping the original dtype. Inputs that must be saved are copied before any
in-place or `out=` overwrite. Only the internal copy temporarily masks the
TorchDispatch and TorchFunction machinery, so the collector does not capture
itself and third-party modes cannot rewrite the snapshot; real model operators
still run once in their original dispatch order. The collector never replaces a
model's inputs, outputs or operator return values.

-   Registered parameters and recognized parameter views: transposes, views,
    detaches: No
-   Weight representations recognized through the explicitly supported copy and
    cast rules: No
-   Unrecognized derived results, such as a `W * scale` outside the supported
    rules: Yes, as an ordinary tensor
-   Input indices, activations, computed results, logits: Yes
-   Buffers and dynamic tensors inside supported caches: Yes, when they appear
    at an observation boundary and hit no parameter exclusion rule

Filtering applies to the inputs and outputs of both modules and operators.
Identity is determined before copying, so a weight is not saved again at the
output of a view or cast operator. Weight representations are recognized only
through explicit parameter aliases and supported conversions; normal results
such as `linear(x, W)` or `embedding(W, ids)` are still captured. "Was computed
from a weight" is not an exclusion criterion, and recognizing arbitrary custom
weight derivations is not promised. There is no embedding-specific option.

Current conversion rules: only `aten.clone.default` and `aten._to_copy.default`
observed inside a root forward propagate identity from a recognized parameter
source; views, transposes and detaches of parameters and derived representations
are verified through the actual storage region. In PyTorch 2.9 a
contiguous/reshape that must copy lowers to that clone; other out/copy overloads
do not propagate automatically. An independent copy whose origin was not
observed is treated as an ordinary tensor. An unknown lower dispatch mode may
change an operator's return value; then copy/cast identity is conservatively not
propagated, so its output is never mistaken for a parameter.

Derived identity uses weak storage references and does not extend the life of a
temporary weight copy. A schema-declared write invalidates the identity of the
corresponding derived storage, which is then captured as an ordinary tensor; the
identity of the registered parameter itself is kept. Between root calls only
write invalidation of existing derived representations is maintained: no
external computation is recorded, no call ids are allocated, no new copies are
recognized. An external operation that raises after a partial write is also
invalidated, so a later forward cannot miss rewritten runtime data.

Buffers are not an active exclusion source; a tensor that also carries parameter
identity follows the parameter exclusion rule regardless of the attribute name
it was reached through. Filtering only decides whether a snapshot is saved; it
never skips operator execution, call identity tracking or integrity checks.

Tensors, tuples, lists and ordinary Mappings (dict, `UserDict`, `ModelOutput`,
with string or integer keys) are expanded. Transformers **5.17.0**'s
non-offloaded `DynamicCache` is supported in addition, accepting only the exact
types below, not their subclasses:

-   `DynamicCache`: the `layers` list, every item one of the layer types below
-   `DynamicLayer`: `layers/i/keys`, `layers/i/values`
-   `DynamicSlidingWindowLayer`: as above, plus
    `layers/i/_sliding_window_tensor`
-   `LinearAttentionLayer`: `layers/i/conv_states/<state>`,
    `layers/i/recurrent_states/<state>`

These paths follow the actual input or output position, e.g. `("kwargs",
"past_key_values", "layers", 0, "keys")`. Uninitialized `None` fields produce no
tensor records; the sliding-window configuration scalar is itself a tensor and
is saved. Every observation immediately copies all fields to capture, sharing
parameter filtering, the cumulative budget and cycle detection; repeated
references are copied once per path. The adaptation reads only loaded
Transformers types and explicit fields; it does not import Transformers and does
not scan arbitrary object attributes. Unknown caches, inner layers, versions or
offloading fail explicitly; type rules are added on demand, never model
configuration.

Gemma's `shared_kv_states` is an ordinary Mapping and is captured in full at its
original path. It may contain this forward's K/V beyond what the sliding-window
cache keeps; these tensors are not derived from, merged with or replaced by the
cache. Layers sharing KV do not necessarily each have a cache entry; the
collector records the actual objects and does not interpret layer counts or
token semantics.

Scalars, `None`, and operator schema configuration such as dtype/device/layout
produce no tensor records; other unsupported custom objects and cyclic
containers raise, while a repeated reference is not a cycle. Argument paths
follow the observed values; operator args/kwargs may already have been
normalized by the dispatch layer.

**All forwards share one snapshot budget that is not reset at the next
forward.** The bytes of tensor data actually saved after parameter filtering
accumulate; snapshots saved separately at different observation points are
charged separately. Excluded parameters are neither copied nor charged. The
remaining budget is checked before copying; exceeding it raises immediately and
fails the whole capture, without skipping tensors, forwards or falling back to
downsampling. The budget excludes the model, Python objects and temporary
allocations, so it is not a bound on process memory; a failed copy allocation
also fails the capture.

## Step 4: verify before exit, then release resources

Every root call must have completed, hook order and module bindings must be
intact, and input/output pairing must be reliable. A forward in which a
late-registered hook made an old value be mistaken (F6), or in which an
interruption swallowed internally caused mispaired calls (F7), must fail the
whole capture.

Cleanup runs in `finally` on success and failure alike: remove our own hooks,
exit our own dispatch mode, release the claim, restore the original gradient
state. Only resources owned by this capture are released; third-party hooks and
other modes stay. An enclosing mode the caller left unexited fails the capture;
our own mode is removed precisely by the pinned PyTorch 2.9 stack and
state-saving rules, preserving the order and later exit behavior of the other
modes.

Exit must happen on the Python thread that entered the scope. An exit request
from the wrong thread raises immediately and voids the result, but keeps the
original thread's hooks, mode, no-grad and model claim for that thread's
`finally` to clean up; those resources are never torn down from another thread.
The entering thread's cleanup still gives priority to the original exception.

**Snapshots are published only when the whole scope exits successfully.** A
capture error raises as soon as it is found and interrupts the current call;
known errors are not deferred to exit. If any forward fails, the whole capture
is unreadable, including the snapshots of earlier successful forwards, and all
staged snapshots are released. The model's original exception takes priority; a
cleanup error cannot replace it. A recorded error does not turn into success
because the caller caught the exception inside the `with`, and later forwards
cannot rescue the result.

**Failure does not roll back the model or the KV cache.** Model state updates
that already happened are not undone; the collector releases only its own
resources. To retry, the caller prepares the cache, inputs and any model state
again and creates a new capture.

## Step 5: read one uniform result

`capture.records` is readable only after a successful exit: a tuple in
observation order covering every forward of the whole scope. Reading while in
progress or after failure raises; there are no partial results. Every module or
operator call has a unique `call_id` shared by its inputs and outputs; a
repeated call gets a new id.

The tuple is a fixed record sequence and `record.tensor` is an independent,
modifiable snapshot. The caller may modify it in place without affecting the
model or the snapshots at other observation points; a modified value no longer
represents the original observation. The first release adds no read-only tensor
wrapper and does not copy again on read.

-   `forward_id`: The target model forward this belongs to, counting up from 0
    within one capture
-   `module`: The current module path; operators use the innermost tracked
    registered module path, the root is `""`
-   `op`: `None` for module records; the full name such as `aten.add.Tensor` for
    operator records
-   `call_id`: The current module or operator call number, unique across the
    whole capture
-   `module_call_id`: The enclosing module call within the same forward; for a
    module record it equals its own `call_id`, kept even when the module has no
    tensor records
-   `edge`: `in` / `out`
-   `path`: Tensor position, e.g. `("args", 0)`, `("kwargs", "x")`, `("output",
    0)`
-   `tensor`: The copied tensor in its original dtype

This is an in-memory list of tensor snapshots. If the target model was not
called inside the scope, a successful exit yields an empty tuple. Conditional
modules that did not run, and operations with no tensor inputs or outputs
meeting the snapshot rules, get no invented records; the snapshot list itself
does not promise a complete execution graph or full parameter replay, which step
6 provides separately through `ops` and `edges`. Parent and child modules and
operator boundaries each save their own snapshots, so the same runtime tensor
may be saved at several observation points.

## Step 6: read the data flow of the same execution

Snapshots answer "what was the value at each observation point", not "who
computed this input": `record.tensor` is a copy whose original identity was lost
at copy time, and parameters and their transposes and casts produce no
snapshots, so those operator calls are invisible in `records`. Data flow is
therefore recorded separately, from the same execution and with the same
`forward_id` / `call_id`, and is on by default.

`Op`, `Edge` and `Port` are NamedTuples: they are constructed per operator call
inside the model forward, and a NamedTuple costs about a third of a frozen
dataclass to build; fields are accessed by name and immutable.

**`capture.ops` is the complete table of operator calls.** Every
dispatcher-visible operator call inside a root forward has an entry whether or
not it produced a snapshot. Fields: `forward_id`, `call_id`, `module`,
`module_call_id`, `scope`, `op`, and `inputs` / `outputs` with the `path`,
`shape` and `dtype` of every tensor port. `scope` is the module call stack from
the root to the innermost call, `((module path, call_id), ...)`, whose last item
is `module` and `module_call_id`; it separates repeated calls of one module and
their inner submodules. Arguments that were written to but not returned are also
listed in `outputs` under their argument position, so every `op` source names
one output port of its producer. `arguments` holds `(name, text)` for non-tensor
arguments, recording only `int`, `float`, `bool`, `None`, `dtype`, `device`,
`layout`, `memory_format` and integer sequences of length at most 8, everything
else as `<type name>`; this distinguishes values such as `dim`, `eps` or a
target `dtype` that decide an operator's semantics without storing arbitrary
objects.

**`capture.edges` gives exactly one source for every tensor input of every
operator.** The consumer is identified by `forward_id`, `call_id` and `path`;
`kind` is one of:

-   `op` (The output of an operator call in some root forward of this capture):
    `source_forward_id`, `source_call_id`, `source_path`; when it comes from an
    earlier forward and was passed into this root call, also `input_path`
-   `input` (An input of the current root forward): `source_path`, e.g.
    `("args", 0)`
-   `param` / `buffer` (A registered parameter or buffer itself): `name`, the
    first name in `named_parameters()` / `named_buffers()`
-   `external` (Produced inside the capture but outside a root forward by a
    visible operator, such as token concatenation between two calls): none
-   `constant` (Produced by no visible operator during the capture, such as a
    `torch.tensor(...)` literal or a tensor that existed before the `with`):
    none

A parameter's transpose or cast is an ordinary operator output and takes `op`;
only the parameter itself takes `param`. Sources may cross forwards: the KV a
decode reads points at the operator that wrote it during prefill.

**Identity is the tensor object, not its value or address.** Under the pinned
CPU eager version, the tensor object an operator returns is the object the next
operator and module hook receive; view operators return new objects. The source
table is a `torch.utils.weak.WeakTensorKeyDictionary`: it does not extend
activation lifetimes, entries disappear when the tensor is collected, and a
reused `id()` cannot hit a stale entry. The table lives as long as the capture
and is not cleared per forward.

**Writes change the producer.** After an operator runs, arguments whose schema
has `alias_info.is_write` are re-registered to the current call, covering
`add_`, `copy_`, `index_put_`, `out=` and the `_foreach_*_` operators that
return `None`. When the write target is a view, its `_base` is registered to the
current call as well, so later reads of the base point at this write. Other
existing views of the same base are not updated; this is a known limitation.

**Failure semantics match snapshots.** `ops` and `edges` are readable only after
the whole scope exits successfully; any exception during tracking voids the
capture. They hold metadata only and are not charged to the snapshot budget.
`constant` is a legitimate source, not a failure; a tracking gap would show up
as a `constant` that should not be there, which acceptance constrains: on
verified models every consumer of a `constant` edge must be one of the entry
operators listed in the acceptance record (pytorch-collector's
`docs/verification/stage5.md`).

**`to_model_explorer(capture, forwards=None, names=None)` returns a dict ready
for `json.dump`.** The structure is the `{"label", "graphs": [...]}` Model
Explorer reads natively, one graph per forward without deduplication; `forwards`
selects a subset. Graphs are named `forward_N` by default: the collector only
knows this is the N-th root call and does not decide whether it is a prefill, a
decode or something else, since a prefill with an existing cache, a chunked
prefill or an encoder would all defeat a guess from shapes or caches. A caller
who knows uses `names={0: "prefill", 1: "decode_1"}`; names must be non-empty,
free of `/` and distinct, and the input nodes' `source_graph` attribute uses the
same names. Each operator call is one node whose `id` is its `call_id`, whose
`label` is the operator name without the overload, and whose `namespace` is the
`/`-separated module path; when one path is called more than once inside the
same enclosing call, the last segment gets `#0`, `#1`, ... and the inner
submodules are separated accordingly. Port metadata is only `tensor_shape`, e.g.
`float32[1, 8, 64]`. `op` edges within the same forward become links in the
graph; everything this forward reads but did not compute is a node under
`inputs`, because Model Explorer links need a source node and cannot cross
graphs: a root input's `id` is its argument path (e.g. `kwargs.input_ids`); an
earlier forward's operator output (such as a KV cache) that was passed into this
forward also uses the passed-in path (`Edge.input_path`) as `id` and points at
the node in its own graph through `source_graph`, `source_node` and
`source_output`; something the model kept across forwards itself, with no
passed-in path, gets `kept:<graph>:<call_id or input>:<path>`; `external` gets
`external:<consumer call_id>:<path>`. `param`, `buffer` and `constant` produce
no nodes and are attributes of the consuming node. Every tensor a root forward
returns is looked up once at the root module's output hook, giving
`capture.outputs` (same type as `Edge`, `call_id` is `None`), and becomes a node
under `outputs` named by its output path. `unused_ops(capture)` starts from
these root outputs and from every operator that writes to an argument (schema
`is_mutable`; an unresolvable overload counts as writing), walks `op` edges
backwards across forwards, and returns the calls of each forward that are not
reached; writing operators must be kept unconditionally because a write through
a view is not attributed to the other views of the same base. Input nodes are
created only by operators that read them: a tensor returned as it came, read by
no operator, has an output node with no link that names its source in a `source`
attribute. `hide_unused=True` merely leaves out those calls and those unlinked
output nodes; it is off by default so the graph stays faithful to the actual
execution. The function writes no files.

Not promised: Python-level scalar and shape computation, untaken branches, write
propagation to other views of the same base, a correspondence between operators
and device kernels, sources outside the capture.

## Step 7: capture only selected modules

```python
targets = ["model.language_model.layers.0", model.lm_head]
with collect(model, modules=targets) as capture:
    model.generate(**inputs, max_new_tokens=24)
```

`collect(model, *, memory_limit_bytes=DEFAULT_MEMORY_LIMIT_BYTES,
modules=None)`. With `modules=None` nothing changes. Otherwise `modules` is an
iterable of module paths, as `named_modules()` names them (`""` is the root), or
of the module objects themselves:

- Paths are resolved by `collect()` against
  `named_modules(remove_duplicate=False)`, so any appearance of a shared module
  may be named; `capture.modules` and the records use the path of its first
  appearance, as everywhere else. Unknown paths raise `ValueError` naming all of
  them; a module object not registered in the model and an empty selection raise
  `ValueError`; a bare string, a single module or a non-iterable raise
  `TypeError`. If the model's structure changes so that a selected module is
  gone or renamed before the `with` is entered, entry raises `CaptureError`.
- Preparation is the same except for what is installed: the claim, `no_grad`,
  the eager checks and the topology and hook verification still cover every
  registered module, but hooks go only on the selected modules and on the root,
  and **no `TorchDispatchMode` is enabled**. The root's hooks only number
  forwards and keep "nothing outside a root forward"; they snapshot nothing
  unless the root is selected. A selected module called outside a root forward
  records nothing.
- Each selected module call is observed exactly as in step 3: the same value
  walk, `DynamicCache` expansion when its arguments carry one, parameter
  exclusion, budget check and `detach().clone()` in the hook. Parameter
  exclusion keeps the registered parameters and their views, transposes and
  detaches; recognizing a `clone` / `_to_copy` of a parameter needs the observed
  operator, so such a copy made inside the forward and passed to a selected
  module is saved as an ordinary tensor.

What a module-only capture contains after a successful exit:

-   `records`: Only the input and output records of the selected modules (`op`
    is `None`, `module_call_id == call_id`), in observation order. Values are
    bit-identical to the module records of a full capture of the same execution;
    `forward_id` is numbered the same way, while `call_id` numbers only the root
    and selected calls, so it does not match a full capture's
-   `modules`: The selected paths in `named_modules()` order (`None` in a full
    capture)
-   `ops`, `edges`, `outputs`: Not recorded: reading them raises `CaptureError`
-   `to_model_explorer`, `unused_ops`: Raise `ValueError`: there are no
    operators to build a graph from

**Devices.** A full capture accepts CPU tensors only (`Snapshots support CPU
tensors only`). The reason is scope, not snapshots: data flow identifies a
tensor by the object an operator returned and tracks copies and writes through
the dispatch mode, and that is verified only for CPU eager. A module-only
capture uses none of it, and parameter exclusion compares storages and byte
regions, which does not depend on the device, so it accepts dense tensors on any
device and clones each snapshot where it was observed; a record's
`tensor.device` is the device of the value it copied (on MPS, Transformers'
`_sliding_window_tensor` is a CPU tensor and stays one). Copying to CPU in the
hook gives the same bits but was measured 3.4–3.8 times slower on MPS without
saving memory on unified memory, so the caller calls `.cpu()` when it needs to.
Verified on CPU and MPS; CUDA is untested.

For one Gemma 4 E2B-it prefill of 13 tokens and 24 decodes, selecting each
layer, its `pre_feedforward_layernorm` and its `per_layer_input_gate` (105
modules) saves 459.5 MB in 48,185 records, against 4.81 GB in 384,410 records
for the full capture, and costs 3.1 s against 13.3 s (2.3 s uncaptured) on CPU;
the measurements are in [step 6 of the plan](PLAN.md). 95% of those bytes are
the `past_key_values` and `shared_kv_states` each decoder layer receives, saved
again at every layer call.

## Support conditions of the first release

- Pinned-version CPU eager (a module-only capture, step 7, also accepts other
  devices), ordinary tensors, a single thread and a stable module structure.
  Global hooks, an enabled `inference_mode`, recursive root calls and
  overlapping claims are rejected; ordinary submodule re-entry must pass pairing
  verification.
- Modules use the public hook API, uniformly `with_kwargs`; operators use
  `torch.utils._python_dispatch.TorchDispatchMode`. The model and PyTorch are
  neither modified nor monkeypatched, and compilation is not enabled. Calling
  `.forward(...)` directly around the hooks is unsupported.
- PyTorch is pinned to 2.9.0, and its eager call frames distinguish a normal
  exit from exception cleanup. Custom `__call__` / `_call_impl` are rejected by
  default; the verified exception is the non-checkpoint branch of Transformers
  5.17.0's `GradientCheckpointingLayer`, whose actual super delegation must go
  directly to `nn.Module.__call__`. That compatibility is reused by shared type,
  not configured per model.
- The collector runtime depends on PyTorch only. A caller using Transformers
  models supplies the verified version through its own environment; cache
  compatibility is handled by type and version. The checkpoint, revision, dtype
  and similar needed to load a model are run or test settings, not per-model
  collector configuration.
- The real visible range is preserved: operator names are not guaranteed to
  correspond one to one with Python functions or device kernels, and fused
  operators expose only their visible inputs and outputs. Internal exceptions
  that never reach an observation point, and configuration changed and restored
  between checkpoints, are not all promised to be recognized. GPU, custom
  operators and tensor subclasses need separate verification.

Implemented step by step per the [implementation plan](PLAN.md). Mechanism
references: [PyTorch 2.9 dispatch modes][pt-modes],
[SDPA fusion boundary][pt-sdpa]. The implementation and acceptance scope follow
the stage status and measured records above.

[pt-modes]: https://docs.pytorch.org/docs/2.9/notes/extending.html#extending-all-torch-api-with-modes
[pt-sdpa]: https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html
