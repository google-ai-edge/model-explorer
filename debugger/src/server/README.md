# Model Debugger Server

The server of the Model Debugger, in Python. It serves the web UI (`src/ui`) and
its HTTP API on `127.0.0.1`, and drives two runners per Session, Reference and
Target, by the gRPC protocol in `src/protocol`: it launches each runner, uploads
the model if the runner lacks it, loads it, starts chats on both sides, sends
each turn to both sides, stores what both runtimes captured, and compares the
two.

A runner holds one model for its whole life, and one chat at a time. Creating a
Session turns its Model Server on (`POST /api/sessions/create`): it launches
both runners and loads both models, and starts no chat. A Session that cannot
start now (a device missing or in use by another Session's Model Server, a model
not on the server) is refused with why, and nothing is saved; there are no saved
Sessions that never started. A chat is started apart (`POST
/api/sessions/{id}/chats`, with each side's settings), and a Session holds
several over its life, `Chat 1`, `Chat 2`, …, one live at a time. Starting a
chat ends the live one; a chat that fails to start on a side is not kept, and
another may be started. Once a chat is over (a reply filled the context, a turn
failed or was stopped, another chat was started) it takes no more turns, and
Debug still shows what it captured; the Model Server stays on. Only a runner
gone ends the Session: the server turns its Model Server off by itself, as Turn
off does, stopping the other runner and releasing both devices, and the Session
says `Runner closed`. A turn longer than what both sides' contexts have left
does not happen: each runner refuses it before its runtime starts, nothing is
captured, the turn's job fails saying how many tokens it needs and how many are
left, and the chat goes on; one side refusing while the other's reply began ends
the chat. One turn, or one chat's start, runs at a time. A Session starts once:
one whose Model Server was turned off, or whose server exited, never runs again,
and neither does one whose start failed, was turned off, or was under way when
its server exited; a Session's configuration never changes, only its name. Such
a Session has ended: the listing shows it `Ended`, or `Ended · no turn` with why
when none of its chats kept a turn, which is not opened; to try again, create a
new Session. Nothing offers to try a start again. A Session an older server
saved without starting it is read as one whose start failed
(`session_store.upgrade()`): `Ended · no turn`, with why.

## Prerequisites

- [Bazelisk](https://github.com/bazelbuild/bazelisk), which runs the Bazel
  version pinned in `.bazelversion`. Bazel fetches everything else: Python 3.12,
  the PyPI packages `requirements_lock.txt` pins, protobuf and gRPC. The first
  build compiles gRPC's Python plugin, which takes a few minutes.
- Xcode on macOS, whose C++ compiler that plugin is built with.

The UI is built in `src/ui` (`npm ci && npm run build`); the command-line runner
in `src/runner` (`bazelisk build //model_debugger/runner/cli:runner`). The
server builds and its tests run without the UI; only the wheel needs it.

## Build and run

From `src/server`, a workspace of its own:

```shell
bazel build //model_debugger/server/cli:server    # the command
bazel build //:wheel                              # the wheel, with the UI
bazel run //model_debugger/server/cli:server
```

The wheel, `bazel-bin/model_debugger_server-0.0.0-py3-none-any.whl`, holds
`model_debugger.server` (not its tests), `model_debugger.protocol` and the built
UI, `src/ui/dist/model_debugger/browser` as it is when the wheel is built, in
`model_debugger/server/ui/`; it requires what `requirements.in` lists, and
installs the command `model-debugger-server`, which serves the UI it carries. It
is the one target that needs the UI, so `//...` leaves it out; without the UI
built, its build fails and says how to build it (`built_ui.bzl`, which takes the
built UI into Bazel and fetches it anew whenever it changes). The server prints
its URL; open it in a browser on the same computer. Relative paths are taken
from the directory it is run in. Flags:

- `--workspace DIR`: where Sessions, models and captures are kept; by default
  `~/Library/Application Support/model_debugger/server` on macOS, else
  `${XDG_DATA_HOME:-~/.local/share}/model_debugger/server`. One server at a time
  may use a workspace. Deleting a Session (its Model Server off) is for good and
  cannot be undone: its record goes first, then its chats' captures
  (`captures/<id>/chat-<n>/`), then the models no other Session uses (the delete
  dialog names them, and the room that frees); a Session's files that no record
  names, left by a crash meanwhile or by an import that did not finish, and the
  records an older server only marked deleted, are removed as the server starts,
  and so are the models no Session uses, such as an upload or a download no
  Session was saved with. The workspace holds every model file the server reads,
  one copy per model shared by every Session (`session/model_store`): those
  uploaded from the browser (`models/`), and those downloaded from Hugging Face
  (`hf/`, a Hugging Face cache of its own, its Xet cache included); the server
  never reads or writes a model file elsewhere, since the browser may be on
  another computer. `models.json` keeps their paths relative to the workspace,
  which may be moved. A model's graph and what an imported Session keeps of it
  (`facts/`) go with the model. An older server's workspace may record models it
  kept outside (a path on its command line, the shared Hugging Face cache):
  those records lose their paths as the server starts, and a Session that names
  one reads what it kept of it but cannot start. The workspace also holds the
  copies of models that observe the tensors Sessions chose (`prepared/`): a
  Session has one copy of each model it observes tensors of, from its start
  until it is deleted. Sessions that choose the same tensors of a model share
  one, which goes with the last of them. What a copy belongs to is read from the
  Sessions' records, not counted apart; the server deletes the copies no Session
  uses as it starts, and never one a start is writing or loading. And the graphs
  of models (`graphs/`), one directory per model file's SHA-256, about 0.1 MB
  per signature: a Session's initialize converts the graphs its models lack
  first, in its own job, before the tensor capture is prepared and before that
  Session's runners launch, and a start whose graph cannot be converted fails
  (see [Graphs](#graphs)). A turn's capture is written only while the
  workspace's disk has room for each file and 1 GB spare, as the runner
  announces it: otherwise, or if the disk fills meanwhile, the turn fails,
  saying how much it needs and how much is free, and keeps none of its files.
  While a turn's capture arrives, once its reply ended, the turn's job says how
  much of it has (`Retrieving the capture · 1.2 GB so far`): the protocol
  announces each file's size as it starts, never the capture's.
- `--port N`: by default 8080, or the next free port up to 8099; a port named
  that is in use is an error. A port counts as in use while any socket listens
  on it, on any address, and for half a minute after connections to it close.
- `--ui DIR`: the built UI; by default the one the wheel carries, or, in a
  source checkout, `src/ui/dist/model_debugger/browser` when it is built. With
  none, the server serves the API alone, and says so as it starts.
- `--kept_builds DIR`: the runner builds kept on this computer, one directory
  per LiteRT-LM commit, as `src/runner`'s `cli:keep` writes them; by default
  `~/Library/Application Support/model_debugger/runners/macos_arm64`. It is read
  anew whenever the editor lists them.
- `--runner FILE`: the development build, a command-line runner offered besides
  the kept builds; by default the one Bazel built last,
  `src/runner/bazel-out/<configuration>/bin/model_debugger/runner/cli/runner`.
- `--graph_timeout SECONDS`: how long converting one model's graph may take
  before the converter is stopped, the graph says it failed and the start fails;
  by default 600 (Gemma 4 E4B's takes about 5 s).
- `--runner_log info|warning|error`: how much of the output of the runners on
  this computer shows, each line at the level the runner gave it: every line,
  its warnings and errors, or its errors; by default `error`, since LiteRT-LM
  warns on every start and every turn about nothing that is trouble
  (accelerators it does not find, sections a model lacks).
- `--verbosity N`: absl's: 0 (the default) shows INFO and above, -1 WARNING and
  above, -2 ERROR and above.

The defaults of `--ui` and `--runner` are found in the source checkout the
command was built from; an installed wheel has no `--runner` default, and its
`--ui` default is the UI it carries.

The server takes no model on its command line: a model is uploaded from the UI
(a LiteRT-LM `.litertlm` file), or named there on Hugging Face
(`session/hf_models`): a `.litertlm` file of a repository, or a whole repository
for PyTorch, downloaded into the workspace unless it is there already, with the
token `HF_TOKEN` or `hf auth login` gives for gated repositories. Before it is
chosen, the editor says whether the server has it, and its size (`On the Server
· 10.3 GB`, `Downloads to the Server · 15.0 GB`). A `.litertlm` file is the
server's by its content: the hub is asked its SHA-256 first, and a model of
those bytes the server keeps, an upload say, is used with no download (offline,
it is fetched as ever). A download runs in a child process of its own, so that
it can be cancelled (`POST /api/jobs/{id}/cancel`): the child is stopped at
once, Xet transfers too, its job ends, and what it wrote is deleted.

The flags are checked before anything starts: what `--ui`, `--runner`,
`--pytorch_runner`, `--kept_builds` and `--workspace` name must be there and of
its kind, and numbers in range. Each mistake, and a port or workspace in use, is
one line on stderr (`model-debugger-server: --ui /x/y: no such directory`), and
the exit status is 1; an unexpected error still shows its traceback.

This computer is a device while it has a build; Android devices (`adb`), booted
iOS simulators and paired iPhones are devices too, each running the Runner App
installed on it. The editor shows which LiteRT-LM commit a phone's App was built
against, as its version says (`versionName`, `CFBundleShortVersionString`), read
without launching it, and says when that is not the commit this server is
written for (`runner/app_version.py`) or no App is installed; a phone without
the App cannot be chosen. A kept build runs only once every file matches the
SHA-256 in its `build.json` and the runner reports that commit in `GetInfo`; a
side that chose a commit is refused on any device whose runner reports another.

### Logs

The server prints its address on stdout and logs to stderr, one line per event,
in absl's format. INFO: what it serves (workspace, UI, runner builds and
models), each Session's Model Server on and off and each turn it completes, each
start, chat or turn refused by the rules (a chat that has ended, an option the
runtime does not take), each runner launched and stopped, each Session imported
or exported, and the access log. The access log has a line for each request that
changes something (a POST other than an analysis read, a DELETE) and, at
WARNING, for each that fails; the UI's polls and the reads of a page as it is
shown, the analyses among them, have none, nor does a request the browser gave
up on (499). ERROR: a start or a turn that fails, or a turn the capture index
rejects, in one line that names the Session (its name and id) and says why as
the UI does, which names the side (a runner on this computer that exited saying
why, as its last line `runner: ...`, gives that line as the reason); a runner on
this computer that ends unasked (killed, or crashed), with its exit and its last
lines; and a request that fails unexpectedly (500), once, with its traceback.

### Stopping

Nothing the server launched outlives it. However it ends (Ctrl-C, SIGTERM,
SIGHUP, an exception as it starts or serves) it first stops every runner it
launched, one still starting too: this computer's runner processes (SIGTERM,
then SIGKILL after 5 s), the Android App and the `adb forward` to it, the App on
an iOS simulator or device. A second Ctrl-C does not cut that short, and it is
bounded: a runner process still there 5 s after SIGTERM is killed, and each
`adb` or `devicectl` command has 30 s. Killed outright (SIGKILL), it stops
nothing; but each launch is recorded in the workspace's `runners/` until it is
stopped (a process by its pid and start time, a forward by its device and port,
an App by its pid), and the next server on that workspace stops what those
records name as it starts, only while it is still the same process or forward.

## Layout

The package `model_debugger.server`, in `model_debugger/server/`, is a facade
(`server.py`), the command line that starts it (`cli/`), and layers below them.
Every directory is a Bazel package with one `py_library` per module, named after
it; a module's tests sit beside it as `foo_test.py`, one `py_test` each, run by
`testing/test_main.py`.

-   `util/`: errors, JSON files, the clock
-   `architectures/`: what a model is, nothing of a runtime: the semantic
    skeleton schema and its checks (`skeleton`), what a definition is written
    with (`builder`), each architecture's definition from a Hugging Face
    config.json (`gemma4`), the architectures known with their pinned configs
    and the skeletons those make (`catalog`, `data/<id>/`), and the generator
    that writes the skeletons or checks them (`generate`); see its README
-   `graph/`: a model's graph: Model Explorer's converter run in a child process
    (`graph_converter`), each signature's graph slimmed and annotated with its
    tap points, and the stem its layers are named by (`model_graph`); how a
    LiteRT-LM graph binds to each architecture (`bindings/<id>.json`), and where
    a model's anchors are in its graphs (`architecture`); the PyTorch module
    each LiteRT-LM point chosen is (`counterparts`)
-   `tap/`: observing tensors of a model: the scan of what can be observed, with
    each tensor's layer and whether it is its layer's output (`tap_scan`), the
    copy that observes the points chosen and its check against the model
    (`tap_model`), the copy of one subgraph's structure without its weights that
    its graph is converted from (`structure_copy`), the `.litertlm` container
    (`litertlm_file`); the TFLite and `.litertlm` schema bindings flatc writes
    at build time
-   `capture/`: a chat's capture on disk: its index (each forward pass's files
    by key, and how many rows it feeds) and tensor files
-   `analysis/`: the comparisons: how a turn's two sides line up, per-token
    metrics and final logits, the KV cache, the intermediate tensors a Session
    observes (`tensor_analysis`: by layer, sync point and position, as the model
    and each step's capture place them, a position read from the step that
    produced its token, the step Final Logits shows there), statistics, what the
    Inspection's charts draw of a whole logits vector (`vector_drawing`:
    line-chart bins, scatter points and extent, computed as the UI computed
    them, to the bit), and what every metric measures
-   `runner/`: driving runners: the gRPC client, the capture stream written to
    disk, the launcher interfaces and the local, Android and iOS simulator and
    device launchers, the records of what they launched (`launch_records`,
    `runners/` in the workspace), this computer's runner builds
-   `session/`: the Session itself: Model Servers and their sides, jobs,
    options, and the stores of Sessions, models and captures, in a workspace;
    the scans of models' tensors and the prepared copies that observe them
    (`prepared/` in the workspace); the models' graphs (`model_graphs`,
    `graphs/` in the workspace); the tensor pairs made by hand, per pair of
    models (`mappings`, `mappings/` in the workspace)
-   `service/`: the interface to the UI: every HTTP route and the UI's files,
    the device and option catalogs, the Session listing and view, the metric
    catalog, the analysis requests
-   `server.py`: `Server`, the facade: binds the port, opens the service, runs
    uvicorn
-   `cli/`: the command line: flags, defaults, logging
-   `testing/`: for the tests only: fake runners, launchers, tools and builds,
    capture fixtures, the real runner when built

The UI never downloads a whole logits vector: final-logits-summary, -bins,
-points and the paged final-logits answer everything it shows.

The protocol's messages and stubs, `model_debugger.protocol`, are
`@protocol//model_debugger/protocol:runner_py_pb2` and `:runner_py_pb2_grpc`,
never generated into the source tree; nor are `tap/`'s schema bindings, which
flatc (`@flatbuffers`) writes from the TFLite schema of the runner's LiteRT and
the `.litertlm` header schema of its LiteRT-LM, both fetched by commit
(MODULE.bazel); `ci/check_schema_pins.py` checks that the commits are the
Runner's, and so is `runner/app_version.py`'s. A copy of a model kept in a
workspace is named by `tap_model.identity()` too: the copy format, the
flatbuffers runtime and the bindings, so one an older server wrote is not
reused. The workspace root holds the pinned PyPI packages, ruff and the wheel.
The server's own checks are in the repository's `ci/`, beside
`ci/check_server.sh`, which runs them.

## Graphs

The Graph Diff view of the intermediate tensors reads a model's graph from the
server. A model's graph is that of its text model's prefill and decode section,
one graph per signature, as Model Explorer's converter writes it: node `<op>` of
graph `<signature>` and its output `<output>` are the tap point
`<signature>:<op>:<output>`, and each output carries its point's facts
(`point_id`, `observable`, `reason`, `layer`, `layer_output`, `context_axes`).
Composite operators are not expanded, and constant values and quantization
longer than 64 characters are not kept. The converter is the prebuilt one of
PyPI's `ai-edge-model-explorer-adapter` (pinned in `requirements.in`), run in a
child process. It is not given the model: it reads a model whole and unpacks it,
weights and all. It is given, one at a time, a structure copy of each subgraph a
signature runs (`tap/ structure_copy`): that subgraph whole, every other
subgraph empty but for its name, and every buffer of more than 64 KiB, the
weights, emptied; so a constant whose buffer is larger has no value in the
graph. Its graph is otherwise the section's own: ids, edges, namespaces,
outputs, shapes and quantization alike (checked on Gemma 4 E2B, E4B and Qwen3
0.6B). It takes about 4 s and 0.7 GB for Gemma 4 E2B, 5 s and 0.9 GB for E4B,
where the whole section took 10 s and 4.7 GB, and 22 s and 8.1 GB. Its wheels
are for macOS on arm64 and Linux on x86_64 and aarch64; elsewhere a model's
graph says "The graph converter is not available on this platform.", and no
Session can start. It is to move into this repository's C++ package later;
everything the server adds is Python (`graph/`).

Model Explorer groups a graph's nodes by their namespaces, and the converter
names some operators of one layer by another's scope: it picks one of a fused
tensor's names. The graph the server serves groups each operator in the layer
its outputs' own names place it in, as the intermediate tensors are placed: in
Gemma 4 E2B's decode graph 166 operators of RoPE, such as layer 1's, which reads
the cosine table layer 0 computes for all, move out of layer 0 (or out of layer
4, the first full-attention layer) into their own. Each moved operator keeps the
converter's namespace as its attribute `converter_namespace`. An operator that
operators of several layers read is shared, and stays where the converter put it
(the cosine table, in layer 0); so does one whose names place it in no layer:
layer 0's KV cache update, whose outputs are named by the signature's function
(`StatefulPartitionedCall:1`), stays in layer 0, where the converter put it by
the operators it reads.

A graph is converted only as a Session is initialized: first in its initialize
job, before its tensor capture is prepared and before its runners launch. No
request converts one; what reads a graph or its layers says where it is instead.
A Session cannot start without its models' graphs: a conversion that fails (the
converter failed, crashed, was killed, took too long, or is not available here)
fails the start as any failed start does: the job fails with the reason, no
runner launches, and the Session has ended with no turn, the listing showing the
reason. Another Session's runners may be running on this computer while one
converts, and the converter's memory is taken beside theirs. One conversion runs
at a time, for at most `--graph_timeout` seconds; a start that waits says for
whose graph and which Session. A start that is turned off stops its conversion
and kills the converter; so does the server as it exits, and the converter also
ends itself when the server's process is gone, killed or not.

What is kept is the converter's slimmed graph, named by the converter's version,
a digest of the slimming code and the layout; the point facts, layers and
architecture are worked out as a graph is read. A failure is kept too, with its
reason, the identity it failed with and the tail of the converter's log
(`graphs/<sha256>/failure.json`, `converter.log`). Nothing converts it again on
its own: a start of another Session of the model, or of that Session after a
restart, tries it once more; a start that was waiting for the conversion that
failed fails with it, unconverted; a new converter identity converts anew. A
negative exit status is told as the signal; "(out of memory?)" is added only to
a SIGKILL the server did not send, after the converter had used most of the
computer's memory.

The layers of a model's tensors come from its graph too: the scope component
`<stem>_<N>` its namespaces repeat (`layer` in Gemma 4, `...Qwen3DecoderLayer`
in Qwen3) names each tensor's layer, read from the tensor's own first name, for
the analyses of intermediate tensors as for the graph; a tensor whose name
places it in no layer is in the layer its operator is grouped in in the graph
served (`model_graph.layer_groups()`): layer 0's KV cache update writes, in
Gemma 4 E2B and E4B (one operator per signature) as in Qwen3 (two), so that the
analyses and the graph place every point alike. A run of numbered scopes is a
model's layers only when it holds at least half of a signature's operators and
every signature names its layers alike; otherwise, or when two runs qualify, the
graph has no layers. A started Session's models have their graphs; a model whose
graph is not kept, such as an old Session's recorded before graphs were, has no
layers, and the analyses of its intermediate tensors say why
(`ModelGraphs.layers()` is the one place that decides it).

A model's architecture is the semantic skeleton the server has of its family and
size (model_arch_explore's schema: `semantic_ops`, `semantic_graph`, `layers`):
what the architecture's definition makes of a checkpoint's config.json, pinned
with the skeleton (`architectures/`, the one source of the definitions, ported
from model_arch_explore's mkjson.py). The architecture answer says where its
skeleton comes from by paths in this repository: `generator` is
`architectures/generate.py`, the definition's module and the architectures'
files' digest, with where the definition was ported from as a note
(`derived_from`); `config` is the pinned config.json, its Hugging Face model,
revision and SHA-256. Only Gemma 4 E2B and E4B have one, for a `.litertlm` holds
no config.json and is matched to a pinned checkpoint by its layers and width.
What the model is lives there; how a LiteRT-LM graph binds to it, which tensor
is which anchor, which signature input, which stages, is `graph/bindings/`.
Gemma 4's decoder has two input ports: `input`, the residual stream, and
`per_layer_input`, which its per-layer embedding stage multiplies by. The
skeleton and the execution graphs (as served, regrouped) are aligned by anchor
alone: per layer and signature, which node and output is the layer's `input`,
`intermediate.1`, `intermediate.2` and `output`, each checked against the
skeleton's residual structure, the first layer's input against the signature's
`embeddings`; a bound anchor says whether it was checked (`checked`), an unbound
one why. A port no anchor is of is bound to nothing, and the architecture says
why (`ports`): `per_layer_input` is not the signature input
`per_layer_embeddings`, which the graph adds the projected embeddings to before
slicing it per layer. Each block of a layer is a region of each signature's
graph, found from its bound anchors and the graph's edges alone: the operators
on a path from the tensor of the anchor before it to the node of the anchor it
ends at (Gemma 4's Attention runs from `input` to `intermediate.1`, Feed Forward
on to `intermediate.2`, Per-Layer Embedding on to `output`); what it reads from
outside, the weights, the per-layer input's slice, a shared KV cache, is its
inputs. A block an unbound anchor bounds is its layer's operators instead, and
says why. The modules not in the skeleton (Gemma 4's embedding, per-layer
embedding, final norm and output head) are named, each with why it is not drawn
(the architecture's) and the stages it runs in, with why (`outside`, the
binding's: the final norm and output head only in LiteRT-LM's decode signatures,
for its prefill produces no logits). A model without a skeleton says so; one
that cannot be bound says why (the architecture answer's `error`), without
failing the graph's answer.

A PyTorch model's graph is the execution graph its runner exports of a turn it
observed on the CPU (pytorch-collector's `to_model_explorer()`;
`execution_graph.json.gz` in the turn's capture): once the turn is recorded, its
Model Server hands the file to model_graphs, which keeps the graph of each
signature (prefill, decode) the model has none of, in `graphs/<sha256>/` as a
LiteRT-LM graph is, the first such turn giving both; until one does, the graph
is unavailable and says how to get one (a CPU side that observes the anchors).
As it is read it is folded (`graph/pytorch_graph.py`): each module LiteRT-LM
runs as one operator, by the binding's `fold` (Gemma4RMSNorm, odml.rms_norm;
Linear, fully_connected; GELUTanh, gelu), is one node of its module path,
labelled by its class, its ATen operators not expanded; everything else stays an
ATen operator. Gemma 4 E2B's decode graph is 5,272 nodes as exported, 2,535
folded, its prefill 5,307 and 2,570. Its layers are the module path's
`layers/<N>` (stem `layers/`); its architecture is the skeleton its config.json
makes, bound to the folded graphs by the same binding's graph rules (each
anchor's `read_by` or `written_by`: the input is what `input_layernorm` reads,
the output the layer's in-place `aten.mul_` by `layer_scalar`), each block a
region between its anchors as for LiteRT-LM, each anchor's point the runner's of
it; each anchor's output carries its point, every other output why it has none.
It binds to its architecture for the intermediate tensors without a graph, by
its runner's module positions: `graph/bindings/<id>_pytorch.json`
(`pytorch_binding`, which architecture.architectures() leaves out) names the
module each anchor is taken at (Gemma 4: a layer's `input` is what
`model.language_model.layers.<i>` reads, `intermediate.1` what its
`pre_feedforward_layernorm` reads, `intermediate.2` what its
`per_layer_input_gate` reads, `output` what it returns), and the stages each
module outside the skeleton runs in (a PyTorch prefill produces logits). Its
skeleton is the one its own config.json makes. A PyTorch run observes by its own
options, not by points chosen and no prepared copy: `observe` 'anchors' (and
`observeMemoryGiB`, its runner's snapshot budget for a turn), which the editor's
tensor capture shows with the server's words (option_catalog.pytorch_tap()); its
runner then writes every layer's four anchors of every forward pass
(`post_anchor_<layer>_<anchor>`) and `observation.json`, which says why when a
turn has none (the budget exceeded). Its first observed turn on the CPU is
captured fully, which exports the execution graph; every other turn, and every
one on the GPU, module-only (src/runner_pytorch). The intermediate tensors of
two runtimes of one architecture, a PyTorch side and a LiteRT-LM side, pair by
anchor: per layer and anchor, the PyTorch tensor and the LiteRT-LM point the
graph binds there, of those the Session observes (a layer's input is often the
output of the layer before, one point at two places), compared in float64
whatever dtype each stores; a side that observes no anchor point says so. Two
sides of one runtime pair as before: one model's points alike, PyTorch's by
anchor, and, in a Session that captures the anchors, LiteRT-LM's by anchor too.

Beside a PyTorch side that observes, each other point chosen of the LiteRT-LM
model (the editor's advanced picker) pairs with the PyTorch module of its
semantic operator where that is certain (`graph/counterparts.py`,
prepared_models.observed()): both bindings' `ops` name, per operator of a layer
as the skeleton names it (`attn/q_proj`, `ffn/act`, `ple/gate_proj`, the norms
`norm.attn.in` and the like; RoPE and the attention itself are inside the
modules and not mapped), the LiteRT-LM tap point that is its output (its
tensor's name directly in the operator's scope, of its operator: q_proj's
DEQUANTIZE, not the int8 FULLY_CONNECTED beside it) and the PyTorch module whose
output (or, for the gates, whose argument 0) it is. A point is paired when it is
exactly one operator's output, the layer has the module (Gemma 4 E2B's KV-shared
layers 15 to 34 have no k_proj, v_proj, k_norm or v_norm) and the shapes agree
after [batch, tokens] without axes of one, or flat on one side (q_proj: [8, 256]
against [2048]); it is placed at (layer, operator), compared flat, and the
PyTorch runner's Load asks for that module (`observeModules`, `{module, edge}`),
which it writes per forward pass under `post_module_<module>_<edge>`. Every
other point chosen says why not ("PyTorch has no counterpart for …; only the
anchors are captured there"), in the editor (`GET
/api/artifacts/{id}/counterparts`) and in Intermediate Tensors
(position-analysis' `unpaired`). Verified on the scan of
gemma-4-E2B-it.litertlm: each of the 21 operators is one point of each layer of
each signature that computes it.

By default a Session captures the anchors of both sides, each side by its own
setting: a PyTorch side by its `observe` option, a LiteRT-LM side by the
Session's `tap_anchors`, by side (two sides of one file, which load one copy,
both or neither; a record written before holds one bool for both)
(`session/anchor_points`): of each LiteRT-LM model, the point of every anchor of
every decoder layer of every signature its graph binds, each point once,
skipping, with the binding's reason, an anchor not bound (a prefill computes
only the layers whose KV cache it fills) or not observable (the first layer's
input is the signature's input embeddings), and a signature that runs only with
speculative decoding. The anchors are found in the graph, so as the Session is
prepared, once its start has the graph, and are kept in the record
(`tap_anchor_points`); the points chosen in the editor's advanced picker
(`tap_points`) are observed beside them, one copy of the model observing both,
of at most `tap_model.MAX_POINTS` (1024) points. Gemma 4 E2B's anchors are 189
points (decode 105, each prefill 42; `verify` skipped); its copy takes about 7 s
to write and check, as one of 2 points does, and a turn captures 0.66 MB a
decode step and 33 MB a 128-token prefill chunk, per side. A model of no
architecture the server has (Qwen) has no anchors, and says why. A record
written before the anchors were (no `tap_anchors`) observes what it chose, as it
did.

- `GET /api/artifacts/{id}/anchors`: the anchors a Session of a LiteRT-LM model
  captures by default, in the server's words (`status` 'found', 'pending' until
  the graph is converted, or 'none'; `text`; per signature how many, of which
  layers, and each anchor skipped with its reason) and what its copy costs on
  disk; `GET /api/runtime/capabilities`' `tap_anchors` is the editor's line of
  it.

- `GET /api/models/compare?ref=<artifact>&target=<artifact>`: the Session
  editor's hint under its two models (`session/model_comparison`): what of the
  two compares, by the analyses' own rules (`analysis/comparability`: one file;
  one vocabulary; tensors by place, by one architecture's anchors, by hand, or
  not; KV caches of one shape), from what the server has of each model without
  converting or running anything (a LiteRT-LM model's architecture is known once
  its graph is converted, and the hint says so until then): `{code, text, label,
  capture_shared}`, `label` what the listing calls the two (the record's
  `model`, which the server names), and `capture_shared` what the editor says of
  two LiteRT-LM sides of one file.

- `GET /api/artifacts/{id}/counterparts?pytorch=<artifact>&points=<id>,…`: how a
  Session of the LiteRT-LM model and the PyTorch one pairs each point chosen
  (`status` 'found', 'pending' until the graph is converted, or 'none'; per
  point `paired` 'anchor', 'module', 'manual' (a pair made by hand names it) or
  '', the `module` and `edge`, and `text`, as the user reads it).

### Pairs made by hand

Beside what the server pairs itself, the user may pair a tensor of one side's
execution graph with one of the other's in the Graph Diff's Mapping pane
(`session/mappings.py`): an operator output of a signature's graph on each side,
as `GET /api/artifacts/{id}/graph/{signature}` serves it. A pair belongs to the
two models, not to a Session: it is kept in the workspace's `mappings/`, one
JSON file per pair of models named by their two SHA-256s in sorted order, and
deleted with either model, and every Session of those two models has it, in
either order (a Session whose Reference loads the file's Target model reads each
pair the other way round). Each pair is `{id, label, created, ref, target}`,
each tensor `{signature, node, output, point}`, the point its output carries
(after pytorch-collector's own record of corresponding operators,
`rmsnorm_hf_vs_litert.json`). As it is saved it is checked: both outputs exist
in their graphs and can be captured (a LiteRT-LM tap point; on a PyTorch side an
anchor, or what a folded module returns or reads: the PyTorch runner captures
modules, not ATen operators, so every folded node's module carries a point in
the graph served); the two shapes hold the same values per position, without
axes of one, flat on one side, a prefill's rows dropped; at least one is in a
decoder layer; and it is no pair the two models have already, by hand or by the
server itself (the anchors, a semantic operator's module). One model on both
sides needs none: each tensor is paired with itself, and the editor says so.

A Session places a pair whose two tensors its sides observe at a place of its
own, (layer, `manual:<id>`), labelled by the pair, in that pair's signatures,
and compares it as any other place (two models of one runtime pair by these
places alone); a PyTorch side that observes asks its runner for the module the
pair names (`observeModules`) from the next start. A pair whose LiteRT-LM point
the Session does not observe says to add it in Tensor capture (`unpaired`,
`tensor_manual_not_observed`); a module the runner was not asked for when the
Session started says it is captured from the next start.

- `GET /api/mappings?session_id=<id>`: the pairs of the Session's two models as
  it has them: `status` 'pairs', 'identity' (one model) or 'none', `text` why,
  and per pair its tensors as this Session's sides are, `place` `{layer,
  anchor}` or null, and `text` why it has no place, or that a module it names is
  captured from the next start.
- `POST /api/mappings?session_id=<id>` `{ref, target, label}`, each tensor
  `{signature, node, output}`: saves one; 400 or 409 with the server's reason.
- `DELETE /api/mappings/{pair}?session_id=<id>`: deletes one.

Each answers the list as it is then; what Sessions' analyses observe is worked
out again.

- `GET /api/artifacts/{id}/graph`: its status, signatures (each with the layers
  it has operators of, `layer_ids`: Gemma 4 E2B's prefill computes layers 0 to
  14, its decode all 35), layer stems and why it has no architecture, if it has
  none; once failed, why.
- `GET /api/artifacts/{id}/graph/{signature}`: that signature's graph, a Model
  Explorer graph collection of its own.
- `GET /api/artifacts/{id}/architecture`: the skeleton, each anchor's binding
  and each block's region.
- `GET /api/position-analysis` names each point's node, `graph_node: {graph,
  node, output}`, the point each side's value of a cell is of (`points`), the
  forward pass each side reads each column from (`columns`: `{signature, step}`
  per side), and why a point it does not list has no value (`not_observed`):
  what the Graph Diff colours the anchors and the execution graph's outputs by.

## Moving a Session to another computer

A Session that has ended (a chat of it kept a turn and its Model Server is off)
is exported as one file, `<name>.mdsession`, with all its chats, and imported on
another computer, where its chats are read (`session/session_archive.py`). The
file is an uncompressed tar, streamed as it is read: `manifest.json` (`format`
`model-debugger-session`, `version` 2, the Session's id, when it was exported),
`record.json` (its chats in it), each chat's capture (`captures/chat-<n>/`),
each model's graph (`graphs/<sha256>/`), the pairs made by hand of its two
models (`mappings/`), and per model its description
(`models/<sha256>/model.json`) and what debugging reads of its file
(`models/<sha256>/facts/`: the tokenizer digest and KV quantization, a LiteRT-LM
model's tensor scans, a PyTorch model's `config.json`). The model files are not
in it: the importing server registers each model with these facts and no file
(the workspace's `facts/<sha256>/`), reads them wherever it would read the file,
and never loads such a model. The imported Session keeps its id unless one there
has it, and its name unless a Session there has it: then it is named `<name>
(2)`, `(3)`, and so on. It says where it came from (`imported`), and has ended:
no new chat, no start. The `bookmarks.json` an older server's export holds is
ignored. A file of `version` 1, an older server's, holds a Session of one chat,
its settings in the record and its capture directly under `captures/`: it is
imported as a Session of one ended chat, `Chat 1`, as the workspace's own
records of an older server are read (`session_store.upgrade()`).

- `GET /api/sessions/{id}/export`: the file, as an attachment; 409 with the
  reason (the listing's `actions.export`) for a Session that has not ended, or
  whose export is being sent. From the start of its stream until the stream
  ends, sent whole or given up by the browser, the Session is exporting: the
  listing says so (`exporting`), and it is neither exported again, deleted nor
  opened.
- `POST /api/sessions/import`, the file as the body: unpacked in `uploads/` and
  checked (a member of another name, a link, a name that leaves its directory,
  or a manifest of another format or version refuses the whole file); answers
  the Session as the listing shows it. An upload the browser cancels leaves
  nothing: one INFO line, `Import cancelled by the client`.

## Rules

The repository's `ci/check_server_layers.py` checks that dependencies only go
down, in this order, lowest first: `util`, `architectures`, `tap`, `graph`,
`capture`, `analysis`, `runner`, `session`, `service`, `server` (the facade),
`cli`. A module imports only from its own layer and those below it. `testing`
stands outside the order, and only tests and `testing` itself import it. A new
directory fails the check until it is given a place in `LAYERS`.

Not checked, but kept: `session` knows neither the HTTP API nor what the UI
lists; `session` and `service` reach devices only through the interface in
`runner/launcher.py`, and `server.py` alone picks its implementation.

## Checks and tests

The full check is the repository's `ci/check_server.sh`, which runs, in order:
buildifier's format and lint over the Starlark files; ruff's check and format
(`ruff.toml`), over the server and the Python in `ci/`; the layer check
(`ci/check_server_layers.py`); the check that each architecture's skeleton files
are byte for byte what its definition makes of its pinned configs (`bazel run
//model_debugger/server/architectures:generate -- --check`; without `--check` it
writes them); the check that the UI's metric catalog fixture is the server's
answer to GET /api/metrics (`ci/check_metric_catalog.py`, which builds the
server and asks it; its `--write` writes the fixture); `bazel test //...`, which
also checks that `requirements_lock.txt` still matches the requirements (which
needs the network); and the wheel: it builds the UI in `src/ui` (`npm run
build`, and `npm ci` first if `node_modules` is missing), then `bazel build
//:wheel`, and checks that the wheel holds
`model_debugger/server/ui/index.html`. The checks in `ci/` need Python's
standard library alone and run as `python3 -I`.

```shell
../../ci/check_server.sh
python3 -I ../../ci/check_server_layers.py "${PWD}/model_debugger/server"
python3 -I ../../ci/check_metric_catalog.py \
  "${PWD}/../ui/src/server/testing/metric_catalog_fixture.json"
bazel test //...
RUNNER_MODEL=/path/to/model.litertlm bazel test //...
```

The tests drive in-process fake runners (`testing/fake_runner.py`); the local
launcher's test runs the real command-line runner when it is built. With a
model, `cli/server_main_test.py` also runs the whole server with two real
runners through the HTTP API, on this computer's CPU and then Target on its GPU,
and the capture index's and analyses' tests compare a turn the real runner
captures (`testing/real_capture.py`). Nothing the runner captures is kept in the
repository. `.bazelrc` passes these through from the environment:

- `RUNNER_MODEL`: a `.litertlm` model; without it, the tests that run the model
  are skipped.
- `RUNNER_BINARY`: a runner other than the one Bazel built last in `src/runner`.
- `REAL_SESSIONS`: a directory of sessions the real runner recorded, for the
  capture index's tests of the real format.

Bazel sees neither these files nor the runner it finds in `src/runner`, so the
tests that read them are tagged `external` and run every time.
`ci/check_server.sh` makes the three paths absolute and stops if one names
nothing.

ruff alone, which Bazel runs from elsewhere, so the files go by absolute path;
the checks in the repository's `ci/` take this directory's `ruff.toml` by name:

```shell
bazel run //:ruff -- check "${PWD}/model_debugger"
bazel run //:ruff -- format "${PWD}/model_debugger"
bazel run //:ruff -- check --config="${PWD}/ruff.toml" "${PWD}"/../../ci/*.py
bazel run //:ruff -- format --config="${PWD}/ruff.toml" "${PWD}"/../../ci/*.py
```

The PyPI packages are pinned, with their hashes, in `requirements_lock.txt`,
from `requirements.in` (what the server needs, which the wheel requires too) and
`requirements_test.in` (what the tests and checks need besides). After changing
either, `bazel run //:requirements.update`.
