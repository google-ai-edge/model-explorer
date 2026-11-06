# Architectures

What a model is, and nothing of how a runtime runs it: each architecture's
definition, which makes a model's semantic skeleton from its Hugging Face
`config.json`; the skeleton schema and its checks; and, per checkpoint the
server knows, its pinned `config.json` and the skeleton it makes. This
package is the one source of the definitions; it depends on nothing of the
server's but itself.

The rule that splits it from the rest: the architecture says what the model
is; a fact a runtime could change belongs to that runtime's side. How a
LiteRT-LM graph binds to an architecture (which tensor of a layer is which
anchor, which signature input the first layer reads, why a port is bound
to nothing, which LiteRT operator an `Add` is, which signatures run the
output head) is in `graph/bindings/<id>.json`, read by
`graph/architecture.py`.

## Layout

-   `skeleton.py`: the schema: the semantic operators' vocabulary
    (`SEMANTIC_OPS`), block names, the file form (`dumps()`, `load()`), and
    every rule (`check()`, `check_anchor_criterion()`, `blocks()`).
-   `builder.py`: what a definition writes with: `Graph`, `anchor()`, display
    names and categories, and the words a config gives (activation, attention
    and FFN algorithm, RoPE).
-   `gemma4.py`: Gemma 4's definition (`model_type` `gemma4_text`): the
    decoder layer definition, each layer's parameters, `build()`.
-   `catalog.py`: the architectures known (`data/<id>/architecture.json`),
    their skeletons, `generate()`: any `config.json` to its architecture and
    checked skeleton, and `provenance()`: where a skeleton comes from, by
    paths in the repository.
-   `generate.py`: the generator: writes each pinned checkpoint's skeleton, or
    with `--check` fails unless every file is what it would write.
-   `data/gemma4/architecture.json`: Gemma 4: id, name, `model_type`, its
    definition's module and where it was ported from (`generator`:
    `definition`, `derived_from`), the modules outside the skeleton
    (`outside`), and per checkpoint its skeleton, SHA-256 and pinned config
    (`pinned`: file, Hugging Face model, revision and source, git blob,
    SHA-256, where it was copied from).
-   `data/gemma4/*.config.json`: the pinned configs, byte for byte Hugging
    Face's.
-   `data/gemma4/*.skeleton.json`: the skeletons the generator writes from
    them.

```shell
# Write.
bazel run //model_debugger/server/architectures:generate
# Check.
bazel run //model_debugger/server/architectures:generate -- --check
```

`ci/check_server.sh` runs the check. A changed skeleton fails it until it
is written again, and then until its entry's `sha256` in
`architecture.json` is updated by hand, so that the change is reviewed.

## Provenance

What the server says a skeleton comes from (`GET
/api/artifacts/{id}/architecture`'s `generator` and `config`,
`Architecture.provenance()`) is this package: `generator` is
`src/server/model_debugger/server/architectures/generate.py`, the
definition's module (`gemma4.py`), the commands that write and check the
skeletons, and the digest of the architectures' files
(`catalog.identity()`); `config` is the pinned config, its path in the
repository, Hugging Face model, revision, source, git blob, SHA-256 and
where it was copied from. Where the definition was ported from is a note
in it, `generator.derived_from`, as below.

Ported on 2026-10-19 from the user's repository `~/model_arch_explore`
(HEAD `5830e01f8207881c2a50a90a03c0ccaf34df132d`), its `mkjson.py` as the
working tree held it: SHA-256
`ca4581c40323dc1c020c51cdd733e42f259afba20703cfdfc5ca8eb7e6e9360f`, last
committed in `7b4f6dba31a458e25223aa5136a06f3ac4ead162`, plus an
uncommitted change: the decoder's second input, `per_layer_input`, which
`ple` reads as its input 1 (`Gemma4TextDecoderLayer.forward`,
`hidden_states * per_layer_input`), and the rule that requires it. Only the
definitions moved; the report and page tools there (`mkaudit.py`,
`build.py`, `mkdemo.py`, `ui/`, `layout.mjs`, `docs/`) stay.

Where each part of mkjson.py went:

-   `SEMANTIC_OPS`, `CAT_LABEL`, `STRUCTURAL`, `attr`, `srcs`, `blocks_of`,
    `check`, `check_anchor_criterion`: `skeleton.py` (`srcs` is `sources`,
    `blocks_of` is `blocks`, `CAT_LABEL` is `CATEGORY_LABELS`).
-   `LABEL`, `CATEGORY`, `ACTIVATION`, `activation_of`, `attn_label`,
    `ffn_label`, `rope_attrs`, `Graph`, `anchor`: `builder.py`.
-   `decoder_def`, `decoder_layer_attrs`, `build`: `gemma4.py`
    (`decoder_definition`, `layer_attrs`, `build`).
-   `main` (`text_config` of the config, build, check, write compact JSON):
    `catalog.generate()`, `skeleton.dumps()`, `generate.py`.

The port's skeletons are byte for byte mkjson.py's for both pinned configs,
and its checks give mkjson.py's verdicts. It differs in how it fails, not
in what: an error is a `ValueError`, not `SystemExit`, and where mkjson.py's
`check()` stops at a block it cannot name, `check()` says so and goes on.

The pinned configs:

-   `google/gemma-4-E2B-it`:
    -   Hugging Face revision: `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
    -   git blob: `923b5e9405e7d319572b0c1b1a89291512262aa3`
    -   SHA-256:
        `1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330`
-   `google/gemma-4-E4B-it`:
    -   Hugging Face revision: `d6436b3d62967e1af08bbb046c6300b2a9ae8e85`
    -   git blob: `d68960fdcce766f2bfe41436325a8a483a74d125`
    -   SHA-256:
        `33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4`

E2B's was copied from the Hugging Face cache's snapshot of that revision;
E4B's from `~/Github/ai_edge_eval/models/gemma-4-E4B-it/`, a local download
whose `.cache/huggingface/download/config.json.metadata` names the revision
and the blob, which the file's git blob id matches.

## A runtime that has the config

A skeleton is made from a `config.json` and the architecture's rules alone:
`catalog.generate(config)` picks the architecture by the text config's
`model_type` and builds and checks its skeleton, for any size of the
family. A PyTorch runner loads its model from a Hugging Face directory, so
it has that `config.json`: the server, given it, gets the skeleton from
`generate()`, with no pinned file. Only a model file that holds no
config, a `.litertlm`, needs the pinned checkpoints, which
`graph/architecture.detect()` matches by its layer count and width.

What a PyTorch runtime adds is its own binding, beside LiteRT-LM's in
`graph/bindings/` and written the same way: where each anchor of a layer is
in the PyTorch module tree (in Gemma 4, `input` is what
`model.layers[i]` reads, `intermediate.1` what its `pre_feedforward_layernorm`
reads, `intermediate.2` what its `per_layer_input_gate` reads, `output` what
it returns), and the stages each module outside the skeleton runs in, which
need not be LiteRT-LM's (its prefill signatures produce no logits; a
PyTorch prefill can). The definitions do not change.
