# Embedded Gemma architecture preview

Packaged from the graph-style-preview design prototype (revision 142) for the
Intermediate Tensors Graph Diff view. This is a static Gemma 4 E2B semantic
architecture preview, not an architecture extracted from the active session.
It intentionally omits the prototype's model name, application toolbar, and
Details pane. The Angular panel owns token position, metric and Details. Graph Diff shows one
aligned token position; Heatmap and Line chart retain their range and aggregation.

`IntermediateGraphArch` exchanges same-origin, source-checked messages:
- `context`: stage, per-layer output metrics at the selected token position, formatted values, panel heat
  palette and theme. Missing values stay absent. No demo metric script is loaded.
- `selection`: semantic node/layer attributes for the host Details panel.

Transformer Layers and Per-Layer Embedding open a read-only Mapping preview, automatically pairing the
identical bundled LiteRT-LM graphs by node/output-port identity. This is explicitly
labeled preview data in the host Details. Other module tensor entries currently inspect
semantic attributes only. Session-specific execution mapping and per-op metrics are not yet
connected. Do not use the bundled sample mapping metadata as session evidence.

The standalone design source remains in the graph-style-preview workspace;
this snapshot is independently served by the Angular application and does not
require port 4202. Shared panel state, position selection, loading, palette and navigation are covered
by GraphMetricPanel tests and browser verification.

The module tree uses full-row metric colors. The shared toolbar's Summary control
(default Output) governs collapsed Decoder and Transformer Layers rows. Output
requires the actual final output; Min/Max reduce only available descendant output
metrics, with coverage in the value tooltip. Missing values remain a dash.
Embedding is disabled because its Arch is unavailable; missing metrics alone do
not disable navigation. The Summary preference is included in saved panel state.

Navigation membership follows the bundled LiteRT execution export's stage scopes:
Prefill exposes layers 0–14 (layer 14 is partial), Decode exposes layers 0–34,
Final Norm and Output Head. Shared embedding and PLE entries remain in both.
This is static export coverage, not executed-session provenance. Missing metrics
do not remove nodes. Replace the catalog with session execution coverage when
that source is connected. Transformer-stack Output still means the final model
layer output, not the last exported partial region.

Transformer defaults pair identical graph/node/output identities automatically.
Semantic anchors use the known Layer 0 Decode bindings, matching compiler scopes
and opcodes in other layer/stage exports, and the actual edge feeding attention's
output reshape. Missing anchors remain unpaired (notably partial Prefill Layer 14).
Runtime regions follow compiler norm-entry boundaries rather than name substring
matches, which misclassify shared/CSE namespaces. Block entries and tensor labels
open the same configuration range; inputs are owned upstream. Hover changes the
background and corresponding edge color, without selecting the tensor.
Rebuild these regions with `scripts/build_layer_mapping_preview.py EXPORT.json`;
validate with `node scripts/check_layer_mapping.mjs` from `src/ui`.

Arch tensor labels have inline metric tags; collapsed blocks show summary tags.
Tags use host-formatted values and host metric colors and share the parent tensor's
interaction. Hover does not recolor the tag. Ops retain their default appearance.
Optional `context.tensorMetrics[semanticScope][outputAnchor]` contains explicit
semantic output metrics. Currently only `context.layers[layer]` supplies the layer
output; internal tensors and block outputs remain neutral `—` until bound data is
provided. Block summaries never substitute the layer or block final output for missing internal data.
Min/Max use unique, available internal outputs. Run `node scripts/check_arch_metrics.mjs`
to verify these ownership, missing-value, aggregation and palette contracts.

### Tensor display and simulated metrics

Arch tensor tags always show Icon + Metric. Tensor names are available only in Details (and accessible labels), never as graph text or hover names. Display controls legends only. Old saved name-display preferences are ignored and dropped when the panel is saved.

Internal metrics use generated fixture data for this preview. This is a data-source implementation detail, not a user-visible simulation mode or Display preference. Values are deterministic per semantic scope, output anchor, stage and token position. The host supplies formatted/color samples using the current metric scale. Explicit tensor bindings (including null) and layer output data take precedence. Legacy simulation preferences are ignored; actual missing values still render as dashes. Folded blocks aggregate unique internal semantic outputs with the topbar Min/Max setting (default Max), excluding the separately displayed final output. Both options remain enabled when data is absent; empty summaries show —. Legacy saved Output settings migrate to Max. Op styles are unchanged.

### Arch inspection and configuration

Clicking a tensor selects it in place and updates Tensor info in the host Details panel. Tensor tooltips do not repeat semantic names; metric tooltips remain numeric/provenance-only. Mapping is opened from the gear configuration buttons. Pointer hover or keyboard focus on a gear outlines all visible tensors in its configuration scope with a dashed rectangle, including internal and output tensors but excluding upstream inputs. Transformer block gears cover internal outputs, the block final output and the following residual Add output. Residual Adds stay outside the block visually but share its configuration button; Layer Scale Multiply shares the Per-Layer Gating button, including the final layer output. There are three configuration buttons in total. Folded summaries still aggregate internal outputs only, excluding block and residual outputs. External layer inputs remain owned by their upstream producers, with ownership explained in Tensor info. Module Arch gears sit beside the producing semantic Op.

### Layer Arch metric readability

Layer tensor icons and values share one heat-colored 24-unit tag. Values use 13px medium-weight, tabular numerals; Ops keep their existing style. Folded block aggregates use the same larger numeric typography. Module Arch keeps its previous rendering. Expanding Layer blocks preserves readable zoom (90–100% minimum according to available width, without shrinking a closer user zoom) and keeps the edited header vertically anchored, centering its block horizontally. Collapse preserves zoom; Zoom to fit remains explicit. Selection, gear-only Mapping and dashed configuration coverage remain unchanged.

Folded Layer blocks place their internal summary on a separate row with a stacked-tensor symbol. Summary selection opens Summary info. Configuration hover/focus includes this summary and visible output tags, using a dashed border plus a translucent mask. Coverage edges and outlines share the native subgraph indicator blue (#135cbb); configuration buttons reuse its background (#d5e7ff), border, icon color, size and placement. Ordinary tensor connectors, neutral edges and arrows use the native Model Explorer edge color (#aaa in light mode), independently of Op borders. Legacy Layer Graph entry buttons and its toolbar title are removed from the host preview.

Layer Mapping ranges match configuration hover ownership. Missing anchors in partial exports produce an explicit unavailable range instead of falling back to the full layer. The mapping check verifies exact output ownership for all three configuration scopes across all 65 bundled layer regions.


PLE preparation Arch wraps Embedding Lookup, Embedding Projection and Combine in one foldable block, with one settings entry covering all three outputs. Its Mapping workspace output selector switches between the separately exported lookup graph and the active-stage projection/combine graph; it does not invent connections between runtime exports. Decoder Layer Arch keeps its original separate gating block, residual Add and final Multiply; the latter two still share its configuration scope.

Layer Arch blocks are fixed overview cards: no in-place expansion or global expand/collapse controls. Their native Model Explorer subgraph arrows enter the corresponding runtime Mapping directly. Internal metric values remain in block Details. Mapping defaults to inspection; Advanced configuration explicitly reveals runtime-source and pair editing controls in a workspace side pane. Closing/reopening Mapping resets to inspection, and closing Advanced configuration restores the full graph width. PLE preparation retains its existing module Arch grouping.

Layer Arch presents Attention, Feed Forward and Per-Layer Gating as fixed semantic nodes, using the same neutral surface, border and medium label weight as Ops. Tensor summary tags retain metric colors. Gears open Mapping for inspection; manual pair edits remain behind Advanced configuration. The separate PLE preparation Arch container is unchanged.
