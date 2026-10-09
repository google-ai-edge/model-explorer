# Pinned viewport and selection bridge

The served `ai-edge-model-explorer-visualizer@0.1.2`
[browser bundle](upstream/model-explorer/dist/main_browser.js) derives from the
style-r6 prototype, with the additional pane background bridge below. These
methods are local extensions, **not upstream public APIs**.
[PROVENANCE.json](upstream/model-explorer/PROVENANCE.json) records the exact hash.

<!-- mdformat off(preserve GFM table layout) -->
| Method | Contract |
| --- | --- |
| `readPaneViewport(paneIndex)` | Returns center, scale, pane/graph IDs and selection, or null before a visible renderer exists. |
| `setPaneViewport(viewport, paneIndex, duration = 0)` | Restores camera/d3 transform, rejects incompatible graph IDs, does not select nodes. |
| `fitPaneGraph(paneIndex)` | Fits the actual graph through the existing renderer. |
| `focusNodeViewport(nodeId, paneIndex, options)` | Frames a rendered node without selecting it; supports `labelSize`, top/center alignment, padding and duration. |
| `setPaneSelection(nodeId, paneIndex)` | Selects or clears that pane without camera navigation or inferred cross-pane mapping. Empty ID clears. |
| `setPaneBackground(background, paneIndex)` | Accepts a six-digit hex color, repaints the existing WebGL scene surface, and retains topology, camera and selection. Returns false while the pane is unavailable. |
<!-- mdformat on -->

The bridge addresses connected visible renderers. False/null means
initialization should retry when the host is visible. It preserves the pinned
worker's layout algorithm and honors reduced motion. The selection helper is for
flat operation/output-tensor graphs.

The host keeps selection and camera movement separate: metric changes and
background clicks preserve inspection; Search explicitly reveals, while
Fit/Readable only move the view. Mapping can select only real captured records
and cancellation can restore the prior camera. Replacing the bundle requires
reapplying and testing these extensions; private renderer names are not stable
API.

The parent passes its current light/dark theme through the checked frame
channel. The frame updates its own CSS tokens, this scene background, and the
existing node-data colors. Theme changes do not remount the visualizer or alter
numerical values.
