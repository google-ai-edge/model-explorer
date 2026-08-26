# Model Explorer Adapter

The Model Explorer Adapter provides graph extraction and visualization
conversion for ML models (TensorFlow SavedModel, GraphDef, FlatBuffer/LiteRT,
StableHLO/MLIR) to Model Explorer's visualization JSON format.

It is implemented as a stable C-ABI library
(`libai_edge_model_explorer_adapter.so` / `.dylib`) with a pure Python `ctypes`
FFI wrapper for universal ABI stability across Python versions without pybind11
compile-time matrix bloat.

## Install from PyPI

Install `ai-edge-model-explorer-adapter` via pip from PyPI. For example, in a
Python virtual environment:

```bash
% python3 -m venv ~/tmp/venv
% source ~/tmp/venv/bin/activate
(venv) $ pip install ai-edge-model-explorer-adapter
```

## Use the Package

After installation, the package can be imported directly:

```python
import ai_edge_model_explorer_adapter as adapter

config = adapter.VisualizeConfig()
json_str = adapter.ConvertFlatbufferDirectlyToJson(config, "model.tflite")
print(json_str)
```

## Build and Install Locally

### Declarative Bazel Build

Build the Python wheel directly using Bazel:

```bash
bazel build -c opt //python/ai_edge_model_explorer_adapter:wheel
```

The resulting wheel is located under `bazel-bin/`:
`bazel-bin/python/ai_edge_model_explorer_adapter/`

### Install

Install the resulting wheel via pip:

```bash
(venv) $ pip install \
  bazel-bin/python/ai_edge_model_explorer_adapter/*.whl
```
