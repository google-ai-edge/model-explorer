# Model Explorer Adapter

## Install from PyPI

Install `ai-edge-model-explorer-adapter` via pip from PyPI. For example, in
a Python virtual environment:

```
% python3 -m venv ~/tmp/venv
% source ~/tmp/venv/bin/activate
(venv) $ pip install ai-edge-model-explorer-adapter
```

## Use the Package
After installation, the package should now be importable and usable. For
example:

```
(venv) $ python3
>>> from ai_edge_model_explorer_adapter import _pywrap_convert_wrapper as convert_wrapper
>>> config = convert_wrapper.VisualizeConfig()
>>> model_path = 'foo.tflite'
>>> json = convert_wrapper.ConvertFlatbufferToJson(config, model_path, True)
>>> print(json)
```

## Build and Install Locally

### Build

The script `python/pip_package/build_pip_package.sh` builds a Python *.whl*
under the output directory `gen/adapter_pip/dist`. The first argument is the
package version, which should be a string of the form "x.x.x". For example:

```
% ./python/pip_package/build_pip_package.sh 0.1.0

% tree gen/adapter_pip/dist
gen/adapter_pip/dist
├── ai_edge_model_explorer_adapter-0.1.0-cp311-cp311-manylinux_2_17_x86_64.whl
└── ai-edge-model-explorer-adapter-0.1.0.manylinux_2_17_x86_64.tar.gz
```

### Install

Install the resulting *.whl* via pip. For example, in a Python virtual
environment:

```
% python3 -m venv ~/tmp/venv
% source ~/tmp/venv/bin/activate
(venv) $ pip install gen/adapter_pip/dist/ai_edge_model_explorer_adapter-0.1.0-cp311-cp311-manylinux_2_17_x86_64.whl
```

The package should now be importable and usable.
