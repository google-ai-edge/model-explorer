# Model Explorer Converter

## Build and Install Locally

### Build

The script `python/pip_package/build_pip_package.sh` builds a Python *.whl*
under the output directory `gen/converter_pip/dist`. For example:

```
% ./python/pip_package/build_pip_package.sh

% tree gen/converter_pip/dist
gen/converter_pip/dist
├── model_explorer_converter-0.1.0-cp311-cp311-linux_x86_64.whl
└── model-explorer-converter-0.1.0.linux-x86_64.tar.gz
```

### Install

Install the resulting *.whl* via pip. For example, in a Python virtual
environment:

```
% python3 -m venv ~/tmp/venv
% source ~/tmp/venv/bin/activate
(venv) $ pip install gen/converter_pip/dist/model_explorer_converter-0.1.0-cp311-cp311-linux_x86_64.whl
```

The package should now be importable and usable. For example:

```
(venv) $ python3
>>> from model_explorer_converter import _pywrap_convert_wrapper as convert_wrapper
>>> config = convert_wrapper.VisualizeConfig()
>>> model_path = 'foo.tflite'
>>> json = convert_wrapper.ConvertFlatbufferToJson(config, model_path, True)
>>> print(json)
```
