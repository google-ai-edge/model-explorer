# Copyright 2026 The AI Edge Model Explorer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_library")
load(
    "//:version.bzl",
    "VERSION",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

genrule(
    name = "version_py",
    outs = ["version.py"],
    cmd = "echo '__version__ = \"'" + VERSION + "'\"' > $@",
)

pytype_strict_library(
    name = "version",
    srcs = [":version_py"],
)

pytype_strict_library(
    name = "ai_edge_model_explorer_adapter",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":_ffi",  # build_cleaner: keep
        ":version",  # build_cleaner: keep
    ],
)

# Declarative PyPI Packaging
# To build wheel: bazel build -c opt //python/ai_edge_model_explorer_adapter:wheel
# copybara:comment_begin(oss-only)
genrule(
    name = "copy_ai_edge_model_explorer_adapter_so",
    srcs = ["//c:ai_edge_model_explorer_adapter"],
    outs = ["libai_edge_model_explorer_adapter.so"],
    cmd = """
        SRCS_ARRAY=($(SRCS))
        OUTS_ARRAY=($(OUTS))
        if [ $${#SRCS_ARRAY[@]} -gt 0 ]; then
            cp $${SRCS_ARRAY[0]} $${OUTS_ARRAY[0]}
        else
            touch $${OUTS_ARRAY[0]}
        fi
    """,
)

genrule(
    name = "copy_ai_edge_model_explorer_adapter_dylib",
    srcs = ["//c:ai_edge_model_explorer_adapter"],
    outs = ["libai_edge_model_explorer_adapter.dylib"],
    cmd = """
        SRCS_ARRAY=($(SRCS))
        OUTS_ARRAY=($(OUTS))
        if [ $${#SRCS_ARRAY[@]} -gt 0 ]; then
            cp $${SRCS_ARRAY[0]} $${OUTS_ARRAY[0]}
        else
            touch $${OUTS_ARRAY[0]}
        fi
    """,
)

genrule(
    name = "copy_ai_edge_model_explorer_adapter_dll",
    srcs = ["//c:ai_edge_model_explorer_adapter"],
    outs = ["ai_edge_model_explorer_adapter.dll"],
    cmd = """
        SRCS_ARRAY=($(SRCS))
        OUTS_ARRAY=($(OUTS))
        if [ $${#SRCS_ARRAY[@]} -gt 0 ]; then
            cp $${SRCS_ARRAY[0]} $${OUTS_ARRAY[0]}
        else
            touch $${OUTS_ARRAY[0]}
        fi
    """,
)

filegroup(
    name = "default_libs",
    srcs = select({
        ":macos_arm64": [":copy_ai_edge_model_explorer_adapter_dylib"],
        ":macos_x86_64": [":copy_ai_edge_model_explorer_adapter_dylib"],
        ":wheel_platform_macos_arm64": [":copy_ai_edge_model_explorer_adapter_dylib"],
        ":wheel_platform_macos_x86_64": [":copy_ai_edge_model_explorer_adapter_dylib"],
        "@platforms//os:macos": [":copy_ai_edge_model_explorer_adapter_dylib"],
        ":windows_x86_64": [":copy_ai_edge_model_explorer_adapter_dll"],
        ":wheel_platform_windows_x86_64": [":copy_ai_edge_model_explorer_adapter_dll"],
        "@platforms//os:windows": [":copy_ai_edge_model_explorer_adapter_dll"],
        "//conditions:default": [":copy_ai_edge_model_explorer_adapter_so"],
    }),
)

pytype_strict_library(
    name = "_ffi",
    srcs = ["_ffi.py"],
    data = [":default_libs"],
    deps = [],
)

# Use "wheel_platform" to allow cross-compiling the wheel package.
config_setting(
    name = "wheel_platform_linux_x86_64",
    define_values = {"wheel_platform": "linux_x86_64"},
)

config_setting(
    name = "wheel_platform_linux_arm64",
    define_values = {"wheel_platform": "linux_arm64"},
)

config_setting(
    name = "wheel_platform_windows_x86_64",
    define_values = {"wheel_platform": "windows_x86_64"},
)

config_setting(
    name = "wheel_platform_macos_arm64",
    define_values = {"wheel_platform": "macos_arm64"},
)

config_setting(
    name = "wheel_platform_macos_x86_64",
    define_values = {"wheel_platform": "macos_x86_64"},
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

config_setting(
    name = "macos_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)

config_setting(
    name = "macos_x86_64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "windows_x86_64",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:x86_64",
    ],
)

py_package(
    name = "package",
    packages = ["python.ai_edge_model_explorer_adapter"],
    deps = [":ai_edge_model_explorer_adapter"],
)

py_wheel(
    name = "wheel",
    abi = "none",
    distribution = "ai_edge_model_explorer_adapter",
    platform = select({
        ":wheel_platform_linux_x86_64": "manylinux_2_27_x86_64",
        ":wheel_platform_linux_arm64": "manylinux_2_27_aarch64",
        ":wheel_platform_macos_arm64": "macosx_12_0_arm64",
        ":wheel_platform_macos_x86_64": "macosx_10_15_x86_64",
        ":wheel_platform_windows_x86_64": "win_amd64",
        ":linux_aarch64": "manylinux_2_27_aarch64",
        ":linux_x86_64": "manylinux_2_27_x86_64",
        ":macos_arm64": "macosx_12_0_arm64",
        ":macos_x86_64": "macosx_10_15_x86_64",
        ":windows_x86_64": "win_amd64",
        "//conditions:default": "any",
    }),
    python_tag = "py3",
    python_requires = ">=3.9",
    summary = "Model Explorer backend adapter for model conversion",
    description_file = "//:README.md",
    license = "Apache-2.0",
    homepage = "https://github.com/google-ai-edge/model-explorer",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    strip_path_prefixes = ["python"],
    version = VERSION,
    deps = [":package"],
)
# copybara:comment_end
