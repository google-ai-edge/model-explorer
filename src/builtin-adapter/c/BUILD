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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "adapter_hdrs",
    hdrs = ["adapter.h"],
)

cc_library(
    name = "adapter",
    srcs = ["adapter.cc"],
    hdrs = ["adapter.h"],
    deps = [
        ":adapter_hdrs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//adapters/litert:direct_flatbuffer_to_json_graph_convert",
        "//adapters/mlir:model_json_graph_convert",
        "//adapters/tensorflow:direct_saved_model_to_json_graph_convert",
        "//common:visualize_config",
    ],
)

cc_binary(
    name = "ai_edge_model_explorer_adapter",
    srcs = ["adapter.cc"],
    copts = ["-fvisibility=hidden"],
    features = ["-legacy_whole_archive"],
    linkopts = select({
        "@platforms//os:macos": [],
        "@platforms//os:windows": [],
        "//conditions:default": [
            "-Wl,--no-undefined",
            "-Wl,--strip-all",
            "-Wl,--gc-sections",
            "-Wl,-Bsymbolic",
            "-Wl,-rpath,$$ORIGIN",
        ],
    }),
    linkshared = 1,
    deps = [
        ":adapter_hdrs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//adapters/litert:direct_flatbuffer_to_json_graph_convert",
        "//adapters/mlir:model_json_graph_convert",
        "//adapters/tensorflow:direct_saved_model_to_json_graph_convert",
        "//common:visualize_config",
    ],
)
