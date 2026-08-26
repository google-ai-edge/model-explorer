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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

exports_files(["README.md"])

cc_library(
    name = "models_to_json_lib",
    srcs = [
        "models_to_json_lib.cc",
    ],
    hdrs = ["models_to_json_lib.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//adapters/litert:direct_flatbuffer_to_json_graph_convert",
        "//adapters/mlir:model_json_graph_convert",
        "//adapters/tensorflow:direct_saved_model_to_json_graph_convert",
        "//common:visualize_config",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_binary(
    name = "models_to_json",
    srcs = [
        "models_to_json_main.cc",
    ],
    deps = [
        ":models_to_json_lib",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//common:visualize_config",
        "@org_tensorflow//tensorflow/compiler/mlir:init_mlir",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/tools:command_line_flags",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
