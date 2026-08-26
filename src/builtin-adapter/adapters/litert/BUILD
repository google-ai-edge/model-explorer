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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "direct_flatbuffer_to_json_graph_convert",
    srcs = ["direct_flatbuffer_to_json_graph_convert.cc"],
    hdrs = ["direct_flatbuffer_to_json_graph_convert.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "//adapters/litert/litertlm:litertlm_header_schema",
        "//adapters/litert/litertlm:litertlm_read",
        "//common:graphnode_builder",
        "//common:schema_structs",
        "//common:visualize_config",
        "//utils:attribute_printer",
        "//utils:convert_type",
        "//utils:diagnostic_collector",
        "//utils:load_opdefs",
        "//utils:namespace_heuristics",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:const_tensor_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_tflite_operator_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:offset_buffer",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/core:absl_error_model_builder",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)
