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
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

test_data = [
    "//tests:tf_simple_add",
    "//tests:tf_simple_v1",
    "//tests:tfl_fully_connected",
    "//tests:jax_sin",
    "//tests:graphdef_foo",
    "//tests:stablehlo_sin",
]

cc_library(
    name = "translate_helpers",
    srcs = ["translate_helpers.cc"],
    hdrs = ["translate_helpers.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers//:runtime_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
        "//common:graphnode_builder",
        "//common:schema_structs",
        "//common:visualize_config",
        "//utils:attribute_printer",
        "//utils:load_opdefs",
        "//utils:namespace_heuristics",
        "//utils:shardy_utils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:stablehlo_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_tflite_operator_lib",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        # copybara:uncomment "@org_tensorflow//tensorflow_text:ops_lib",
    ],
)

cc_library(
    name = "translations",
    hdrs = ["translations.h"],
    deps = [
        ":translate_helpers",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "//adapters/mlir/transforms:conversion",
        "//common:schema_structs",
        "//common:visualize_config",
    ],
    alwayslink = True,
)

cc_library(
    name = "model_json_graph_convert",
    srcs = ["model_json_graph_convert.cc"],
    hdrs = ["model_json_graph_convert.h"],
    deps = [
        ":translate_helpers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
        "//adapters/mlir/transforms:conversion",
        "//common:schema_structs",
        "//common:visualize_config",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//stablehlo/tests:check_ops",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:vhlo_ops",
        "@org_tensorflow//tensorflow/cc/saved_model:reader",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_import",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/stablehlo:legalize_tf_xla_call_module_to_stablehlo_pass",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:error_util",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:verify_no_outside_compilation_markers_pass",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core/protobuf:for_core_protos_cc",
    ] + select({
        "//conditions:default": [
        ],
    }),
)

cc_binary(
    name = "tensor-graph-vis-translate",
    srcs = ["tensor_graph_vis_translate_main.cc"],
    deps = [
        ":translations",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirTranslateMain",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TranslateLib",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//stablehlo/tests:check_ops",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_tflite_operator_lib",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
    ],
)
