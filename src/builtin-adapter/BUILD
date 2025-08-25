load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "translate_helpers",
    srcs = ["translate_helpers.cc"],
    hdrs = ["translate_helpers.h"],
    deps = [
        ":graphnode_builder",
        ":status_macros",
        ":visualize_config",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
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
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:stablehlo_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_tflite_operator_lib",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//formats:schema_structs",
        "//tools:attribute_printer",
        "//tools:load_opdefs",
        "//tools:namespace_heuristics",
        "//tools:shardy_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        # copybara:uncomment "@org_tensorflow//tensorflow_text:ops_lib",
    ],
)

# Library of translation functions.
cc_library(
    name = "translations",
    hdrs = ["translations.h"],
    deps = [
        ":translate_helpers",
        ":visualize_config",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "//formats:schema_structs",
        "//transforms:conversion",
    ],
    alwayslink = True,
)

cc_library(
    name = "model_json_graph_convert",
    srcs = ["model_json_graph_convert.cc"],
    hdrs = ["model_json_graph_convert.h"],
    deps = [
        ":status_macros",
        ":translate_helpers",
        ":visualize_config",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
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
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//stablehlo/tests:check_ops",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:vhlo_ops",
        "@org_tensorflow//tensorflow/cc/saved_model:reader",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_import",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//formats:schema_structs",
        "//transforms:conversion",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/stablehlo:legalize_tf_xla_call_module_to_stablehlo_pass",
        "@org_tensorflow//tensorflow/compiler/mlir/stablehlo:rename_entrypoint_to_main",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:verify_no_outside_compilation_markers_pass",
        "@local_xla//xla/mlir_hlo",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core/protobuf:for_core_protos_cc",
    ] + select({
        "//conditions:default": [
        ],
    }),
)

cc_library(
    name = "visualize_config",
    hdrs = ["visualize_config.h"],
)

cc_library(
    name = "graphnode_builder",
    srcs = ["graphnode_builder.cc"],
    hdrs = ["graphnode_builder.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//formats:schema_structs",
    ],
)

cc_library(
    name = "status_macros",
    hdrs = ["status_macros.h"],
)

cc_library(
    name = "models_to_json_lib",
    srcs = [
        "models_to_json_lib.cc",
    ],
    hdrs = ["models_to_json_lib.h"],
    deps = [
        ":direct_flatbuffer_to_json_graph_convert",
        ":direct_saved_model_to_json_graph_convert",
        ":model_json_graph_convert",
        ":status_macros",
        ":visualize_config",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//mediapipe_adapter:mediapipe_to_json",
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
        ":visualize_config",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/compiler/mlir:init_mlir",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/tools:command_line_flags",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "direct_flatbuffer_to_json_graph_convert",
    srcs = ["direct_flatbuffer_to_json_graph_convert.cc"],
    hdrs = ["direct_flatbuffer_to_json_graph_convert.h"],
    deps = [
        ":graphnode_builder",
        ":status_macros",
        ":visualize_config",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers//:runtime_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:const_tensor_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_tflite_operator_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:offset_buffer",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/core:absl_error_model_builder",
        "//formats:schema_structs",
        "//tools:attribute_printer",
        "//tools:convert_type",
        "//tools:load_opdefs",
        "//tools:namespace_heuristics",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "direct_saved_model_to_json_graph_convert",
    srcs = ["direct_saved_model_to_json_graph_convert.cc"],
    hdrs = ["direct_saved_model_to_json_graph_convert.h"],
    deps = [
        ":graphnode_builder",
        ":status_macros",
        ":visualize_config",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@com_google_protobuf//:protobuf_lite",
        "@org_tensorflow//tensorflow/cc/saved_model:reader",
        "//formats:schema_structs",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)
