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
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@stablehlo//:stablehlo_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//formats:schema_structs",
        "//tools:attribute_printer",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
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
        "@llvm-project//mlir:Support",
        "//formats:schema_structs",
    ],
    alwayslink = True,
)

cc_library(
    name = "model_json_graph_convert",
    srcs = ["model_json_graph_convert.cc"],
    hdrs = ["model_json_graph_convert.h"],
    deps = [
        ":status_macros",
        ":translations",
        ":visualize_config",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/cc/saved_model:reader",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_import",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/stablehlo:legalize_tf_xla_call_module_to_stablehlo_pass",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/stablehlo:rename_entrypoint_to_main",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow:quantize_preprocess",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:verify_no_outside_compilation_markers_pass",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/protobuf:for_core_protos_cc",
    ],
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
        ":status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//formats:schema_structs",
    ],
)

cc_library(
    name = "status_macros",
    hdrs = ["status_macros.h"],
)
