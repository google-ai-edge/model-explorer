#load("@llvm-project//llvm/build_defs:lit.bzl", "glob_lit_tests")
#load("@org_tensorflow//tensorflow:tensorflow.google.bzl", "filegroup")

package(
    #    default_applicable_licenses = ["@org_tensorflow//:LICENSE"],  # copybara:comment
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "translate_helpers",
    srcs = ["translate_helpers.cc"],
    hdrs = ["translate_helpers.h"],
    compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":graphnode_builder",
        ":visualize_config",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:string_view",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:FuncDialect",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/stablehlo:stablehlo_ops",
        "//third_party/tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/formats:schema_structs",
        "//third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/tools:attribute_printer",
        "//third_party/tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//util/task:status",
    ],
)

# Library of translation functions.
cc_library(
    name = "translations",
    hdrs = ["translations.h"],
    deps = [
        ":translate_helpers",
        ":visualize_config",
        "//third_party/absl/status:statusor",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:Support",
        "//third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/formats:schema_structs",
    ],
    alwayslink = True,
)

cc_library(
    name = "model_json_graph_convert",
    srcs = ["model_json_graph_convert.cc"],
    hdrs = ["model_json_graph_convert.h"],
    deps = [
        ":translations",
        ":visualize_config",
        "//file/base",
        "//file/base:options_cc",
        "//file/colossus/public:cns",  # buildcleaner: keep
        "//third_party/absl/log",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:FuncDialect",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:Pass",
        "//third_party/llvm/llvm-project/mlir:Support",
        "//third_party/tensorflow/cc/saved_model:reader",
        "//third_party/tensorflow/compiler/mlir/lite:flatbuffer_import",
        "//third_party/tensorflow/compiler/mlir/lite/stablehlo:legalize_tf_xla_call_module_to_stablehlo_pass",
        "//third_party/tensorflow/compiler/mlir/lite/stablehlo:rename_entrypoint_to_main",
        "//third_party/tensorflow/compiler/mlir/quantization/tensorflow:quantize_preprocess",
        "//third_party/tensorflow/compiler/mlir/tensorflow",
        "//third_party/tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "//third_party/tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//third_party/tensorflow/compiler/mlir/tensorflow/transforms:verify_no_outside_compilation_markers_pass",
        "//third_party/tensorflow/core/protobuf:for_core_protos_cc",
        "//util/task:status",
    ],
)

cc_library(
    name = "graphnode_builder",
    srcs = ["graphnode_builder.cc"],
    hdrs = ["graphnode_builder.h"],
    compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        "//third_party/absl/status",
        "//third_party/absl/strings",
        "//third_party/absl/strings:string_view",
        "//third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/formats:schema_structs",
        "//util/task:status",
    ],
)

cc_test(
    name = "graphnode_builder_test",
    srcs = ["graphnode_builder_test.cc"],
    deps = [
        ":graphnode_builder",
        "//testing/base/public:gunit_main",
        "//third_party/absl/status",
        "//third_party/absl/strings",
        "//third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/formats:schema_structs",
    ],
)

cc_library(
    name = "visualize_config",
    hdrs = ["visualize_config.h"],
    #compatible_with = ["//buildenv/target:non_prod"],
)

cc_library(
    name = "direct_saved_model_to_json_graph_convert",
    srcs = ["direct_saved_model_to_json_graph_convert.cc"],
    hdrs = ["direct_saved_model_to_json_graph_convert.h"],
    deps = [
        ":graphnode_builder",
        ":visualize_config",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/container:flat_hash_set",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/absl/strings:string_view",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/protobuf:protobuf_lite",
        "//third_party/tensorflow/cc/saved_model:reader",
        "//third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/formats:schema_structs",
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:protos_all_cc",
        "//util/gtl:any_span",
        "//util/regexp/re2",
        "//util/task:status",
    ],
)

cc_library(
    name = "direct_flatbuffer_to_json_graph_convert",
    srcs = ["direct_flatbuffer_to_json_graph_convert.cc"],
    hdrs = ["direct_flatbuffer_to_json_graph_convert.h"],
    deps = [
        ":graphnode_builder",
        ":visualize_config",
        "//file/base",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/container:flat_hash_set",
        "//third_party/absl/log",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:string_view",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/tensorflow/compiler/mlir/lite:flatbuffer_tflite_operator_lib",
        "//third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/formats:schema_structs",
        "//third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/tools:attribute_printer",
        "//third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/tools:convert_type",
        "//third_party/tensorflow/lite/core:model_builder",
        "//third_party/tensorflow/lite/schema:schema_fbs",
        "//third_party/tensorflow/lite/schema:schema_utils",
        "//util/task:status",
    ],
)

cc_test(
    name = "direct_saved_model_to_json_graph_convert_test",
    srcs = ["direct_saved_model_to_json_graph_convert_test.cc"],
    deps = [
        ":direct_saved_model_to_json_graph_convert",
        ":visualize_config",
        "//file/base",
        "//file/base:options_cc",
        "//file/base:path",
        "//net/proto2/contrib/parse_proto:parse_text_proto",
        "//testing/base/public:gunit_main",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/strings:string_view",
        "//third_party/json",
        "//third_party/tensorflow/core:protos_all_cc",
        "//third_party/tensorflow/core/protobuf:for_core_protos_cc",
        "//util/math:mathutil",
    ],
)

# LLVM command line tool for translating mlir to visualizer json.
cc_binary(
    name = "tensor-graph-vis-translate",
    srcs = ["tensor_graph_vis_translate_main.cc"],
    deps = [
        ":translations",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:FuncDialect",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:MlirTranslateMain",
        "//third_party/llvm/llvm-project/mlir:QuantOps",
        "//third_party/llvm/llvm-project/mlir:Support",
        "//third_party/llvm/llvm-project/mlir:TranslateLib",
        "//third_party/stablehlo:stablehlo_ops",
        "//third_party/tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//third_party/tensorflow/compiler/mlir/tensorflow:tensorflow_types",
    ],
)

cc_binary(
    name = "models_to_json",
    srcs = ["models_to_json_main.cc"],
    deps = [
        ":direct_flatbuffer_to_json_graph_convert",
        ":model_json_graph_convert",
        ":visualize_config",
        "//file/base",
        "//file/base:options_cc",
        "//third_party/absl/log",
        "//third_party/absl/status:statusor",
        "//third_party/tensorflow/compiler/mlir:init_mlir",
        "//third_party/tensorflow/core:portable_gif_internal",
        "//third_party/tensorflow/lite/tools:command_line_flags",
    ],
)

# Standard file check tests setup.
# glob_lit_tests(
#     name = "all_tests",
#     data = [":test_utilities"],
#     driver = "//third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling:run_lit.sh",
#     test_file_exts = ["mlir"],
# )

exports_files(
    ["tflite-to-visualizer-json.sh"],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":tensor-graph-vis-translate",
        "//third_party/llvm/llvm-project/llvm:FileCheck",
        "//third_party/llvm/llvm-project/mlir:run_lit.sh",
        "//third_party/tensorflow/compiler/mlir/lite:flatbuffer_translate",
    ],
)
