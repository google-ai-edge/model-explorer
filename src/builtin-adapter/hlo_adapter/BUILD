load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "direct_hlo_to_json_graph_convert",
    srcs = ["direct_hlo_to_json_graph_convert.cc"],
    hdrs = ["direct_hlo_to_json_graph_convert.h"],
    compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "//:graphnode_builder",
        "//:status_macros",
        "//formats:schema_structs",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/hlo/ir:hlo",
        "@org_tensorflow//tensorflow/core/profiler/utils:hlo_module_utils",
    ],
)
