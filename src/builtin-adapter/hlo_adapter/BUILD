load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "direct_hlo_to_json_graph_convert",
    hdrs = ["direct_hlo_to_json_graph_convert.h"],
    compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        "@xla//xla/hlo/tools/hlo_diff/render/hlo_adapter:direct_hlo_to_json_graph_convert",
    ],
)
