load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "schema_structs",
    hdrs = ["schema_structs.h"],
    deps = ["@xla//xla/hlo/tools/hlo_diff/render/hlo_adapter:schema_structs"],
)
