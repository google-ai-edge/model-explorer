load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "schema_structs",
    srcs = ["schema_structs.cc"],
    hdrs = ["schema_structs.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//llvm:Support",
    ],
)
