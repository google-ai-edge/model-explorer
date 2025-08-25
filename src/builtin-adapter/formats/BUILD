load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "schema_structs",
    srcs = ["schema_structs.cc"],
    hdrs = ["schema_structs.h"],
    deps = ["@llvm-project//llvm:Support"],
)
