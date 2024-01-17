# Tools that Model Explore backend utilizes to help parsing data.

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

# Attribute printer library.
cc_library(
    name = "attribute_printer",
    srcs = ["attribute_printer.cc"],
    hdrs = ["attribute_printer.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
