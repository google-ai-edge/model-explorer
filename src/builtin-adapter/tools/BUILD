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

cc_library(
    name = "convert_type",
    srcs = ["convert_type.cc"],
    hdrs = ["convert_type.h"],
    deps = ["@org_tensorflow//tensorflow/compiler/mlir/lite/schema:schema_fbs"],
)

cc_library(
    name = "load_opdefs",
    srcs = ["load_opdefs.cc"],
    hdrs = ["load_opdefs.h"],
    deps = ["@com_google_absl//absl/container:flat_hash_map"],
)
