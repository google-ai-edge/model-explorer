# This package contains MLIR passes created and used by Model Explorer.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//:__subpackages__",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    tbl_outs = {"passes.h.inc": [
        "-gen-pass-decls",
        "-name=VisualizationClient",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    hdrs = ["passes.h"],
    textual_hdrs = ["passes.h.inc"],
    deps = [
        ":passes_inc_gen",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "unique_op_names",
    srcs = ["unique_op_names.cc"],
    deps = [
        ":passes",
        ":passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "conversion",
    hdrs = ["conversion.h"],
    deps = [
        ":passes",
        ":unique_op_names",  # buildcleaner: keep
    ],
)
