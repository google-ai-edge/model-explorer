# Copyright 2026 The AI Edge Model Explorer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This package contains MLIR passes created and used by Model Explorer.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
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
    alwayslink = 1,
)

cc_library(
    name = "conversion",
    hdrs = ["conversion.h"],
    deps = [
        ":passes",
        ":unique_op_names",  # buildcleaner: keep
    ],
)
