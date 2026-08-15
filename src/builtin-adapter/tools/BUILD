# Copyright 2026 The AI Edge Model Explorer Authors. All Rights Reserved.
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
# ==============================================================================

# Tools that Model Explore backend utilizes to help parsing data.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

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
        ":shardy_utils",
        "@com_google_absl//absl/log",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

# Utils for pretty printing Shardy attributes.
cc_library(
    name = "shardy_utils",
    srcs = ["shardy_utils.cc"],
    hdrs = ["shardy_utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "//:status_macros",
        "//formats:schema_structs",
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

cc_library(
    name = "namespace_heuristics",
    srcs = ["namespace_heuristics.cc"],
    hdrs = ["namespace_heuristics.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)
