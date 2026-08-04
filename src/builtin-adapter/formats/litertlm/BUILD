# Copyright 2026 The AI Edge Model Explorer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    srcs = ["litertlm_header_schema.fbs"],
)

flatbuffer_cc_library(
    name = "litertlm_header_schema",
    srcs = ["litertlm_header_schema.fbs"],
    flatc_args = [
        "--gen-mutable",
        "--gen-object-api",
    ],
)

cc_library(
    name = "litertlm_read",
    srcs = ["litertlm_read.cc"],
    hdrs = ["litertlm_read.h"],
    deps = [
        ":litertlm_header_schema",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers//:runtime_cc",
    ],
)
