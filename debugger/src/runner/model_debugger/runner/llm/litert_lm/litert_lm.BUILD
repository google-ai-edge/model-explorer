# LiteRT-LM for {platform}, built from commit {commit} by
# litert_lm_library.bzl. Only the Runner's LiteRT-LM package sees it.

load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["@model_debugger_runner//model_debugger/runner/llm/litert_lm:__pkg__"])

# The C API's headers, included as third_party/LiteRT-LM/c/<name>.h.
cc_library(
    name = "c_api",
    hdrs = [{headers}],
    include_prefix = "third_party/LiteRT-LM/c",
    strip_include_prefix = "c",
)

# The commit the library was built from, as litert_lm/version.h.
cc_library(
    name = "version",
    hdrs = ["version.h"],
    include_prefix = "litert_lm",
)

# The GPU plugins' names, in load order, as litert_lm/plugins.h.
cc_library(
    name = "plugins",
    hdrs = ["plugins.h"],
    include_prefix = "litert_lm",
)

# The runtime: exports the C API alone, with the debugger on. What it links
# travels with it, as imports, into an App's bundle.
cc_import(
    name = "litert_lm",
    shared_library = "{library}",
    deps = [{linked}],
)

# A placeholder for the cc_import of each library the runtime links.
# buildifier: disable=no-effect
{imports}

# The GPU plugins the runtime loads at run time from its own directory; an
# App bundles them beside the runtime: as files (macOS) or, where only linked
# libraries are packaged (Android), as imports.
filegroup(
    name = "gpu_libraries",
    srcs = [{plugins}],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gpu_library_imports",
    visibility = ["//visibility:public"],
    deps = [{plugin_labels}],
)

# A placeholder for the cc_import of each GPU plugin.
# buildifier: disable=no-effect
{plugin_imports}
