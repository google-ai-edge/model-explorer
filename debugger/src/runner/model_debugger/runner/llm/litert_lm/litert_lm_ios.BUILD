# LiteRT-LM for {platform}, built from commit {commit} by
# litert_lm_library.bzl. Only the Runner's LiteRT-LM package sees it.
#
# An iOS App bundles a dynamic library only as a framework, so each library
# is one: <name>.framework/<name>, found by the App's rpath in Frameworks/.

load("@rules_apple//apple:apple.bzl", "apple_dynamic_framework_import")
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

# The runtime: exports the C API alone, with the debugger on. It links the
# two below by @rpath; as frameworks they travel with it into the bundle.
apple_dynamic_framework_import(
    name = "litert_lm",
    framework_imports = glob(["liblitert-lm.framework/**"]),
    deps = [
        ":gemma_model_constraint_provider",
        ":litert",
    ],
)

apple_dynamic_framework_import(
    name = "litert",
    framework_imports = glob(["libLiteRt.framework/**"]),
)

apple_dynamic_framework_import(
    name = "gemma_model_constraint_provider",
    framework_imports = glob(["libGemmaModelConstraintProvider.framework/**"]),
)

# The GPU plugins the runtime loads at run time, as frameworks the App
# carries but does not link: the runtime opens them, from beside its own
# framework, only for a model on the GPU.
cc_library(
    name = "gpu_library_imports",
    visibility = ["//visibility:public"],
    deps = [{plugin_labels}],
)

# A placeholder for the framework import of each GPU plugin.
# buildifier: disable=no-effect
{plugin_framework_imports}
