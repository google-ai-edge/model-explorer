load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//third_party/tensorflow:tensorflow.google.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//third_party/tensorflow:license"],  # copybara:comment
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

pybind_extension(
    name = "_pywrap_convert_wrapper",
    srcs = [
        "convert_wrapper.cc",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_convert_wrapper.pyi",
    ],
    deps = [
        "//third_party/absl/strings:string_view",
        "//third_party/pybind11",
        "//third_party/pybind11_abseil:import_status_module",
        "//third_party/pybind11_abseil:status_casters",
        "//third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling:model_json_graph_convert",
        "//third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling:visualize_config",
    ],
)

py_strict_test(
    name = "convert_wrapper_test",
    srcs = ["convert_wrapper_test.py"],
    data = [
        "//third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/tests:tf_simple_add",
        "//third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/tests:tfl_fully_connected",
    ],
    python_version = "PY3",
    deps = [
        ":_pywrap_convert_wrapper",
        "//pyglib:resources",
        "//third_party/pybind11_abseil:status",
        "//third_party/tensorflow/python/framework:test_lib",
        "//third_party/tensorflow/python/platform:client_testlib",
        "//third_party/tensorflow/python/platform:resource_loader",
    ],
)
