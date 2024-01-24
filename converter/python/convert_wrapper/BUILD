load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
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
        "@com_google_absl//absl/strings:string_view",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:import_status_module",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "//:model_json_graph_convert",
        "//:visualize_config",
    ],
)
