/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"
#include "pybind11_abseil/import_status_module.h"
#include "pybind11_abseil/status_casters.h"  // IWYU pragma : keep
#include "converter/model_json_graph_convert.h"
#include "converter/visualize_config.h"

using tooling::visualization_client::VisualizeConfig;

namespace pybind11 {

PYBIND11_MODULE(_pywrap_convert_wrapper, m) {
  pybind11::google::ImportStatusModule();

  class_<VisualizeConfig>(m, "VisualizeConfig")
      .def(init<>())
      .def_readwrite("const_element_count_limit",
                     &VisualizeConfig::const_element_count_limit);

  m.def(
      "ConvertSavedModelV1ToJson",
      [](const VisualizeConfig& config, absl::string_view model_path) {
        return ::tooling::visualization_client::ConvertSavedModelV1ToJson(
            config, model_path);
      },
      R"pbdoc(
      Converts a SavedModel v1 to visualizer JSON string if succeeded,
      otherwise raises `StatusNotOk` exception.
      )pbdoc");

  m.def(
      "ConvertSavedModelV2ToJson",
      [](const VisualizeConfig& config, absl::string_view model_path) {
        return ::tooling::visualization_client::ConvertSavedModelV2ToJson(
            config, model_path);
      },
      R"pbdoc(
      Converts a SavedModel v2 to visualizer JSON string if succeeded,
      otherwise raises `StatusNotOk` exception.
      )pbdoc");

  m.def(
      "ConvertFlatbufferToJson",
      [](const VisualizeConfig& config, absl::string_view model_path,
         bool is_modelpath) {
        return ::tooling::visualization_client::ConvertFlatbufferToJson(
            config, model_path, is_modelpath);
      },
      R"pbdoc(
      Converts a Flatbuffer to visualizer JSON string if succeeded,
      otherwise raises `StatusNotOk` exception.
      )pbdoc");
}

}  // namespace pybind11
