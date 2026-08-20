// Copyright 2026 The AI Edge Model Explorer Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "c/c_api.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace {
inline void EnsureInitialized() {}
}  // namespace

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "direct_flatbuffer_to_json_graph_convert.h"
#include "direct_saved_model_to_json_graph_convert.h"
#include "model_json_graph_convert.h"
#include "visualize_config.h"

namespace {

using ::tooling::visualization_client::VisualizeConfig;

VisualizeConfig ToCppConfig(const AdapterVisualizeConfig* c_config) {
  EnsureInitialized();
  VisualizeConfig config;
  if (c_config != nullptr) {
    config.const_element_count_limit = c_config->const_element_count_limit;
    config.add_tensor_name_attribute = c_config->add_tensor_name_attribute;
  }
  return config;
}

char* DuplicateString(absl::string_view str) {
  char* result = static_cast<char*>(malloc(str.size() + 1));
  if (result != nullptr) {
    memcpy(result, str.data(), str.size());
    result[str.size()] = '\0';
  }
  return result;
}

AdapterStatusCode HandleResult(
    const absl::StatusOr<std::string>& status_or_json, char** out_json,
    char** out_error_message) {
  EnsureInitialized();
  if (out_json == nullptr || out_error_message == nullptr) {
    return ADAPTER_STATUS_INVALID_ARGUMENT;
  }
  *out_json = nullptr;
  *out_error_message = nullptr;

  if (!status_or_json.ok()) {
    *out_error_message = DuplicateString(status_or_json.status().ToString());
    if (status_or_json.status().code() == absl::StatusCode::kNotFound) {
      return ADAPTER_STATUS_NOT_FOUND;
    }
    if (status_or_json.status().code() == absl::StatusCode::kInvalidArgument) {
      return ADAPTER_STATUS_INVALID_ARGUMENT;
    }
    return ADAPTER_STATUS_ERROR;
  }

  *out_json = DuplicateString(*status_or_json);
  return ADAPTER_STATUS_OK;
}

}  // namespace

extern "C" {

void AdapterGetDefaultConfig(AdapterVisualizeConfig* config) {
  if (config != nullptr) {
    VisualizeConfig default_config;
    config->const_element_count_limit =
        default_config.const_element_count_limit;
    config->add_tensor_name_attribute =
        default_config.add_tensor_name_attribute;
  }
}

void AdapterFreeString(char* str) {
  if (str != nullptr) {
    free(str);
  }
}

AdapterStatusCode AdapterConvertSavedModelToJson(
    const AdapterVisualizeConfig* config, const char* model_path,
    char** out_json, char** out_error_message) {
  if (model_path == nullptr) return ADAPTER_STATUS_INVALID_ARGUMENT;
  auto result = ::tooling::visualization_client::ConvertSavedModelToJson(
      ToCppConfig(config), model_path);
  return HandleResult(result, out_json, out_error_message);
}

AdapterStatusCode AdapterConvertFlatbufferToJson(
    const AdapterVisualizeConfig* config, const char* model_path,
    bool is_modelpath, char** out_json, char** out_error_message) {
  if (model_path == nullptr) return ADAPTER_STATUS_INVALID_ARGUMENT;
  auto result = ::tooling::visualization_client::ConvertFlatbufferToJson(
      ToCppConfig(config), model_path, is_modelpath);
  return HandleResult(result, out_json, out_error_message);
}

AdapterStatusCode AdapterConvertFlatbufferDirectlyToJson(
    const AdapterVisualizeConfig* config, const char* model_path,
    char** out_json, char** out_error_message) {
  if (model_path == nullptr) return ADAPTER_STATUS_INVALID_ARGUMENT;
  auto result =
      ::tooling::visualization_client::ConvertFlatbufferDirectlyToJson(
          ToCppConfig(config), model_path);
  return HandleResult(result, out_json, out_error_message);
}

AdapterStatusCode AdapterConvertSavedModelDirectlyToJson(
    const AdapterVisualizeConfig* config, const char* model_path,
    char** out_json, char** out_error_message) {
  if (model_path == nullptr) return ADAPTER_STATUS_INVALID_ARGUMENT;
  auto result =
      ::tooling::visualization_client::ConvertSavedModelDirectlyToJson(
          ToCppConfig(config), model_path);
  return HandleResult(result, out_json, out_error_message);
}

AdapterStatusCode AdapterConvertGraphDefDirectlyToJson(
    const AdapterVisualizeConfig* config, const char* model_path,
    char** out_json, char** out_error_message) {
  if (model_path == nullptr) return ADAPTER_STATUS_INVALID_ARGUMENT;
  auto result = ::tooling::visualization_client::ConvertGraphDefDirectlyToJson(
      ToCppConfig(config), model_path);
  return HandleResult(result, out_json, out_error_message);
}

AdapterStatusCode AdapterConvertMlirToJson(const AdapterVisualizeConfig* config,
                                           const char* model_path,
                                           char** out_json,
                                           char** out_error_message) {
  if (model_path == nullptr) return ADAPTER_STATUS_INVALID_ARGUMENT;
  auto result = ::tooling::visualization_client::ConvertMlirToJson(
      ToCppConfig(config), model_path);
  return HandleResult(result, out_json, out_error_message);
}

}  // extern "C"
